using System.IO.Hashing;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HybridSearch.Broker.Experiments;

public enum ExperimentKind { Ab, Interleave, Aa }

/// <summary>One ranker configuration (an experiment arm). Null fields fall back to the request / server defaults.</summary>
public sealed record RankerConfig
{
    public string? Mode { get; init; }
    public bool? Rerank { get; init; }
    public int? RerankDepth { get; init; }
    public string? Fusion { get; init; }          // "rrf" | "weighted"
    public int? RrfK { get; init; }
    public double? Alpha { get; init; }
    public string? Normalization { get; init; }   // "minmax" | "zscore"
}

public sealed record ExperimentDefinition
{
    public required string Id { get; init; }
    public required string Kind { get; init; }           // "ab" | "interleave" | "aa"
    public double Allocation { get; init; }
    public required string Salt { get; init; }
    /// <summary>"running" (default) or "paused". Paused experiments enroll nobody but keep their results.</summary>
    public string Status { get; init; } = "running";
    public string? Description { get; init; }
    /// <summary>
    /// For kind "aa" only: "ab" (default, split sessions over two identical arms) or "interleave"
    /// (team-draft interleave a config with itself; credit should split evenly).
    /// </summary>
    public string? Method { get; init; }
    public RankerConfig Control { get; init; } = new();
    public RankerConfig? Treatment { get; init; }

    [JsonIgnore]
    public ExperimentKind ParsedKind => Kind.ToLowerInvariant() switch
    {
        "ab" => ExperimentKind.Ab,
        "interleave" => ExperimentKind.Interleave,
        "aa" => ExperimentKind.Aa,
        _ => throw new InvalidOperationException($"experiment {Id}: unknown kind '{Kind}'"),
    };

    [JsonIgnore] public bool IsRunning => string.Equals(Status, "running", StringComparison.OrdinalIgnoreCase);

    /// <summary>True when served results are a team-draft interleaving of the two arms.</summary>
    [JsonIgnore]
    public bool Interleaves => ParsedKind == ExperimentKind.Interleave ||
                               (ParsedKind == ExperimentKind.Aa && string.Equals(Method, "interleave", StringComparison.OrdinalIgnoreCase));

    /// <summary>A/A uses the control config for both arms, whatever the file says.</summary>
    [JsonIgnore] public RankerConfig EffectiveTreatment => ParsedKind == ExperimentKind.Aa ? Control : Treatment ?? Control;
}

public sealed record ExperimentsFile
{
    public List<ExperimentDefinition> Experiments { get; init; } = [];

    public static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static ExperimentsFile Parse(string json)
    {
        var file = JsonSerializer.Deserialize<ExperimentsFile>(json, JsonOptions) ?? new ExperimentsFile();
        file.Validate();
        return file;
    }

    public void Validate()
    {
        var ids = new HashSet<string>(StringComparer.Ordinal);
        foreach (var e in Experiments)
        {
            if (string.IsNullOrWhiteSpace(e.Id)) throw new InvalidOperationException("experiment id is required");
            if (!ids.Add(e.Id)) throw new InvalidOperationException($"duplicate experiment id '{e.Id}'");
            _ = e.ParsedKind;
            if (e.Allocation is < 0 or > 1 || double.IsNaN(e.Allocation))
                throw new InvalidOperationException($"experiment {e.Id}: allocation must be in [0,1]");
            if (string.IsNullOrEmpty(e.Salt)) throw new InvalidOperationException($"experiment {e.Id}: salt is required");
            if (e.ParsedKind != ExperimentKind.Aa && e.Treatment is null)
                throw new InvalidOperationException($"experiment {e.Id}: treatment is required for kind '{e.Kind}'");
        }
    }
}

public enum Variant { Control, Treatment }

public sealed record Assignment(ExperimentDefinition Experiment, Variant Variant, uint Bucket)
{
    public string VariantName => Variant == Variant.Control ? "control" : "treatment";
}

/// <summary>
/// Deterministic, sticky assignment (docs/ARCHITECTURE.md, Experiments):
/// <c>bucket = xxhash64(salt + sessionId) mod 10000</c>; enrolled iff <c>bucket &lt; allocation·10000</c>.
/// The arm comes from a second, independent hash: xxhash64 of the same bytes with seed
/// <see cref="ArmSeed"/>, mod 2 (0 = control). Using a different hash for the arm means the arm
/// split inside the enrolled population is not correlated with the enrollment bucket, so
/// raising the allocation later never moves an already-enrolled session to the other arm.
/// </summary>
public static class ExperimentAssigner
{
    public const int Buckets = 10_000;
    public const ulong ArmSeed = 0x5EED_A5B1_0000_0001UL;

    public static uint Bucket(string salt, string sessionId) =>
        (uint)(XxHash64.HashToUInt64(Encoding.UTF8.GetBytes(salt + sessionId)) % Buckets);

    public static bool IsEnrolled(double allocation, uint bucket) => bucket < (uint)Math.Round(allocation * Buckets);

    public static Variant Arm(string salt, string sessionId) =>
        XxHash64.HashToUInt64(Encoding.UTF8.GetBytes(salt + sessionId), unchecked((long)ArmSeed)) % 2 == 0
            ? Variant.Control
            : Variant.Treatment;

    /// <summary>
    /// First running experiment (in file order) that enrolls this session, or null. Experiments
    /// are not layered: a session is in at most one experiment at a time, which keeps the
    /// analysis free of interaction effects at the cost of traffic.
    /// </summary>
    public static Assignment? Assign(IReadOnlyList<ExperimentDefinition> experiments, string? sessionId)
    {
        if (string.IsNullOrEmpty(sessionId)) return null;
        foreach (var e in experiments)
        {
            if (!e.IsRunning) continue;
            var bucket = Bucket(e.Salt, sessionId);
            if (!IsEnrolled(e.Allocation, bucket)) continue;
            return new Assignment(e, Arm(e.Salt, sessionId), bucket);
        }
        return null;
    }

    /// <summary>Per-query interleaving seed: stable for the same session + query (a reload shows the same page).</summary>
    public static ulong InterleaveSeed(string salt, string sessionId, string query) =>
        XxHash64.HashToUInt64(Encoding.UTF8.GetBytes($"{salt}\u001f{sessionId}\u001f{query}"));
}
