namespace HybridSearch.Broker;

/// <summary>Root of the "Broker" configuration section (appsettings.json / env vars Broker__...).</summary>
public sealed class BrokerOptions
{
    public const string Section = "Broker";

    /// <summary>Data root; relative paths below resolve against it. Relative to the content root if itself relative.</summary>
    public string DataRoot { get; set; } = "../../data";

    public TopologyOptions Topology { get; set; } = new();
    public SearchOptions Search { get; set; } = new();
    public HedgingOptions Hedging { get; set; } = new();
    public DegradationOptions Degradation { get; set; } = new();
    public ModelOptions Models { get; set; } = new();
    public AutocompleteOptions Autocomplete { get; set; } = new();
    public EventOptions Events { get; set; } = new();
    public ExperimentOptions Experiments { get; set; } = new();
    public HealthOptions Health { get; set; } = new();
    public CorsOptions Cors { get; set; } = new();
}

public sealed class TopologyOptions
{
    /// <summary>One entry per slice (disjoint doc-ID range); each lists ≥1 replica endpoint.</summary>
    public List<SliceOptions> Slices { get; set; } = [];
}

public sealed class SliceOptions
{
    public int Id { get; set; }
    public List<string> Replicas { get; set; } = [];
}

public sealed class SearchOptions
{
    public int DefaultK { get; set; } = 10;
    public int MaxK { get; set; } = 100;
    public int MaxQueryChars { get; set; } = 512;
    public int DefaultDeadlineMs { get; set; } = 300;
    public int MinDeadlineMs { get; set; } = 10;
    public int MaxDeadlineMs { get; set; } = 10_000;
    /// <summary>Per-retriever candidates requested from every slice (and kept after the merge).</summary>
    public int CandidateDepth { get; set; } = 100;
    public int RerankDepth { get; set; } = 50;
    public string Fusion { get; set; } = "rrf";
    public int RrfK { get; set; } = 60;
    public double Alpha { get; set; } = 0.5;
    public string Normalization { get; set; } = "minmax";
    public int SnippetChars { get; set; } = 240;
    /// <summary>Dense beam width (L_search) sent to shards; 0 = shard default.</summary>
    public uint DenseBeamWidth { get; set; }
    /// <summary>Beam width used when the degradation policy shrinks the dense beam.</summary>
    public uint DenseBeamWidthShrunk { get; set; } = 32;
    /// <summary>Time kept back from each shard's budget so a partial answer can travel back before the gRPC deadline.</summary>
    public double ShardGraceMs { get; set; } = 3;
    /// <summary>Snippet fetches get at least this long even if the search budget is spent (a result without text is useless).</summary>
    public int MinFetchMs { get; set; } = 30;
}

public sealed class HedgingOptions
{
    public bool Enabled { get; set; } = true;
    /// <summary>Hedge when the primary has not answered by this latency quantile of the slice.</summary>
    public double Quantile { get; set; } = 0.95;
    /// <summary>Token-bucket refill per primary request: hedges ≤ BudgetRatio × primaries (+ nothing: bucket starts empty).</summary>
    public double BudgetRatio { get; set; } = 0.05;
    /// <summary>Maximum banked hedge tokens (limits bursts after a quiet period).</summary>
    public double BurstTokens { get; set; } = 10;
    /// <summary>No hedging until the slice has this many latency samples in the window.</summary>
    public int MinSamples { get; set; } = 50;
    public double MinDelayMs { get; set; } = 1;
    public int WindowSeconds { get; set; } = 60;
    /// <summary>Retry immediately on another replica when an attempt fails fast (not budgeted: the failed attempt did no work).</summary>
    public bool FailoverOnError { get; set; } = true;
}

public sealed class DegradationOptions
{
    /// <summary>Quantile of recorded stage latencies used to plan whether a stage fits the remaining budget.</summary>
    public double PlanningQuantile { get; set; } = 0.9;
    public int MinSamples { get; set; } = 20;
    /// <summary>Prior cost estimates used until enough samples exist.</summary>
    public double DefaultEncodeMs { get; set; } = 15;
    public double DefaultShardFullMs { get; set; } = 40;
    public double DefaultShardShrunkMs { get; set; } = 25;
    public double DefaultShardLexicalMs { get; set; } = 15;
    public double DefaultRerankPerDocMs { get; set; } = 1.5;
    public double DefaultFetchMs { get; set; } = 5;
    /// <summary>In-flight search counts at which load alone forces level 1, 2, 3. Empty = no load shedding.</summary>
    public List<int> InFlightThresholds { get; set; } = []; // appsettings.json ships [64, 128, 256]; list defaults here would be appended to by the binder
}

public sealed class ModelOptions
{
    /// <summary>"onnx" (default) or "hashing-dev" (a feature-hashing stand-in for local UI work with FakeShard; never a real model).</summary>
    public string QueryEncoderKind { get; set; } = "onnx";
    public string QueryEncoderDir { get; set; } = "models/query-encoder";
    public string RerankerDir { get; set; } = "models/reranker";
    public bool PreferInt8Reranker { get; set; } = true;
    public string QueryPrefix { get; set; } = "Represent this sentence for searching relevant passages: ";
    public int QueryMaxTokens { get; set; } = 64;
    public int RerankMaxTokens { get; set; } = 256;
    public int RerankBatchSize { get; set; } = 16;
    public int IntraOpThreads { get; set; }
    public int HashingDevDimension { get; set; } = 64;
}

public sealed class AutocompleteOptions
{
    public string QueryLogPath { get; set; } = "raw/msmarco/queries.train.tsv";
    public int TopK { get; set; } = 10;
    public bool SpellCorrection { get; set; } = true;
    public long SpellMinWordFrequency { get; set; } = 3;
}

public sealed class EventOptions
{
    public string Directory { get; set; } = "events";
    public int QueueCapacity { get; set; } = 10_000;
    public int MaxBatch { get; set; } = 100;
    public bool LogServedResults { get; set; } = true;
}

public sealed class ExperimentOptions
{
    public string ConfigPath { get; set; } = "experiments.json";
    public int BootstrapIterations { get; set; } = 2000;
}

public sealed class HealthOptions
{
    public int IntervalMs { get; set; } = 2000;
    public int TimeoutMs { get; set; } = 1000;
}

public sealed class CorsOptions
{
    public List<string> AllowedOrigins { get; set; } = []; // empty → http://localhost:5173 (PostConfigure in Program.cs)
}

public static class PathResolver
{
    public static string DataRoot(BrokerOptions o, string contentRoot) =>
        Path.GetFullPath(Path.IsPathRooted(o.DataRoot) ? o.DataRoot : Path.Combine(contentRoot, o.DataRoot));

    public static string UnderData(BrokerOptions o, string contentRoot, string path) =>
        Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(DataRoot(o, contentRoot), path));
}
