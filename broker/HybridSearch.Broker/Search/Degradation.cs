using HybridSearch.Broker.Shards;

namespace HybridSearch.Broker.Search;

/// <summary>Degradation steps, in the order they are applied. The numeric value is the response's <c>degradation.level</c>.</summary>
public enum DegradationKind
{
    RerankerSkipped = 1,
    DenseBeamShrunk = 2,
    LexicalOnly = 3,
}

public readonly record struct DegradationStep(DegradationKind Kind, string Reason)
{
    public string KindName => Kind switch
    {
        DegradationKind.RerankerSkipped => "reranker_skipped",
        DegradationKind.DenseBeamShrunk => "dense_beam_shrunk",
        DegradationKind.LexicalOnly => "lexical_only",
        _ => "unknown",
    };

    /// <summary>Wire form, e.g. "reranker_skipped:deadline".</summary>
    public override string ToString() => $"{KindName}:{Reason}";
}

/// <summary>What the caller asked for (after experiment overrides).</summary>
public sealed record RequestedPlan(bool Lexical, bool Dense, bool Rerank);

/// <summary>What is actually available right now.</summary>
public sealed record Capabilities(bool EncoderAvailable, bool VectorIndexReady, bool RerankerAvailable);

/// <summary>Estimated stage costs in milliseconds, from recent history (or priors).</summary>
public sealed record CostEstimates(
    double EncodeMs, double ShardFullMs, double ShardShrunkMs, double ShardLexicalMs, double RerankMs, double FetchMs);

/// <summary>The decided plan. <see cref="Level"/> = max step level (0 when nothing was degraded).</summary>
public sealed record ExecutionPlan(bool Lexical, bool Dense, bool ShrinkBeam, bool Rerank, IReadOnlyList<DegradationStep> Steps)
{
    public int Level => Steps.Count == 0 ? 0 : Steps.Max(s => (int)s.Kind);
}

/// <summary>
/// The degradation policy, as a pure function. Order is fixed: skip the reranker, then shrink
/// the dense beam, then answer lexical-only. The ordering reflects cost per unit of quality:
/// the cross-encoder is the most expensive stage (tens of ms per query on CPU) and only
/// re-orders a candidate set that already exists; a smaller beam keeps dense recall mostly
/// intact for a fraction of the SSD reads; dropping dense retrieval loses the most quality
/// (on MS MARCO dev, BM25 is ~half of BGE's MRR@10), so it is the last resort.
///
/// Budget-driven degradation is cumulative (level 2 implies the reranker is also skipped).
/// Capability-driven steps (a model file missing, a vector index not loaded) are recorded
/// independently: with no query encoder the reranker still runs on BM25 candidates if the budget allows.
/// </summary>
public static class DegradationPolicy
{
    public static ExecutionPlan Plan(RequestedPlan req, Capabilities caps, double remainingMs, CostEstimates est, int pressureLevel)
    {
        var steps = new List<DegradationStep>();
        bool dense = req.Dense, rerank = req.Rerank, shrink = false;
        bool lexical = req.Lexical; // dense-only requests fall back to lexical when dense is dropped (see return)

        if (dense && !caps.EncoderAvailable)
        {
            dense = false;
            steps.Add(new DegradationStep(DegradationKind.LexicalOnly, "encoder_unavailable"));
        }
        else if (dense && !caps.VectorIndexReady)
        {
            dense = false;
            steps.Add(new DegradationStep(DegradationKind.LexicalOnly, "vector_index_unavailable"));
        }
        if (rerank && !caps.RerankerAvailable)
        {
            rerank = false;
            steps.Add(new DegradationStep(DegradationKind.RerankerSkipped, "model_unavailable"));
        }

        // Least-degraded level (≥ the load-shedding floor) whose estimated cost fits the budget.
        int floor = Math.Clamp(pressureLevel, 0, 3);
        int chosen = 3;
        for (int level = floor; level <= 3; level++)
        {
            if (level == 3 || Cost(level, dense, rerank, est) <= remainingMs) { chosen = level; break; }
        }
        string ReasonFor(int level) => level <= floor ? "load" : "deadline";

        if (chosen >= 1 && rerank)
        {
            rerank = false;
            steps.Add(new DegradationStep(DegradationKind.RerankerSkipped, ReasonFor(1)));
        }
        if (chosen >= 2 && dense && chosen < 3)
        {
            shrink = true;
            steps.Add(new DegradationStep(DegradationKind.DenseBeamShrunk, ReasonFor(2)));
        }
        if (chosen >= 3 && dense)
        {
            dense = false;
            steps.Add(new DegradationStep(DegradationKind.LexicalOnly, ReasonFor(3)));
        }

        // Keep steps in policy order for the response, whatever produced them.
        steps.Sort((a, b) => a.Kind.CompareTo(b.Kind));
        return new ExecutionPlan(lexical || !dense, dense, shrink, rerank, steps);
    }

    /// <summary>Estimated wall time of a plan at a given degradation level.</summary>
    public static double Cost(int level, bool dense, bool rerank, CostEstimates e)
    {
        bool useDense = dense && level < 3;
        double shard = !useDense ? e.ShardLexicalMs : level >= 2 ? e.ShardShrunkMs : e.ShardFullMs;
        return (useDense ? e.EncodeMs : 0) + shard + (rerank && level < 1 ? e.RerankMs : 0) + e.FetchMs;
    }

    /// <summary>Load-shedding floor from the number of searches in flight: one level per threshold crossed.</summary>
    public static int PressureLevel(long inFlight, IReadOnlyList<int> thresholds)
    {
        int level = 0;
        for (int i = 0; i < thresholds.Count && i < 3; i++)
            if (inFlight > thresholds[i]) level = i + 1;
        return level;
    }
}

/// <summary>Rolling stage-cost history feeding <see cref="CostEstimates"/>.</summary>
public sealed class CostModel(DegradationOptions options)
{
    private static LatencyTracker New() => new(TimeSpan.FromSeconds(60));

    public LatencyTracker Encode { get; } = New();
    public LatencyTracker ShardFull { get; } = New();
    public LatencyTracker ShardShrunk { get; } = New();
    public LatencyTracker ShardLexical { get; } = New();
    /// <summary>Recorded as µs per reranked candidate, so estimates scale with rerank depth.</summary>
    public LatencyTracker RerankPerDoc { get; } = New();
    public LatencyTracker Fetch { get; } = New();

    public CostEstimates Estimate(int rerankDepth)
    {
        double Q(LatencyTracker t, double fallback) =>
            t.Quantile(options.PlanningQuantile, options.MinSamples)?.TotalMilliseconds ?? fallback;
        return new CostEstimates(
            Q(Encode, options.DefaultEncodeMs),
            Q(ShardFull, options.DefaultShardFullMs),
            Q(ShardShrunk, options.DefaultShardShrunkMs),
            Q(ShardLexical, options.DefaultShardLexicalMs),
            Q(RerankPerDoc, options.DefaultRerankPerDocMs) * rerankDepth,
            Q(Fetch, options.DefaultFetchMs));
    }
}
