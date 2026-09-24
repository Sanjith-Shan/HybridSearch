using System.Diagnostics;
using Prometheus;

namespace HybridSearch.Broker.Observability;

/// <summary>
/// Prometheus metrics. Names and labels are a contract with deploy/prometheus/slo_rules.yml and
/// the Grafana dashboard — do not rename. Each broker instance owns its own registry, so
/// several in-process brokers (integration tests) never share counters.
/// </summary>
public sealed class BrokerMetrics
{
    public static readonly double[] LatencyBuckets = [.005, .01, .025, .05, .1, .2, .3, .5, 1, 2.5];
    private static readonly double[] FastBuckets = [.000_001, .000_002, .000_005, .000_01, .000_025, .000_05, .000_1, .000_25, .000_5, .001, .005];

    public CollectorRegistry Registry { get; }

    public Counter SearchRequests { get; }
    public Histogram SearchDuration { get; }
    public Counter ShardRequests { get; }
    public Histogram ShardDuration { get; }
    public Counter ShardFailures { get; }
    public Counter ShardPartial { get; }
    public Counter HedgesSent { get; }
    public Counter HedgesWon { get; }
    public Counter HedgesDenied { get; }
    public Counter ShardPrimaryRequests { get; }
    public Counter ShardFailovers { get; }
    public Gauge HedgeExtraLoadRatio { get; }
    public Counter Degradation { get; }
    public Histogram RerankDuration { get; }
    public Histogram EncodeDuration { get; }
    public Counter EventsIngested { get; }
    public Counter EventsDropped { get; }
    public Counter EventsWritten { get; }
    public Histogram SuggestDuration { get; }
    public Counter ExperimentAssignments { get; }
    public Gauge InFlightSearches { get; }

    public BrokerMetrics()
    {
        Registry = Metrics.NewCustomRegistry();
        var f = Metrics.WithCustomRegistry(Registry);

        SearchRequests = f.CreateCounter("hs_search_requests_total", "Search requests by mode and outcome (ok | error | invalid).", "mode", "outcome");
        SearchDuration = f.CreateHistogram("hs_search_duration_seconds", "End-to-end search latency.",
            new HistogramConfiguration { LabelNames = ["mode"], Buckets = LatencyBuckets });
        ShardRequests = f.CreateCounter("hs_shard_requests_total", "Search RPC attempts sent to a shard replica (primaries, hedges and failovers).", "shard", "replica");
        ShardDuration = f.CreateHistogram("hs_shard_duration_seconds", "Time until a slice answered (first successful attempt), as the broker saw it.",
            new HistogramConfiguration { LabelNames = ["shard"], Buckets = LatencyBuckets });
        ShardFailures = f.CreateCounter("hs_shard_failures_total", "Failed shard attempts (reason = timeout | error | unavailable). Cancelled hedge losers are not failures.", "shard", "reason");
        ShardPartial = f.CreateCounter("hs_shard_partial_total", "Slice answers cut short by the deadline (partial top-k).", "shard");
        HedgesSent = f.CreateCounter("hs_hedges_sent_total", "Hedge requests sent (each one is extra load).", "shard");
        HedgesWon = f.CreateCounter("hs_hedges_won_total", "Hedge requests that answered before the primary.", "shard");
        HedgesDenied = f.CreateCounter("hs_hedges_denied_total", "Hedges wanted (primary slower than p95) but refused by the load budget.", "shard");
        ShardPrimaryRequests = f.CreateCounter("hs_shard_primary_requests_total", "Primary (non-hedge) slice requests; denominator of the hedge extra-load ratio.", "shard");
        ShardFailovers = f.CreateCounter("hs_shard_failovers_total", "Immediate retries on another replica after a fast failure.", "shard");
        HedgeExtraLoadRatio = f.CreateGauge("hs_hedge_extra_load_ratio", "hedges_sent / primary requests since start, per slice.", "shard");
        Degradation = f.CreateCounter("hs_degradation_total", "Degradation steps applied (reranker_skipped | dense_beam_shrunk | lexical_only).", "step");
        RerankDuration = f.CreateHistogram("hs_rerank_duration_seconds", "Cross-encoder rerank latency (whole cascade stage).",
            new HistogramConfiguration { Buckets = LatencyBuckets });
        EncodeDuration = f.CreateHistogram("hs_encode_duration_seconds", "Query encoder latency.",
            new HistogramConfiguration { Buckets = LatencyBuckets });
        EventsIngested = f.CreateCounter("hs_events_ingested_total", "UI events accepted into the event log queue.", "type");
        EventsDropped = f.CreateCounter("hs_events_dropped_total", "UI events dropped (reason = invalid | queue_full | write_error).", "reason");
        EventsWritten = f.CreateCounter("hs_events_written_total", "Event log lines written to disk (UI events and server 'served' records).");
        SuggestDuration = f.CreateHistogram("hs_suggest_duration_seconds", "Autocomplete lookup latency (trie only).",
            new HistogramConfiguration { Buckets = FastBuckets });
        ExperimentAssignments = f.CreateCounter("hs_experiment_assignments_total", "Searches served under an experiment arm.", "experiment", "variant");
        InFlightSearches = f.CreateGauge("hs_search_inflight", "Searches currently in flight.");
    }

    /// <summary>
    /// Creates zero-valued children for every known label value, so series exist (and rate() works)
    /// before the first event, and dashboards never show "no data" for a healthy zero.
    /// </summary>
    public void Initialize(IEnumerable<string> shards, IEnumerable<(string Experiment, string Variant)> experimentArms)
    {
        foreach (var s in shards)
        {
            ShardPartial.WithLabels(s);
            HedgesSent.WithLabels(s);
            HedgesWon.WithLabels(s);
            HedgesDenied.WithLabels(s);
            ShardPrimaryRequests.WithLabels(s);
            ShardFailovers.WithLabels(s);
            HedgeExtraLoadRatio.WithLabels(s);
            foreach (var reason in new[] { "timeout", "error", "unavailable" }) ShardFailures.WithLabels(s, reason);
        }
        foreach (var step in new[] { "reranker_skipped", "dense_beam_shrunk", "lexical_only" }) Degradation.WithLabels(step);
        foreach (var t in new[] { "impression", "click", "dwell", "query", "abandon" }) EventsIngested.WithLabels(t);
        foreach (var r in new[] { "invalid", "queue_full", "write_error" }) EventsDropped.WithLabels(r);
        foreach (var (e, v) in experimentArms) ExperimentAssignments.WithLabels(e, v);
    }

    public static double Seconds(long startTimestamp) => Stopwatch.GetElapsedTime(startTimestamp).TotalSeconds;
}
