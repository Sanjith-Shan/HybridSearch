using System.Diagnostics;
using Grpc.Core;
using HybridSearch.Broker.Observability;
using HybridSearch.Proto.V1;
using Microsoft.Extensions.Options;

namespace HybridSearch.Broker.Shards;

public enum SliceStatus { Ok, Partial, Failed }

/// <summary>What one slice contributed to a query.</summary>
public sealed record SliceOutcome(
    int SliceId,
    SliceStatus Status,
    SearchResponse? Response,
    bool HedgeWon,
    string? FailureReason,
    TimeSpan Elapsed,
    int Attempts);

internal enum AttemptKind { Primary, Hedge, Failover }

internal enum AttemptStatus { Success, Timeout, Unavailable, Error, Cancelled }

internal sealed record AttemptResult(AttemptStatus Status, SearchResponse? Response, TimeSpan Elapsed, string? Detail);

/// <summary>
/// Parallel fan-out to every slice under a deadline, with hedged requests.
///
/// Per slice: send to one replica; if no reply by the slice's rolling p95 (and the hedge budget
/// has a token, and another replica exists) send the same request to a second replica; take the
/// first success and cancel the other (gRPC cancellation reaches the loser's server). The
/// remaining budget is propagated both in <c>deadline_budget_us</c> (minus a small grace, so the
/// shard can return a partial top-k before the call's own deadline) and as the gRPC deadline.
/// </summary>
public sealed class ShardFanOut(
    ShardTopology topology,
    IOptions<BrokerOptions> options,
    BrokerMetrics metrics,
    ILogger<ShardFanOut> logger)
{
    private readonly HedgingOptions _hedging = options.Value.Hedging;
    private readonly SearchOptions _search = options.Value.Search;

    public ShardTopology Topology => topology;

    public Task<SliceOutcome[]> SearchAsync(SearchRequest template, Deadline deadline, CancellationToken ct) =>
        Task.WhenAll(topology.Slices.Select(s => SearchSliceAsync(s, template, deadline, ct)));

    /// <summary>The delay after which a hedge would be sent, or null if this slice cannot hedge right now.</summary>
    public TimeSpan? HedgeDelay(SliceState slice)
    {
        if (!_hedging.Enabled || slice.Replicas.Count < 2) return null;
        var q = slice.Tracker.Quantile(_hedging.Quantile, _hedging.MinSamples);
        if (q is null) return null;
        var min = TimeSpan.FromMilliseconds(_hedging.MinDelayMs);
        return q.Value < min ? min : q.Value;
    }

    private sealed class Attempt(ReplicaState replica, AttemptKind kind, CancellationTokenSource cts)
    {
        public ReplicaState Replica { get; } = replica;
        public AttemptKind Kind { get; } = kind;
        public CancellationTokenSource Cts { get; } = cts;
        public long StartTimestamp { get; } = Stopwatch.GetTimestamp();
        public Task<AttemptResult> Task { get; set; } = null!;
        public bool Done { get; set; }
    }

    private async Task<SliceOutcome> SearchSliceAsync(SliceState slice, SearchRequest template, Deadline deadline, CancellationToken ct)
    {
        long start = Stopwatch.GetTimestamp();
        using var sliceCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var attempts = new List<Attempt>(2);

        Attempt Start(ReplicaState replica, AttemptKind kind)
        {
            var cts = CancellationTokenSource.CreateLinkedTokenSource(sliceCts.Token);
            var a = new Attempt(replica, kind, cts);
            a.Task = RunAttemptAsync(slice, replica, kind, template, deadline, cts.Token);
            attempts.Add(a);
            return a;
        }

        slice.CountPrimary();
        metrics.ShardPrimaryRequests.WithLabels(slice.Label).Inc();
        Start(slice.PickPrimary(), AttemptKind.Primary);

        var hedgeDelay = HedgeDelay(slice);
        Task? hedgeTimer = hedgeDelay is { } d && d < deadline.Remaining ? Task.Delay(d, sliceCts.Token) : null;
        bool failoverUsed = false;
        AttemptResult? lastFailure = null;

        try
        {
            while (true)
            {
                var pending = new List<Task>(3);
                foreach (var a in attempts) if (!a.Done) pending.Add(a.Task);

                if (pending.Count == 0)
                {
                    // Everything in flight failed. One immediate failover if the failure was fast.
                    if (!failoverUsed && _hedging.FailoverOnError && lastFailure is { Status: AttemptStatus.Unavailable or AttemptStatus.Error }
                        && !deadline.Expired && slice.PickAlternate(attempts.Select(x => x.Replica).ToList()) is { } alt)
                    {
                        failoverUsed = true;
                        metrics.ShardFailovers.WithLabels(slice.Label).Inc();
                        Start(alt, AttemptKind.Failover);
                        continue;
                    }
                    return new SliceOutcome(slice.Id, SliceStatus.Failed, null, false,
                        lastFailure is null ? "unknown" : Reason(lastFailure), Stopwatch.GetElapsedTime(start), attempts.Count);
                }

                if (hedgeTimer is not null) pending.Add(hedgeTimer);
                var done = await Task.WhenAny(pending).ConfigureAwait(false);

                if (done == hedgeTimer)
                {
                    hedgeTimer = null; // at most one hedge per slice request
                    if (done.IsCanceled) continue;
                    var inFlight = attempts.Where(a => !a.Done).Select(a => a.Replica).ToList();
                    if (inFlight.Count == 0 || deadline.Expired) continue;
                    var alt = slice.PickAlternate(attempts.Select(a => a.Replica).ToList());
                    if (alt is null) continue;
                    if (slice.Budget.TryAcquire())
                    {
                        slice.CountHedge();
                        metrics.HedgesSent.WithLabels(slice.Label).Inc();
                        Start(alt, AttemptKind.Hedge);
                    }
                    else
                    {
                        metrics.HedgesDenied.WithLabels(slice.Label).Inc();
                    }
                    metrics.HedgeExtraLoadRatio.WithLabels(slice.Label).Set(slice.ExtraLoadRatio);
                    continue;
                }

                var finished = attempts.First(a => a.Task == done);
                finished.Done = true;
                var result = await finished.Task.ConfigureAwait(false); // attempts never throw

                if (result.Status == AttemptStatus.Success)
                {
                    foreach (var other in attempts)
                    {
                        if (other.Done) continue;
                        other.Cts.Cancel();
                        // A primary that lost to a hedge is right-censored at ≥ the hedge delay (≥ p95).
                        // Recording its elapsed time keeps the tail mass in the histogram; dropping it
                        // would make the observed p95 fall, hedges fire earlier, and the budget drain.
                        if (other.Kind == AttemptKind.Primary)
                            slice.Tracker.Record(Stopwatch.GetElapsedTime(other.StartTimestamp));
                    }
                    slice.Tracker.Record(result.Elapsed);
                    var elapsed = Stopwatch.GetElapsedTime(start);
                    metrics.ShardDuration.WithLabels(slice.Label).Observe(elapsed.TotalSeconds);
                    bool hedgeWon = finished.Kind == AttemptKind.Hedge;
                    if (hedgeWon) metrics.HedgesWon.WithLabels(slice.Label).Inc();
                    bool partial = result.Response!.Partial;
                    if (partial) metrics.ShardPartial.WithLabels(slice.Label).Inc();
                    return new SliceOutcome(slice.Id, partial ? SliceStatus.Partial : SliceStatus.Ok, result.Response,
                        hedgeWon, null, elapsed, attempts.Count);
                }

                if (result.Status == AttemptStatus.Timeout) slice.Tracker.Record(result.Elapsed);
                if (result.Status != AttemptStatus.Cancelled) lastFailure = result;
            }
        }
        finally
        {
            // Stops the hedge timer and any attempt still running (e.g. the caller gave up).
            await sliceCts.CancelAsync().ConfigureAwait(false);
            foreach (var a in attempts) a.Cts.Dispose();
        }
    }

    private static string Reason(AttemptResult r) => r.Status switch
    {
        AttemptStatus.Timeout => "timeout",
        AttemptStatus.Unavailable => "unavailable",
        _ => "error",
    };

    private async Task<AttemptResult> RunAttemptAsync(
        SliceState slice, ReplicaState replica, AttemptKind kind, SearchRequest template, Deadline deadline, CancellationToken ct)
    {
        long t0 = Stopwatch.GetTimestamp();
        var remaining = deadline.Remaining;
        if (remaining <= TimeSpan.Zero)
        {
            metrics.ShardFailures.WithLabels(slice.Label, "timeout").Inc();
            return new AttemptResult(AttemptStatus.Timeout, null, TimeSpan.Zero, "no budget left");
        }

        using var activity = Telemetry.Source.StartActivity(
            kind == AttemptKind.Hedge ? Telemetry.ShardHedgeSpan : Telemetry.ShardSearchSpan, ActivityKind.Client);
        activity?.SetTag("shard", slice.Id);
        activity?.SetTag("replica", replica.Index);
        activity?.SetTag("attempt.kind", kind.ToString().ToLowerInvariant());
        activity?.SetTag("rpc.system", "grpc");
        activity?.SetTag("server.address", replica.Endpoint);

        var request = template.Clone();
        double graceMs = Math.Min(_search.ShardGraceMs, remaining.TotalMilliseconds / 2);
        request.DeadlineBudgetUs = (ulong)Math.Max(1, (remaining.TotalMilliseconds - graceMs) * 1000);
        activity?.SetTag("deadline_budget_us", request.DeadlineBudgetUs);
        metrics.ShardRequests.WithLabels(slice.Label, replica.Index.ToString(System.Globalization.CultureInfo.InvariantCulture)).Inc();

        try
        {
            var call = replica.Client.SearchAsync(request, Telemetry.TraceMetadata(activity ?? Activity.Current),
                deadline: DateTime.UtcNow + remaining, cancellationToken: ct);
            var response = await call.ResponseAsync.ConfigureAwait(false);
            activity?.SetTag("partial", response.Partial);
            return new AttemptResult(AttemptStatus.Success, response, Stopwatch.GetElapsedTime(t0), null);
        }
        catch (Exception ex) when (ct.IsCancellationRequested && ex is RpcException or OperationCanceledException)
        {
            activity?.SetTag("cancelled", true);
            return new AttemptResult(AttemptStatus.Cancelled, null, Stopwatch.GetElapsedTime(t0), "cancelled");
        }
        catch (RpcException ex)
        {
            var status = ex.StatusCode switch
            {
                StatusCode.DeadlineExceeded => AttemptStatus.Timeout,
                StatusCode.Unavailable => AttemptStatus.Unavailable,
                _ => AttemptStatus.Error,
            };
            if (status == AttemptStatus.Unavailable) replica.MarkUnhealthy();
            string reason = status switch { AttemptStatus.Timeout => "timeout", AttemptStatus.Unavailable => "unavailable", _ => "error" };
            metrics.ShardFailures.WithLabels(slice.Label, reason).Inc();
            activity?.SetStatus(ActivityStatusCode.Error, ex.Status.Detail);
            logger.LogDebug("shard {Slice} replica {Replica} {Kind} failed: {Status}", slice.Id, replica.Index, kind, ex.StatusCode);
            return new AttemptResult(status, null, Stopwatch.GetElapsedTime(t0), ex.Status.Detail);
        }
    }

    /// <summary>
    /// Batch fetch, grouped by the slice that returned each doc. No hedging (fetch is cheap and
    /// off the tail), but one failover to another replica on error. Missing docs are simply absent.
    /// </summary>
    public async Task<(Dictionary<ulong, Document> Docs, List<int> FailedSlices)> FetchAsync(
        IReadOnlyDictionary<int, List<ulong>> docsBySlice, IReadOnlyList<string> highlightTerms, uint snippetChars,
        Deadline deadline, CancellationToken ct)
    {
        var tasks = docsBySlice
            .Where(kv => kv.Value.Count > 0)
            .Select(async kv =>
            {
                var slice = topology.Slices.First(s => s.Id == kv.Key);
                var req = new FetchRequest { SnippetChars = snippetChars };
                req.DocIds.AddRange(kv.Value);
                req.HighlightTerms.AddRange(highlightTerms);
                var tried = new List<ReplicaState>();
                var replica = slice.PickPrimary();
                for (int attempt = 0; attempt < 2 && replica is not null; attempt++)
                {
                    tried.Add(replica);
                    using var activity = Telemetry.Source.StartActivity(Telemetry.FetchSpan, ActivityKind.Client);
                    activity?.SetTag("shard", slice.Id);
                    activity?.SetTag("replica", replica.Index);
                    activity?.SetTag("docs", kv.Value.Count);
                    try
                    {
                        var remaining = deadline.Remaining;
                        if (remaining <= TimeSpan.Zero) break;
                        var resp = await replica.Client.FetchAsync(req, Telemetry.TraceMetadata(activity ?? Activity.Current),
                            deadline: DateTime.UtcNow + remaining, cancellationToken: ct).ResponseAsync.ConfigureAwait(false);
                        return (slice.Id, (IReadOnlyList<Document>?)resp.Documents);
                    }
                    catch (RpcException ex) when (!ct.IsCancellationRequested)
                    {
                        activity?.SetStatus(ActivityStatusCode.Error, ex.Status.Detail);
                        if (ex.StatusCode == StatusCode.Unavailable) replica.MarkUnhealthy();
                        if (ex.StatusCode == StatusCode.DeadlineExceeded) break;
                        replica = slice.PickAlternate(tried);
                    }
                }
                return (slice.Id, (IReadOnlyList<Document>?)null);
            });

        var results = await Task.WhenAll(tasks).ConfigureAwait(false);
        var docs = new Dictionary<ulong, Document>();
        var failed = new List<int>();
        foreach (var (sliceId, list) in results)
        {
            if (list is null) { failed.Add(sliceId); continue; }
            foreach (var d in list) docs[d.DocId] = d;
        }
        return (docs, failed);
    }

    /// <summary>One Health probe; updates the replica's state.</summary>
    public async Task<HealthResponse?> ProbeAsync(ReplicaState replica, TimeSpan timeout, CancellationToken ct)
    {
        try
        {
            var h = await replica.Client.HealthAsync(new HealthRequest(), deadline: DateTime.UtcNow + timeout, cancellationToken: ct)
                .ResponseAsync.ConfigureAwait(false);
            replica.MarkHealthy(h);
            return h;
        }
        catch (RpcException) when (!ct.IsCancellationRequested)
        {
            replica.MarkUnhealthy();
            return null;
        }
    }
}
