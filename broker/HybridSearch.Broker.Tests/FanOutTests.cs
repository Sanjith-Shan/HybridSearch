using System.Diagnostics;
using Grpc.Core;
using HybridSearch.Broker.Shards;
using HybridSearch.Proto.V1;

namespace HybridSearch.Broker.Tests;

/// <summary>
/// Fan-out against real gRPC FakeShards. Timing assertions use injected latencies that are large
/// relative to scheduling noise (the dev machine can be heavily loaded), and compare shapes, not
/// absolute numbers.
/// </summary>
[Collection("timing")]
public class FanOutTests
{
    private static SearchRequest Req(string q = "capital of peru") => new()
    {
        Query = q,
        RequestId = "t",
        Lexical = new LexicalParams { Enabled = true, K = 10 },
        Vector = new VectorParams { Enabled = false },
    };

    [Fact]
    public async Task Deadline_is_propagated_in_the_request_and_as_the_grpc_deadline()
    {
        await using var h = await FanOutHarness.StartAsync(2, 1);
        var budget = TimeSpan.FromMilliseconds(2000);
        var outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(budget), CancellationToken.None);
        Assert.All(outcomes, o => Assert.Equal(SliceStatus.Ok, o.Status));
        foreach (var server in h.Servers.SelectMany(x => x))
        {
            var seen = Assert.Single(server.Seen.Searches);
            Assert.InRange(seen.DeadlineBudgetUs, 1UL, (ulong)budget.TotalMicroseconds);
            Assert.NotNull(seen.GrpcDeadlineRemaining);
            // Shard budget = remaining − grace, so the gRPC deadline is later than the shard's own budget.
            Assert.True(seen.GrpcDeadlineRemaining!.Value.TotalMicroseconds > seen.DeadlineBudgetUs - 500_000,
                $"grpc deadline {seen.GrpcDeadlineRemaining} vs budget {seen.DeadlineBudgetUs}us");
            Assert.True(seen.GrpcDeadlineRemaining.Value <= budget);
        }
    }

    [Fact]
    public async Task Budget_shrinks_with_time_already_spent()
    {
        await using var h = await FanOutHarness.StartAsync(1, 1);
        var deadline = Deadline.FromNow(TimeSpan.FromMilliseconds(3000));
        await Task.Delay(1000);
        await h.FanOut.SearchAsync(Req(), deadline, CancellationToken.None);
        var seen = Assert.Single(h.Servers[0][0].Seen.Searches);
        Assert.InRange(seen.DeadlineBudgetUs, 1UL, 2_000_000UL);
    }

    [Fact]
    public async Task Slow_shard_returns_partial_at_its_budget_and_is_reported()
    {
        // Grace (budget kept back for the reply to travel) is raised so a loaded CI box does not turn the partial into a timeout.
        await using var h = await FanOutHarness.StartAsync(2, 1, (s, _, f) => { if (s == 1) f.LatencyMs = 5000; }, o => o.Search.ShardGraceMs = 150);
        var sw = Stopwatch.StartNew();
        var outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromMilliseconds(700)), CancellationToken.None);
        Assert.True(sw.ElapsedMilliseconds < 3000);
        Assert.Equal(SliceStatus.Ok, outcomes[0].Status);
        Assert.Equal(SliceStatus.Partial, outcomes[1].Status);
        Assert.True(outcomes[1].Response!.Partial);
    }

    [Fact]
    public async Task Shard_ignoring_its_budget_times_out_and_is_failed_not_awaited()
    {
        await using var h = await FanOutHarness.StartAsync(2, 1, (s, _, f) => { if (s == 1) { f.LatencyMs = 10_000; f.HonorBudget = false; } });
        var sw = Stopwatch.StartNew();
        var outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromMilliseconds(500)), CancellationToken.None);
        Assert.True(sw.ElapsedMilliseconds < 4000, $"took {sw.ElapsedMilliseconds} ms");
        Assert.Equal(SliceStatus.Ok, outcomes[0].Status);
        Assert.Equal(SliceStatus.Failed, outcomes[1].Status);
        Assert.Equal("timeout", outcomes[1].FailureReason);
    }

    [Fact]
    public async Task Erroring_shard_is_reported_failed()
    {
        await using var h = await FanOutHarness.StartAsync(3, 1, (s, _, f) => { if (s == 2) { f.FailRate = 1; f.FailStatus = StatusCode.Internal; } });
        var outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None);
        Assert.Equal([SliceStatus.Ok, SliceStatus.Ok, SliceStatus.Failed], outcomes.Select(o => o.Status));
        Assert.Equal("error", outcomes[2].FailureReason);
        Assert.Contains("hs_shard_failures_total", await Scrape(h));
    }

    [Fact]
    public async Task Slice_with_every_replica_unhealthy_fails_fast_instead_of_burning_the_deadline()
    {
        // Slice 1 is "hung": it would answer only after 2 s. Once health checks have marked both of
        // its replicas unhealthy, the fan-out must not spend the request's budget waiting on it.
        await using var h = await FanOutHarness.StartAsync(2, 2, (s, _, f) => { if (s == 1) f.LatencyMs = 2000; });
        foreach (var r in h.Topology.Slices[1].Replicas) r.MarkUnhealthy();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromMilliseconds(800)), CancellationToken.None);
        sw.Stop();
        Assert.Equal(SliceStatus.Ok, outcomes[0].Status);
        Assert.Equal(SliceStatus.Failed, outcomes[1].Status);
        Assert.Equal("slice_down", outcomes[1].FailureReason);
        Assert.True(sw.ElapsedMilliseconds < 400, $"took {sw.ElapsedMilliseconds} ms; should not wait for the dead slice");
        Assert.Equal(0L, h.Servers[1][0].Seen.SearchCalls + h.Servers[1][1].Seen.SearchCalls);

        // A successful health probe brings the slice back: it is queried again, not skipped.
        h.Topology.Slices[1].Replicas[0].MarkHealthy(new HybridSearch.Proto.V1.HealthResponse());
        outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromMilliseconds(300)), CancellationToken.None);
        Assert.NotEqual("slice_down", outcomes[1].FailureReason);
        Assert.True(h.Servers[1][0].Seen.SearchCalls + h.Servers[1][1].Seen.SearchCalls >= 1);
    }

    [Fact]
    public async Task Fail_fast_can_be_disabled()
    {
        await using var h = await FanOutHarness.StartAsync(1, 1, (_, _, f) => f.LatencyMs = 2000, o => o.Hedging.FailFastWhenSliceDown = false);
        h.Topology.Slices[0].Replicas[0].MarkUnhealthy();
        var outcomes = await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromMilliseconds(300)), CancellationToken.None);
        Assert.NotEqual("slice_down", outcomes[0].FailureReason);
        Assert.Equal(1L, h.Servers[0][0].Seen.SearchCalls);
    }

    [Fact]
    public async Task Unavailable_replica_fails_over_to_the_other_replica()
    {
        await using var h = await FanOutHarness.StartAsync(1, 2, (_, r, f) => { if (r == 0) f.FailRate = 1; });
        for (int i = 0; i < 6; i++)
        {
            var o = (await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None))[0];
            Assert.Equal(SliceStatus.Ok, o.Status);
        }
        Assert.True(h.Servers[0][1].Seen.SearchCompleted >= 6);
    }

    [Fact]
    public async Task Hedging_cuts_the_tail_and_its_extra_load_is_counted_and_bounded()
    {
        // Two replicas; 3% of requests take +300 ms. Hedge at the rolling p95 with a 10% load budget.
        const int n = 600;
        static void Faults(int s, int r, HybridSearch.FakeShard.FaultOptions f)
        {
            f.LatencyMs = 4; f.JitterMs = 4; f.TailProbability = 0.03; f.TailMs = 300;
        }

        async Task<(double P50, double P99, FanOutHarness H)> Run(bool hedging)
        {
            var h = await FanOutHarness.StartAsync(1, 2, Faults, o =>
            {
                o.Hedging.Enabled = hedging;
                o.Hedging.MinSamples = 40;
                o.Hedging.BudgetRatio = 0.10;
                o.Hedging.BurstTokens = 5;
                o.Hedging.Quantile = 0.95;
            });
            var lat = new List<double>();
            for (int i = 0; i < n; i++)
            {
                var sw = Stopwatch.StartNew();
                var o = (await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(5)), CancellationToken.None))[0];
                Assert.Equal(SliceStatus.Ok, o.Status);
                if (i >= 50) lat.Add(sw.Elapsed.TotalMilliseconds); // skip warm-up (JIT, connections, min samples)
            }
            lat.Sort();
            return (lat[lat.Count / 2], lat[(int)(lat.Count * 0.99) - 1], h);
        }

        var (_, p99Off, hOff) = await Run(false);
        await using (hOff) { Assert.Equal(0, hOff.Topology.Slices[0].Hedges); }
        var (_, p99On, h) = await Run(true);
        await using (h)
        {
            var slice = h.Topology.Slices[0];
            long shardCalls = h.Servers[0].Sum(s => Interlocked.Read(ref s.Seen.SearchCalls));
            // Extra-load accounting: every call a shard saw is a primary or a hedge (no failures injected).
            // "Hedges sent" is an upper bound on server-side extra load: a hedge cancelled because the
            // primary answered first can be torn down before its request reaches the server.
            Assert.InRange(shardCalls, slice.Primaries + 1, slice.Primaries + slice.Hedges);
            Assert.True(slice.Primaries + slice.Hedges - shardCalls <= Math.Max(2, slice.Hedges / 4),
                $"too many hedges never reached a shard: sent {slice.Hedges}, shard calls {shardCalls}");
            Assert.Equal(n, slice.Primaries);
            Assert.True(slice.Hedges > 0);
            Assert.True(slice.Hedges <= 0.10 * slice.Primaries, $"hedges {slice.Hedges} > 10% of {slice.Primaries}");
            Assert.Equal(slice.Hedges / (double)slice.Primaries, slice.ExtraLoadRatio, 9);
            var metrics = await Scrape(h);
            Assert.Contains($"hs_hedges_sent_total{{shard=\"0\"}} {slice.Hedges}", metrics);
            // Losers are cancelled: the shard observed cancellations.
            long cancelled = h.Servers[0].Sum(s => Interlocked.Read(ref s.Seen.SearchCancelled));
            Assert.True(cancelled > 0);
            Assert.True(p99On < p99Off * 0.6, $"p99 with hedging {p99On:F1} ms vs without {p99Off:F1} ms");
        }
    }

    [Fact]
    public async Task No_hedging_with_a_single_replica()
    {
        await using var h = await FanOutHarness.StartAsync(1, 1, (_, _, f) => { f.TailProbability = 0.5; f.TailMs = 50; },
            o => o.Hedging.MinSamples = 5);
        for (int i = 0; i < 30; i++) await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None);
        Assert.Equal(0, h.Topology.Slices[0].Hedges);
        Assert.Null(h.FanOut.HedgeDelay(h.Topology.Slices[0]));
    }

    [Fact]
    public async Task Hedge_budget_denies_when_every_request_is_slow()
    {
        // An incident: everything is slower than the p95 learned earlier. Hedging must not double load.
        await using var h = await FanOutHarness.StartAsync(1, 2, (_, _, f) => f.LatencyMs = 2, o =>
        {
            o.Hedging.MinSamples = 20; o.Hedging.BudgetRatio = 0.05; o.Hedging.BurstTokens = 2;
        });
        for (int i = 0; i < 40; i++) await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None);
        foreach (var s in h.Servers[0]) s.Faults.LatencyMs = 150;
        for (int i = 0; i < 40; i++) await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None);
        var slice = h.Topology.Slices[0];
        Assert.True(slice.Hedges <= 0.05 * slice.Primaries);
        Assert.Contains("hs_hedges_denied_total", await Scrape(h));
    }

    [Fact]
    public async Task Trace_context_is_forwarded_in_grpc_metadata()
    {
        using var listener = new ActivityListener
        {
            ShouldListenTo = s => s.Name == Observability.Telemetry.SourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
        };
        ActivitySource.AddActivityListener(listener);
        await using var h = await FanOutHarness.StartAsync(2, 1);
        using var root = Observability.Telemetry.Source.StartActivity("test.root")!;
        await h.FanOut.SearchAsync(Req(), Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None);
        foreach (var s in h.Servers.SelectMany(x => x))
        {
            var tp = Assert.Single(Assert.Single(s.Seen.Searches).Traceparents);
            Assert.Equal(root.TraceId.ToHexString(), tp.Split('-')[1]);
            Assert.NotEqual(root.SpanId.ToHexString(), tp.Split('-')[2]); // parent is the shard.search span, not the root
        }
    }

    [Fact]
    public async Task Fetch_groups_by_slice_and_returns_snippets_with_byte_highlights()
    {
        await using var h = await FanOutHarness.StartAsync(2, 1);
        var bySlice = new Dictionary<int, List<ulong>> { [0] = [1001, 1016], [1] = [1028, 999_999] };
        var (docs, failed) = await h.FanOut.FetchAsync(bySlice, ["paris", "capital"], 0, Deadline.FromNow(TimeSpan.FromSeconds(3)), CancellationToken.None);
        Assert.Empty(failed);
        Assert.Equal(new ulong[] { 1001, 1016, 1028 }, docs.Keys.Order().ToArray());
        var cafe = docs[1016];
        var hl = Assert.Single(cafe.Highlights);
        var bytes = System.Text.Encoding.UTF8.GetBytes(cafe.Text);
        Assert.Equal("Paris", System.Text.Encoding.UTF8.GetString(bytes, (int)hl.Start, (int)(hl.End - hl.Start)));
    }

    [Fact]
    public async Task Health_probe_marks_replicas_and_readiness()
    {
        await using var h = await FanOutHarness.StartAsync(2, 1, (s, _, f) => { if (s == 1) f.HealthDown = true; });
        foreach (var r in h.Topology.Slices.SelectMany(s => s.Replicas)) await h.FanOut.ProbeAsync(r, TimeSpan.FromSeconds(2), CancellationToken.None);
        Assert.True(h.Topology.Slices[0].IsReady);
        Assert.False(h.Topology.Slices[1].IsReady);
        Assert.False(h.Topology.IsReady);
        Assert.Equal((1001UL, 1016UL), h.Topology.Slices[0].Range);
        Assert.Same(h.Topology.Slices[0], h.Topology.SliceForDoc(1005));
    }

    private static async Task<string> Scrape(FanOutHarness h)
    {
        using var ms = new MemoryStream();
        await h.Metrics.Registry.CollectAndExportAsTextAsync(ms);
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }
}

[CollectionDefinition("timing", DisableParallelization = true)]
public class TimingCollection;
