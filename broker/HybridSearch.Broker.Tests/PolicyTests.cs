using System.Diagnostics;
using HybridSearch.Broker.Api;
using HybridSearch.Broker.Experiments;
using HybridSearch.Broker.Search;
using HybridSearch.Broker.Shards;
using HybridSearch.Proto.V1;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Primitives;

namespace HybridSearch.Broker.Tests;

public class DegradationPolicyTests
{
    private static readonly CostEstimates Est = new(EncodeMs: 10, ShardFullMs: 50, ShardShrunkMs: 30, ShardLexicalMs: 15, RerankMs: 100, FetchMs: 5);
    private static readonly Capabilities All = new(true, true, true);
    private static readonly RequestedPlan Hybrid = new(true, true, true);

    [Fact]
    public void Enough_budget_means_no_degradation()
    {
        var p = DegradationPolicy.Plan(Hybrid, All, 1000, Est, 0);
        Assert.Equal(0, p.Level);
        Assert.True(p.Dense && p.Rerank && p.Lexical && !p.ShrinkBeam);
    }

    [Theory]
    [InlineData(165, 0)]   // 10+50+100+5
    [InlineData(164, 1)]   // no room for the reranker
    [InlineData(65, 1)]    // 10+50+5 fits
    [InlineData(64, 2)]    // shrink the beam: 10+30+5
    [InlineData(45, 2)]
    [InlineData(44, 3)]    // lexical only: 15+5
    [InlineData(1, 3)]     // floor: always answer lexical-only
    public void Budget_thresholds(double budget, int level) =>
        Assert.Equal(level, DegradationPolicy.Plan(Hybrid, All, budget, Est, 0).Level);

    [Fact]
    public void Levels_are_monotone_and_steps_follow_the_fixed_order()
    {
        int prev = 0;
        string[] order = ["reranker_skipped", "dense_beam_shrunk", "lexical_only"];
        for (double b = 300; b >= 0; b -= 0.5)
        {
            var p = DegradationPolicy.Plan(Hybrid, All, b, Est, 0);
            Assert.True(p.Level >= prev, $"level went down at budget {b}");
            prev = p.Level;
            var kinds = p.Steps.Select(s => s.KindName).ToList();
            // Cumulative: steps are exactly a prefix of the order (except shrink is replaced by lexical_only at level 3).
            if (p.Level == 3) Assert.Equal(["reranker_skipped", "lexical_only"], kinds);
            else Assert.Equal(order.Take(p.Level), kinds);
            Assert.All(p.Steps, s => Assert.Equal("deadline", s.Reason));
            Assert.Equal(p.Level < 1, p.Rerank);
            Assert.Equal(p.Level < 3, p.Dense);
            Assert.Equal(p.Level == 2, p.ShrinkBeam);
        }
        Assert.Equal(3, prev);
    }

    [Fact]
    public void Missing_models_are_reported_not_silent()
    {
        var p = DegradationPolicy.Plan(Hybrid, new Capabilities(false, true, false), 1000, Est, 0);
        Assert.Equal(["reranker_skipped:model_unavailable", "lexical_only:encoder_unavailable"], p.Steps.Select(s => s.ToString()));
        Assert.Equal(3, p.Level);
        Assert.False(p.Dense);
        Assert.True(p.Lexical);
    }

    [Fact]
    public void Missing_encoder_alone_still_reranks_bm25_candidates()
    {
        var p = DegradationPolicy.Plan(Hybrid, new Capabilities(false, true, true), 1000, Est, 0);
        Assert.True(p.Rerank);
        Assert.Equal(["lexical_only:encoder_unavailable"], p.Steps.Select(s => s.ToString()));
    }

    [Fact]
    public void Dense_only_request_falls_back_to_lexical()
    {
        var p = DegradationPolicy.Plan(new RequestedPlan(false, true, false), new Capabilities(true, false, true), 1000, Est, 0);
        Assert.True(p.Lexical);
        Assert.False(p.Dense);
        Assert.Equal(["lexical_only:vector_index_unavailable"], p.Steps.Select(s => s.ToString()));
    }

    [Fact]
    public void Lexical_request_never_reports_dense_steps()
    {
        var p = DegradationPolicy.Plan(new RequestedPlan(true, false, false), All, 1, Est, 0);
        Assert.Empty(p.Steps);
        Assert.Equal(0, p.Level);
    }

    [Fact]
    public void Load_pressure_sets_a_floor_with_reason_load()
    {
        var p = DegradationPolicy.Plan(Hybrid, All, 1000, Est, pressureLevel: 2);
        Assert.Equal(["reranker_skipped:load", "dense_beam_shrunk:load"], p.Steps.Select(s => s.ToString()));
        var q = DegradationPolicy.Plan(Hybrid, All, 30, Est, pressureLevel: 1);
        Assert.Equal(["reranker_skipped:load", "lexical_only:deadline"], q.Steps.Select(s => s.ToString()));
        Assert.Equal(0, DegradationPolicy.PressureLevel(10, [64, 128, 256]));
        Assert.Equal(1, DegradationPolicy.PressureLevel(65, [64, 128, 256]));
        Assert.Equal(3, DegradationPolicy.PressureLevel(1000, [64, 128, 256]));
        Assert.Equal(0, DegradationPolicy.PressureLevel(1000, []));
    }
}

public class LatencyTrackerTests
{
    [Fact]
    public void Quantiles_are_within_bucket_error_of_exact()
    {
        var t = new LatencyTracker(TimeSpan.FromMinutes(1), cacheFor: TimeSpan.Zero);
        var rng = new Random(5);
        var xs = Enumerable.Range(0, 20_000).Select(_ => (long)Math.Exp(rng.NextDouble() * 12)).ToArray(); // 1 µs .. ~160 ms
        foreach (var x in xs) t.RecordMicros(x);
        Array.Sort(xs);
        foreach (var q in new[] { 0.5, 0.9, 0.95, 0.99 })
        {
            double exact = xs[(int)Math.Ceiling(q * xs.Length) - 1];
            double est = t.Quantile(q)!.Value.TotalMicroseconds;
            Assert.InRange(est, exact, Math.Max(exact * 1.0325, exact + 1)); // upper-edge rounding: never below, ≤ 1/32 above
        }
    }

    [Fact]
    public void Buckets_are_monotone_and_upper_bound_their_values()
    {
        int prev = -1;
        for (long v = 0; v < 5_000_000; v = v < 100 ? v + 1 : (long)(v * 1.01))
        {
            int b = LatencyTracker.BucketOf(v);
            Assert.True(b >= prev);
            Assert.True(LatencyTracker.BucketUpperMicros(b) >= v);
            prev = b;
        }
        Assert.Equal(LatencyTracker.BucketCount - 1, LatencyTracker.BucketOf(long.MaxValue));
    }

    [Fact]
    public void Samples_expire_after_the_window()
    {
        long now = 0;
        long sec = Stopwatch.Frequency;
        var t = new LatencyTracker(TimeSpan.FromSeconds(6), slots: 6, clock: () => now, cacheFor: TimeSpan.Zero);
        for (int i = 0; i < 100; i++) t.RecordMicros(1000);
        Assert.Equal(100, t.Count);
        now += 3 * sec;
        for (int i = 0; i < 100; i++) t.RecordMicros(50_000);
        Assert.Equal(200, t.Count);
        Assert.True(t.Quantile(0.95)!.Value.TotalMilliseconds >= 50);
        now += 4 * sec; // first batch is > 6 s old
        Assert.Equal(100, t.Count);
        Assert.True(t.Quantile(0.05)!.Value.TotalMilliseconds >= 50);
        now += 10 * sec;
        Assert.Equal(0, t.Count);
        Assert.Null(t.Quantile(0.95));
    }

    [Fact]
    public void MinSamples_gates_the_quantile()
    {
        var t = new LatencyTracker(TimeSpan.FromMinutes(1), cacheFor: TimeSpan.Zero);
        for (int i = 0; i < 10; i++) t.RecordMicros(100);
        Assert.Null(t.Quantile(0.95, minSamples: 11));
        Assert.NotNull(t.Quantile(0.95, minSamples: 10));
    }

    [Fact]
    public async Task Concurrent_recording_loses_nothing_within_a_slot()
    {
        var t = new LatencyTracker(TimeSpan.FromHours(1), cacheFor: TimeSpan.Zero);
        await Task.WhenAll(Enumerable.Range(0, 8).Select(w => Task.Run(() =>
        {
            for (int i = 0; i < 50_000; i++) t.RecordMicros(i % 5000);
        })));
        Assert.Equal(400_000, t.Count);
    }

    [Fact]
    public async Task Hedge_budget_never_exceeds_ratio_under_concurrency()
    {
        var b = new HedgeBudget(0.05, 10);
        long primaries = 0, hedges = 0;
        await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
        {
            for (int i = 0; i < 100_000; i++)
            {
                Interlocked.Increment(ref primaries); // count first, so the check never sees a deposit before its primary
                b.OnPrimary();
                if (b.TryAcquire())
                {
                    long h = Interlocked.Increment(ref hedges);
                    Assert.True(h <= 0.05 * Interlocked.Read(ref primaries) + 1e-6);
                }
            }
        })));
        Assert.InRange(hedges, 39_000, 40_000);
    }

    [Fact]
    public void Hedge_budget_caps_the_burst()
    {
        var b = new HedgeBudget(0.5, 2);
        for (int i = 0; i < 100; i++) b.OnPrimary();
        Assert.Equal(2.0, b.Tokens, 6);
        Assert.True(b.TryAcquire());
        Assert.True(b.TryAcquire());
        Assert.False(b.TryAcquire());
    }
}

public class Utf8OffsetTests
{
    private static Highlight H(uint s, uint e) => new() { Start = s, End = e };

    [Fact]
    public void Ascii_is_identity()
    {
        var r = Utf8Offsets.ToUtf16("hello world", [H(6, 11)]);
        Assert.Equal([new HighlightDto(6, 11)], r);
    }

    [Fact]
    public void Multibyte_characters_shift_offsets()
    {
        // "Café — Paris": é = 2 bytes, — = 3 bytes. "Paris" starts at UTF-16 7, byte 10.
        const string text = "Café — Paris";
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        int byteStart = System.Text.Encoding.UTF8.GetByteCount("Café — ");
        Assert.Equal(10, byteStart);
        var r = Utf8Offsets.ToUtf16(text, [H(0, 5), H((uint)byteStart, (uint)bytes.Length)]);
        Assert.Equal([new HighlightDto(0, 4), new HighlightDto(7, 12)], r);
        Assert.Equal("Paris", text[7..12]);
    }

    [Fact]
    public void Surrogate_pairs_count_as_two_units()
    {
        const string text = "a😀b";  // 😀 = 4 UTF-8 bytes, 2 UTF-16 units
        var r = Utf8Offsets.ToUtf16(text, [H(5, 6)]);
        Assert.Equal([new HighlightDto(3, 4)], r);
        Assert.Equal("b", text[3..4]);
    }

    [Fact]
    public void Out_of_range_and_empty_spans_are_clamped_or_dropped()
    {
        var r = Utf8Offsets.ToUtf16("abc", [H(1, 100), H(2, 2)]);
        Assert.Equal([new HighlightDto(1, 3)], r);
    }
}

public class EventValidationTests
{
    private static readonly DateTimeOffset Now = DateTimeOffset.Parse("2026-09-23T20:00:00Z");

    private static UiEvent Click() => new() { Type = "click", SessionId = "s-1", RequestId = "01JABC", DocId = 7, Rank = 1, Query = "q", ClientTs = "2026-09-23T20:10:11.123Z", Team = "A" };

    [Fact]
    public void Valid_click_is_stamped_with_server_time()
    {
        var r = EventValidator.Validate(Click(), Now, out var err);
        Assert.Null(err);
        Assert.Equal(Now, r!.ServerTs);
        Assert.Equal("event", r.Kind);
        Assert.Equal(7UL, r.DocId);
        Assert.Equal(DateTimeOffset.Parse("2026-09-23T20:10:11.123Z"), r.ClientTs);
    }

    public static IEnumerable<object[]> Invalid() =>
    [
        [Click() with { Type = "hover" }],
        [Click() with { Type = null }],
        [Click() with { SessionId = null }],
        [Click() with { SessionId = "has space" }],
        [Click() with { SessionId = new string('a', 129) }],
        [Click() with { RequestId = null }],
        [Click() with { DocId = null }],
        [Click() with { DocId = -1 }],
        [Click() with { Rank = 0 }],
        [Click() with { Rank = 5000 }],
        [Click() with { Team = "C" }],
        [Click() with { ClientTs = "yesterday" }],
        [Click() with { Query = new string('q', 513) }],
        [Click() with { Type = "dwell", DwellMs = null }],
        [Click() with { Type = "dwell", DwellMs = -5 }],
    ];

    [Theory]
    [MemberData(nameof(Invalid))]
    public void Invalid_events_are_rejected_with_a_reason(UiEvent e)
    {
        Assert.Null(EventValidator.Validate(e, Now, out var err));
        Assert.False(string.IsNullOrEmpty(err));
    }

    [Fact]
    public void Query_and_abandon_need_no_doc()
    {
        Assert.NotNull(EventValidator.Validate(new UiEvent { Type = "query", SessionId = "s", Query = "x" }, Now, out _));
        Assert.NotNull(EventValidator.Validate(new UiEvent { Type = "abandon", SessionId = "s", RequestId = "r" }, Now, out _));
    }

    [Fact]
    public void Dwell_keeps_dwell_ms_and_other_types_drop_it()
    {
        Assert.Equal(1200, EventValidator.Validate(Click() with { Type = "dwell", DwellMs = 1200 }, Now, out _)!.DwellMs);
        Assert.Null(EventValidator.Validate(Click() with { DwellMs = 1200 }, Now, out _)!.DwellMs);
    }

    [Fact]
    public void Log_record_has_no_network_identity_fields()
    {
        var names = typeof(LogRecord).GetProperties().Select(p => p.Name.ToLowerInvariant()).ToList();
        Assert.DoesNotContain(names, n => n.Contains("ip") || n.Contains("agent") || n.Contains("user") || n.Contains("address"));
    }

    [Fact]
    public void Hourly_file_names()
    {
        Assert.Equal("events-20260923-20.jsonl", EventLogWriter.FileNameFor(DateTimeOffset.Parse("2026-09-23T20:59:59Z")));
        Assert.Equal("events-20260923-21.jsonl", EventLogWriter.FileNameFor(DateTimeOffset.Parse("2026-09-23T17:00:00-04:00")));
    }
}

public class SearchQueryParserTests
{
    private static SearchQuery? Parse(string qs, out Dictionary<string, string[]> errors)
    {
        var dict = Microsoft.AspNetCore.WebUtilities.QueryHelpers.ParseQuery(qs);
        return SearchQueryParser.Parse(new QueryCollection(dict), new SearchOptions(), out errors);
    }

    [Fact]
    public void Defaults()
    {
        var q = Parse("?q=peru", out _)!;
        Assert.Equal(new SearchQuery("peru", 10, SearchMode.Hybrid, true, false, 300, null), q);
    }

    [Theory]
    [InlineData("?q=")]
    [InlineData("?k=5")]
    [InlineData("?q=a&k=0")]
    [InlineData("?q=a&k=101")]
    [InlineData("?q=a&k=abc")]
    [InlineData("?q=a&mode=semantic")]
    [InlineData("?q=a&rerank=maybe")]
    [InlineData("?q=a&deadlineMs=0")]
    [InlineData("?q=a&deadlineMs=999999")]
    [InlineData("?q=a&sessionId=bad%20id")]
    public void Rejects(string qs)
    {
        Assert.Null(Parse(qs, out var errors));
        Assert.NotEmpty(errors);
    }

    [Fact]
    public void Query_length_limit_is_512_chars()
    {
        Assert.NotNull(Parse("?q=" + new string('a', 512), out _));
        Assert.Null(Parse("?q=" + new string('a', 513), out var e));
        Assert.Contains("q", e.Keys);
    }

    [Fact]
    public void Parses_all_params()
    {
        var q = Parse("?q=x&k=100&mode=dense&rerank=false&explain=true&deadlineMs=50&sessionId=ab-C_9", out _)!;
        Assert.Equal(new SearchQuery("x", 100, SearchMode.Dense, false, true, 50, "ab-C_9"), q);
    }

    [Fact]
    public void Ulids_are_26_chars_and_sortable()
    {
        var a = Ulid.NewString();
        Thread.Sleep(2);
        var b = Ulid.NewString();
        Assert.Equal(26, a.Length);
        Assert.True(string.CompareOrdinal(a, b) < 0);
        Assert.StartsWith("01", a);
    }
}

public class DegradationProbeTests
{
    private static readonly RequestedPlan Full = new(true, true, true);
    private static readonly Capabilities All = new(true, true, true);
    // Estimates far over budget: without probing this must go lexical-only.
    private static readonly CostEstimates Slow = new(500, 500, 400, 10, 900, 5);

    [Fact]
    public void Without_probe_an_over_budget_plan_degrades_to_lexical_only()
    {
        var plan = DegradationPolicy.Plan(Full, All, 300, Slow, 0);
        Assert.False(plan.Dense);
        Assert.Equal(3, plan.Level);
    }

    [Fact]
    public void A_probe_runs_the_full_plan_so_skipped_stages_get_fresh_estimates()
    {
        var plan = DegradationPolicy.Plan(Full, All, 300, Slow, 0, probe: true);
        Assert.True(plan.Dense);
        Assert.True(plan.Rerank);
        Assert.Equal(0, plan.Level);
    }

    [Fact]
    public void A_probe_still_respects_load_shedding_and_missing_models()
    {
        var plan = DegradationPolicy.Plan(Full, All, 300, Slow, 2, probe: true);
        Assert.False(plan.Rerank);
        Assert.True(plan.ShrinkBeam);
        var noEncoder = DegradationPolicy.Plan(Full, new Capabilities(false, true, true), 300, Slow, 0, probe: true);
        Assert.False(noEncoder.Dense);
    }
}

