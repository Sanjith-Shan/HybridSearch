using HybridSearch.Broker.Experiments;

namespace HybridSearch.Broker.Tests;

public class AssignmentTests
{
    private static ExperimentDefinition Exp(string id, double alloc, string salt = "s1", string kind = "ab", string status = "running") => new()
    {
        Id = id, Kind = kind, Allocation = alloc, Salt = salt, Status = status,
        Control = new RankerConfig { Mode = "lexical" }, Treatment = new RankerConfig { Mode = "hybrid" },
    };

    private static IEnumerable<string> Sessions(int n) => Enumerable.Range(0, n).Select(i => $"sess-{i:x8}-{i * 7919}");

    [Fact]
    public void Assignment_is_deterministic_and_sticky()
    {
        var exps = new[] { Exp("e", 0.5) };
        foreach (var s in Sessions(1000))
        {
            var a = ExperimentAssigner.Assign(exps, s);
            var b = ExperimentAssigner.Assign(exps, s);
            Assert.Equal(a?.Variant, b?.Variant);
            Assert.Equal(a?.Bucket, b?.Bucket);
        }
    }

    [Fact]
    public void Bucket_matches_the_documented_formula()
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes("2026-09-23-a" + "abc");
        var expected = System.IO.Hashing.XxHash64.HashToUInt64(bytes) % 10000;
        Assert.Equal((uint)expected, ExperimentAssigner.Bucket("2026-09-23-a", "abc"));
    }

    [Theory]
    [InlineData(0.2)]
    [InlineData(0.05)]
    [InlineData(0.5)]
    public void Allocation_proportion_is_respected(double allocation)
    {
        const int n = 100_000;
        var exps = new[] { Exp("e", allocation) };
        var assigned = Sessions(n).Select(s => ExperimentAssigner.Assign(exps, s)).Where(a => a is not null).ToList();
        double sd = Math.Sqrt(n * allocation * (1 - allocation));
        Assert.InRange(assigned.Count, n * allocation - 5 * sd, n * allocation + 5 * sd);

        int control = assigned.Count(a => a!.Variant == Variant.Control);
        double sd2 = Math.Sqrt(assigned.Count * 0.25);
        Assert.InRange(control, assigned.Count / 2.0 - 5 * sd2, assigned.Count / 2.0 + 5 * sd2);
        var (_, p) = Stats.SampleRatioMismatch(control, assigned.Count - control);
        Assert.True(p > 1e-4);
    }

    [Fact]
    public void Arm_is_independent_of_enrollment_bucket()
    {
        // Among enrolled sessions, low and high buckets should split 50/50 alike.
        var sessions = Sessions(60_000).ToList();
        var low = sessions.Where(s => ExperimentAssigner.Bucket("s1", s) < 1000).ToList();
        var high = sessions.Where(s => ExperimentAssigner.Bucket("s1", s) is >= 1000 and < 2000).ToList();
        double fl = low.Count(s => ExperimentAssigner.Arm("s1", s) == Variant.Treatment) / (double)low.Count;
        double fh = high.Count(s => ExperimentAssigner.Arm("s1", s) == Variant.Treatment) / (double)high.Count;
        Assert.InRange(fl, 0.45, 0.55);
        Assert.InRange(fh, 0.45, 0.55);
    }

    [Fact]
    public void Raising_allocation_never_moves_enrolled_sessions()
    {
        foreach (var s in Sessions(5000))
        {
            var a = ExperimentAssigner.Assign([Exp("e", 0.1)], s);
            var b = ExperimentAssigner.Assign([Exp("e", 0.3)], s);
            if (a is not null) Assert.Equal(a.Variant, b!.Variant);
        }
    }

    [Fact]
    public void Paused_experiments_enroll_nobody_and_first_match_wins()
    {
        Assert.All(Sessions(200), s => Assert.Null(ExperimentAssigner.Assign([Exp("p", 1.0, status: "paused")], s)));
        Assert.All(Sessions(200), s => Assert.Equal("first", ExperimentAssigner.Assign([Exp("first", 1.0), Exp("second", 1.0, "s2")], s)!.Experiment.Id));
        Assert.Null(ExperimentAssigner.Assign([Exp("e", 1.0)], null));
        Assert.Null(ExperimentAssigner.Assign([Exp("e", 0.0)], "x"));
    }

    [Fact]
    public void Salt_decorrelates_experiments()
    {
        var sessions = Sessions(20_000).ToList();
        int both = sessions.Count(s => ExperimentAssigner.IsEnrolled(0.5, ExperimentAssigner.Bucket("a", s)) &&
                                       ExperimentAssigner.IsEnrolled(0.5, ExperimentAssigner.Bucket("b", s)));
        Assert.InRange(both, 5000 - 400, 5000 + 400);
    }

    [Fact]
    public void Config_parsing_and_validation()
    {
        var file = ExperimentsFile.Parse("""
        { // comment allowed
          "experiments": [
            { "id": "x", "kind": "interleave", "allocation": 0.2, "salt": "s",
              "control": { "mode": "lexical", "rerank": false },
              "treatment": { "mode": "hybrid", "rerank": true, "rerankDepth": 50, "fusion": "rrf", "rrfK": 60 } },
            { "id": "aa", "kind": "aa", "allocation": 0.1, "salt": "t", "method": "interleave", "control": { "mode": "hybrid" } },
          ]
        }
        """);
        Assert.Equal(2, file.Experiments.Count);
        Assert.Equal(ExperimentKind.Interleave, file.Experiments[0].ParsedKind);
        Assert.Equal(50, file.Experiments[0].Treatment!.RerankDepth);
        Assert.True(file.Experiments[1].Interleaves);
        Assert.Same(file.Experiments[1].Control, file.Experiments[1].EffectiveTreatment);

        Assert.Throws<InvalidOperationException>(() => ExperimentsFile.Parse("""{"experiments":[{"id":"a","kind":"ab","allocation":2,"salt":"s","treatment":{}}]}"""));
        Assert.Throws<InvalidOperationException>(() => ExperimentsFile.Parse("""{"experiments":[{"id":"a","kind":"nope","allocation":0.1,"salt":"s"}]}"""));
        Assert.Throws<InvalidOperationException>(() => ExperimentsFile.Parse("""{"experiments":[{"id":"a","kind":"ab","allocation":0.1,"salt":"s"}]}"""));
        Assert.Throws<InvalidOperationException>(() => ExperimentsFile.Parse("""{"experiments":[{"id":"a","kind":"aa","allocation":0.1,"salt":"s"},{"id":"a","kind":"aa","allocation":0.1,"salt":"s"}]}"""));
    }

    [Fact]
    public void Shipped_experiments_json_is_valid()
    {
        var path = Path.Combine(TestPaths.BrokerProjectDir, "experiments.json");
        var file = ExperimentsFile.Parse(File.ReadAllText(path));
        Assert.Contains(file.Experiments, e => e.Id == "hybrid-vs-lexical" && e.Interleaves);
    }
}

public class StatsTests
{
    [Theory]
    [InlineData(10, 0, 2.0 / 1024)]
    [InlineData(0, 10, 2.0 / 1024)]
    [InlineData(5, 5, 1.0)]
    [InlineData(0, 0, 1.0)]
    [InlineData(8, 2, 0.109375)]     // 2·(1+10+45)/1024
    [InlineData(60, 40, 0.0569)]     // exact binomial, R: binom.test(60,100)$p.value = 0.05689
    public void Sign_test_known_values(int w, int l, double expected) =>
        Assert.Equal(expected, Stats.SignTestTwoSided(w, l), expected < 0.06 && expected > 0.05 ? 3 : 9);

    [Fact]
    public void Sign_test_large_n_does_not_underflow()
    {
        double p = Stats.SignTestTwoSided(5200, 4800);
        Assert.InRange(p, 5e-5, 1e-4); // z ≈ 4.0 → two-sided ≈ 6.3e-5
        Assert.Equal(1.0, Stats.SignTestTwoSided(50_000, 50_000), 6);
    }

    [Fact]
    public void Erfc_and_srm()
    {
        Assert.Equal(1.0, Stats.Erfc(0), 6);
        Assert.Equal(0.157299, Stats.Erfc(1), 5);
        Assert.Equal(1.842701, Stats.Erfc(-1), 5);
        var (chi, p) = Stats.SampleRatioMismatch(500, 500);
        Assert.Equal(0, chi);
        Assert.Equal(1.0, p, 6);
        var (chi2, p2) = Stats.SampleRatioMismatch(5300, 4700);
        Assert.Equal(36.0, chi2, 6);
        Assert.True(p2 < 1e-8);
    }

    [Fact]
    public void Bootstrap_is_reproducible_and_brackets_the_estimate()
    {
        var rng = new Random(1);
        var clusters = Enumerable.Range(0, 400).Select(_ => ((double)rng.Next(0, 3), (double)rng.Next(1, 4))).ToList();
        var a = Stats.RatioBootstrap(clusters, 1000, seed: 9);
        var b = Stats.RatioBootstrap(clusters, 1000, seed: 9);
        Assert.Equal(a, b);
        Assert.InRange(a.Estimate, a.Low, a.High);
        Assert.True(a.High - a.Low < 0.2);
    }

    [Fact]
    public void Difference_ci_covers_zero_for_identical_arms_and_excludes_it_for_different()
    {
        var rng = new Random(2);
        List<(double, double)> Arm(double p) => Enumerable.Range(0, 2000).Select(_ => (rng.NextDouble() < p ? 1.0 : 0.0, 1.0)).ToList();
        var arm = Arm(0.3);
        var same = Stats.RatioDifferenceBootstrap(arm, arm, 1000);
        Assert.Equal(0, same.Estimate);
        Assert.InRange(0.0, same.Low, same.High);
        var diff = Stats.RatioDifferenceBootstrap(Arm(0.3), Arm(0.4), 1000);
        Assert.True(diff.Low > 0);
    }

    [Fact]
    public void Interleaving_delta_matches_formula()
    {
        var i = Stats.InterleavingDeltaBootstrap([(3, 1, 1), (2, 0, 0), (0, 1, 2)], 500);
        Assert.Equal((5.0 - 2) / (5 + 2 + 3), i.Estimate, 9);
    }
}

public class ExperimentAnalyzerTests
{
    private static readonly DateTimeOffset T = DateTimeOffset.Parse("2026-09-23T12:00:00Z");

    private static LogRecord Served(string exp, string session, string req, string variant, params (ulong Doc, string? Team)[] results) => new()
    {
        Kind = "served", ServerTs = T, ExperimentId = exp, SessionId = session, RequestId = req, Variant = variant,
        Results = results.Select((r, i) => new ServedResult(r.Doc, i + 1, r.Team)).ToList(),
    };

    private static LogRecord Click(string session, string req, ulong doc) => new()
    {
        Kind = "event", ServerTs = T, Type = "click", SessionId = session, RequestId = req, DocId = doc, Rank = 1,
    };

    [Fact]
    public void Interleaving_counts_wins_losses_ties_and_ignores_foreign_clicks()
    {
        var exp = new ExperimentDefinition { Id = "il", Kind = "interleave", Allocation = 1, Salt = "s", Treatment = new() };
        var recs = new List<LogRecord>
        {
            Served("il", "s1", "r1", "interleaved", (1, "A"), (2, "B")), Click("s1", "r1", 2),            // B wins
            Served("il", "s1", "r2", "interleaved", (1, "A"), (2, "B")), Click("s1", "r2", 1),            // A wins
            Served("il", "s2", "r3", "interleaved", (1, "B"), (2, "A")), Click("s2", "r3", 1), Click("s2", "r3", 2), // tie
            Served("il", "s2", "r4", "interleaved", (1, "A"), (2, "B")),                                   // no click
            Served("il", "s3", "r5", "interleaved", (1, "A"), (2, "B")), Click("s3", "r5", 2), Click("s3", "r5", 2), // B wins (dup click)
            Click("intruder", "r4", 1),   // other session: ignored
            Click("s2", "r4", 99),        // not served: ignored
            Served("other-exp", "s9", "r9", "interleaved", (1, "A")), Click("s9", "r9", 1),
        };
        var r = ExperimentAnalyzer.Analyze(exp, recs, 200, T);
        Assert.NotNull(r.Interleaving);
        var il = r.Interleaving!;
        Assert.Equal(5, r.Queries);
        Assert.Equal((2, 1, 1, 1), (il.Wins, il.Losses, il.Ties, il.NoClicks));
        Assert.Equal((2.0 - 1) / 4, il.DeltaPreference.Value, 6);
        Assert.Equal(1.0, il.PValue, 6);
        Assert.Null(r.Srm);
        Assert.False(r.Simulated);
        var v = Assert.Single(r.Variants);
        Assert.Equal(("interleaved", 3, 5), (v.Name, v.Sessions, v.Queries));
    }

    [Fact]
    public void Ab_metrics_per_variant_and_srm()
    {
        var exp = new ExperimentDefinition { Id = "ab", Kind = "ab", Allocation = 1, Salt = "s", Treatment = new() };
        var recs = new List<LogRecord>
        {
            Served("ab", "c1", "q1", "control", (1, null), (2, null)), Click("c1", "q1", 2),
            Served("ab", "c1", "q2", "control", (1, null), (2, null)),
            Served("ab", "t1", "q3", "treatment", (3, null), (4, null)), Click("t1", "q3", 3), Click("t1", "q3", 4),
            Served("ab", "t2", "q4", "treatment", (3, null), (4, null)), Click("t2", "q4", 3),
        };
        var r = ExperimentAnalyzer.Analyze(exp, recs, 200, T);
        var c = r.Variants.Single(v => v.Name == "control");
        var t = r.Variants.Single(v => v.Name == "treatment");
        Assert.Equal((1, 2, 1), (c.Sessions, c.Queries, c.Clicks));
        Assert.Equal(0.5, c.Metrics["ctr"].Value, 6);          // 1 click / 2 impressions
        Assert.Equal(0.0, c.Metrics["clicksAt1"].Value, 6);
        Assert.Equal(0.5, c.Metrics["abandonment"].Value, 6);
        Assert.Equal(0.25, c.Metrics["mrrFirstClick"].Value, 6); // (1/2 + 0)/2
        Assert.Equal(1.5, t.Metrics["ctr"].Value, 6);
        Assert.Equal(1.0, t.Metrics["clicksAt1"].Value, 6);
        Assert.Equal(0.0, t.Metrics["abandonment"].Value, 6);
        Assert.Equal(1.0, r.Differences!["ctr"].Value, 6);
        Assert.Equal((1, 2), (r.Srm!.ControlSessions, r.Srm.TreatmentSessions));
        Assert.Null(r.Interleaving);
        Assert.Equal(0.95, r.ConfidenceLevel);
    }

    [Fact]
    public void Aa_interleaving_with_random_clicks_shows_no_preference()
    {
        var exp = new ExperimentDefinition { Id = "aa", Kind = "aa", Method = "interleave", Allocation = 1, Salt = "s" };
        var rng = new Random(4);
        var recs = new List<LogRecord>();
        var ranking = Enumerable.Range(1, 10).Select(i => (ulong)i).ToList();
        for (int i = 0; i < 3000; i++)
        {
            var inter = TeamDraftInterleaver.Interleave(ranking, ranking, 10, (ulong)i);
            recs.Add(Served("aa", $"s{i % 500}", $"r{i}", "interleaved", inter.Select(d => (d.DocId, (string?)d.Team.ToString())).ToArray()));
            if (rng.NextDouble() < 0.6) recs.Add(Click($"s{i % 500}", $"r{i}", (ulong)rng.Next(1, 4)));
        }
        var il = ExperimentAnalyzer.Analyze(exp, recs, 500, T).Interleaving!;
        Assert.InRange(0.0, il.DeltaPreference.CiLow, il.DeltaPreference.CiHigh);
        Assert.True(il.PValue > 0.001);
    }

    [Fact]
    public void Simulated_clicks_are_labelled_with_their_click_model()
    {
        var exp = new ExperimentDefinition { Id = "il", Kind = "interleave", Allocation = 1, Salt = "s", Treatment = new() };
        var recs = new List<LogRecord>
        {
            Served("il", "s1", "r1", "interleaved", (1, "A"), (2, "B")),
            Click("s1", "r1", 2) with { Simulated = true, ClickModel = "pbm" },
        };
        var r = ExperimentAnalyzer.Analyze(exp, recs, 100, T);
        Assert.True(r.Simulated);
        Assert.Equal(["pbm"], r.ClickModels);
        Assert.Contains("SIMULATED", r.Note);
    }
}
