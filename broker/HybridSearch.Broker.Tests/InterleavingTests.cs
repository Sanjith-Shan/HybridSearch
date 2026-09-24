using HybridSearch.Broker.Experiments;

namespace HybridSearch.Broker.Tests;

public class InterleavingTests
{
    private static List<ulong> RandomRanking(Random rng, int n, int universe) =>
        Enumerable.Range(0, universe).Select(i => (ulong)i).OrderBy(_ => rng.Next()).Take(n).ToList();

    public static IEnumerable<object[]> Seeds() => Enumerable.Range(0, 20).Select(i => new object[] { i });

    [Theory]
    [MemberData(nameof(Seeds))]
    public void Properties_hold_on_random_rankings(int seed)
    {
        var rng = new Random(seed);
        for (int trial = 0; trial < 50; trial++)
        {
            int k = rng.Next(1, 30);
            int universe = rng.Next(k, 3 * k + 2);
            var a = RandomRanking(rng, rng.Next(k, universe + 1), universe);
            var b = RandomRanking(rng, rng.Next(k, universe + 1), universe);
            ulong s = (ulong)rng.NextInt64();
            var inter = TeamDraftInterleaver.Interleave(a, b, k, s);

            // Length: k (both lists have ≥ k docs).
            Assert.Equal(k, inter.Count);
            // No duplicates.
            Assert.Equal(inter.Count, inter.Select(d => d.DocId).Distinct().Count());
            // Each team's docs keep that team's relative order and come from that team's list.
            AssertSubsequence(inter.Where(d => d.Team == Team.A).Select(d => d.DocId).ToList(), a);
            AssertSubsequence(inter.Where(d => d.Team == Team.B).Select(d => d.DocId).ToList(), b);
            // Every prefix: team sizes differ by at most 1.
            int na = 0, nb = 0;
            foreach (var d in inter)
            {
                if (d.Team == Team.A) na++; else nb++;
                Assert.InRange(na - nb, -1, 1);
            }
            // Team-draft picks the highest-ranked unused doc: each team's i-th pick is the first doc
            // of its list not already shown before that pick.
            var shown = new HashSet<ulong>();
            foreach (var d in inter)
            {
                var list = d.Team == Team.A ? a : b;
                Assert.Equal(list.First(x => !shown.Contains(x)), d.DocId);
                shown.Add(d.DocId);
            }
            // Deterministic under the seed.
            Assert.Equal(inter, TeamDraftInterleaver.Interleave(a, b, k, s));
        }
    }

    private static void AssertSubsequence(List<ulong> sub, List<ulong> of)
    {
        int j = 0;
        foreach (var x in sub)
        {
            while (j < of.Count && of[j] != x) j++;
            Assert.True(j < of.Count, $"{x} out of order or not in team list");
            j++;
        }
    }

    [Fact]
    public void Identical_rankings_keep_the_ranking_and_split_teams()
    {
        var a = Enumerable.Range(1, 10).Select(i => (ulong)i).ToList();
        var inter = TeamDraftInterleaver.Interleave(a, a, 10, 99);
        Assert.Equal(a, inter.Select(d => d.DocId));
        Assert.Equal(5, inter.Count(d => d.Team == Team.A));
    }

    [Fact]
    public void Coin_is_fair_across_seeds()
    {
        var a = new List<ulong> { 1, 2 };
        var b = new List<ulong> { 3, 4 };
        int aFirst = Enumerable.Range(0, 10_000).Count(s => TeamDraftInterleaver.Interleave(a, b, 1, (ulong)s)[0].Team == Team.A);
        Assert.InRange(aFirst, 4_700, 5_300); // ±6σ
    }

    [Fact]
    public void Different_seeds_give_different_interleavings()
    {
        var a = Enumerable.Range(0, 20).Select(i => (ulong)i).ToList();
        var b = Enumerable.Range(100, 20).Select(i => (ulong)i).ToList();
        var distinct = Enumerable.Range(0, 50).Select(s => string.Join(",", TeamDraftInterleaver.Interleave(a, b, 10, (ulong)s).Select(d => d.Team))).Distinct().Count();
        Assert.True(distinct > 10);
    }

    [Fact]
    public void Short_list_is_exhausted_then_other_team_fills()
    {
        var a = new List<ulong> { 1 };
        var b = new List<ulong> { 1, 2, 3, 4 };
        var inter = TeamDraftInterleaver.Interleave(a, b, 4, 5);
        Assert.Equal(4, inter.Count);
        Assert.Equal(inter.Count, inter.Select(d => d.DocId).Distinct().Count());
        Assert.True(inter.Count(d => d.Team == Team.A) <= 1);
    }

    [Fact]
    public void Empty_inputs_and_k_zero()
    {
        Assert.Empty(TeamDraftInterleaver.Interleave([], [], 5, 1));
        Assert.Empty(TeamDraftInterleaver.Interleave([1], [2], 0, 1));
    }

    [Fact]
    public void SplitMix64_reference_values()
    {
        // Reference sequence for seed 0 (Vigna's splitmix64.c).
        var r = new SplitMix64(0);
        Assert.Equal(0xE220A8397B1DCDAFUL, r.Next());
        Assert.Equal(0x6E789E6AA1B965F4UL, r.Next());
        Assert.Equal(0x06C45D188009454FUL, r.Next());
    }
}

public class CreditAssignmentTests
{
    private static readonly List<InterleavedDoc> Shown =
    [
        new(10, Team.A), new(20, Team.B), new(30, Team.A), new(40, Team.B),
    ];

    [Fact]
    public void Clicks_credit_the_contributing_team()
    {
        var c = InterleavingAttribution.Credit(Shown, [20, 40]);
        Assert.Equal(new InterleavingCredit(0, 2), c);
        Assert.Equal(InterleavingOutcome.WinB, c.Outcome);
    }

    [Fact]
    public void Repeat_clicks_count_once()
    {
        var c = InterleavingAttribution.Credit(Shown, [10, 10, 10, 20]);
        Assert.Equal(new InterleavingCredit(1, 1), c);
        Assert.Equal(InterleavingOutcome.Tie, c.Outcome);
    }

    [Fact]
    public void Clicks_on_unserved_docs_are_ignored()
    {
        var c = InterleavingAttribution.Credit(Shown, [999, 30]);
        Assert.Equal(new InterleavingCredit(1, 0), c);
        Assert.Equal(InterleavingOutcome.WinA, c.Outcome);
    }

    [Fact]
    public void No_clicks_is_no_preference() =>
        Assert.Equal(InterleavingOutcome.NoClicks, InterleavingAttribution.Credit(Shown, []).Outcome);
}
