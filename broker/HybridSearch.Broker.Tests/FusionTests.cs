using HybridSearch.Broker.Search;

namespace HybridSearch.Broker.Tests;

public class FusionTests
{
    private static List<ScoredDoc> L(params (ulong Id, float S)[] xs) => xs.Select(x => new ScoredDoc(x.Id, x.S)).ToList();

    [Fact]
    public void Rrf_matches_hand_computed_scores()
    {
        var lex = L((1, 10), (2, 9), (3, 8));
        var den = L((3, .9f), (4, .8f), (1, .7f));
        var fused = Fusion.ReciprocalRank([lex, den], k: 60);

        double s1 = 1.0 / 61 + 1.0 / 63, s3 = 1.0 / 63 + 1.0 / 61, s2 = 1.0 / 62, s4 = 1.0 / 62;
        Assert.Equal([1UL, 3, 2, 4], fused.Select(f => f.DocId)); // 1 and 3 tie → lower id first; 2 and 4 tie likewise
        Assert.Equal(s1, fused[0].Score, 12);
        Assert.Equal(s3, fused[1].Score, 12);
        Assert.Equal(s2, fused[2].Score, 12);
        Assert.Equal(s4, fused[3].Score, 12);
    }

    [Fact]
    public void Rrf_k_changes_weighting_of_top_ranks()
    {
        // X = rank 1 in one list; Y = rank 4 in both. Y wins iff 2/(k+4) > 1/(k+1) ⇔ k > 2.
        var a = L((1, 9), (7, 8), (8, 7), (2, 6));
        var b = L((3, 9), (9, 8), (10, 7), (2, 6));
        Assert.Equal(1UL, Fusion.ReciprocalRank([a, b], k: 0)[0].DocId);
        Assert.Equal(2UL, Fusion.ReciprocalRank([a, b], k: 60)[0].DocId);
    }

    [Fact]
    public void Rrf_ignores_duplicate_within_list_and_uses_first_rank()
    {
        var fused = Fusion.ReciprocalRank([L((5, 3), (5, 2), (6, 1))], k: 60);
        Assert.Equal(1.0 / 61, fused.Single(f => f.DocId == 5).Score, 12);
        Assert.Equal(1.0 / 62, fused.Single(f => f.DocId == 6).Score, 12); // rank counts distinct docs
    }

    [Fact]
    public void Weighted_minmax_hand_computed()
    {
        var lex = L((1, 10), (2, 5), (3, 0));     // norm 1, .5, 0
        var den = L((2, .9f), (4, .5f));          // norm 1, 0
        var fused = Fusion.Weighted(lex, den, alpha: 0.4, ScoreNormalization.MinMax);
        var byId = fused.ToDictionary(f => f.DocId, f => f.Score);
        Assert.Equal(0.6 * 1 + 0.4 * 0, byId[1], 9);    // missing in dense → dense floor 0
        Assert.Equal(0.6 * .5 + 0.4 * 1, byId[2], 9);
        Assert.Equal(0.0, byId[3], 9);
        Assert.Equal(0.6 * 0 + 0.4 * 0, byId[4], 9);
        Assert.Equal([2UL, 1, 3, 4], fused.Select(f => f.DocId));
    }

    [Fact]
    public void Weighted_zscore_uses_population_sd_and_min_floor()
    {
        var lex = L((1, 3), (2, 1));  // mean 2, sd 1 → z = 1, -1
        var den = L((1, 2), (3, 0));  // z = 1, -1
        var byId = Fusion.Weighted(lex, den, 0.5, ScoreNormalization.ZScore).ToDictionary(f => f.DocId, f => f.Score);
        Assert.Equal(1.0, byId[1], 9);
        Assert.Equal(0.5 * -1 + 0.5 * -1, byId[2], 9); // missing from dense → dense min z (-1)
        Assert.Equal(0.5 * -1 + 0.5 * -1, byId[3], 9);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void Weighted_alpha_extremes_reproduce_single_list_order(double alpha)
    {
        var rng = new Random(7);
        var lex = Enumerable.Range(0, 30).Select(i => new ScoredDoc((ulong)i, (float)rng.NextDouble())).OrderByDescending(d => d.Score).ToList();
        var den = Enumerable.Range(0, 30).Select(i => new ScoredDoc((ulong)i, (float)rng.NextDouble())).OrderByDescending(d => d.Score).ToList();
        var fused = Fusion.Weighted(lex, den, alpha, ScoreNormalization.MinMax);
        var expected = (alpha == 0 ? lex : den).Select(d => d.DocId);
        Assert.Equal(expected, fused.Select(f => f.DocId));
    }

    [Fact]
    public void MinMax_and_ZScore_edge_cases()
    {
        Assert.Equal([1.0, 1.0], Fusion.MinMax([2f, 2f]));
        Assert.Equal([0.0, 0.0], Fusion.ZScore([2f, 2f]));
        Assert.Empty(Fusion.MinMax([]));
        Assert.Equal([1.0, 0.0, 0.5], Fusion.MinMax([4f, 2f, 3f]));
    }

    [Fact]
    public void Weighted_rejects_bad_alpha() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => Fusion.Weighted([], [], 1.5, ScoreNormalization.MinMax));

    [Fact]
    public void MergeTopK_orders_by_score_then_id_dedupes_and_truncates()
    {
        var merged = Fusion.MergeTopK([L((10, 5), (11, 3)), L((20, 5), (21, 4)), L((11, 7))], k: 3);
        Assert.Equal([11UL, 10, 20], merged.Select(d => d.DocId));
        Assert.Equal(7f, merged[0].Score);
    }

    [Fact]
    public void Fuse_output_is_always_in_ranking_order()
    {
        var rng = new Random(3);
        for (int t = 0; t < 50; t++)
        {
            var lex = Enumerable.Range(0, 40).Select(_ => new ScoredDoc((ulong)rng.Next(60), rng.Next(5))).ToList();
            var den = Enumerable.Range(0, 40).Select(_ => new ScoredDoc((ulong)rng.Next(60), rng.Next(5))).ToList();
            foreach (var opt in new[] { new FusionOptions(), new FusionOptions(FusionMethod.Weighted, Alpha: .3), new FusionOptions(FusionMethod.Weighted, Normalization: ScoreNormalization.ZScore) })
            {
                var f = Fusion.Fuse(lex, den, opt);
                Assert.Equal(f.Select(x => x.DocId).Distinct().Count(), f.Count);
                for (int i = 1; i < f.Count; i++)
                    Assert.True(Fusion.CompareRanking(f[i - 1].Score, f[i - 1].DocId, f[i].Score, f[i].DocId) < 0);
            }
        }
    }
}
