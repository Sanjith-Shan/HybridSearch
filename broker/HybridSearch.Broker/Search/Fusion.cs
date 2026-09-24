namespace HybridSearch.Broker.Search;

/// <summary>A candidate from one retriever: global passage ID and that retriever's score.</summary>
public readonly record struct ScoredDoc(ulong DocId, float Score);

/// <summary>A fused candidate. <see cref="Score"/> is the fused score (higher is better).</summary>
public readonly record struct FusedDoc(ulong DocId, double Score);

public enum FusionMethod { Rrf, Weighted }

public enum ScoreNormalization { MinMax, ZScore }

/// <summary>Fusion configuration. <see cref="Alpha"/> weights the dense list: fused = α·dense + (1−α)·lexical.</summary>
public sealed record FusionOptions(
    FusionMethod Method = FusionMethod.Rrf,
    int RrfK = 60,
    double Alpha = 0.5,
    ScoreNormalization Normalization = ScoreNormalization.MinMax);

/// <summary>
/// Pure rank/score fusion functions. Every output list is in the project's ranking order:
/// score descending, ties broken by lower global doc ID (docs/ARCHITECTURE.md, Conventions).
/// Inputs are assumed to already be in ranking order; a doc's rank is its 1-based position
/// (first occurrence wins if a list contains a duplicate).
/// </summary>
public static class Fusion
{
    /// <summary>Total order used everywhere: higher score first, then lower doc ID.</summary>
    public static int CompareRanking(double scoreA, ulong idA, double scoreB, ulong idB)
    {
        int c = scoreB.CompareTo(scoreA);
        return c != 0 ? c : idA.CompareTo(idB);
    }

    public static void SortRanking(List<FusedDoc> docs) =>
        docs.Sort(static (a, b) => CompareRanking(a.Score, a.DocId, b.Score, b.DocId));

    public static void SortRanking(List<ScoredDoc> docs) =>
        docs.Sort(static (a, b) => CompareRanking(a.Score, a.DocId, b.Score, b.DocId));

    /// <summary>
    /// Merge per-slice top-k lists into a global top-k. Slices own disjoint doc ranges, but the
    /// merge still de-duplicates (keeping the best score) so a misconfigured topology cannot
    /// produce the same passage twice.
    /// </summary>
    public static List<ScoredDoc> MergeTopK(IEnumerable<IReadOnlyList<ScoredDoc>> perSlice, int k)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(k);
        var best = new Dictionary<ulong, float>();
        foreach (var list in perSlice)
        {
            foreach (var d in list)
            {
                if (!best.TryGetValue(d.DocId, out var s) || d.Score > s)
                    best[d.DocId] = d.Score;
            }
        }
        var all = new List<ScoredDoc>(best.Count);
        foreach (var (id, s) in best) all.Add(new ScoredDoc(id, s));
        SortRanking(all);
        if (all.Count > k) all.RemoveRange(k, all.Count - k);
        return all;
    }

    /// <summary>
    /// Reciprocal rank fusion (Cormack, Clarke, Büttcher 2009): score(d) = Σ_lists 1 / (k + rank(d)).
    /// A doc absent from a list contributes nothing for that list.
    /// </summary>
    public static List<FusedDoc> ReciprocalRank(IReadOnlyList<IReadOnlyList<ScoredDoc>> lists, int k = 60)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(k);
        var acc = new Dictionary<ulong, double>();
        foreach (var list in lists)
        {
            var seen = new HashSet<ulong>();
            int rank = 0;
            foreach (var d in list)
            {
                if (!seen.Add(d.DocId)) continue;
                rank++;
                acc[d.DocId] = acc.GetValueOrDefault(d.DocId) + 1.0 / (k + rank);
            }
        }
        return ToSorted(acc);
    }

    /// <summary>
    /// Weighted combination of normalised scores: fused = α·norm(dense) + (1−α)·norm(lexical).
    /// A doc missing from a list gets that list's minimum normalised value (0 for min-max, the
    /// lowest observed z for z-score): "not retrieved" is treated as "no better than the worst
    /// retrieved", never as a bonus. With a single non-empty list the result is that list's order.
    /// </summary>
    public static List<FusedDoc> Weighted(
        IReadOnlyList<ScoredDoc> lexical, IReadOnlyList<ScoredDoc> dense, double alpha, ScoreNormalization normalization)
    {
        if (double.IsNaN(alpha) || alpha < 0 || alpha > 1)
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "alpha must be in [0, 1]");

        var lex = Normalize(lexical, normalization);
        var den = Normalize(dense, normalization);
        double lexFloor = lex.Count == 0 ? 0 : lex.Values.Min();
        double denFloor = den.Count == 0 ? 0 : den.Values.Min();

        var acc = new Dictionary<ulong, double>();
        foreach (var id in lex.Keys.Concat(den.Keys))
        {
            if (acc.ContainsKey(id)) continue;
            double l = lex.TryGetValue(id, out var lv) ? lv : lexFloor;
            double d = den.TryGetValue(id, out var dv) ? dv : denFloor;
            acc[id] = alpha * d + (1 - alpha) * l;
        }
        return ToSorted(acc);
    }

    /// <summary>Dispatches on <paramref name="options"/>.</summary>
    public static List<FusedDoc> Fuse(IReadOnlyList<ScoredDoc> lexical, IReadOnlyList<ScoredDoc> dense, FusionOptions options) =>
        options.Method switch
        {
            FusionMethod.Rrf => ReciprocalRank([lexical, dense], options.RrfK),
            FusionMethod.Weighted => Weighted(lexical, dense, options.Alpha, options.Normalization),
            _ => throw new ArgumentOutOfRangeException(nameof(options)),
        };

    /// <summary>Min-max to [0,1]. If every score is equal, every doc maps to 1 (they tie for best).</summary>
    public static double[] MinMax(ReadOnlySpan<float> scores)
    {
        var r = new double[scores.Length];
        if (scores.Length == 0) return r;
        double min = double.MaxValue, max = double.MinValue;
        foreach (var s in scores) { min = Math.Min(min, s); max = Math.Max(max, s); }
        double range = max - min;
        for (int i = 0; i < scores.Length; i++) r[i] = range > 0 ? (scores[i] - min) / range : 1.0;
        return r;
    }

    /// <summary>Z-score using the population standard deviation. Zero variance maps every doc to 0.</summary>
    public static double[] ZScore(ReadOnlySpan<float> scores)
    {
        var r = new double[scores.Length];
        if (scores.Length == 0) return r;
        double mean = 0;
        foreach (var s in scores) mean += s;
        mean /= scores.Length;
        double var = 0;
        foreach (var s in scores) var += (s - mean) * (s - mean);
        double sd = Math.Sqrt(var / scores.Length);
        for (int i = 0; i < scores.Length; i++) r[i] = sd > 0 ? (scores[i] - mean) / sd : 0.0;
        return r;
    }

    private static Dictionary<ulong, double> Normalize(IReadOnlyList<ScoredDoc> list, ScoreNormalization n)
    {
        // First occurrence of each doc only (ranking order input).
        var ids = new List<ulong>(list.Count);
        var scores = new List<float>(list.Count);
        var seen = new HashSet<ulong>();
        foreach (var d in list)
        {
            if (!seen.Add(d.DocId)) continue;
            ids.Add(d.DocId);
            scores.Add(d.Score);
        }
        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(scores);
        var norm = n == ScoreNormalization.MinMax ? MinMax(span) : ZScore(span);
        var map = new Dictionary<ulong, double>(ids.Count);
        for (int i = 0; i < ids.Count; i++) map[ids[i]] = norm[i];
        return map;
    }

    private static List<FusedDoc> ToSorted(Dictionary<ulong, double> acc)
    {
        var list = new List<FusedDoc>(acc.Count);
        foreach (var (id, s) in acc) list.Add(new FusedDoc(id, s));
        SortRanking(list);
        return list;
    }
}
