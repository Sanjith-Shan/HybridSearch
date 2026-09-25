using System.Text.Json;
using HybridSearch.Broker.Search;

namespace HybridSearch.Broker.Tests;

/// <summary>
/// The broker's fusion against tests/golden/fusion.json, generated from the Python fusion used
/// in the M4 study. The configuration the broker serves was tuned in Python, so this is what
/// makes the served ranking the one that was evaluated.
/// </summary>
public class FusionGoldenTests
{
    public static IEnumerable<object[]> Cases()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestPaths.RepoRoot, "tests", "golden", "fusion.json")));
        int i = 0;
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray()) yield return [i++, c.GetRawText()];
    }

    private static List<ScoredDoc> Docs(JsonElement arr) =>
        arr.EnumerateArray().Select(p => new ScoredDoc(p[0].GetUInt64(), (float)p[1].GetDouble())).ToList();

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesPythonFusion(int index, string json)
    {
        using var doc = JsonDocument.Parse(json);
        var c = doc.RootElement;
        var lex = Docs(c.GetProperty("lexical"));
        var den = Docs(c.GetProperty("dense"));
        var got = c.GetProperty("method").GetString() == "rrf"
            ? Fusion.ReciprocalRank([lex, den], c.GetProperty("k").GetInt32())
            : Fusion.Weighted(lex, den, c.GetProperty("alpha").GetDouble(),
                c.GetProperty("norm").GetString() == "zscore" ? ScoreNormalization.ZScore : ScoreNormalization.MinMax);
        var exp = c.GetProperty("expected").EnumerateArray().Select(p => (Id: p[0].GetUInt64(), Score: p[1].GetDouble())).ToList();

        Assert.Equal(exp.Count, got.Count);
        for (int i = 0; i < exp.Count; i++)
        {
            Assert.True(Math.Abs(exp[i].Score - got[i].Score) < 1e-9, $"case {index} pos {i}: {exp[i].Score} vs {got[i].Score}");
            // Order must match exactly, except among docs whose scores are equal to within rounding.
            if (exp[i].Id != got[i].DocId)
                Assert.True(Math.Abs(exp[i].Score - got[i].Score) < 1e-9 && got.Any(g => g.DocId == exp[i].Id && Math.Abs(g.Score - exp[i].Score) < 1e-9),
                    $"case {index} pos {i}: expected doc {exp[i].Id}, got {got[i].DocId}");
        }
    }
}
