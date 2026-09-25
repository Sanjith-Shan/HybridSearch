using System.Text.Json;
using HybridSearch.Broker.Experiments;

namespace HybridSearch.Broker.Tests;

/// <summary>
/// The broker's team-draft interleaver against tests/golden/interleaving.json, the fixture the
/// Python click simulator is also tested against. If these pass, an interleaving experiment
/// analysed offline in Python means the same thing as one served by the broker.
/// </summary>
public class InterleavingGoldenTests
{
    public static IEnumerable<object[]> Cases()
    {
        var path = Path.Combine(TestPaths.RepoRoot, "tests", "golden", "interleaving.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
            yield return [c.GetProperty("name").GetString()!, c.GetRawText()];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesGoldenCase(string name, string json)
    {
        using var doc = JsonDocument.Parse(json);
        var c = doc.RootElement;
        var a = c.GetProperty("rankingA").EnumerateArray().Select(x => x.GetUInt64()).ToList();
        var b = c.GetProperty("rankingB").EnumerateArray().Select(x => x.GetUInt64()).ToList();
        var coins = c.GetProperty("coins").EnumerateArray().Select(x => x.GetBoolean()).ToList();
        int consumed = 0;
        var shown = TeamDraftInterleaver.Interleave(a, b, c.GetProperty("k").GetInt32(), () => coins[consumed++]);

        var exp = c.GetProperty("expected");
        Assert.Equal(exp.GetProperty("interleaved").EnumerateArray().Select(x => x.GetUInt64()), shown.Select(d => d.DocId));
        Assert.Equal(exp.GetProperty("teams").EnumerateArray().Select(x => x.GetString()), shown.Select(d => d.Team.ToString()));
        Assert.True(exp.GetProperty("coinsConsumed").GetInt32() == consumed, $"{name}: coins consumed");

        foreach (var t in c.GetProperty("clickTests").EnumerateArray())
        {
            var credit = InterleavingAttribution.Credit(shown, t.GetProperty("clickedDocIds").EnumerateArray().Select(x => x.GetUInt64()));
            Assert.Equal(t.GetProperty("clicksA").GetInt32(), credit.ClicksA);
            Assert.Equal(t.GetProperty("clicksB").GetInt32(), credit.ClicksB);
            // The fixture calls 0-0 a tie; the broker reports it as NoClicks so that impressions
            // without a click carry no preference in delta. Both mean "no winner".
            var expected = t.GetProperty("winner").GetString() switch
            {
                "A" => new[] { InterleavingOutcome.WinA },
                "B" => new[] { InterleavingOutcome.WinB },
                _ => new[] { InterleavingOutcome.Tie, InterleavingOutcome.NoClicks },
            };
            Assert.Contains(credit.Outcome, expected);
            if (credit.ClicksA == 0 && credit.ClicksB == 0) Assert.Equal(InterleavingOutcome.NoClicks, credit.Outcome);
        }
    }
}
