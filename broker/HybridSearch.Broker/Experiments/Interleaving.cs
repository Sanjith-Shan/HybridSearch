namespace HybridSearch.Broker.Experiments;

public enum Team { A, B }

public readonly record struct InterleavedDoc(ulong DocId, Team Team);

/// <summary>
/// SplitMix64 (Steele, Lea, Flood 2014). Used instead of System.Random so a seed means the
/// same sequence on every runtime and platform, forever — interleavings must be reproducible
/// from the logged seed.
/// </summary>
public struct SplitMix64(ulong seed)
{
    private ulong _state = seed;

    public ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public bool NextBool() => (Next() >> 63) == 1;

    /// <summary>Uniform integer in [0, n) (Lemire's multiply-shift; bias ≤ n/2^64, negligible here).</summary>
    public int NextInt(int n)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        return (int)(((UInt128)Next() * (ulong)n) >> 64);
    }
}

/// <summary>
/// Team-draft interleaving (Radlinski, Kurup, Joachims, CIKM 2008).
///
/// Each round, the team with fewer picks so far picks next (a coin flip breaks ties); a team
/// picks its highest-ranked document not already in the interleaved list. This gives the
/// properties the tests check: each team's documents appear in the same relative order as in
/// that team's ranking, no duplicates, and in every prefix the team sizes differ by at most 1
/// (while both rankings still have unused documents).
///
/// When one ranking runs out of unused documents, the other fills the remaining slots (still
/// tagged with its own team) so the page is never short; with both input lists at least
/// <c>k</c> long this cannot happen before <c>k</c> results.
/// </summary>
public static class TeamDraftInterleaver
{
    public static List<InterleavedDoc> Interleave(IReadOnlyList<ulong> rankingA, IReadOnlyList<ulong> rankingB, int k, ulong seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(k);
        var rng = new SplitMix64(seed);
        var result = new List<InterleavedDoc>(k);
        var used = new HashSet<ulong>();
        int ia = 0, ib = 0, sizeA = 0, sizeB = 0;

        while (result.Count < k)
        {
            AdvancePastUsed(rankingA, ref ia, used);
            AdvancePastUsed(rankingB, ref ib, used);
            bool aHas = ia < rankingA.Count, bHas = ib < rankingB.Count;
            if (!aHas && !bHas) break;

            Team pick;
            if (aHas && bHas)
            {
                // The coin is drawn only on ties so the RNG stream is consumed identically for
                // identical inputs, which is what makes the result deterministic under the seed.
                pick = sizeA < sizeB ? Team.A : sizeB < sizeA ? Team.B : (rng.NextBool() ? Team.A : Team.B);
            }
            else
            {
                pick = aHas ? Team.A : Team.B;
            }

            if (pick == Team.A)
            {
                used.Add(rankingA[ia]);
                result.Add(new InterleavedDoc(rankingA[ia++], Team.A));
                sizeA++;
            }
            else
            {
                used.Add(rankingB[ib]);
                result.Add(new InterleavedDoc(rankingB[ib++], Team.B));
                sizeB++;
            }
        }
        return result;
    }

    private static void AdvancePastUsed(IReadOnlyList<ulong> ranking, ref int i, HashSet<ulong> used)
    {
        while (i < ranking.Count && used.Contains(ranking[i])) i++;
    }
}

public enum InterleavingOutcome { WinA, WinB, Tie, NoClicks }

public readonly record struct InterleavingCredit(int ClicksA, int ClicksB)
{
    /// <summary>
    /// Team-draft attribution: the team with more clicked documents wins the impression.
    /// Equal non-zero credit is a tie; an impression with no attributable click is
    /// <see cref="InterleavingOutcome.NoClicks"/> and carries no preference.
    /// </summary>
    public InterleavingOutcome Outcome =>
        ClicksA == 0 && ClicksB == 0 ? InterleavingOutcome.NoClicks
        : ClicksA > ClicksB ? InterleavingOutcome.WinA
        : ClicksB > ClicksA ? InterleavingOutcome.WinB
        : InterleavingOutcome.Tie;
}

public static class InterleavingAttribution
{
    /// <summary>
    /// Credits each distinct clicked document to the team that contributed it to the shown list.
    /// Repeat clicks on the same document count once; clicks on documents that were not in the
    /// served list are ignored (they cannot be attributed).
    /// </summary>
    public static InterleavingCredit Credit(IReadOnlyList<InterleavedDoc> shown, IEnumerable<ulong> clickedDocIds)
    {
        var teamOf = new Dictionary<ulong, Team>(shown.Count);
        foreach (var d in shown) teamOf.TryAdd(d.DocId, d.Team);
        int a = 0, b = 0;
        foreach (var id in clickedDocIds.Distinct())
        {
            if (!teamOf.TryGetValue(id, out var t)) continue;
            if (t == Team.A) a++; else b++;
        }
        return new InterleavingCredit(a, b);
    }
}
