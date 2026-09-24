using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;

namespace HybridSearch.Broker.Autocomplete;

/// <summary>
/// SymSpell-style correction (Garbe's symmetric delete) over the query-log vocabulary.
///
/// Offline, every vocabulary word contributes the deletes (up to <see cref="MaxDistance"/>) of
/// its first <see cref="PrefixLength"/> characters. Online, the misspelling's own deletes are
/// looked up; any word sharing a delete is a candidate, verified with the real
/// optimal-string-alignment distance. Deletes are stored as a sorted (hash, wordId) array rather
/// than a dictionary of lists: ~12 bytes per entry, binary-searched. Hash collisions only add
/// candidates that the exact distance check then rejects.
/// </summary>
public sealed class SpellCorrector
{
    public const int MaxDistance = 2;
    public const int PrefixLength = 7;

    private readonly string[] _words;
    private readonly long[] _freq;
    private readonly Dictionary<string, int> _index;
    private readonly ulong[] _deleteHash;
    private readonly int[] _deleteWord;
    private readonly long _minKnownFrequency;

    public int VocabularySize => _words.Length;

    private SpellCorrector(string[] words, long[] freq, ulong[] deleteHash, int[] deleteWord, long minKnownFrequency)
    {
        _words = words; _freq = freq; _deleteHash = deleteHash; _deleteWord = deleteWord;
        _minKnownFrequency = minKnownFrequency;
        _index = new Dictionary<string, int>(words.Length, StringComparer.Ordinal);
        for (int i = 0; i < words.Length; i++) _index[words[i]] = i;
    }

    /// <param name="wordCounts">Word → frequency, from normalised queries.</param>
    /// <param name="minFrequency">Words rarer than this are not correction targets (log noise, typos themselves).</param>
    public static SpellCorrector Build(IEnumerable<KeyValuePair<string, long>> wordCounts, long minFrequency = 3)
    {
        var words = new List<string>();
        var freq = new List<long>();
        foreach (var (w, c) in wordCounts)
        {
            if (c < minFrequency || !IsCorrectable(w)) continue;
            words.Add(w); freq.Add(c);
        }
        var pairs = new List<(ulong Hash, int Word)>();
        var deletes = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < words.Count; i++)
        {
            deletes.Clear();
            var key = words[i].Length > PrefixLength ? words[i][..PrefixLength] : words[i];
            deletes.Add(key);
            AddDeletes(key, MaxDistance, deletes);
            foreach (var d in deletes) pairs.Add((Hash(d), i));
        }
        pairs.Sort(static (a, b) => a.Hash != b.Hash ? a.Hash.CompareTo(b.Hash) : a.Word.CompareTo(b.Word));
        var hashes = new ulong[pairs.Count];
        var ids = new int[pairs.Count];
        for (int i = 0; i < pairs.Count; i++) { hashes[i] = pairs[i].Hash; ids[i] = pairs[i].Word; }
        return new SpellCorrector([.. words], [.. freq], hashes, ids, minFrequency);
    }

    /// <summary>Best correction for one word, or null if it is already known or nothing is close enough.</summary>
    public string? CorrectWord(string word)
    {
        if (!IsCorrectable(word) || _index.ContainsKey(word)) return null;
        var key = word.Length > PrefixLength ? word[..PrefixLength] : word;
        var probes = new HashSet<string>(StringComparer.Ordinal) { key };
        AddDeletes(key, MaxDistance, probes);

        int bestWord = -1, bestDist = int.MaxValue;
        var seen = new HashSet<int>();
        foreach (var p in probes)
        {
            var h = Hash(p);
            int i = LowerBound(h);
            for (; i < _deleteHash.Length && _deleteHash[i] == h; i++)
            {
                int w = _deleteWord[i];
                if (!seen.Add(w)) continue;
                var cand = _words[w];
                if (Math.Abs(cand.Length - word.Length) > MaxDistance) continue;
                int d = OsaDistance(word, cand, MaxDistance);
                if (d > MaxDistance) continue;
                if (d < bestDist || (d == bestDist && (_freq[w] > _freq[bestWord] ||
                    (_freq[w] == _freq[bestWord] && string.CompareOrdinal(cand, _words[bestWord]) < 0))))
                {
                    bestDist = d; bestWord = w;
                }
            }
        }
        return bestWord >= 0 ? _words[bestWord] : null;
    }

    /// <summary>
    /// "Did you mean" for a whole query: corrects each unknown word independently. Returns null
    /// when nothing changes. Words with digits, very short words and known words are left alone.
    /// </summary>
    public string? Suggest(string normalizedQuery)
    {
        var tokens = normalizedQuery.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        bool changed = false;
        for (int i = 0; i < tokens.Length; i++)
        {
            var c = CorrectWord(tokens[i]);
            if (c is not null && c != tokens[i]) { tokens[i] = c; changed = true; }
        }
        return changed ? string.Join(' ', tokens) : null;
    }

    public bool IsKnown(string word) => _index.ContainsKey(word);

    /// <summary>Only purely alphabetic words of length ≥ 3 are corrected (numbers, codes, "a", "of" are not).</summary>
    public static bool IsCorrectable(string w)
    {
        if (w.Length < 3 || w.Length > 30) return false;
        foreach (var ch in w) if (!char.IsLetter(ch)) return false;
        return true;
    }

    /// <summary>Optimal string alignment distance (Damerau–Levenshtein without repeated edits), early-exit above <paramref name="max"/>.</summary>
    public static int OsaDistance(string a, string b, int max)
    {
        int n = a.Length, m = b.Length;
        if (Math.Abs(n - m) > max) return max + 1;
        var prev2 = new int[m + 1];
        var prev = new int[m + 1];
        var cur = new int[m + 1];
        for (int j = 0; j <= m; j++) prev[j] = j;
        for (int i = 1; i <= n; i++)
        {
            cur[0] = i;
            int rowMin = cur[0];
            for (int j = 1; j <= m; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                int v = Math.Min(Math.Min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1])
                    v = Math.Min(v, prev2[j - 2] + 1);
                cur[j] = v;
                rowMin = Math.Min(rowMin, v);
            }
            if (rowMin > max) return max + 1;
            (prev2, prev, cur) = (prev, cur, prev2);
        }
        return prev[m];
    }

    private static void AddDeletes(string s, int depth, HashSet<string> into)
    {
        if (depth == 0 || s.Length <= 1) return;
        for (int i = 0; i < s.Length; i++)
        {
            var d = s.Remove(i, 1);
            if (into.Add(d)) AddDeletes(d, depth - 1, into);
        }
    }

    private static ulong Hash(string s) => XxHash64.HashToUInt64(MemoryMarshal.AsBytes(s.AsSpan()));

    private int LowerBound(ulong h)
    {
        int lo = 0, hi = _deleteHash.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (_deleteHash[mid] < h) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
}
