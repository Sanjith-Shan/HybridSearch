using HybridSearch.Broker.Autocomplete;

namespace HybridSearch.Broker.Tests;

public class CompletionTrieTests
{
    private static List<Completion> BruteForce(Dictionary<string, long> counts, string prefix, int k) =>
        counts.Where(kv => kv.Key.StartsWith(prefix, StringComparison.Ordinal))
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => System.Text.Encoding.UTF8.GetBytes(kv.Key), ByteComparer.Instance)
            .Take(k)
            .Select(kv => new Completion(kv.Key, kv.Value))
            .ToList();

    private sealed class ByteComparer : IComparer<byte[]>
    {
        public static readonly ByteComparer Instance = new();
        public int Compare(byte[]? x, byte[]? y) => x.AsSpan().SequenceCompareTo(y);
    }

    public static IEnumerable<object[]> Seeds() => Enumerable.Range(0, 12).Select(i => new object[] { i });

    [Theory]
    [MemberData(nameof(Seeds))]
    public void Matches_brute_force_on_random_logs(int seed)
    {
        var rng = new Random(seed);
        // Small alphabet (incl. multi-byte chars and space) → deep shared prefixes, many ties.
        char[] alphabet = ['a', 'b', 'c', ' ', 'é', 'z', '日'];
        var counts = new Dictionary<string, long>();
        int n = rng.Next(1, 400);
        for (int i = 0; i < n; i++)
        {
            int len = rng.Next(1, 9);
            var s = new string(Enumerable.Range(0, len).Select(_ => alphabet[rng.Next(alphabet.Length)]).ToArray());
            counts[s] = counts.GetValueOrDefault(s) + rng.Next(1, 5);
        }
        int maxK = rng.Next(1, 12);
        var trie = CompletionTrie.Build(counts, maxK);
        Assert.Equal(counts.Count, trie.DistinctQueries);
        Assert.Equal(counts.Values.Sum(), trie.TotalQueries);

        var prefixes = new List<string> { "" };
        foreach (var q in counts.Keys) for (int i = 1; i <= q.Length; i++) prefixes.Add(q[..i]);
        for (int i = 0; i < 200; i++)
            prefixes.Add(new string(Enumerable.Range(0, rng.Next(0, 5)).Select(_ => alphabet[rng.Next(alphabet.Length)]).ToArray()));

        foreach (var p in prefixes.Distinct())
        {
            foreach (var k in new[] { 1, maxK, maxK + 3 })
            {
                var expected = BruteForce(counts, p, Math.Min(k, maxK));
                var actual = trie.Lookup(p, k);
                Assert.Equal(expected, actual);
            }
        }
        foreach (var (q, c) in counts) Assert.Equal(c, trie.CountOf(q));
        Assert.Equal(0, trie.CountOf("not-in-log-" + seed));
    }

    [Fact]
    public void Prefix_ending_inside_a_compressed_edge()
    {
        var trie = CompletionTrie.Build(new Dictionary<string, long> { ["how to lose weight"] = 5, ["how to tie a tie"] = 9, ["hello"] = 1 });
        Assert.Equal(["how to tie a tie", "how to lose weight"], trie.Lookup("how t", 5).Select(c => c.Text));
        Assert.Equal(["how to lose weight"], trie.Lookup("how to l", 5).Select(c => c.Text));
        Assert.Empty(trie.Lookup("how to x", 5));
        Assert.Equal(3, trie.Lookup("", 5).Count);
        Assert.Equal("how to tie a tie", trie.Lookup("", 1)[0].Text);
    }

    [Fact]
    public void Empty_trie_and_build_with_duplicates()
    {
        var empty = CompletionTrie.Build([]);
        Assert.Empty(empty.Lookup("a", 5));
        var trie = CompletionTrie.Build([new("x", 1), new("x", 2), new("", 5)]);
        Assert.Equal(new Completion("x", 3), trie.Lookup("", 5).Single());
    }

    [Fact]
    public async Task Query_log_loader_normalises_and_counts()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(path, ["1\tWhat  is   BM25", "2\twhat is bm25 ", "3\tWhat is Lucene", "4\t   ", "5\tno-tab-line"]);
            var (queries, words) = await AutocompleteService.LoadQueryLogAsync(path, CancellationToken.None);
            Assert.Equal(2, queries["what is bm25"]);
            Assert.Equal(1, queries["what is lucene"]);
            Assert.Equal(3, words["what"]);
            Assert.False(queries.ContainsKey(""));
        }
        finally { File.Delete(path); }
    }
}

public class QueryNormalizerTests
{
    [Theory]
    [InlineData("  How   TO\tlose\nweight ", "how to lose weight")]
    [InlineData("", "")]
    [InlineData("   ", "")]
    [InlineData("ÉCOLE", "école")]
    public void Normalize(string input, string expected) => Assert.Equal(expected, QueryNormalizer.Normalize(input));

    [Theory]
    [InlineData("how to ", "how to ")]
    [InlineData("how to   ", "how to ")]
    [InlineData("  How", "how")]
    [InlineData("   ", "")]
    public void NormalizePrefix_keeps_one_trailing_space(string input, string expected) =>
        Assert.Equal(expected, QueryNormalizer.NormalizePrefix(input));
}

public class SpellCorrectorTests
{
    private static SpellCorrector Build() => SpellCorrector.Build(new Dictionary<string, long>
    {
        ["capital"] = 100, ["peru"] = 50, ["weight"] = 80, ["eight"] = 10, ["photosynthesis"] = 20,
        ["capitol"] = 5, ["rare"] = 1, ["what"] = 1000,
    }, minFrequency: 3);

    [Theory]
    [InlineData("capitl", "capital")]
    [InlineData("captial", "capital")]    // transposition = 1 under OSA
    [InlineData("wieght", "weight")]
    [InlineData("photosynthesys", "photosynthesis")]
    [InlineData("peur", "peru")]
    public void Corrects_common_misspellings(string wrong, string right) => Assert.Equal(right, Build().CorrectWord(wrong));

    [Theory]
    [InlineData("capital")]   // known
    [InlineData("zzzzzzzz")]  // nothing close
    [InlineData("ab")]        // too short
    [InlineData("b52")]       // not alphabetic
    [InlineData("rarr")]      // only candidate is below min frequency
    public void Leaves_alone(string word) => Assert.Null(Build().CorrectWord(word));

    [Fact]
    public void Prefers_smaller_distance_then_frequency()
    {
        // "capitel": capital (1) and capitol (1) tie on distance; capital is more frequent.
        Assert.Equal("capital", Build().CorrectWord("capitel"));
    }

    [Fact]
    public void Suggest_whole_query()
    {
        var s = Build();
        Assert.Equal("what capital peru", s.Suggest("what capitl peur"));
        Assert.Null(s.Suggest("what capital peru"));
    }

    [Theory]
    [InlineData("abc", "abc", 0)]
    [InlineData("abc", "acb", 1)]
    [InlineData("abc", "", 3)]
    [InlineData("kitten", "sitting", 3)]
    [InlineData("ca", "abc", 3)] // OSA (not full Damerau): 3
    public void Osa_distance(string a, string b, int d) => Assert.Equal(Math.Min(d, 4), SpellCorrector.OsaDistance(a, b, 3));

    [Fact]
    public void Matches_brute_force_on_random_vocab()
    {
        var rng = new Random(11);
        var vocab = new Dictionary<string, long>();
        for (int i = 0; i < 300; i++)
        {
            var w = new string(Enumerable.Range(0, rng.Next(3, 9)).Select(_ => (char)('a' + rng.Next(6))).ToArray());
            vocab[w] = rng.Next(3, 100);
        }
        var s = SpellCorrector.Build(vocab, 3);
        for (int t = 0; t < 300; t++)
        {
            var q = new string(Enumerable.Range(0, rng.Next(3, 9)).Select(_ => (char)('a' + rng.Next(6))).ToArray());
            if (vocab.ContainsKey(q)) continue;
            var expected = vocab
                .Select(kv => (kv.Key, kv.Value, D: SpellCorrector.OsaDistance(q, kv.Key, 2)))
                .Where(x => x.D <= 2)
                .OrderBy(x => x.D).ThenByDescending(x => x.Value).ThenBy(x => x.Key, StringComparer.Ordinal)
                .Select(x => x.Key).FirstOrDefault();
            // Words longer than PrefixLength are matched on their prefix, so only compare short words.
            if (q.Length <= SpellCorrector.PrefixLength && vocab.Keys.All(k => k.Length <= SpellCorrector.PrefixLength))
                Assert.Equal(expected, s.CorrectWord(q));
        }
    }
}
