using BenchmarkDotNet.Attributes;
using HybridSearch.Broker.Autocomplete;
using HybridSearch.Broker.Experiments;
using HybridSearch.Broker.Search;

namespace HybridSearch.Broker.Bench;

/// <summary>Shared query-log trie (built once per benchmark process).</summary>
public static class QueryLog
{
    public static string Path => Environment.GetEnvironmentVariable("HS_QUERY_LOG")
        ?? System.IO.Path.Combine(Meta.RepoRoot, "data", "raw", "msmarco", "queries.train.tsv");

    private static (CompletionTrie Trie, List<string> Queries)? _cache;

    public static (CompletionTrie Trie, List<string> Queries) Load()
    {
        if (_cache is { } c) return c;
        var (queries, _) = AutocompleteService.LoadQueryLogAsync(Path, CancellationToken.None).GetAwaiter().GetResult();
        var trie = CompletionTrie.Build(queries, 10);
        _cache = (trie, queries.Keys.ToList());
        return _cache.Value;
    }

    /// <summary>Realistic keystroke prefixes: random real queries cut at a random length (1..len), seeded.</summary>
    public static string[] Prefixes(List<string> queries, int n, int seed = 20260923)
    {
        var rng = new Random(seed);
        var r = new string[n];
        for (int i = 0; i < n; i++)
        {
            var q = queries[rng.Next(queries.Count)];
            r[i] = q[..rng.Next(1, q.Length + 1)];
        }
        return r;
    }
}

[MemoryDiagnoser]
public class SuggestBench
{
    private const int N = 10_000;
    private CompletionTrie _trie = null!;
    private string[] _prefixes = null!;
    private string[] _raw = null!;

    [GlobalSetup]
    public void Setup()
    {
        var (trie, queries) = QueryLog.Load();
        _trie = trie;
        _prefixes = QueryLog.Prefixes(queries, N);
        _raw = _prefixes.Select(p => "  " + p.ToUpperInvariant()).ToArray();
    }

    /// <summary>Trie lookup of an already-normalised prefix, k = 8 (the UI default). Per-op time = per keystroke.</summary>
    [Benchmark(OperationsPerInvoke = N)]
    public int Lookup_k8()
    {
        int total = 0;
        foreach (var p in _prefixes) total += _trie.Lookup(p, 8).Count;
        return total;
    }

    /// <summary>What /api/suggest does per request: normalise the raw prefix, then look up.</summary>
    [Benchmark(OperationsPerInvoke = N)]
    public int NormalizeAndLookup_k8()
    {
        int total = 0;
        foreach (var p in _raw) total += _trie.Lookup(QueryNormalizer.NormalizePrefix(p), 8).Count;
        return total;
    }
}

[MemoryDiagnoser]
public class FusionBench
{
    private List<ScoredDoc> _lex = null!, _dense = null!;

    [Params(100, 1000)]
    public int Depth { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(1);
        _lex = Enumerable.Range(0, Depth).Select(i => new ScoredDoc((ulong)rng.Next(0, Depth * 3), (float)(30 - i * 0.01))).ToList();
        _dense = Enumerable.Range(0, Depth).Select(i => new ScoredDoc((ulong)rng.Next(0, Depth * 3), (float)(0.9 - i * 0.0001))).ToList();
    }

    [Benchmark] public int Rrf() => Fusion.ReciprocalRank([_lex, _dense], 60).Count;
    [Benchmark] public int WeightedMinMax() => Fusion.Weighted(_lex, _dense, 0.5, ScoreNormalization.MinMax).Count;
    [Benchmark] public int WeightedZScore() => Fusion.Weighted(_lex, _dense, 0.5, ScoreNormalization.ZScore).Count;
}

[MemoryDiagnoser]
public class InterleaveBench
{
    private List<ulong> _a = null!, _b = null!;
    private ulong _seed;

    [Params(10, 100)]
    public int K { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(2);
        _a = Enumerable.Range(0, 100).Select(_ => (ulong)rng.Next(0, 150)).Distinct().ToList();
        _b = Enumerable.Range(0, 100).Select(_ => (ulong)rng.Next(0, 150)).Distinct().ToList();
    }

    [Benchmark] public int TeamDraft() => TeamDraftInterleaver.Interleave(_a, _b, K, _seed++).Count;
}
