using System.Diagnostics;
using HybridSearch.Broker.Autocomplete;
using HybridSearch.Broker.Experiments;
using HybridSearch.Broker.Models;
using HybridSearch.Broker.Shards;
using Microsoft.Extensions.Options;

namespace HybridSearch.Broker;

/// <summary>Experiments from experiments.json (loaded once at startup; a bad file fails startup loudly).</summary>
public sealed class ExperimentRegistry
{
    public ExperimentRegistry(IOptions<BrokerOptions> options, IHostEnvironment env, ILogger<ExperimentRegistry> logger)
    {
        var path = options.Value.Experiments.ConfigPath;
        ConfigPath = Path.IsPathRooted(path) ? path : Path.Combine(env.ContentRootPath, path);
        if (File.Exists(ConfigPath))
        {
            Experiments = ExperimentsFile.Parse(File.ReadAllText(ConfigPath)).Experiments;
            logger.LogInformation("loaded {Count} experiments from {Path}", Experiments.Count, ConfigPath);
        }
        else
        {
            Experiments = [];
            logger.LogWarning("no experiments file at {Path}; experiments disabled", ConfigPath);
        }
    }

    /// <summary>For tests and tools: a registry over an explicit list.</summary>
    public ExperimentRegistry(IReadOnlyList<ExperimentDefinition> experiments)
    {
        ConfigPath = "(in-memory)";
        Experiments = experiments;
    }

    public string ConfigPath { get; }
    public IReadOnlyList<ExperimentDefinition> Experiments { get; }
    public ExperimentDefinition? Find(string id) => Experiments.FirstOrDefault(e => e.Id == id);
}

public enum ComponentState { Loading, Ready, Unavailable, Failed }

/// <summary>
/// Autocomplete + spelling data, built in the background at startup from the train query log
/// (~808K lines). Until it finishes, /api/suggest answers 503 and /readyz reports "loading".
/// A missing log is "unavailable" (reported, not fatal).
/// </summary>
public sealed class AutocompleteService(IOptions<BrokerOptions> options, IHostEnvironment env, ILogger<AutocompleteService> logger)
    : BackgroundService
{
    private volatile CompletionTrie? _trie;
    private volatile SpellCorrector? _speller;
    private volatile int _state = (int)ComponentState.Loading;

    public ComponentState State => (ComponentState)_state;
    public string Detail { get; private set; } = "loading";
    public CompletionTrie? Trie => _trie;
    public SpellCorrector? Speller => _speller;
    public TimeSpan? BuildTime { get; private set; }

    /// <summary>For tests: install prebuilt structures.</summary>
    public void Install(CompletionTrie trie, SpellCorrector? speller)
    {
        _trie = trie;
        _speller = speller;
        Detail = $"{trie.DistinctQueries} distinct queries (in-memory)";
        _state = (int)ComponentState.Ready;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (_trie is not null) return;
        var o = options.Value;
        var path = PathResolver.UnderData(o, env.ContentRootPath, o.Autocomplete.QueryLogPath);
        if (!File.Exists(path))
        {
            Detail = $"query log not found: {path}";
            _state = (int)ComponentState.Unavailable;
            logger.LogWarning("autocomplete unavailable: {Detail}", Detail);
            return;
        }
        try
        {
            // Yield so host startup is not delayed by the first synchronous stretch of work.
            await Task.Yield();
            var sw = Stopwatch.StartNew();
            var (queries, words) = await LoadQueryLogAsync(path, stoppingToken).ConfigureAwait(false);
            var trie = CompletionTrie.Build(queries, o.Autocomplete.TopK);
            var speller = o.Autocomplete.SpellCorrection ? SpellCorrector.Build(words, o.Autocomplete.SpellMinWordFrequency) : null;
            BuildTime = sw.Elapsed;
            _trie = trie;
            _speller = speller;
            Detail = $"{trie.DistinctQueries} distinct / {trie.TotalQueries} queries, {trie.NodeCount} nodes, " +
                     $"~{trie.ApproximateBytes / (1 << 20)} MiB, built in {sw.Elapsed.TotalSeconds:F1}s" +
                     (speller is null ? "" : $"; speller vocabulary {speller.VocabularySize}");
            _state = (int)ComponentState.Ready;
            logger.LogInformation("autocomplete ready: {Detail}", Detail);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            Detail = $"failed: {ex.Message}";
            _state = (int)ComponentState.Failed;
            logger.LogError(ex, "autocomplete build failed");
        }
    }

    /// <summary>Reads "qid \t text" lines; returns normalised query counts and word counts.</summary>
    public static async Task<(Dictionary<string, long> Queries, Dictionary<string, long> Words)> LoadQueryLogAsync(string path, CancellationToken ct)
    {
        var queries = new Dictionary<string, long>(StringComparer.Ordinal);
        var words = new Dictionary<string, long>(StringComparer.Ordinal);
        await using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        using var sr = new StreamReader(fs);
        string? line;
        while ((line = await sr.ReadLineAsync(ct).ConfigureAwait(false)) is not null)
        {
            int tab = line.IndexOf('\t');
            var text = QueryNormalizer.Normalize(tab >= 0 ? line.AsSpan(tab + 1) : line.AsSpan());
            if (text.Length == 0) continue;
            queries[text] = queries.GetValueOrDefault(text) + 1;
            foreach (var w in text.Split(' '))
                words[w] = words.GetValueOrDefault(w) + 1;
        }
        return (queries, words);
    }
}

/// <summary>Loads the ONNX models if their files exist; otherwise reports why dense / rerank are unavailable.</summary>
public sealed class ModelProvider : IDisposable
{
    public ModelProvider(IOptions<BrokerOptions> options, IHostEnvironment env, ILogger<ModelProvider> logger)
    {
        var o = options.Value;
        var m = o.Models;
        Encoder = CreateEncoder(o, m, env.ContentRootPath, logger);
        Reranker = CreateReranker(o, m, env.ContentRootPath, logger);
        logger.LogInformation("query encoder: {Encoder}; reranker: {Reranker}", Encoder.Status, Reranker.Status);
    }

    public ModelProvider(IQueryEncoder encoder, IReranker reranker)
    {
        Encoder = encoder;
        Reranker = reranker;
    }

    public IQueryEncoder Encoder { get; }
    public IReranker Reranker { get; }

    private static IQueryEncoder CreateEncoder(BrokerOptions o, ModelOptions m, string contentRoot, ILogger logger)
    {
        if (string.Equals(m.QueryEncoderKind, "hashing-dev", StringComparison.OrdinalIgnoreCase))
        {
            logger.LogWarning("query encoder is the hashing-dev stand-in: dense results are NOT from a real model");
            return new HashingDevEncoder(m.HashingDevDimension);
        }
        var dir = PathResolver.UnderData(o, contentRoot, m.QueryEncoderDir);
        string model = Path.Combine(dir, "model.onnx"), vocab = Path.Combine(dir, "vocab.txt");
        if (!File.Exists(model) || !File.Exists(vocab))
            return new UnavailableEncoder($"query encoder missing ({model} / vocab.txt not found): dense retrieval unavailable");
        try
        {
            return new OnnxQueryEncoder(model, vocab, m.QueryPrefix, m.QueryMaxTokens, m.IntraOpThreads);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "failed to load query encoder");
            return new UnavailableEncoder($"query encoder failed to load: {ex.Message}");
        }
    }

    private static IReranker CreateReranker(BrokerOptions o, ModelOptions m, string contentRoot, ILogger logger)
    {
        var dir = PathResolver.UnderData(o, contentRoot, m.RerankerDir);
        string int8 = Path.Combine(dir, "model.int8.onnx"), fp32 = Path.Combine(dir, "model.onnx"), vocab = Path.Combine(dir, "vocab.txt");
        string? model = m.PreferInt8Reranker && File.Exists(int8) ? int8 : File.Exists(fp32) ? fp32 : File.Exists(int8) ? int8 : null;
        if (model is null || !File.Exists(vocab))
            return new UnavailableReranker($"reranker missing (no model.onnx / model.int8.onnx + vocab.txt in {dir})");
        try
        {
            return new OnnxCrossEncoder(model, vocab, m.RerankMaxTokens, m.RerankBatchSize, m.IntraOpThreads);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "failed to load reranker");
            return new UnavailableReranker($"reranker failed to load: {ex.Message}");
        }
    }

    public void Dispose()
    {
        (Encoder as IDisposable)?.Dispose();
        (Reranker as IDisposable)?.Dispose();
    }
}

/// <summary>Probes every replica's Health RPC periodically; drives readiness and primary selection.</summary>
public sealed class ShardHealthMonitor(ShardFanOut fanOut, IOptions<BrokerOptions> options, ILogger<ShardHealthMonitor> logger) : BackgroundService
{
    private int _rounds;
    public int Rounds => Volatile.Read(ref _rounds);

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var h = options.Value.Health;
        using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(Math.Max(50, h.IntervalMs)));
        do
        {
            await ProbeAllAsync(TimeSpan.FromMilliseconds(h.TimeoutMs), stoppingToken).ConfigureAwait(false);
        }
        while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false));
    }

    public async Task ProbeAllAsync(TimeSpan timeout, CancellationToken ct)
    {
        var replicas = fanOut.Topology.Slices.SelectMany(s => s.Replicas).ToList();
        await Task.WhenAll(replicas.Select(async r =>
        {
            var before = r.Health;
            var resp = await fanOut.ProbeAsync(r, timeout, ct).ConfigureAwait(false);
            if (before != r.Health)
                logger.LogInformation("shard {Slice} replica {Replica} ({Endpoint}) is now {Health}", r.SliceId, r.Index, r.Endpoint, r.Health);
            _ = resp;
        })).ConfigureAwait(false);
        Interlocked.Increment(ref _rounds);
    }
}
