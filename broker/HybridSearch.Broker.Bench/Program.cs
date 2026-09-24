using System.Diagnostics;
using System.Globalization;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using HybridSearch.Broker;
using HybridSearch.Broker.Autocomplete;
using HybridSearch.Broker.Bench;
using HybridSearch.Broker.Observability;
using HybridSearch.Broker.Shards;
using HybridSearch.FakeShard;
using HybridSearch.Proto.V1;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

// Usage:
//   dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- bdn --filter '*'     BenchmarkDotNet suites
//   dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- hedging [rate] [seconds]
//   dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- trie-stats
var command = "dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- " + string.Join(' ', args);
var load0 = Meta.LoadAverage();

switch (args.FirstOrDefault())
{
    case "hedging":
        await HedgingExperiment.RunAsync(args.Skip(1).ToArray(), command, load0);
        break;
    case "trie-stats":
        TrieStats.Run(command, load0);
        break;
    case "bdn":
        var artifacts = Path.Combine(Meta.ResultsDir, "bdn");
        var summaries = BenchmarkSwitcher.FromAssembly(typeof(SuggestBench).Assembly)
            .Run(args.Skip(1).ToArray(), DefaultConfig.Instance.WithArtifactsPath(artifacts));
        foreach (var s in summaries)
        {
            Meta.Write("bdn-" + s.Title.Split('-')[0].Split('.').Last(), new
            {
                reports = s.Reports.Select(r => new
                {
                    benchmark = r.BenchmarkCase.DisplayInfo,
                    meanNs = r.ResultStatistics?.Mean,
                    medianNs = r.ResultStatistics?.Median,
                    stdDevNs = r.ResultStatistics?.StandardDeviation,
                    p95Ns = r.ResultStatistics?.Percentiles?.P95,
                    allocatedBytesPerOp = r.GcStats.GetBytesAllocatedPerOperation(r.BenchmarkCase),
                }),
                bdnReportDir = Path.GetRelativePath(Meta.RepoRoot, s.ResultsDirectoryPath),
            }, Meta.Collect(command, load0, new() { ["tool"] = "BenchmarkDotNet 0.15.8", ["configuration"] = "Release" }));
        }
        break;
    default:
        Console.WriteLine("usage: bdn [--filter ...] | hedging [ratePerSec] [seconds] | trie-stats");
        break;
}

static class TrieStats
{
    public static void Run(string command, string load0)
    {
        var sw = Stopwatch.StartNew();
        var (queries, words) = AutocompleteService.LoadQueryLogAsync(QueryLog.Path, CancellationToken.None).GetAwaiter().GetResult();
        var readMs = sw.Elapsed.TotalMilliseconds;
        long before = GC.GetTotalMemory(true);
        sw.Restart();
        var trie = CompletionTrie.Build(queries, 10);
        var buildMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        var speller = SpellCorrector.Build(words, 3);
        var spellMs = sw.Elapsed.TotalMilliseconds;
        long after = GC.GetTotalMemory(true);
        GC.KeepAlive(trie); GC.KeepAlive(speller);
        Meta.Write("autocomplete_build", new
        {
            queryLog = Path.GetRelativePath(Meta.RepoRoot, QueryLog.Path),
            totalQueries = trie.TotalQueries,
            distinctNormalizedQueries = trie.DistinctQueries,
            trieNodes = trie.NodeCount,
            trieArrayBytes = trie.ApproximateBytes,
            topK = trie.MaxK,
            spellerVocabulary = speller.VocabularySize,
            readAndNormalizeMs = Math.Round(readMs, 1),
            trieBuildMs = Math.Round(buildMs, 1),
            spellerBuildMs = Math.Round(spellMs, 1),
            managedHeapDeltaBytes = after - before,
            note = "single run, not repeated; wall-clock on a shared dev laptop",
        }, Meta.Collect(command, load0));
    }
}

static class HedgingExperiment
{
    /// <summary>
    /// Open-loop (constant arrival rate) load against one slice with two FakeShard replicas whose
    /// latency has an injected tail. Latency is measured from each request's *scheduled* send time,
    /// so a stalled client cannot hide queueing (no coordinated omission). The same seed-free fault
    /// distribution is used for every configuration. This measures the hedging mechanism, not the
    /// C++ engine: FakeShard's latency is synthetic.
    /// </summary>
    public static async Task RunAsync(string[] args, string command, string load0)
    {
        double rate = args.Length > 0 ? double.Parse(args[0], CultureInfo.InvariantCulture) : 100;
        int seconds = args.Length > 1 ? int.Parse(args[1], CultureInfo.InvariantCulture) : 30;
        var faults = (Base: 5, Jitter: 5, TailProb: 0.02, TailMs: 200);
        var configs = new (string Name, bool Enabled, double Budget)[]
        {
            ("no-hedging", false, 0), ("hedge-p95-budget-5pct", true, 0.05), ("hedge-p95-budget-10pct", true, 0.10),
        };
        var rows = new List<object>();
        foreach (var (name, enabled, budget) in configs)
        {
            var corpus = new FakeCorpus(FakeCorpus.BuiltIn);
            var servers = new List<FakeShardServer>();
            for (int r = 0; r < 2; r++)
                servers.Add(await FakeShardServer.StartAsync(corpus, 0, 0, new FaultOptions
                {
                    LatencyMs = faults.Base, JitterMs = faults.Jitter, TailProbability = faults.TailProb, TailMs = faults.TailMs,
                }));
            var o = new BrokerOptions
            {
                Topology = new TopologyOptions { Slices = [new SliceOptions { Id = 0, Replicas = servers.Select(s => s.Address).ToList() }] },
                Hedging = new HedgingOptions { Enabled = enabled, BudgetRatio = budget, BurstTokens = 10, MinSamples = 50, Quantile = 0.95 },
            };
            using var topology = new ShardTopology(o.Topology, o.Hedging);
            var fanOut = new ShardFanOut(topology, Options.Create(o), new BrokerMetrics(), NullLogger<ShardFanOut>.Instance);
            var req = new SearchRequest { Query = "capital of peru", Lexical = new LexicalParams { Enabled = true, K = 10 } };

            // Warm-up (connections, JIT, the p95 tracker's MinSamples) at the same rate, not measured.
            await Drive(fanOut, req, rate, 5);
            long p0 = topology.Slices[0].Primaries, h0 = topology.Slices[0].Hedges;
            long calls0 = servers.Sum(s => Interlocked.Read(ref s.Seen.SearchCalls));
            var (lat, failures) = await Drive(fanOut, req, rate, seconds);
            long primaries = topology.Slices[0].Primaries - p0, hedges = topology.Slices[0].Hedges - h0;
            long calls = servers.Sum(s => Interlocked.Read(ref s.Seen.SearchCalls)) - calls0;
            lat.Sort();
            double P(double q) => Math.Round(lat[Math.Min(lat.Count - 1, (int)Math.Ceiling(q * lat.Count) - 1)], 2);
            var row = new
            {
                config = name, hedgingEnabled = enabled, budgetRatio = budget, requests = lat.Count, failures,
                p50Ms = P(0.50), p90Ms = P(0.90), p95Ms = P(0.95), p99Ms = P(0.99), p999Ms = P(0.999), maxMs = Math.Round(lat[^1], 2),
                primaries, hedgesSent = hedges,
                extraLoadRatioBrokerCounted = primaries == 0 ? 0 : Math.Round((double)hedges / primaries, 4),
                shardCallsObserved = calls,
                extraLoadRatioShardObserved = primaries == 0 ? 0 : Math.Round((double)(calls - primaries) / primaries, 4),
            };
            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(row));
            rows.Add(row);
            foreach (var s in servers) await s.DisposeAsync();
        }
        Meta.Write("hedging_fakeshard", new
        {
            description = "Open-loop hedging experiment: 1 slice x 2 FakeShard replicas (in-process, loopback gRPC), injected latency " +
                          $"{faults.Base}ms + U[0,{faults.Jitter})ms, {faults.TailProb:P0} of requests +{faults.TailMs}ms. " +
                          "Latency from scheduled send time. Synthetic shard latency: measures the hedging mechanism, not the engine.",
            ratePerSecond = rate, measuredSecondsPerConfig = seconds, warmupSeconds = 5, rows,
        }, Meta.Collect(command, load0, new() { ["configuration"] = "Release", ["arrival"] = "constant-rate open loop" }));
    }

    private static async Task<(List<double> Latencies, int Failures)> Drive(ShardFanOut fanOut, SearchRequest req, double rate, int seconds)
    {
        int n = (int)(rate * seconds);
        var tasks = new Task<(double, bool)>[n];
        var start = Stopwatch.GetTimestamp();
        double intervalTicks = Stopwatch.Frequency / rate;
        for (int i = 0; i < n; i++)
        {
            long scheduled = start + (long)(i * intervalTicks);
            var wait = TimeSpan.FromSeconds((double)(scheduled - Stopwatch.GetTimestamp()) / Stopwatch.Frequency);
            if (wait > TimeSpan.FromMilliseconds(1)) await Task.Delay(wait);
            tasks[i] = One(fanOut, req, scheduled);
        }
        var results = await Task.WhenAll(tasks);
        return (results.Where(r => r.Item2).Select(r => r.Item1).ToList(), results.Count(r => !r.Item2));
    }

    private static async Task<(double, bool)> One(ShardFanOut fanOut, SearchRequest req, long scheduled)
    {
        var o = await fanOut.SearchAsync(req, Deadline.FromNow(TimeSpan.FromSeconds(2)), CancellationToken.None);
        double ms = (Stopwatch.GetTimestamp() - scheduled) * 1000.0 / Stopwatch.Frequency;
        return (ms, o[0].Status != SliceStatus.Failed);
    }
}
