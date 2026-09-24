using System.Globalization;
using Grpc.Core;

namespace HybridSearch.FakeShard;

/// <summary>
/// dotnet run --project broker/HybridSearch.FakeShard -- --port 50051 --slice 0 [--num-slices N] [--corpus file.tsv]
///     [--max-docs N] [--dim 64] [--latency-ms 5] [--jitter-ms 2] [--tail-prob 0.01] [--tail-ms 100]
///     [--fail-rate 0] [--partial-rate 0] [--no-vector] [--ignore-budget]
/// </summary>
public static class FakeShardProgram
{
    public static async Task<int> Main(string[] args)
    {
        var a = ParseArgs(args);
        if (a.ContainsKey("help") || a.ContainsKey("h"))
        {
            Console.WriteLine("usage: FakeShard --port 50051 --slice 0 [--num-slices 1] [--corpus <tsv>] [--max-docs N] [--dim 64] " +
                              "[--latency-ms 0] [--jitter-ms 0] [--tail-prob 0] [--tail-ms 0] [--fail-rate 0] [--partial-rate 0] [--no-vector] [--ignore-budget]");
            return 0;
        }
        int port = Int(a, "port", 50051);
        int slice = Int(a, "slice", 0);
        int numSlices = Int(a, "num-slices", 1);
        int dim = Int(a, "dim", 64);
        int maxDocs = Int(a, "max-docs", int.MaxValue);

        IEnumerable<Passage> passages = a.TryGetValue("corpus", out var path) ? FakeCorpus.ReadTsv(path).Take(maxDocs) : FakeCorpus.BuiltIn;
        var corpus = new FakeCorpus(FakeCorpus.Slice(passages, slice, numSlices), dim);

        var faults = new FaultOptions
        {
            LatencyMs = Int(a, "latency-ms", 0),
            JitterMs = Int(a, "jitter-ms", 0),
            TailProbability = Dbl(a, "tail-prob", 0),
            TailMs = Int(a, "tail-ms", 0),
            FailRate = Dbl(a, "fail-rate", 0),
            FailStatus = StatusCode.Unavailable,
            PartialRate = Dbl(a, "partial-rate", 0),
            VectorReady = !a.ContainsKey("no-vector"),
            HonorBudget = !a.ContainsKey("ignore-budget"),
        };

        await using var server = await FakeShardServer.StartAsync(corpus, (uint)slice, port, faults, quiet: false, anyInterface: true);
        Console.WriteLine($"FakeShard slice {slice}/{numSlices}: {corpus.Passages.Count} docs [{corpus.FirstDocId}..{corpus.LastDocId}], " +
                          $"dim {dim}, listening on {server.Address} (h2c). Ctrl+C to stop.");
        var done = new TaskCompletionSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; done.TrySetResult(); };
        AppDomain.CurrentDomain.ProcessExit += (_, _) => done.TrySetResult();
        await done.Task;
        return 0;
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) && !args[i].StartsWith('-')) continue;
            var key = args[i].TrimStart('-');
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal)) d[key] = args[++i];
            else d[key] = "true";
        }
        return d;
    }

    private static int Int(Dictionary<string, string> a, string k, int def) =>
        a.TryGetValue(k, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : def;

    private static double Dbl(Dictionary<string, string> a, string k, double def) =>
        a.TryGetValue(k, out var v) ? double.Parse(v, CultureInfo.InvariantCulture) : def;
}
