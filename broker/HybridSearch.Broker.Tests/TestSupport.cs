using HybridSearch.Broker.Observability;
using HybridSearch.Broker.Shards;
using HybridSearch.FakeShard;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace HybridSearch.Broker.Tests;

public static class TestPaths
{
    public static string BrokerDir
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "HybridSearch.sln"))) dir = dir.Parent;
            return dir?.FullName ?? throw new InvalidOperationException("broker/ not found");
        }
    }

    public static string BrokerProjectDir => Path.Combine(BrokerDir, "HybridSearch.Broker");
}

/// <summary>Builds a ShardFanOut against in-process FakeShards (real gRPC over loopback h2c).</summary>
public sealed class FanOutHarness : IAsyncDisposable
{
    public required List<List<FakeShardServer>> Servers { get; init; }
    public required ShardFanOut FanOut { get; init; }
    public required ShardTopology Topology { get; init; }
    public required BrokerMetrics Metrics { get; init; }

    public static async Task<FanOutHarness> StartAsync(int slices, int replicas, Action<int, int, FaultOptions>? faults = null,
        Action<BrokerOptions>? configure = null)
    {
        var servers = new List<List<FakeShardServer>>();
        var topo = new TopologyOptions();
        for (int s = 0; s < slices; s++)
        {
            var corpus = new FakeCorpus(FakeCorpus.Slice(FakeCorpus.BuiltIn, s, slices));
            var reps = new List<FakeShardServer>();
            for (int r = 0; r < replicas; r++)
            {
                var f = new FaultOptions();
                faults?.Invoke(s, r, f);
                reps.Add(await FakeShardServer.StartAsync(corpus, (uint)s, 0, f));
            }
            servers.Add(reps);
            topo.Slices.Add(new SliceOptions { Id = s, Replicas = reps.Select(x => x.Address).ToList() });
        }
        var o = new BrokerOptions { Topology = topo };
        configure?.Invoke(o);
        var topology = new ShardTopology(o.Topology, o.Hedging);
        var metrics = new BrokerMetrics();
        var fanOut = new ShardFanOut(topology, Options.Create(o), metrics, NullLogger<ShardFanOut>.Instance);
        return new FanOutHarness { Servers = servers, FanOut = fanOut, Topology = topology, Metrics = metrics };
    }

    public async ValueTask DisposeAsync()
    {
        Topology.Dispose();
        foreach (var s in Servers.SelectMany(x => x)) await s.DisposeAsync();
    }
}
