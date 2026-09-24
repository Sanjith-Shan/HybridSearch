using System.Diagnostics;
using Grpc.Net.Client;
using HybridSearch.Proto.V1;

namespace HybridSearch.Broker.Shards;

public enum ReplicaHealth { Unknown = 0, Healthy = 1, Unhealthy = 2 }

public sealed class ReplicaState : IDisposable
{
    private int _health;
    private HealthResponse? _lastHealth;

    public ReplicaState(int sliceId, int index, string endpoint)
    {
        SliceId = sliceId;
        Index = index;
        Endpoint = endpoint;
        var handler = new SocketsHttpHandler
        {
            EnableMultipleHttp2Connections = true,
            PooledConnectionIdleTimeout = Timeout.InfiniteTimeSpan,
            KeepAlivePingDelay = TimeSpan.FromSeconds(30),
            KeepAlivePingTimeout = TimeSpan.FromSeconds(10),
            ConnectTimeout = TimeSpan.FromSeconds(2),
            // Trace context is written into gRPC metadata explicitly (Telemetry.TraceMetadata);
            // turn off HttpClient's own injection so the shard sees exactly one traceparent.
            ActivityHeadersPropagator = null,
        };
        Channel = GrpcChannel.ForAddress(endpoint, new GrpcChannelOptions { HttpHandler = handler, DisposeHttpClient = true });
        Client = new Shard.ShardClient(Channel);
    }

    public int SliceId { get; }
    public int Index { get; }
    public string Endpoint { get; }
    public GrpcChannel Channel { get; }
    public Shard.ShardClient Client { get; }

    public ReplicaHealth Health => (ReplicaHealth)Volatile.Read(ref _health);
    public HealthResponse? LastHealth => Volatile.Read(ref _lastHealth);

    public void MarkHealthy(HealthResponse h)
    {
        Volatile.Write(ref _lastHealth, h);
        Volatile.Write(ref _health, (int)ReplicaHealth.Healthy);
    }

    public void MarkUnhealthy() => Volatile.Write(ref _health, (int)ReplicaHealth.Unhealthy);

    public void Dispose() => Channel.Dispose();
}

/// <summary>One disjoint doc-ID slice and its replicas, with the slice's latency tracker and hedge budget.</summary>
public sealed class SliceState : IDisposable
{
    private int _rr = -1;
    private long _primaries;
    private long _hedges;

    public SliceState(int id, IReadOnlyList<ReplicaState> replicas, HedgingOptions hedging)
    {
        if (replicas.Count == 0) throw new ArgumentException($"slice {id} has no replicas");
        Id = id;
        Replicas = replicas;
        Tracker = new LatencyTracker(TimeSpan.FromSeconds(Math.Max(1, hedging.WindowSeconds)));
        Budget = new HedgeBudget(hedging.BudgetRatio, hedging.BurstTokens);
    }

    public int Id { get; }
    public string Label => Id.ToString(System.Globalization.CultureInfo.InvariantCulture);
    public IReadOnlyList<ReplicaState> Replicas { get; }
    public LatencyTracker Tracker { get; }
    public HedgeBudget Budget { get; }
    public long Primaries => Interlocked.Read(ref _primaries);
    public long Hedges => Interlocked.Read(ref _hedges);

    /// <summary>hedges sent / primary requests, i.e. the fractional extra load hedging put on this slice.</summary>
    public double ExtraLoadRatio => Primaries == 0 ? 0 : (double)Hedges / Primaries;

    internal void CountPrimary() { Interlocked.Increment(ref _primaries); Budget.OnPrimary(); }
    internal void CountHedge() => Interlocked.Increment(ref _hedges);

    /// <summary>Doc-ID range the slice reported in Health, if any replica has answered.</summary>
    public (ulong First, ulong Last)? Range
    {
        get
        {
            foreach (var r in Replicas)
                if (r.LastHealth is { } h && h.NumDocs > 0) return (h.FirstDocId, h.LastDocId);
            return null;
        }
    }

    public bool IsReady => Replicas.Any(r => r.Health == ReplicaHealth.Healthy);

    /// <summary>Round-robin over replicas not known to be unhealthy; if all are unhealthy, over all of them.</summary>
    public ReplicaState PickPrimary()
    {
        int n = Replicas.Count;
        int start = Interlocked.Increment(ref _rr) & int.MaxValue;
        for (int i = 0; i < n; i++)
        {
            var r = Replicas[(start + i) % n];
            if (r.Health != ReplicaHealth.Unhealthy) return r;
        }
        return Replicas[start % n];
    }

    /// <summary>Another replica for a hedge / failover, preferring healthy ones; null if there is none.</summary>
    public ReplicaState? PickAlternate(IReadOnlyCollection<ReplicaState> exclude)
    {
        ReplicaState? fallback = null;
        foreach (var r in Replicas)
        {
            if (exclude.Contains(r)) continue;
            if (r.Health != ReplicaHealth.Unhealthy) return r;
            fallback ??= r;
        }
        return fallback;
    }

    public void Dispose()
    {
        foreach (var r in Replicas) r.Dispose();
    }
}

public sealed class ShardTopology : IDisposable
{
    public ShardTopology(TopologyOptions topology, HedgingOptions hedging)
    {
        var slices = new List<SliceState>();
        var ids = new HashSet<int>();
        foreach (var s in topology.Slices)
        {
            if (!ids.Add(s.Id)) throw new InvalidOperationException($"duplicate slice id {s.Id}");
            var replicas = s.Replicas.Select((ep, i) => new ReplicaState(s.Id, i, ep)).ToList();
            slices.Add(new SliceState(s.Id, replicas, hedging));
        }
        Slices = slices;
    }

    public IReadOnlyList<SliceState> Slices { get; }

    public bool IsReady => Slices.Count > 0 && Slices.All(s => s.IsReady);

    /// <summary>The slice whose reported doc range contains <paramref name="docId"/>, or null if unknown.</summary>
    public SliceState? SliceForDoc(ulong docId) =>
        Slices.FirstOrDefault(s => s.Range is { } r && docId >= r.First && docId <= r.Last);

    public void Dispose()
    {
        foreach (var s in Slices) s.Dispose();
    }
}

/// <summary>An absolute point in monotonic time; the budget of a request or of one of its phases.</summary>
public readonly record struct Deadline(long EndTimestamp)
{
    public static Deadline FromNow(TimeSpan budget) =>
        new(Stopwatch.GetTimestamp() + (long)(budget.TotalSeconds * Stopwatch.Frequency));

    public TimeSpan Remaining
    {
        get
        {
            var ticks = EndTimestamp - Stopwatch.GetTimestamp();
            return ticks <= 0 ? TimeSpan.Zero : TimeSpan.FromSeconds((double)ticks / Stopwatch.Frequency);
        }
    }

    public bool Expired => Stopwatch.GetTimestamp() >= EndTimestamp;

    /// <summary>A deadline <paramref name="reserve"/> earlier (budget kept back for later stages), never earlier than now + <paramref name="floor"/>.</summary>
    public Deadline Reserve(TimeSpan reserve, TimeSpan floor = default)
    {
        long end = EndTimestamp - (long)(reserve.TotalSeconds * Stopwatch.Frequency);
        long min = Stopwatch.GetTimestamp() + (long)(floor.TotalSeconds * Stopwatch.Frequency);
        return new Deadline(Math.Max(end, Math.Min(min, EndTimestamp)));
    }

    /// <summary>At least <paramref name="floor"/> from now, even past this deadline.</summary>
    public Deadline AtLeast(TimeSpan floor)
    {
        long min = Stopwatch.GetTimestamp() + (long)(floor.TotalSeconds * Stopwatch.Frequency);
        return new Deadline(Math.Max(EndTimestamp, min));
    }
}
