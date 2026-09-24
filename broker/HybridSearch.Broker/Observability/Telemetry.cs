using System.Diagnostics;
using Grpc.Core;

namespace HybridSearch.Broker.Observability;

/// <summary>Span names from docs/ARCHITECTURE.md (Operations).</summary>
public static class Telemetry
{
    public const string SourceName = "HybridSearch.Broker";
    public static readonly ActivitySource Source = new(SourceName, "1.0.0");

    public const string SearchSpan = "broker.search";
    public const string EncodeSpan = "broker.encode";
    public const string ShardSearchSpan = "shard.search";
    public const string ShardHedgeSpan = "shard.hedge";
    public const string FuseSpan = "broker.fuse";
    public const string RerankSpan = "broker.rerank";
    public const string FetchSpan = "shard.fetch";

    /// <summary>
    /// Write W3C trace context for <paramref name="activity"/> into gRPC metadata. Done explicitly
    /// rather than relying on HttpClient's automatic header injection so the hop is visible in
    /// code and covered by a test (the shard must see the broker's span as parent).
    /// </summary>
    public static Metadata TraceMetadata(Activity? activity)
    {
        var md = new Metadata();
        if (activity is null) return md;
        DistributedContextPropagator.Current.Inject(activity, md, static (carrier, key, value) =>
        {
            if (carrier is Metadata m && key is not null && value is not null && m.Get(key) is null)
                m.Add(key, value);
        });
        return md;
    }
}
