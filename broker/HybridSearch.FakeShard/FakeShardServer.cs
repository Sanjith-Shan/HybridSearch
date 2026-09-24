using System.Globalization;
using System.Net;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Server.Kestrel.Core;

namespace HybridSearch.FakeShard;

/// <summary>A FakeShard hosted on real Kestrel (h2c), in-process. Port 0 picks a free port.</summary>
public sealed class FakeShardServer : IAsyncDisposable
{
    private readonly WebApplication _app;

    private FakeShardServer(WebApplication app, FakeCorpus corpus, FaultOptions faults, ShardObservations seen, string address)
    {
        _app = app; Corpus = corpus; Faults = faults; Seen = seen; Address = address;
    }

    public FakeCorpus Corpus { get; }
    public FaultOptions Faults { get; }
    public ShardObservations Seen { get; }
    public string Address { get; }

    public static async Task<FakeShardServer> StartAsync(FakeCorpus corpus, uint shardId, int port = 0,
        FaultOptions? faults = null, bool quiet = true, bool anyInterface = false, CancellationToken ct = default)
    {
        faults ??= new FaultOptions();
        var seen = new ShardObservations();
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.ConfigureKestrel(k =>
            k.Listen(anyInterface ? IPAddress.Any : IPAddress.Loopback, port, lo => lo.Protocols = HttpProtocols.Http2));
        if (quiet) builder.Logging.ClearProviders();
        else { builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning); builder.Logging.AddFilter("Grpc", LogLevel.Warning); }
        builder.Services.AddGrpc();
        builder.Services.AddSingleton(new FakeShardService(corpus, shardId, faults, seen));
        var app = builder.Build();
        app.MapGrpcService<FakeShardService>();
        await app.StartAsync(ct);
        var addr = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        var boundPort = new Uri(addr.Replace("0.0.0.0", "127.0.0.1").Replace("[::]", "127.0.0.1")).Port;
        return new FakeShardServer(app, corpus, faults, seen, $"http://127.0.0.1:{boundPort.ToString(CultureInfo.InvariantCulture)}");
    }

    public async ValueTask DisposeAsync()
    {
        await _app.StopAsync(TimeSpan.FromSeconds(2));
        await _app.DisposeAsync();
    }
}
