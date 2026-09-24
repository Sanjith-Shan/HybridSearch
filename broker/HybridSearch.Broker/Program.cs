using System.Diagnostics;
using HybridSearch.Broker;
using HybridSearch.Broker.Api;
using HybridSearch.Broker.Experiments;
using HybridSearch.Broker.Observability;
using HybridSearch.Broker.Search;
using HybridSearch.Broker.Shards;
using Microsoft.Extensions.Options;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;
using Prometheus;

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<BrokerOptions>(builder.Configuration.GetSection(BrokerOptions.Section));
builder.Services.PostConfigure<BrokerOptions>(o =>
{
    if (o.Cors.AllowedOrigins.Count == 0) o.Cors.AllowedOrigins.Add("http://localhost:5173");
});
builder.Services.AddProblemDetails();
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase);

builder.Services.AddCors();
builder.Services.AddOptions<Microsoft.AspNetCore.Cors.Infrastructure.CorsOptions>()
    .Configure<IOptions<BrokerOptions>>((cors, broker) => cors.AddDefaultPolicy(p => p
        .WithOrigins([.. broker.Value.Cors.AllowedOrigins])
        .WithMethods("GET", "POST")
        .WithHeaders("content-type", "traceparent", "tracestate", "x-request-id")
        .WithExposedHeaders("x-request-id", "traceparent")));

builder.Services.AddSingleton<BrokerMetrics>();
builder.Services.AddSingleton(sp => new ShardTopology(
    sp.GetRequiredService<IOptions<BrokerOptions>>().Value.Topology,
    sp.GetRequiredService<IOptions<BrokerOptions>>().Value.Hedging));
builder.Services.AddSingleton<ShardFanOut>();
builder.Services.AddSingleton(sp => new CostModel(sp.GetRequiredService<IOptions<BrokerOptions>>().Value.Degradation));
builder.Services.AddSingleton<ModelProvider>();
builder.Services.AddSingleton<ExperimentRegistry>();
builder.Services.AddSingleton<SearchService>();
builder.Services.AddSingleton<EventLogWriter>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<EventLogWriter>());
builder.Services.AddSingleton<AutocompleteService>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<AutocompleteService>());
builder.Services.AddSingleton<ShardHealthMonitor>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<ShardHealthMonitor>());

// Tracing: always collected (spans are created only when a listener exists); exported via OTLP
// unless Broker:Otel:Exporter = "none". The endpoint comes from OTEL_EXPORTER_OTLP_ENDPOINT
// (default http://localhost:4317), read by the exporter itself.
var exporter = builder.Configuration["Broker:Otel:Exporter"] ?? "otlp";
builder.Services.AddOpenTelemetry()
    .ConfigureResource(r => r.AddService("hybridsearch-broker"))
    .WithTracing(t =>
    {
        t.AddSource(Telemetry.SourceName)
         .AddAspNetCoreInstrumentation(o => o.Filter = ctx => ctx.Request.Path.StartsWithSegments("/api"));
        if (exporter.Equals("otlp", StringComparison.OrdinalIgnoreCase)) t.AddOtlpExporter();
    });

var app = builder.Build();

app.Services.GetRequiredService<BrokerMetrics>().Initialize(
    app.Services.GetRequiredService<ShardTopology>().Slices.Select(s => s.Label),
    app.Services.GetRequiredService<ExperimentRegistry>().Experiments
        .SelectMany(e => e.Interleaves ? new[] { (e.Id, "interleaved") } : [(e.Id, "control"), (e.Id, "treatment")]));

app.UseExceptionHandler();
app.UseStatusCodePages();
app.Use(async (ctx, next) =>
{
    // Request id: keep a sane client-supplied one, else mint a ULID. Echoed with the trace id.
    string? incoming = ctx.Request.Headers["x-request-id"];
    var id = incoming is { Length: > 0 and <= 128 } && incoming.All(SearchQueryParser.IsSessionChar) ? incoming : Ulid.NewString();
    ctx.Items[Endpoints.RequestIdItem] = id;
    ctx.Response.OnStarting(() =>
    {
        ctx.Response.Headers["x-request-id"] = id;
        if (Activity.Current is { } a && a.IdFormat == ActivityIdFormat.W3C)
            ctx.Response.Headers["traceparent"] = a.Id;
        return Task.CompletedTask;
    });
    await next(ctx);
});
app.UseCors();

app.MapBrokerApi();
app.MapMetrics(registry: app.Services.GetRequiredService<BrokerMetrics>().Registry);

app.Run();

/// <summary>Entry point marker for WebApplicationFactory.</summary>
public partial class Program;
