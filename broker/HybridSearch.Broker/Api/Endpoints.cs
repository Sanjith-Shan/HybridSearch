using System.Diagnostics;
using System.Text.Json;
using HybridSearch.Broker.Autocomplete;
using HybridSearch.Broker.Experiments;
using HybridSearch.Broker.Observability;
using HybridSearch.Broker.Search;
using HybridSearch.Broker.Shards;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.Extensions.Options;

namespace HybridSearch.Broker.Api;

public static class Endpoints
{
    public const string RequestIdItem = "hs.requestId";

    public static string RequestId(this HttpContext ctx) =>
        ctx.Items.TryGetValue(RequestIdItem, out var v) && v is string s ? s : Ulid.NewString();

    public static void MapBrokerApi(this WebApplication app)
    {
        var api = app.MapGroup("/api");
        api.MapGet("/search", SearchAsync);
        api.MapGet("/search/stream", SearchStreamAsync);
        api.MapGet("/suggest", Suggest);
        api.MapGet("/doc/{docId}", GetDocAsync);
        api.MapPost("/events", PostEvents).WithMetadata(new Microsoft.AspNetCore.Mvc.RequestSizeLimitAttribute(512 * 1024));
        api.MapGet("/experiments", ListExperiments);
        api.MapGet("/experiments/{id}/results", ExperimentResultsAsync);

        app.MapGet("/healthz", () => Results.Ok(new { status = "ok" }));
        app.MapGet("/readyz", Ready);
    }

    private static async Task<IResult> SearchAsync(HttpContext ctx, SearchService svc, IOptions<BrokerOptions> o, BrokerMetrics metrics)
    {
        var q = SearchQueryParser.Parse(ctx.Request.Query, o.Value.Search, out var errors);
        if (q is null)
        {
            metrics.SearchRequests.WithLabels(ModeLabel(ctx), "invalid").Inc();
            return Results.ValidationProblem(errors);
        }
        long t0 = Stopwatch.GetTimestamp();
        string mode = SearchModes.Name(q.Mode);
        try
        {
            var r = await svc.SearchAsync(q, ctx.RequestId(), null, ctx.RequestAborted);
            metrics.SearchRequests.WithLabels(mode, "ok").Inc();
            return Results.Ok(r);
        }
        catch (AllShardsFailedException ex)
        {
            metrics.SearchRequests.WithLabels(mode, "error").Inc();
            return Results.Problem(title: "All shards failed", detail: ex.Message, statusCode: StatusCodes.Status503ServiceUnavailable,
                extensions: new Dictionary<string, object?> { ["failedShards"] = ex.FailedShards, ["requestId"] = ctx.RequestId() });
        }
        catch (Exception) when (!ctx.RequestAborted.IsCancellationRequested)
        {
            metrics.SearchRequests.WithLabels(mode, "error").Inc();
            throw;
        }
        finally
        {
            metrics.SearchDuration.WithLabels(mode).Observe(BrokerMetrics.Seconds(t0));
        }
    }

    /// <summary>SSE: <c>event: lexical</c> as soon as shards answer, then <c>event: final</c>; <c>event: error</c> on failure.</summary>
    private static async Task<IResult> SearchStreamAsync(HttpContext ctx, SearchService svc, IOptions<BrokerOptions> o,
        BrokerMetrics metrics, IOptions<JsonOptions> json, ILoggerFactory lf)
    {
        var q = SearchQueryParser.Parse(ctx.Request.Query, o.Value.Search, out var errors);
        if (q is null)
        {
            metrics.SearchRequests.WithLabels(ModeLabel(ctx), "invalid").Inc();
            return Results.ValidationProblem(errors);
        }
        var ser = json.Value.SerializerOptions;
        var ct = ctx.RequestAborted;
        ctx.Response.StatusCode = StatusCodes.Status200OK;
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.Headers["X-Accel-Buffering"] = "no";

        async Task Send(string evt, object payload)
        {
            await ctx.Response.WriteAsync($"event: {evt}\ndata: {JsonSerializer.Serialize(payload, ser)}\n\n", ct);
            await ctx.Response.Body.FlushAsync(ct);
        }

        string mode = SearchModes.Name(q.Mode);
        long t0 = Stopwatch.GetTimestamp();
        try
        {
            var final = await svc.SearchAsync(q, ctx.RequestId(), page => Send("lexical", page), ct);
            await Send("final", final);
            metrics.SearchRequests.WithLabels(mode, "ok").Inc();
        }
        catch (AllShardsFailedException ex)
        {
            metrics.SearchRequests.WithLabels(mode, "error").Inc();
            await Send("error", new { message = ex.Message, failedShards = ex.FailedShards });
        }
        catch (Exception ex) when (!ct.IsCancellationRequested)
        {
            metrics.SearchRequests.WithLabels(mode, "error").Inc();
            lf.CreateLogger("SearchStream").LogError(ex, "stream search failed");
            await Send("error", new { message = "internal error" });
        }
        finally
        {
            metrics.SearchDuration.WithLabels(mode).Observe(BrokerMetrics.Seconds(t0));
        }
        return Results.Empty;
    }

    private static string ModeLabel(HttpContext ctx) =>
        SearchModes.TryParse(ctx.Request.Query["mode"], out var m) ? SearchModes.Name(m) : "invalid";

    private static IResult Suggest(string? prefix, int? k, AutocompleteService ac, IOptions<BrokerOptions> o, BrokerMetrics metrics)
    {
        int maxK = o.Value.Autocomplete.TopK;
        var errors = new Dictionary<string, string[]>();
        if (prefix is null) errors["prefix"] = ["prefix is required"];
        else if (prefix.Length > o.Value.Search.MaxQueryChars) errors["prefix"] = [$"prefix must be at most {o.Value.Search.MaxQueryChars} characters"];
        int kk = k ?? Math.Min(8, maxK);
        if (kk < 1 || kk > maxK) errors["k"] = [$"k must be in [1, {maxK}]"];
        if (errors.Count > 0) return Results.ValidationProblem(errors);

        var trie = ac.Trie;
        if (trie is null)
            return Results.Problem(title: "Autocomplete not available", detail: $"{ac.State}: {ac.Detail}",
                statusCode: StatusCodes.Status503ServiceUnavailable);

        long t0 = Stopwatch.GetTimestamp();
        var norm = QueryNormalizer.NormalizePrefix(prefix!);
        var list = trie.Lookup(norm, kk);
        var elapsed = Stopwatch.GetElapsedTime(t0);
        metrics.SuggestDuration.Observe(elapsed.TotalSeconds);
        return Results.Ok(new SuggestResponseDto(prefix!, list.Select(c => new SuggestionDto(c.Text, c.Count)).ToList(),
            Math.Round(elapsed.TotalMicroseconds, 2)));
    }

    private static async Task<IResult> GetDocAsync(string docId, string? q, SearchService svc, HttpContext ctx)
    {
        if (!ulong.TryParse(docId, out var id))
            return Results.ValidationProblem(new Dictionary<string, string[]> { ["docId"] = ["docId must be a non-negative integer"] });
        if (q is { Length: > 512 })
            return Results.ValidationProblem(new Dictionary<string, string[]> { ["q"] = ["q must be at most 512 characters"] });
        var doc = await svc.GetDocAsync(id, q, ctx.RequestAborted);
        return doc is null
            ? Results.Problem(title: "Document not found", detail: $"no shard returned passage {id}", statusCode: StatusCodes.Status404NotFound)
            : Results.Ok(doc);
    }

    private static IResult PostEvents(EventBatch? batch, EventLogWriter log, IOptions<BrokerOptions> o, BrokerMetrics metrics)
    {
        if (batch?.Events is null)
            return Results.ValidationProblem(new Dictionary<string, string[]> { ["events"] = ["body must be { \"events\": [...] }"] });
        int max = o.Value.Events.MaxBatch;
        if (batch.Events.Count > max)
            return Results.Problem(title: "Batch too large", detail: $"at most {max} events per batch", statusCode: StatusCodes.Status413PayloadTooLarge);

        var now = DateTimeOffset.UtcNow;
        int accepted = 0, dropped = 0;
        var errors = new List<EventErrorDto>();
        for (int i = 0; i < batch.Events.Count; i++)
        {
            var e = batch.Events[i];
            string? error = "event is null";
            var rec = e is null ? null : EventValidator.Validate(e, now, out error);
            if (rec is null)
            {
                errors.Add(new EventErrorDto(i, error ?? "invalid event"));
                metrics.EventsDropped.WithLabels("invalid").Inc();
                continue;
            }
            if (log.TryEnqueue(rec))
            {
                accepted++;
                metrics.EventsIngested.WithLabels(rec.Type!).Inc();
            }
            else dropped++;
        }
        return Results.Accepted(value: new EventsAcceptedDto(accepted, errors.Count, dropped, errors));
    }

    private static IResult ListExperiments(ExperimentRegistry reg) =>
        Results.Ok(new
        {
            experiments = reg.Experiments.Select(e => new
            {
                e.Id,
                e.Kind,
                e.Status,
                e.Allocation,
                control = e.Control,
                treatment = e.EffectiveTreatment,
                e.Description,
                interleaved = e.Interleaves,
                resultsUrl = $"/api/experiments/{Uri.EscapeDataString(e.Id)}/results",
            }),
        });

    private static async Task<IResult> ExperimentResultsAsync(string id, ExperimentRegistry reg, EventLogWriter log,
        IOptions<BrokerOptions> o, CancellationToken ct)
    {
        var exp = reg.Find(id);
        if (exp is null)
            return Results.Problem(title: "Unknown experiment", detail: id, statusCode: StatusCodes.Status404NotFound);
        await log.FlushAsync(TimeSpan.FromSeconds(1), ct);
        var records = new List<LogRecord>();
        await foreach (var r in ExperimentAnalyzer.ReadLogAsync(log.Directory, ct)) records.Add(r);
        return Results.Ok(ExperimentAnalyzer.Analyze(exp, records, o.Value.Experiments.BootstrapIterations, DateTimeOffset.UtcNow));
    }

    private static IResult Ready(ShardTopology topology, AutocompleteService ac, ModelProvider models, EventLogWriter events)
    {
        var slices = topology.Slices.Select(s => new
        {
            shard = s.Id,
            ready = s.IsReady,
            range = s.Range is { } r ? new[] { r.First, r.Last } : null,
            replicas = s.Replicas.Select(r => new { r.Index, r.Endpoint, health = r.Health.ToString().ToLowerInvariant() }),
            hedgeExtraLoadRatio = Math.Round(s.ExtraLoadRatio, 6),
        }).ToList();
        bool shardsReady = topology.IsReady;
        bool acSettled = ac.State != ComponentState.Loading;
        var body = new
        {
            ready = shardsReady && acSettled,
            shards = slices,
            autocomplete = new { state = ac.State.ToString().ToLowerInvariant(), detail = ac.Detail },
            queryEncoder = new { available = models.Encoder.IsAvailable, status = models.Encoder.Status },
            reranker = new { available = models.Reranker.IsAvailable, status = models.Reranker.Status },
            eventLog = new { status = events.Status, directory = events.Directory },
        };
        return shardsReady && acSettled ? Results.Ok(body) : Results.Json(body, statusCode: StatusCodes.Status503ServiceUnavailable);
    }
}

/// <summary>ULID (Crockford base32: 48-bit ms timestamp + 80 random bits): sortable, URL-safe request IDs like "01J…".</summary>
public static class Ulid
{
    private const string Alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";

    public static string NewString()
    {
        Span<byte> b = stackalloc byte[16];
        long ms = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        for (int i = 5; i >= 0; i--) { b[i] = (byte)ms; ms >>= 8; }
        System.Security.Cryptography.RandomNumberGenerator.Fill(b[6..]);
        var hi = ((UInt128)System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(b) << 64) |
                 System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(b[8..]);
        Span<char> c = stackalloc char[26];
        for (int i = 25; i >= 0; i--) { c[i] = Alphabet[(int)(hi & 31)]; hi >>= 5; }
        return new string(c);
    }
}
