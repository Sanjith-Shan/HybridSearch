using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using HybridSearch.Broker.Experiments;
using HybridSearch.FakeShard;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;

namespace HybridSearch.Broker.Tests;

/// <summary>A broker (WebApplicationFactory, in-memory HTTP) wired to two in-process FakeShards.</summary>
public sealed class BrokerApp : IAsyncDisposable
{
    public required WebApplicationFactory<Program> Factory { get; init; }
    public required HttpClient Client { get; init; }
    public required List<FakeShardServer> Shards { get; init; }
    public required string TempDir { get; init; }

    public const string InterleaveSession = "il-session";

    public static async Task<BrokerApp> StartAsync(Action<int, FaultOptions>? faults = null, Dictionary<string, string?>? extra = null,
        string? experimentsJson = null, bool waitReady = true)
    {
        var tmp = Directory.CreateTempSubdirectory("hs-broker-test-").FullName;
        var shards = new List<FakeShardServer>();
        for (int s = 0; s < 2; s++)
        {
            var f = new FaultOptions();
            faults?.Invoke(s, f);
            shards.Add(await FakeShardServer.StartAsync(new FakeCorpus(FakeCorpus.Slice(FakeCorpus.BuiltIn, s, 2)), (uint)s, 0, f));
        }
        File.WriteAllLines(Path.Combine(tmp, "queries.tsv"),
        [
            "1\tcapital of peru", "2\tcapital of peru", "3\tcapital of france", "4\tcapital one bank",
            "5\thow to lose weight", "6\thow to lose weight", "7\thow to lose weight", "8\thow to tie a tie", "9\tphotosynthesis definition",
            "10\tphotosynthesis definition", "11\tphotosynthesis definition",
        ]);
        File.WriteAllText(Path.Combine(tmp, "experiments.json"), experimentsJson ?? """
        { "experiments": [
          { "id": "il", "kind": "interleave", "allocation": 1.0, "salt": "il-salt",
            "control": { "mode": "lexical", "rerank": false }, "treatment": { "mode": "dense", "rerank": false } } ] }
        """);

        var settings = new Dictionary<string, string?>
        {
            ["Broker:DataRoot"] = tmp,
            ["Broker:Topology:Slices:0:Id"] = "0",
            ["Broker:Topology:Slices:0:Replicas:0"] = shards[0].Address,
            ["Broker:Topology:Slices:1:Id"] = "1",
            ["Broker:Topology:Slices:1:Replicas:0"] = shards[1].Address,
            ["Broker:Models:QueryEncoderKind"] = "hashing-dev",
            ["Broker:Models:RerankerDir"] = Path.Combine(tmp, "no-reranker"),
            ["Broker:Autocomplete:QueryLogPath"] = Path.Combine(tmp, "queries.tsv"),
            ["Broker:Autocomplete:TopK"] = "10",
            ["Broker:Events:Directory"] = Path.Combine(tmp, "events"),
            ["Broker:Experiments:ConfigPath"] = Path.Combine(tmp, "experiments.json"),
            ["Broker:Experiments:BootstrapIterations"] = "200",
            ["Broker:Health:IntervalMs"] = "200",
            ["Broker:Health:TimeoutMs"] = "3000",
            ["Broker:Otel:Exporter"] = "none",
            ["Broker:Search:DefaultDeadlineMs"] = "5000",
        };
        foreach (var (k, v) in extra ?? []) settings[k] = v;

        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            foreach (var (k, v) in settings) b.UseSetting(k, v);
            b.UseEnvironment("Testing");
        });
        var client = factory.CreateClient();
        var app = new BrokerApp { Factory = factory, Client = client, Shards = shards, TempDir = tmp };
        if (waitReady) await app.WaitReadyAsync();
        return app;
    }

    public async Task WaitReadyAsync()
    {
        var sw = Stopwatch.StartNew();
        while (sw.Elapsed < TimeSpan.FromSeconds(30))
        {
            var r = await Client.GetAsync("/readyz");
            if (r.StatusCode == HttpStatusCode.OK) return;
            await Task.Delay(100);
        }
        throw new TimeoutException("broker not ready: " + await Client.GetStringAsync("/readyz"));
    }

    public T Service<T>() where T : notnull => Factory.Services.GetRequiredService<T>();

    public async ValueTask DisposeAsync()
    {
        Client.Dispose();
        await Factory.DisposeAsync();
        foreach (var s in Shards) await s.DisposeAsync();
        try { Directory.Delete(TempDir, true); } catch (IOException) { }
    }
}

[Collection("timing")]
public class ApiTests
{
    private static async Task<JsonElement> GetJson(HttpClient c, string url, HttpStatusCode expected = HttpStatusCode.OK)
    {
        var r = await c.GetAsync(url);
        var body = await r.Content.ReadAsStringAsync();
        Assert.True(expected == r.StatusCode, $"{url} → {(int)r.StatusCode}: {body}");
        return JsonDocument.Parse(body).RootElement;
    }

    [Fact]
    public async Task Search_returns_the_contract_shape_with_snippets_and_scores()
    {
        await using var app = await BrokerApp.StartAsync();
        var r = await app.Client.GetAsync("/api/search?q=capital%20of%20peru&k=3&explain=true");
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.True(r.Headers.Contains("x-request-id"));
        var j = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal(r.Headers.GetValues("x-request-id").Single(), j.GetProperty("requestId").GetString());
        Assert.Equal("hybrid", j.GetProperty("mode").GetString());
        var results = j.GetProperty("results");
        Assert.Equal(3, results.GetArrayLength());
        var top = results[0];
        Assert.Equal(1001UL, top.GetProperty("docId").GetUInt64());
        Assert.Equal(1, top.GetProperty("rank").GetInt32());
        var text = top.GetProperty("text").GetString()!;
        var hl = top.GetProperty("highlights")[0];
        Assert.Equal("capital", text[hl.GetProperty("start").GetInt32()..hl.GetProperty("end").GetInt32()]);
        var scores = top.GetProperty("scores");
        Assert.Equal(1, scores.GetProperty("bm25Rank").GetInt32());
        Assert.Equal(JsonValueKind.Number, scores.GetProperty("dense").ValueKind);
        Assert.Equal(JsonValueKind.Null, scores.GetProperty("rerank").ValueKind);
        Assert.Equal(JsonValueKind.Null, top.GetProperty("team").ValueKind);
        Assert.True(top.GetProperty("terms").GetArrayLength() >= 1);
        // No reranker model in the test data root: the response must say so.
        var deg = j.GetProperty("degradation");
        Assert.Equal(1, deg.GetProperty("level").GetInt32());
        Assert.Equal("reranker_skipped:model_unavailable", deg.GetProperty("steps")[0].GetString());
        Assert.Equal(0, deg.GetProperty("failedShards").GetArrayLength());
        foreach (var t in new[] { "totalMs", "encodeMs", "shardsMs", "fuseMs", "rerankMs", "fetchMs" })
            Assert.True(j.GetProperty("timings").TryGetProperty(t, out _));
        Assert.Equal(JsonValueKind.Null, j.GetProperty("experiment").ValueKind);
    }

    [Fact]
    public async Task Lexical_mode_and_rerank_false_report_no_degradation()
    {
        await using var app = await BrokerApp.StartAsync();
        var j = await GetJson(app.Client, "/api/search?q=paris&mode=lexical&rerank=false");
        Assert.Equal("lexical", j.GetProperty("mode").GetString());
        Assert.Equal(0, j.GetProperty("degradation").GetProperty("level").GetInt32());
        Assert.All(j.GetProperty("results").EnumerateArray(), x => Assert.Equal(JsonValueKind.Null, x.GetProperty("scores").GetProperty("dense").ValueKind));
        // FakeShard saw lexical-only requests.
        Assert.All(app.Shards.SelectMany(s => s.Seen.Searches), s => Assert.False(s.Vector));
    }

    [Fact]
    public async Task Validation_errors_are_problem_details()
    {
        await using var app = await BrokerApp.StartAsync();
        foreach (var url in new[] { "/api/search", "/api/search?q=a&k=101", "/api/search?q=a&k=0", "/api/search?q=" + new string('x', 513), "/api/search?q=a&mode=x" })
        {
            var r = await app.Client.GetAsync(url);
            Assert.Equal(HttpStatusCode.BadRequest, r.StatusCode);
            Assert.Equal("application/problem+json", r.Content.Headers.ContentType!.MediaType);
            var j = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
            Assert.True(j.TryGetProperty("errors", out _));
        }
        var metrics = await app.Client.GetStringAsync("/metrics");
        Assert.Contains("hs_search_requests_total{mode=\"hybrid\",outcome=\"invalid\"}", metrics);
    }

    [Fact]
    public async Task Stream_sends_lexical_then_final()
    {
        await using var app = await BrokerApp.StartAsync();
        var r = await app.Client.GetAsync("/api/search/stream?q=capital&k=2", HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("text/event-stream", r.Content.Headers.ContentType!.MediaType);
        var body = await r.Content.ReadAsStringAsync();
        var events = body.Split("\n\n", StringSplitOptions.RemoveEmptyEntries)
            .Select(block => block.Split('\n'))
            .Select(lines => (Event: lines[0]["event: ".Length..], Data: JsonDocument.Parse(lines[1]["data: ".Length..]).RootElement))
            .ToList();
        Assert.Equal(["lexical", "final"], events.Select(e => e.Event));
        Assert.Equal("lexical", events[0].Data.GetProperty("mode").GetString());
        Assert.Equal(events[0].Data.GetProperty("requestId").GetString(), events[1].Data.GetProperty("requestId").GetString());
        Assert.Equal(2, events[1].Data.GetProperty("results").GetArrayLength());
    }

    [Fact]
    public async Task Stream_reports_error_event_when_all_shards_fail()
    {
        await using var app = await BrokerApp.StartAsync(faults: (_, f) => { f.FailRate = 1; f.FailStatus = Grpc.Core.StatusCode.Internal; });
        var body = await app.Client.GetStringAsync("/api/search/stream?q=capital");
        Assert.StartsWith("event: error", body);
        var r = await app.Client.GetAsync("/api/search?q=capital");
        Assert.Equal(HttpStatusCode.ServiceUnavailable, r.StatusCode);
        var j = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal(2, j.GetProperty("failedShards").GetArrayLength());
    }

    [Fact]
    public async Task Partial_and_failed_shards_are_reported()
    {
        await using var app = await BrokerApp.StartAsync(faults: (s, f) =>
        {
            if (s == 1) { f.FailRate = 1; f.FailStatus = Grpc.Core.StatusCode.Internal; }
            else f.PartialRate = 1;
        });
        var j = await GetJson(app.Client, "/api/search?q=capital&k=5");
        var deg = j.GetProperty("degradation");
        Assert.Equal([0], deg.GetProperty("partialShards").EnumerateArray().Select(x => x.GetInt32()));
        Assert.Equal([1], deg.GetProperty("failedShards").EnumerateArray().Select(x => x.GetInt32()));
        Assert.True(j.GetProperty("results").GetArrayLength() > 0);
        var metrics = await app.Client.GetStringAsync("/metrics");
        Assert.Contains("hs_shard_partial_total{shard=\"0\"}", metrics);
        Assert.Contains("hs_shard_failures_total{shard=\"1\",reason=\"error\"}", metrics);
    }

    [Fact]
    public async Task Tight_deadline_degrades_in_order_and_is_visible()
    {
        // Priors say a full hybrid query costs far more than 15 ms → the policy must go lexical-only.
        await using var app = await BrokerApp.StartAsync(extra: new()
        {
            ["Broker:Degradation:DefaultShardFullMs"] = "400",
            ["Broker:Degradation:DefaultShardShrunkMs"] = "300",
            ["Broker:Degradation:DefaultShardLexicalMs"] = "5",
            ["Broker:Degradation:DefaultFetchMs"] = "1",
            ["Broker:Degradation:MinSamples"] = "1000000",
        });
        var j = await GetJson(app.Client, "/api/search?q=capital&deadlineMs=250");
        var steps = j.GetProperty("degradation").GetProperty("steps").EnumerateArray().Select(s => s.GetString()).ToList();
        Assert.Equal(3, j.GetProperty("degradation").GetProperty("level").GetInt32());
        Assert.Equal(["reranker_skipped:model_unavailable", "lexical_only:deadline"], steps);
        Assert.All(app.Shards.SelectMany(s => s.Seen.Searches), s => Assert.False(s.Vector));

        var k = await GetJson(app.Client, "/api/search?q=capital&deadlineMs=350");
        Assert.Contains("dense_beam_shrunk:deadline", k.GetProperty("degradation").GetProperty("steps").EnumerateArray().Select(s => s.GetString()));
        Assert.Contains(app.Shards.SelectMany(s => s.Seen.Searches), s => s.Vector && s.BeamWidth == 32);
        var metrics = await app.Client.GetStringAsync("/metrics");
        Assert.Contains("hs_degradation_total{step=\"lexical_only\"}", metrics);
        Assert.Contains("hs_degradation_total{step=\"dense_beam_shrunk\"}", metrics);
    }

    [Fact]
    public async Task Missing_encoder_is_reported_as_lexical_only()
    {
        await using var app = await BrokerApp.StartAsync(extra: new() { ["Broker:Models:QueryEncoderKind"] = "onnx", ["Broker:Models:QueryEncoderDir"] = "nope" });
        var j = await GetJson(app.Client, "/api/search?q=capital");
        Assert.Contains("lexical_only:encoder_unavailable", j.GetProperty("degradation").GetProperty("steps").EnumerateArray().Select(s => s.GetString()));
        var ready = await GetJson(app.Client, "/readyz");
        Assert.False(ready.GetProperty("queryEncoder").GetProperty("available").GetBoolean());
    }

    [Fact]
    public async Task Traceparent_from_the_browser_reaches_the_shards()
    {
        await using var app = await BrokerApp.StartAsync();
        const string traceId = "4bf92f3577b34da6a3ce929d0e0e4736";
        var req = new HttpRequestMessage(HttpMethod.Get, "/api/search?q=capital&mode=lexical");
        req.Headers.Add("traceparent", $"00-{traceId}-00f067aa0ba902b7-01");
        var r = await app.Client.SendAsync(req);
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        Assert.Contains(traceId, r.Headers.GetValues("traceparent").Single());
        foreach (var s in app.Shards)
        {
            var search = s.Seen.Searches.Last();
            var tp = Assert.Single(search.Traceparents);
            Assert.Equal(traceId, tp.Split('-')[1]);
        }
    }

    [Fact]
    public async Task Doc_endpoint_returns_full_passage_with_utf16_highlights()
    {
        await using var app = await BrokerApp.StartAsync();
        var j = await GetJson(app.Client, "/api/doc/1016?q=paris%20culture");
        var text = j.GetProperty("text").GetString()!;
        Assert.StartsWith("Café culture in Paris", text);
        var spans = j.GetProperty("highlights").EnumerateArray().Select(h => text[h.GetProperty("start").GetInt32()..h.GetProperty("end").GetInt32()]).ToList();
        Assert.Equal(["culture", "Paris"], spans);
        await GetJson(app.Client, "/api/doc/424242", HttpStatusCode.NotFound);
        await GetJson(app.Client, "/api/doc/abc", HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task Suggest_serves_top_k_from_the_query_log()
    {
        await using var app = await BrokerApp.StartAsync();
        var j = await GetJson(app.Client, "/api/suggest?prefix=How%20to%20&k=5");
        Assert.Equal("How to ", j.GetProperty("prefix").GetString());
        var s = j.GetProperty("suggestions");
        Assert.Equal("how to lose weight", s[0].GetProperty("text").GetString());
        Assert.Equal(3, s[0].GetProperty("count").GetInt64());
        Assert.Equal(2, s.GetArrayLength());
        Assert.True(j.GetProperty("micros").GetDouble() >= 0);
        Assert.Equal(0, (await GetJson(app.Client, "/api/suggest?prefix=zzz")).GetProperty("suggestions").GetArrayLength());
        await GetJson(app.Client, "/api/suggest?prefix=a&k=0", HttpStatusCode.BadRequest);
        await GetJson(app.Client, "/api/suggest", HttpStatusCode.BadRequest);
        Assert.Contains("hs_suggest_duration_seconds", await app.Client.GetStringAsync("/metrics"));
    }

    [Fact]
    public async Task Did_you_mean_from_the_query_log_vocabulary()
    {
        await using var app = await BrokerApp.StartAsync(extra: new() { ["Broker:Autocomplete:SpellMinWordFrequency"] = "1" });
        var j = await GetJson(app.Client, "/api/search?q=photosynthsis%20definition&mode=lexical");
        Assert.Equal("photosynthesis definition", j.GetProperty("didYouMean").GetString());
        var k = await GetJson(app.Client, "/api/search?q=capital%20of%20peru&mode=lexical");
        Assert.Equal(JsonValueKind.Null, k.GetProperty("didYouMean").ValueKind);
    }

    [Fact]
    public async Task Readiness_waits_for_autocomplete_and_healthy_shards()
    {
        await using var app = await BrokerApp.StartAsync(faults: (s, f) => { if (s == 1) f.HealthDown = true; }, waitReady: false);
        await Task.Delay(1500);
        var r = await app.Client.GetAsync("/readyz");
        Assert.Equal(HttpStatusCode.ServiceUnavailable, r.StatusCode);
        var j = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
        Assert.False(j.GetProperty("shards")[1].GetProperty("ready").GetBoolean());
        app.Shards[1].Faults.HealthDown = false;
        await app.WaitReadyAsync();
        Assert.Equal(HttpStatusCode.OK, (await app.Client.GetAsync("/healthz")).StatusCode);
    }

    [Fact]
    public async Task Cors_allows_the_web_dev_server_only()
    {
        await using var app = await BrokerApp.StartAsync();
        var ok = new HttpRequestMessage(HttpMethod.Get, "/api/suggest?prefix=h");
        ok.Headers.Add("Origin", "http://localhost:5173");
        var r = await app.Client.SendAsync(ok);
        Assert.Equal("http://localhost:5173", r.Headers.GetValues("Access-Control-Allow-Origin").Single());
        var bad = new HttpRequestMessage(HttpMethod.Get, "/api/suggest?prefix=h");
        bad.Headers.Add("Origin", "http://evil.example");
        Assert.False((await app.Client.SendAsync(bad)).Headers.Contains("Access-Control-Allow-Origin"));
    }

    [Fact]
    public async Task Events_are_validated_capped_stamped_and_written_without_ip()
    {
        await using var app = await BrokerApp.StartAsync();
        var batch = new
        {
            events = new object[]
            {
                new { type = "click", sessionId = "s1", requestId = "r1", docId = 1001, rank = 1, query = "q", team = "A", clientTs = "2026-09-23T20:10:11.123Z" },
                new { type = "hover", sessionId = "s1" },
                new { type = "dwell", sessionId = "s1", requestId = "r1", docId = 1001, rank = 1, dwellMs = 1200 },
            },
        };
        var r = await app.Client.PostAsJsonAsync("/api/events", batch);
        Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
        var j = JsonDocument.Parse(await r.Content.ReadAsStringAsync()).RootElement;
        Assert.Equal(2, j.GetProperty("accepted").GetInt32());
        Assert.Equal(1, j.GetProperty("rejected").GetInt32());
        Assert.Equal(1, j.GetProperty("errors")[0].GetProperty("index").GetInt32());

        var tooMany = new { events = Enumerable.Range(0, 101).Select(_ => new { type = "query", sessionId = "s" }).ToArray() };
        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, (await app.Client.PostAsJsonAsync("/api/events", tooMany)).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await app.Client.PostAsJsonAsync("/api/events", new { nope = 1 })).StatusCode);

        var log = app.Service<EventLogWriter>();
        Assert.True(await log.FlushAsync(TimeSpan.FromSeconds(10), CancellationToken.None));
        var lines = Directory.GetFiles(log.Directory, "events-*.jsonl").SelectMany(File.ReadAllLines).ToList();
        Assert.Equal(2, lines.Count);
        foreach (var line in lines)
        {
            var rec = JsonDocument.Parse(line).RootElement;
            Assert.True(rec.TryGetProperty("serverTs", out _));
            Assert.DoesNotContain("127.0.0.1", line);
            Assert.DoesNotContain("\"ip", line, StringComparison.OrdinalIgnoreCase);
        }
        var metrics = await app.Client.GetStringAsync("/metrics");
        Assert.Contains("hs_events_ingested_total{type=\"click\"} 1", metrics);
        Assert.Contains("hs_events_dropped_total{reason=\"invalid\"} 1", metrics);
    }

    [Fact]
    public async Task Interleaved_experiment_tags_teams_logs_served_pages_and_computes_results()
    {
        await using var app = await BrokerApp.StartAsync();
        var list = await GetJson(app.Client, "/api/experiments");
        Assert.Equal("il", list.GetProperty("experiments")[0].GetProperty("id").GetString());

        int queries = 0;
        foreach (var q in new[] { "capital of peru", "paris", "weight loss diet", "heart chambers", "coffee caffeine", "speed of light" })
        {
            for (int s = 0; s < 3; s++)
            {
                var session = $"sess{s}";
                var j = await GetJson(app.Client, $"/api/search?q={Uri.EscapeDataString(q)}&k=6&sessionId={session}");
                queries++;
                var exp = j.GetProperty("experiment");
                Assert.Equal("il", exp.GetProperty("id").GetString());
                Assert.True(exp.GetProperty("interleaved").GetBoolean());
                var results = j.GetProperty("results").EnumerateArray().ToList();
                Assert.All(results, x => Assert.Contains(x.GetProperty("team").GetString(), new[] { "A", "B" }));
                // Same session + query → same interleaving.
                var again = await GetJson(app.Client, $"/api/search?q={Uri.EscapeDataString(q)}&k=6&sessionId={session}");
                Assert.Equal(results.Select(x => x.GetProperty("team").GetString()), again.GetProperty("results").EnumerateArray().Select(x => x.GetProperty("team").GetString()));
                queries++;
                // Click the first team-B result: treatment should win every impression that got a click.
                var b = results.FirstOrDefault(x => x.GetProperty("team").GetString() == "B");
                if (b.ValueKind == JsonValueKind.Undefined) continue;
                var ev = new { events = new[] { new { type = "click", sessionId = session, requestId = j.GetProperty("requestId").GetString(), docId = b.GetProperty("docId").GetUInt64(), rank = b.GetProperty("rank").GetInt32(), team = "B" } } };
                Assert.Equal(HttpStatusCode.Accepted, (await app.Client.PostAsJsonAsync("/api/events", ev)).StatusCode);
            }
        }
        var res = await GetJson(app.Client, "/api/experiments/il/results");
        Assert.Equal("il", res.GetProperty("id").GetString());
        Assert.False(res.GetProperty("simulated").GetBoolean());
        Assert.Equal(0.95, res.GetProperty("confidenceLevel").GetDouble());
        Assert.Equal(queries, res.GetProperty("queries").GetInt32());
        var v = res.GetProperty("variants")[0];
        Assert.Equal(queries, v.GetProperty("queries").GetInt32());
        foreach (var m in new[] { "ctr", "clicksAt1", "abandonment", "mrrFirstClick" })
            foreach (var f in new[] { "value", "ciLow", "ciHigh" })
                Assert.Equal(JsonValueKind.Number, v.GetProperty("metrics").GetProperty(m).GetProperty(f).ValueKind);
        var il = res.GetProperty("interleaving");
        Assert.True(il.GetProperty("wins").GetInt32() > 0);
        Assert.Equal(0, il.GetProperty("losses").GetInt32());
        Assert.Equal(1.0, il.GetProperty("deltaPreference").GetProperty("value").GetDouble(), 6);
        Assert.True(il.GetProperty("pValue").GetDouble() < 0.05);
        await GetJson(app.Client, "/api/experiments/nope/results", HttpStatusCode.NotFound);
        Assert.Contains("hs_experiment_assignments_total{experiment=\"il\",variant=\"interleaved\"}", await app.Client.GetStringAsync("/metrics"));
    }

    [Fact]
    public async Task Ab_experiment_applies_the_arm_config()
    {
        await using var app = await BrokerApp.StartAsync(experimentsJson: """
        { "experiments": [ { "id": "ab", "kind": "ab", "allocation": 1.0, "salt": "x",
          "control": { "mode": "lexical" }, "treatment": { "mode": "dense" } } ] }
        """);
        for (int i = 0; i < 20; i++)
        {
            var session = $"user{i}";
            var j = await GetJson(app.Client, $"/api/search?q=capital&sessionId={session}");
            var variant = j.GetProperty("experiment").GetProperty("variant").GetString();
            Assert.Equal(ExperimentAssigner.Arm("x", session) == Variant.Control ? "control" : "treatment", variant);
            Assert.Equal(variant == "control" ? "lexical" : "dense", j.GetProperty("mode").GetString());
        }
        var res = await GetJson(app.Client, "/api/experiments/ab/results");
        Assert.Equal(20, res.GetProperty("queries").GetInt32());
        Assert.Equal(["control", "treatment"], res.GetProperty("variants").EnumerateArray().Select(x => x.GetProperty("name").GetString()));
        Assert.Equal(20, res.GetProperty("variants").EnumerateArray().Sum(x => x.GetProperty("sessions").GetInt32()));
        Assert.Equal(JsonValueKind.Null, res.GetProperty("interleaving").ValueKind);
        Assert.True(res.GetProperty("srm").GetProperty("pValue").GetDouble() >= 0);
        var list = await GetJson(app.Client, "/api/experiments");
        var e = list.GetProperty("experiments")[0];
        foreach (var f in new[] { "id", "kind", "status", "allocation", "control", "treatment" }) Assert.True(e.TryGetProperty(f, out _), f);
    }

    [Fact]
    public async Task Metrics_endpoint_exposes_the_contract_names()
    {
        await using var app = await BrokerApp.StartAsync();
        await GetJson(app.Client, "/api/search?q=capital");
        var m = await app.Client.GetStringAsync("/metrics");
        foreach (var name in new[] { "hs_search_requests_total", "hs_search_duration_seconds_bucket", "hs_shard_requests_total", "hs_shard_duration_seconds_bucket",
                                     "hs_degradation_total", "hs_encode_duration_seconds", "hs_hedges_sent_total", "hs_hedges_won_total",
                                     "hs_events_ingested_total", "hs_suggest_duration_seconds", "hs_experiment_assignments_total" })
            Assert.Contains(name, m);
        Assert.Contains("le=\"0.3\"", m);
        Assert.Contains("hs_search_requests_total{mode=\"hybrid\",outcome=\"ok\"} 1", m);
    }
}

[Collection("timing")]
public class EventLogFailureTests
{
    [Fact]
    public async Task Unwritable_event_log_degrades_instead_of_stopping_the_broker()
    {
        var blocker = Path.GetTempFileName(); // a *file* where the events directory should be
        try
        {
            await using var app = await BrokerApp.StartAsync(extra: new() { ["Broker:Events:Directory"] = Path.Combine(blocker, "events") });
            var r = await app.Client.PostAsJsonAsync("/api/events", new { events = new[] { new { type = "query", sessionId = "s" } } });
            Assert.Equal(HttpStatusCode.Accepted, r.StatusCode);
            Assert.Equal(HttpStatusCode.OK, (await app.Client.GetAsync("/api/search?q=capital&mode=lexical")).StatusCode);
            var log = app.Service<EventLogWriter>();
            Assert.True(await log.FlushAsync(TimeSpan.FromSeconds(10), CancellationToken.None));
            Assert.StartsWith("unwritable", log.Status);
            Assert.Contains("hs_events_dropped_total{reason=\"write_error\"} 1", await app.Client.GetStringAsync("/metrics"));
        }
        finally { File.Delete(blocker); }
    }
}
