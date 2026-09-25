using System.Collections.Concurrent;
using System.Diagnostics;
using HybridSearch.Broker.Api;
using HybridSearch.Broker.Autocomplete;
using HybridSearch.Broker.Experiments;
using HybridSearch.Broker.Models;
using HybridSearch.Broker.Observability;
using HybridSearch.Broker.Shards;
using HybridSearch.Proto.V1;
using Microsoft.Extensions.Options;

namespace HybridSearch.Broker.Search;

/// <summary>Thrown when no slice answered at all: there is nothing honest to return.</summary>
public sealed class AllShardsFailedException(IReadOnlyList<int> failed)
    : Exception($"all shards failed: {string.Join(",", failed)}")
{
    public IReadOnlyList<int> FailedShards { get; } = failed;
}

/// <summary>One ranker configuration after applying defaults and experiment overrides.</summary>
public sealed record RankerSpec(SearchMode Mode, bool Rerank, int RerankDepth, FusionOptions Fusion)
{
    public bool NeedsLexical => Mode is SearchMode.Hybrid or SearchMode.Lexical;
    public bool NeedsDense => Mode is SearchMode.Hybrid or SearchMode.Dense;
}

/// <summary>
/// The query path: plan (degradation policy) → encode → fan-out → merge → fuse → rerank cascade →
/// fetch snippets → respond. Also runs the experiment arm / interleaving and logs the served page.
/// </summary>
public sealed class SearchService(
    ShardFanOut fanOut,
    ModelProvider models,
    ExperimentRegistry experiments,
    AutocompleteService autocomplete,
    EventLogWriter eventLog,
    CostModel costs,
    BrokerMetrics metrics,
    IOptions<BrokerOptions> options,
    ILogger<SearchService> logger)
{
    private readonly BrokerOptions _o = options.Value;
    private readonly ConcurrentDictionary<string, string[]> _analyzedTerms = new(StringComparer.Ordinal);
    private long _inFlight;

    private sealed class Candidates
    {
        public List<ScoredDoc> Lexical { get; init; } = [];
        public List<ScoredDoc> Dense { get; init; } = [];
        public Dictionary<ulong, int> OriginSlice { get; } = [];
        public Dictionary<ulong, IReadOnlyList<TermContribution>> Explain { get; } = [];
        public string[] AnalyzedTerms { get; set; } = [];
    }

    private sealed class ArmRanking
    {
        public required RankerSpec Spec { get; init; }
        public List<FusedDoc> Fused { get; set; } = [];
        public List<ulong> Final { get; set; } = [];
    }

    public RankerSpec DefaultSpec(SearchQuery q) => new(
        q.Mode, q.Rerank, _o.Search.RerankDepth,
        new FusionOptions(ParseFusion(_o.Search.Fusion), _o.Search.RrfK, _o.Search.Alpha, ParseNorm(_o.Search.Normalization)));

    public static RankerSpec Apply(RankerSpec baseSpec, RankerConfig? cfg)
    {
        if (cfg is null) return baseSpec;
        var mode = baseSpec.Mode;
        if (cfg.Mode is not null && SearchModes.TryParse(cfg.Mode, out var m)) mode = m;
        var f = baseSpec.Fusion;
        f = f with
        {
            Method = cfg.Fusion is null ? f.Method : ParseFusion(cfg.Fusion),
            RrfK = cfg.RrfK ?? f.RrfK,
            Alpha = cfg.Alpha ?? f.Alpha,
            Normalization = cfg.Normalization is null ? f.Normalization : ParseNorm(cfg.Normalization),
        };
        return new RankerSpec(mode, cfg.Rerank ?? baseSpec.Rerank, cfg.RerankDepth ?? baseSpec.RerankDepth, f);
    }

    public static FusionMethod ParseFusion(string s) => s.Equals("weighted", StringComparison.OrdinalIgnoreCase) ? FusionMethod.Weighted : FusionMethod.Rrf;
    public static ScoreNormalization ParseNorm(string s) => s.Equals("zscore", StringComparison.OrdinalIgnoreCase) ? ScoreNormalization.ZScore : ScoreNormalization.MinMax;

    public long InFlight => Interlocked.Read(ref _inFlight);

    /// <param name="onLexical">If set, called with a lexical-only first page as soon as shards answer (SSE).</param>
    public async Task<SearchResponseDto> SearchAsync(
        SearchQuery q, string requestId, Func<SearchResponseDto, Task>? onLexical, CancellationToken ct)
    {
        long t0 = Stopwatch.GetTimestamp();
        var deadline = Deadline.FromNow(TimeSpan.FromMilliseconds(q.DeadlineMs));
        long inFlight = Interlocked.Increment(ref _inFlight);
        metrics.InFlightSearches.Inc();
        using var activity = Telemetry.Source.StartActivity(Telemetry.SearchSpan);
        activity?.SetTag("request.id", requestId);
        activity?.SetTag("search.mode", SearchModes.Name(q.Mode));
        activity?.SetTag("search.k", q.K);
        activity?.SetTag("search.deadline_ms", q.DeadlineMs);
        try
        {
            return await SearchCoreAsync(q, requestId, deadline, t0, inFlight, onLexical, activity, ct).ConfigureAwait(false);
        }
        finally
        {
            Interlocked.Decrement(ref _inFlight);
            metrics.InFlightSearches.Dec();
        }
    }

    private async Task<SearchResponseDto> SearchCoreAsync(
        SearchQuery q, string requestId, Deadline deadline, long t0, long inFlight,
        Func<SearchResponseDto, Task>? onLexical, Activity? activity, CancellationToken ct)
    {
        // ---- experiment arm(s) ----
        var assignment = ExperimentAssigner.Assign(experiments.Experiments, q.SessionId);
        var baseSpec = DefaultSpec(q);
        ArmRanking[] arms;
        ExperimentDto? expDto = null;
        if (assignment is null)
        {
            arms = [new ArmRanking { Spec = baseSpec }];
        }
        else
        {
            var e = assignment.Experiment;
            var control = Apply(baseSpec, e.Control);
            var treatment = Apply(baseSpec, e.EffectiveTreatment);
            if (e.Interleaves)
            {
                arms = [new ArmRanking { Spec = control }, new ArmRanking { Spec = treatment }];
                expDto = new ExperimentDto(e.Id, "interleaved", true);
            }
            else
            {
                arms = [new ArmRanking { Spec = assignment.Variant == Variant.Control ? control : treatment }];
                expDto = new ExperimentDto(e.Id, assignment.VariantName, false);
            }
            metrics.ExperimentAssignments.WithLabels(e.Id, expDto.Variant).Inc();
            activity?.SetTag("experiment.id", e.Id);
            activity?.SetTag("experiment.variant", expDto.Variant);
        }

        // ---- plan ----
        var requested = new RequestedPlan(arms.Any(a => a.Spec.NeedsLexical), arms.Any(a => a.Spec.NeedsDense), arms.Any(a => a.Spec.Rerank));
        int rerankDepth = Math.Max(q.K, arms.Where(a => a.Spec.Rerank).Select(a => a.Spec.RerankDepth).DefaultIfEmpty(0).Max());
        var caps = new Capabilities(models.Encoder.IsAvailable, VectorIndexReady(), models.Reranker.IsAvailable);
        var est = costs.Estimate(rerankDepth);
        int pressure = DegradationPolicy.PressureLevel(inFlight, _o.Degradation.InFlightThresholds);
        bool probe = Random.Shared.NextDouble() < _o.Degradation.ProbeFraction;
        var plan = DegradationPolicy.Plan(requested, caps, deadline.Remaining.TotalMilliseconds, est, pressure, probe);
        var steps = new List<DegradationStep>(plan.Steps);
        bool dense = plan.Dense, rerank = plan.Rerank;

        // ---- encode ----
        double encodeMs = 0;
        float[]? embedding = null;
        if (dense)
        {
            long te = Stopwatch.GetTimestamp();
            using var span = Telemetry.Source.StartActivity(Telemetry.EncodeSpan);
            using var encodeCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            encodeCts.CancelAfter(deadline.Reserve(TimeSpan.FromMilliseconds(est.ShardLexicalMs + est.FetchMs)).Remaining);
            try
            {
                embedding = models.Encoder.Encode(q.Q, encodeCts.Token);
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                dense = false;
                steps.Add(new DegradationStep(DegradationKind.LexicalOnly, "encode_deadline"));
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                dense = false;
                steps.Add(new DegradationStep(DegradationKind.LexicalOnly, "encode_failed"));
                logger.LogWarning(ex, "query encoding failed");
            }
            encodeMs = Stopwatch.GetElapsedTime(te).TotalMilliseconds;
            if (embedding is not null)
            {
                costs.Encode.Record(TimeSpan.FromMilliseconds(encodeMs));
                metrics.EncodeDuration.Observe(encodeMs / 1000);
            }
        }
        bool lexical = plan.Lexical || !dense;

        // ---- fan-out ----
        int depth = Math.Max(_o.Search.CandidateDepth, rerankDepth);
        var req = new SearchRequest
        {
            Query = q.Q,
            RequestId = requestId,
            Lexical = new LexicalParams { Enabled = lexical, K = (uint)depth, Explain = q.Explain },
            Vector = new VectorParams
            {
                Enabled = dense,
                K = (uint)depth,
                BeamWidth = plan.ShrinkBeam ? _o.Search.DenseBeamWidthShrunk : _o.Search.DenseBeamWidth,
            },
        };
        if (embedding is not null && dense) req.Vector.QueryEmbedding.AddRange(embedding);

        var reserve = est.FetchMs + (rerank ? est.RerankMs : 0);
        var shardDeadline = deadline.Reserve(TimeSpan.FromMilliseconds(reserve), floor: TimeSpan.FromMilliseconds(Math.Min(10, deadline.Remaining.TotalMilliseconds)));
        long ts = Stopwatch.GetTimestamp();
        var outcomes = await fanOut.SearchAsync(req, shardDeadline, ct).ConfigureAwait(false);
        double shardsMs = Stopwatch.GetElapsedTime(ts).TotalMilliseconds;
        (dense ? plan.ShrinkBeam ? costs.ShardShrunk : costs.ShardFull : costs.ShardLexical).Record(TimeSpan.FromMilliseconds(shardsMs));

        var failed = outcomes.Where(o => o.Status == SliceStatus.Failed).Select(o => o.SliceId).Order().ToList();
        var partial = outcomes.Where(o => o.Status == SliceStatus.Partial).Select(o => o.SliceId).Order().ToList();
        var hedged = outcomes.Where(o => o.HedgeWon).Select(o => o.SliceId).Order().ToList();
        activity?.SetTag("shards.failed", failed.Count);
        activity?.SetTag("shards.partial", partial.Count);
        activity?.SetTag("shards.hedged", hedged.Count);
        if (failed.Count == outcomes.Length) throw new AllShardsFailedException(failed);

        var cands = Merge(outcomes, depth);
        if (cands.AnalyzedTerms.Length > 0) RememberTerms(q.Q, cands.AnalyzedTerms);

        string? didYouMean = DidYouMean(q.Q);
        var degradationSoFar = () => BuildDegradation(steps, partial, failed, hedged);

        double fetchMs = 0;
        // ---- lexical-first page (SSE) ----
        if (onLexical is not null)
        {
            var lexIds = cands.Lexical.Take(q.K).Select(d => d.DocId).ToList();
            var (docs, _, ms) = await FetchSnippetsAsync(lexIds, cands, deadline, ct).ConfigureAwait(false);
            fetchMs += ms;
            var lexResults = BuildResults(lexIds, null, cands, new Dictionary<ulong, float>(), docs, null, q.Explain, lexicalPage: true);
            var page = new SearchResponseDto(requestId, q.Q, didYouMean, "lexical", lexResults, degradationSoFar(),
                new TimingsDto(Ms(t0), encodeMs, shardsMs, 0, 0, fetchMs), expDto);
            await onLexical(page).ConfigureAwait(false);
        }

        // ---- fuse ----
        long tf = Stopwatch.GetTimestamp();
        using (Telemetry.Source.StartActivity(Telemetry.FuseSpan))
        {
            foreach (var arm in arms) arm.Fused = RankArm(arm.Spec, cands, dense);
        }
        double fuseMs = Stopwatch.GetElapsedTime(tf).TotalMilliseconds;

        // ---- rerank cascade ----
        double rerankMs = 0;
        var rerankScores = new Dictionary<ulong, float>();
        var rerankArms = arms.Where(a => a.Spec.Rerank).ToList();
        if (rerank && rerankArms.Count > 0)
        {
            var est2 = costs.Estimate(rerankDepth);
            if (deadline.Remaining.TotalMilliseconds < est2.RerankMs + est2.FetchMs)
            {
                rerank = false;
                steps.Add(new DegradationStep(DegradationKind.RerankerSkipped, "deadline"));
            }
            else
            {
                var pool = rerankArms.SelectMany(a => a.Fused.Take(a.Spec.RerankDepth)).Select(d => d.DocId).Distinct().ToList();
                var (ok, ms, fms) = await RerankAsync(q.Q, pool, cands, rerankScores, deadline, est2.FetchMs, ct).ConfigureAwait(false);
                rerankMs = ms;
                fetchMs += fms;
                if (ok is not null)
                {
                    rerank = false;
                    rerankScores.Clear();
                    steps.Add(new DegradationStep(DegradationKind.RerankerSkipped, ok));
                }
            }
        }

        foreach (var arm in arms)
        {
            var ids = arm.Fused.Select(d => d.DocId).ToList();
            if (rerank && arm.Spec.Rerank && rerankScores.Count > 0)
            {
                int n = Math.Min(arm.Spec.RerankDepth, ids.Count);
                var head = ids.Take(n).Select((id, i) => (id, i, s: rerankScores.GetValueOrDefault(id, float.NegativeInfinity)))
                    .OrderByDescending(x => x.s).ThenBy(x => x.i).Select(x => x.id);
                ids = head.Concat(ids.Skip(n)).ToList();
            }
            arm.Final = ids;
        }

        // ---- final page ----
        List<ulong> pageIds;
        Dictionary<ulong, Team>? teams = null;
        if (arms.Length == 2)
        {
            var seed = ExperimentAssigner.InterleaveSeed(assignment!.Experiment.Salt, q.SessionId!, q.Q);
            var inter = TeamDraftInterleaver.Interleave(arms[0].Final, arms[1].Final, q.K, seed);
            pageIds = inter.Select(d => d.DocId).ToList();
            teams = inter.ToDictionary(d => d.DocId, d => d.Team);
        }
        else
        {
            pageIds = arms[0].Final.Take(q.K).ToList();
        }

        var (snips, fetchFailed, fetchMs2) = await FetchSnippetsAsync(pageIds, cands, deadline, ct).ConfigureAwait(false);
        fetchMs += fetchMs2;
        if (fetchFailed.Count > 0) activity?.SetTag("fetch.failed_shards", string.Join(",", fetchFailed));
        var missingText = pageIds.Where(id => !snips.ContainsKey(id)).ToList();

        var results = BuildResults(pageIds, teams, cands, rerank ? rerankScores : [], snips, arms, q.Explain, lexicalPage: false);

        foreach (var s in steps) metrics.Degradation.WithLabels(s.KindName).Inc();
        var degradation = BuildDegradation(steps, partial, failed, hedged);
        if (missingText.Count > 0)
        {
            // Not a ranking degradation (level unchanged) but never silent: results without text are flagged.
            degradation = degradation with { Steps = [.. degradation.Steps, $"snippets_unavailable:{missingText.Count}"] };
        }
        activity?.SetTag("degradation.level", degradation.Level);

        string modeName = SearchModes.Name(arms.Length == 1 ? arms[0].Spec.Mode : q.Mode);
        var response = new SearchResponseDto(requestId, q.Q, didYouMean, modeName, results, degradation,
            new TimingsDto(Ms(t0), Round(encodeMs), Round(shardsMs), Round(fuseMs), Round(rerankMs), Round(fetchMs)), expDto);

        LogServed(q, requestId, response, assignment);
        return response;
    }

    private static double Ms(long t0) => Round(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
    private static double Round(double v) => Math.Round(v, 3);

    private bool VectorIndexReady()
    {
        // Unknown (no health answer yet) counts as ready: the shard will say so if it is not.
        foreach (var s in fanOut.Topology.Slices)
            foreach (var r in s.Replicas)
                if (r.LastHealth is null || r.LastHealth.VectorReady) return true;
        return false;
    }

    private static Candidates Merge(IReadOnlyList<SliceOutcome> outcomes, int depth)
    {
        var answered = outcomes.Where(o => o.Response is not null).ToList();
        var c = new Candidates
        {
            Lexical = Fusion.MergeTopK(answered.Select(o => (IReadOnlyList<ScoredDoc>)o.Response!.LexicalHits.Select(h => new ScoredDoc(h.DocId, h.Score)).ToList()), depth),
            Dense = Fusion.MergeTopK(answered.Select(o => (IReadOnlyList<ScoredDoc>)o.Response!.VectorHits.Select(h => new ScoredDoc(h.DocId, h.Score)).ToList()), depth),
        };
        foreach (var o in answered)
        {
            foreach (var h in o.Response!.LexicalHits)
            {
                c.OriginSlice.TryAdd(h.DocId, o.SliceId);
                if (h.Terms.Count > 0) c.Explain.TryAdd(h.DocId, h.Terms);
            }
            foreach (var h in o.Response.VectorHits) c.OriginSlice.TryAdd(h.DocId, o.SliceId);
            if (c.AnalyzedTerms.Length == 0 && o.Response.AnalyzedQueryTerms.Count > 0) c.AnalyzedTerms = [.. o.Response.AnalyzedQueryTerms];
        }
        return c;
    }

    private static List<FusedDoc> RankArm(RankerSpec spec, Candidates c, bool denseAvailable)
    {
        var mode = spec.Mode;
        if (!denseAvailable && mode != SearchMode.Lexical) mode = SearchMode.Lexical;
        return mode switch
        {
            SearchMode.Lexical => c.Lexical.Select(d => new FusedDoc(d.DocId, d.Score)).ToList(),
            SearchMode.Dense => c.Dense.Select(d => new FusedDoc(d.DocId, d.Score)).ToList(),
            _ => Fusion.Fuse(c.Lexical, c.Dense, spec.Fusion),
        };
    }

    /// <summary>Returns (failureReason or null, rerankMs, fetchMs).</summary>
    private async Task<(string? Failure, double RerankMs, double FetchMs)> RerankAsync(
        string query, List<ulong> pool, Candidates cands, Dictionary<ulong, float> into, Deadline deadline, double fetchReserveMs, CancellationToken ct)
    {
        if (pool.Count == 0) return (null, 0, 0);
        long tf = Stopwatch.GetTimestamp();
        var bySlice = GroupBySlice(pool, cands);
        var (docs, failedSlices) = await fanOut.FetchAsync(bySlice, [], 0, deadline.Reserve(TimeSpan.FromMilliseconds(fetchReserveMs)), ct).ConfigureAwait(false);
        double fetchMs = Stopwatch.GetElapsedTime(tf).TotalMilliseconds;
        var ids = pool.Where(docs.ContainsKey).ToList();
        if (ids.Count == 0) return ("passages_unavailable", 0, fetchMs);

        long tr = Stopwatch.GetTimestamp();
        using var span = Telemetry.Source.StartActivity(Telemetry.RerankSpan);
        span?.SetTag("rerank.candidates", ids.Count);
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(deadline.Reserve(TimeSpan.FromMilliseconds(fetchReserveMs)).Remaining);
        try
        {
            var scores = models.Reranker.Score(query, ids.Select(id => docs[id].Text).ToList(), cts.Token);
            for (int i = 0; i < ids.Count; i++) into[ids[i]] = scores[i];
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            span?.SetTag("rerank.terminated", true);
            return ("deadline", Stopwatch.GetElapsedTime(tr).TotalMilliseconds, fetchMs);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            logger.LogWarning(ex, "reranker failed");
            return ("error", Stopwatch.GetElapsedTime(tr).TotalMilliseconds, fetchMs);
        }
        var rerankMs = Stopwatch.GetElapsedTime(tr).TotalMilliseconds;
        costs.RerankPerDoc.Record(TimeSpan.FromMilliseconds(rerankMs / ids.Count));
        costs.Fetch.Record(TimeSpan.FromMilliseconds(fetchMs));
        metrics.RerankDuration.Observe(rerankMs / 1000);
        return (null, rerankMs, fetchMs);
    }

    private Dictionary<int, List<ulong>> GroupBySlice(IEnumerable<ulong> ids, Candidates c)
    {
        var map = new Dictionary<int, List<ulong>>();
        foreach (var id in ids)
        {
            int slice = c.OriginSlice.TryGetValue(id, out var s) ? s : fanOut.Topology.SliceForDoc(id)?.Id ?? -1;
            if (slice < 0) continue;
            if (!map.TryGetValue(slice, out var list)) map[slice] = list = [];
            list.Add(id);
        }
        return map;
    }

    private async Task<(Dictionary<ulong, Document> Docs, List<int> Failed, double Ms)> FetchSnippetsAsync(
        List<ulong> ids, Candidates c, Deadline deadline, CancellationToken ct)
    {
        long t = Stopwatch.GetTimestamp();
        var fetchDeadline = deadline.AtLeast(TimeSpan.FromMilliseconds(_o.Search.MinFetchMs));
        var (docs, failed) = await fanOut.FetchAsync(GroupBySlice(ids, c), c.AnalyzedTerms, (uint)_o.Search.SnippetChars, fetchDeadline, ct)
            .ConfigureAwait(false);
        var ms = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
        costs.Fetch.Record(TimeSpan.FromMilliseconds(ms));
        return (docs, failed, ms);
    }

    private static List<ResultDto> BuildResults(
        List<ulong> ids, Dictionary<ulong, Team>? teams, Candidates c, Dictionary<ulong, float> rerank,
        Dictionary<ulong, Document> docs, ArmRanking[]? arms, bool explain, bool lexicalPage)
    {
        var lexRank = c.Lexical.Select((d, i) => (d, i)).ToDictionary(x => x.d.DocId, x => (x.d.Score, Rank: x.i + 1));
        var denRank = c.Dense.Select((d, i) => (d, i)).ToDictionary(x => x.d.DocId, x => (x.d.Score, Rank: x.i + 1));
        var fusedByArm = arms?.Select(a => a.Fused.Select((d, i) => (d, i)).ToDictionary(x => x.d.DocId, x => (x.d.Score, Rank: x.i + 1))).ToArray();

        var results = new List<ResultDto>(ids.Count);
        for (int i = 0; i < ids.Count; i++)
        {
            var id = ids[i];
            Team? team = teams is not null && teams.TryGetValue(id, out var t) ? t : null;
            (double Score, int Rank)? fused = null;
            if (!lexicalPage && fusedByArm is not null)
            {
                var armIdx = team == Team.B ? 1 : 0;
                if (fusedByArm[armIdx].TryGetValue(id, out var f)) fused = f;
            }
            else if (lexicalPage && lexRank.TryGetValue(id, out var lr)) fused = (lr.Score, lr.Rank);

            var scores = new ScoresDto(
                lexRank.TryGetValue(id, out var l) ? l.Score : null, lexRank.TryGetValue(id, out l) ? l.Rank : null,
                denRank.TryGetValue(id, out var d) ? d.Score : null, denRank.TryGetValue(id, out d) ? d.Rank : null,
                fused?.Score, fused?.Rank,
                rerank.TryGetValue(id, out var r) ? r : null);

            string text = "";
            List<HighlightDto> highlights = [];
            bool isSnippet = false;
            if (docs.TryGetValue(id, out var doc))
            {
                text = doc.Text;
                highlights = Utf8Offsets.ToUtf16(doc.Text, doc.Highlights);
                isSnippet = doc.IsSnippet;
            }

            IReadOnlyList<TermDto>? terms = null;
            if (explain)
                terms = c.Explain.TryGetValue(id, out var tc) ? tc.Select(x => new TermDto(x.Term, x.Score, x.Tf, x.Df)).ToList() : [];

            results.Add(new ResultDto(id, i + 1, text, highlights, isSnippet, team?.ToString(), scores, terms));
        }
        return results;
    }

    private static DegradationDto BuildDegradation(List<DegradationStep> steps, List<int> partial, List<int> failed, List<int> hedged)
    {
        var ordered = steps.OrderBy(s => s.Kind).ToList();
        int level = ordered.Count == 0 ? 0 : ordered.Max(s => (int)s.Kind);
        return new DegradationDto(level, ordered.Select(s => s.ToString()).ToList(), partial, failed, hedged);
    }

    private string? DidYouMean(string query)
    {
        var speller = autocomplete.Speller;
        var trie = autocomplete.Trie;
        if (speller is null) return null;
        var norm = QueryNormalizer.Normalize(query);
        if (trie is not null && trie.CountOf(norm) > 0) return null; // a real logged query: leave it alone
        var s = speller.Suggest(norm);
        return s == norm ? null : s;
    }

    private void LogServed(SearchQuery q, string requestId, SearchResponseDto r, Assignment? a)
    {
        if (!_o.Events.LogServedResults || q.SessionId is null) return;
        eventLog.TryEnqueue(new LogRecord
        {
            Kind = "served",
            ServerTs = DateTimeOffset.UtcNow,
            SessionId = q.SessionId,
            RequestId = requestId,
            Query = q.Q,
            ExperimentId = a?.Experiment.Id,
            Variant = r.Experiment?.Variant,
            Interleaved = r.Experiment?.Interleaved,
            Results = r.Results.Select(x => new ServedResult(x.DocId, x.Rank, x.Team)).ToList(),
        });
    }

    private void RememberTerms(string q, string[] terms)
    {
        if (_analyzedTerms.Count > 10_000) _analyzedTerms.Clear();
        _analyzedTerms[q] = terms;
    }

    /// <summary>
    /// Full passage for /api/doc. Highlight terms come from the analyzed form of <paramref name="q"/>:
    /// cached from a recent search, else one cheap lexical Search (k=1) on the owning slice, since the
    /// broker has no analyzer of its own (the shard owns Lucene-parity analysis).
    /// </summary>
    public async Task<DocResponseDto?> GetDocAsync(ulong docId, string? q, CancellationToken ct)
    {
        var deadline = Deadline.FromNow(TimeSpan.FromMilliseconds(Math.Max(_o.Search.DefaultDeadlineMs, 500)));
        var owner = fanOut.Topology.SliceForDoc(docId);
        var slices = owner is not null ? [owner] : fanOut.Topology.Slices;
        string[] terms = [];
        if (!string.IsNullOrWhiteSpace(q) && !_analyzedTerms.TryGetValue(q.Trim(), out terms!))
        {
            terms = [];
            var req = new SearchRequest
            {
                Query = q.Trim(),
                RequestId = "doc-terms",
                Lexical = new LexicalParams { Enabled = true, K = 1 },
                Vector = new VectorParams { Enabled = false },
            };
            var outcomes = await fanOut.SearchAsync(req, deadline.Reserve(TimeSpan.FromMilliseconds(50)), ct).ConfigureAwait(false);
            var any = outcomes.FirstOrDefault(o => o.Response is { AnalyzedQueryTerms.Count: > 0 });
            if (any is not null) terms = [.. any.Response!.AnalyzedQueryTerms];
        }
        var bySlice = slices.ToDictionary(s => s.Id, _ => new List<ulong> { docId });
        var (docs, _) = await fanOut.FetchAsync(bySlice, terms, 0, deadline, ct).ConfigureAwait(false);
        if (!docs.TryGetValue(docId, out var doc)) return null;
        return new DocResponseDto(docId, doc.Text, Utf8Offsets.ToUtf16(doc.Text, doc.Highlights));
    }
}
