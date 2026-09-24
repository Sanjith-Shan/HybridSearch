using System.Collections.Concurrent;
using Grpc.Core;
using HybridSearch.Proto.V1;

namespace HybridSearch.FakeShard;

/// <summary>
/// Fault-injection knobs. Mutable at runtime (tests flip them mid-run); every field is read
/// once per call, so a change applies from the next request.
/// </summary>
public sealed class FaultOptions
{
    /// <summary>Base latency of every Search.</summary>
    public volatile int LatencyMs;
    /// <summary>Uniform extra latency in [0, JitterMs).</summary>
    public volatile int JitterMs;
    /// <summary>Probability that a Search takes TailMs extra (a slow outlier: GC, cold cache...).</summary>
    public double TailProbability;
    public volatile int TailMs;
    /// <summary>Probability that a Search fails with <see cref="FailStatus"/>.</summary>
    public double FailRate;
    public StatusCode FailStatus = StatusCode.Unavailable;
    /// <summary>Probability that a Search returns half its hits with partial=true.</summary>
    public double PartialRate;
    /// <summary>If true (default), a Search whose injected latency exceeds deadline_budget_us stops at the budget and returns partial=true, like the real shard.</summary>
    public volatile bool HonorBudget = true;
    public volatile bool VectorReady = true;
    /// <summary>If true, Health fails (Unavailable).</summary>
    public volatile bool HealthDown;
    public volatile int FetchLatencyMs;
}

/// <summary>What the fake saw — tests assert on these (budget propagation, cancellation, trace context).</summary>
public sealed class ShardObservations
{
    public long SearchCalls;
    public long SearchCompleted;
    public long SearchCancelled;
    public long FetchCalls;
    public long HealthCalls;
    public readonly ConcurrentQueue<ObservedSearch> Searches = new();
}

public sealed record ObservedSearch(
    string RequestId, ulong DeadlineBudgetUs, TimeSpan? GrpcDeadlineRemaining, string[] Traceparents,
    bool Lexical, bool Vector, uint BeamWidth, DateTime ReceivedUtc);

public sealed class FakeShardService(FakeCorpus corpus, uint shardId, FaultOptions faults, ShardObservations seen) : Shard.ShardBase
{
    private static readonly ThreadLocal<Random> Rng = new(() => new Random(Environment.CurrentManagedThreadId * 7919 + Environment.TickCount));

    public override async Task<SearchResponse> Search(SearchRequest request, ServerCallContext context)
    {
        Interlocked.Increment(ref seen.SearchCalls);
        var now = DateTime.UtcNow;
        TimeSpan? grpcRemaining = context.Deadline == DateTime.MaxValue ? null : context.Deadline - now;
        seen.Searches.Enqueue(new ObservedSearch(request.RequestId, request.DeadlineBudgetUs, grpcRemaining,
            context.RequestHeaders.Where(h => h.Key == "traceparent").Select(h => h.Value).ToArray(),
            request.Lexical?.Enabled ?? false, request.Vector?.Enabled ?? false, request.Vector?.BeamWidth ?? 0, now));
        while (seen.Searches.Count > 10_000) seen.Searches.TryDequeue(out _);

        var rng = Rng.Value!;
        int latency = faults.LatencyMs + (faults.JitterMs > 0 ? rng.Next(faults.JitterMs) : 0);
        if (faults.TailProbability > 0 && rng.NextDouble() < faults.TailProbability) latency += faults.TailMs;
        bool partial = false;
        if (request.DeadlineBudgetUs > 0 && faults.HonorBudget && latency * 1000L > (long)request.DeadlineBudgetUs)
        {
            latency = (int)(request.DeadlineBudgetUs / 1000);
            partial = true;
        }
        try
        {
            if (latency > 0) await Task.Delay(latency, context.CancellationToken);
        }
        catch (OperationCanceledException)
        {
            Interlocked.Increment(ref seen.SearchCancelled);
            throw new RpcException(new Status(StatusCode.Cancelled, "cancelled by client"));
        }

        if (faults.FailRate > 0 && rng.NextDouble() < faults.FailRate)
            throw new RpcException(new Status(faults.FailStatus, "injected failure"));
        if (faults.PartialRate > 0 && rng.NextDouble() < faults.PartialRate) partial = true;

        var terms = FakeCorpus.Analyze(request.Query).ToList();
        var resp = new SearchResponse { ShardId = shardId, Partial = partial };
        resp.AnalyzedQueryTerms.AddRange(terms.Distinct());
        if (request.Lexical is { Enabled: true } lp)
        {
            var hits = corpus.SearchLexical(terms, (int)Math.Max(1, lp.K), lp.K1 > 0 ? lp.K1 : 0.9f, lp.B > 0 ? lp.B : 0.4f);
            if (partial) hits = hits.Take(Math.Max(1, hits.Count / 2)).ToList();
            foreach (var h in hits)
            {
                var hit = new Hit { DocId = h.DocId, Score = h.Score };
                if (lp.Explain)
                    hit.Terms.AddRange(h.Terms.Select(t => new TermContribution { Term = t.Term, Score = t.Score, Tf = (uint)t.Tf, Df = (uint)t.Df }));
                resp.LexicalHits.Add(hit);
            }
            resp.Stats = new SearchStats { DocsScored = (ulong)corpus.Passages.Count };
        }
        if (request.Vector is { Enabled: true } vp)
        {
            if (!faults.VectorReady) throw new RpcException(new Status(StatusCode.FailedPrecondition, "vector index not loaded"));
            if (vp.QueryEmbedding.Count != corpus.Dimension)
                throw new RpcException(new Status(StatusCode.InvalidArgument, $"query_embedding has {vp.QueryEmbedding.Count} dims, index has {corpus.Dimension}"));
            var hits = corpus.SearchVector(vp.QueryEmbedding, (int)Math.Max(1, vp.K));
            if (partial) hits = hits.Take(Math.Max(1, hits.Count / 2)).ToList();
            foreach (var (id, s) in hits) resp.VectorHits.Add(new Hit { DocId = id, Score = s });
        }
        Interlocked.Increment(ref seen.SearchCompleted);
        return resp;
    }

    public override async Task<FetchResponse> Fetch(FetchRequest request, ServerCallContext context)
    {
        Interlocked.Increment(ref seen.FetchCalls);
        if (faults.FetchLatencyMs > 0) await Task.Delay(faults.FetchLatencyMs, context.CancellationToken);
        var resp = new FetchResponse();
        foreach (var id in request.DocIds)
        {
            var p = corpus.Get(id);
            if (p is null) continue;
            var (text, hl, isSnippet) = FakeCorpus.Snippet(p.Text, request.HighlightTerms, (int)request.SnippetChars);
            var doc = new Document { DocId = id, Text = text, IsSnippet = isSnippet, DocLength = (uint)corpus.DocLength(id) };
            doc.Highlights.AddRange(hl.Select(h => new Highlight { Start = h.Start, End = h.End }));
            resp.Documents.Add(doc);
        }
        return resp;
    }

    public override Task<HealthResponse> Health(HealthRequest request, ServerCallContext context)
    {
        Interlocked.Increment(ref seen.HealthCalls);
        if (faults.HealthDown) throw new RpcException(new Status(StatusCode.Unavailable, "health down (injected)"));
        return Task.FromResult(new HealthResponse
        {
            ShardId = shardId,
            FirstDocId = corpus.FirstDocId,
            LastDocId = corpus.LastDocId,
            NumDocs = (ulong)corpus.Passages.Count,
            LexicalReady = true,
            VectorReady = faults.VectorReady,
            VectorDim = (uint)corpus.Dimension,
            BuildInfo = "HybridSearch.FakeShard (test double, not the C++ engine)",
        });
    }
}
