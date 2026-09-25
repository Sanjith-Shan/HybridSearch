using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using HybridSearch.Contracts;

namespace HybridSearch.Broker.Models;

public interface IQueryEncoder
{
    bool IsAvailable { get; }
    /// <summary>Human-readable status for /readyz and logs ("onnx model.onnx (768-d)", "missing: …").</summary>
    string Status { get; }
    int Dimension { get; }
    /// <summary>L2-normalised query embedding. CPU-bound and synchronous; <paramref name="ct"/> terminates the ONNX run.</summary>
    float[] Encode(string query, CancellationToken ct);
}

public interface IReranker
{
    bool IsAvailable { get; }
    string Status { get; }
    /// <summary>One relevance logit per passage (higher is more relevant). <paramref name="ct"/> terminates the ONNX run.</summary>
    float[] Score(string query, IReadOnlyList<string> passages, CancellationToken ct);
}

public sealed class UnavailableEncoder(string why) : IQueryEncoder
{
    public bool IsAvailable => false;
    public string Status => why;
    public int Dimension => 0;
    public float[] Encode(string query, CancellationToken ct) => throw new InvalidOperationException(why);
}

public sealed class UnavailableReranker(string why) : IReranker
{
    public bool IsAvailable => false;
    public string Status => why;
    public float[] Score(string query, IReadOnlyList<string> passages, CancellationToken ct) => throw new InvalidOperationException(why);
}

/// <summary>Shared ONNX plumbing: session creation, input-name discovery, cancellable runs.</summary>
internal static class Onnx
{
    public static InferenceSession CreateSession(string modelPath, int intraOpThreads)
    {
        var so = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
        if (intraOpThreads > 0) so.IntraOpNumThreads = intraOpThreads;
        return new InferenceSession(modelPath, so);
    }

    public static BertTokenizer CreateTokenizer(string vocabPath) =>
        BertTokenizer.Create(vocabPath, new BertOptions { LowerCaseBeforeTokenization = true });

    /// <summary>Runs the session; a cancellation sets RunOptions.Terminate, which aborts the native run.</summary>
    public static IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(
        InferenceSession session, IReadOnlyCollection<NamedOnnxValue> inputs, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        using var ro = new RunOptions();
        using var reg = ct.Register(static o => ((RunOptions)o!).Terminate = true, ro);
        try
        {
            return session.Run(inputs, session.OutputMetadata.Keys.Take(1).ToList(), ro);
        }
        catch (OnnxRuntimeException) when (ct.IsCancellationRequested)
        {
            throw new OperationCanceledException(ct);
        }
    }

    /// <summary>
    /// Builds input_ids / attention_mask / token_type_ids tensors for whichever of those names the
    /// model declares (exports differ: some drop token_type_ids).
    /// </summary>
    public static List<NamedOnnxValue> BertInputs(InferenceSession session, long[][] ids, long[][] typeIds)
    {
        int batch = ids.Length;
        int len = ids.Max(r => r.Length);
        var inputIds = new DenseTensor<long>([batch, len]);
        var mask = new DenseTensor<long>([batch, len]);
        var types = new DenseTensor<long>([batch, len]);
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < ids[b].Length; t++)
            {
                inputIds[b, t] = ids[b][t];
                mask[b, t] = 1;
                types[b, t] = typeIds[b][t];
            }
            // Padding: id 0 ([PAD] in BERT vocabularies), mask 0, type 0 — tensors start zeroed.
        }
        var inputs = new List<NamedOnnxValue>(3);
        foreach (var name in session.InputMetadata.Keys)
        {
            if (name.Contains("input_ids", StringComparison.OrdinalIgnoreCase)) inputs.Add(NamedOnnxValue.CreateFromTensor(name, inputIds));
            else if (name.Contains("attention_mask", StringComparison.OrdinalIgnoreCase)) inputs.Add(NamedOnnxValue.CreateFromTensor(name, mask));
            else if (name.Contains("token_type_ids", StringComparison.OrdinalIgnoreCase)) inputs.Add(NamedOnnxValue.CreateFromTensor(name, types));
            else throw new InvalidOperationException($"unexpected model input '{name}'");
        }
        return inputs;
    }
}

/// <summary>
/// BGE-base-en-v1.5 query encoder (docs/ARCHITECTURE.md: CLS pooling, L2-normalised, query
/// instruction prefix, max 64 tokens). Accepts either an export whose first output is the
/// token-level last_hidden_state [1, L, d] (CLS = position 0) or one that already pools to [1, d].
/// </summary>
public sealed class OnnxQueryEncoder : IQueryEncoder, IDisposable
{
    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;
    private readonly string _prefix;
    private readonly int _maxTokens;

    public OnnxQueryEncoder(string modelPath, string vocabPath, string prefix, int maxTokens, int intraOpThreads)
    {
        _session = Onnx.CreateSession(modelPath, intraOpThreads);
        _tokenizer = Onnx.CreateTokenizer(vocabPath);
        _prefix = prefix;
        _maxTokens = maxTokens;
        var output = _session.OutputMetadata.First().Value;
        Dimension = output.Dimensions[^1];
        Status = $"onnx {Path.GetFileName(modelPath)} ({(Dimension > 0 ? Dimension.ToString() : "?")}-d)";
    }

    public bool IsAvailable => true;
    public string Status { get; }
    public int Dimension { get; private set; }

    /// <summary>[CLS] + wordpieces(prefix + query), truncated, + [SEP].</summary>
    public long[] Tokenize(string query)
    {
        var pieces = _tokenizer.EncodeToIds(_prefix + query, addSpecialTokens: false, considerPreTokenization: true, considerNormalization: true);
        int keep = Math.Min(pieces.Count, _maxTokens - 2);
        var ids = new long[keep + 2];
        ids[0] = _tokenizer.ClassificationTokenId;
        for (int i = 0; i < keep; i++) ids[i + 1] = pieces[i];
        ids[^1] = _tokenizer.SeparatorTokenId;
        return ids;
    }

    public float[] Encode(string query, CancellationToken ct)
    {
        var ids = Tokenize(query);
        var inputs = Onnx.BertInputs(_session, [ids], [new long[ids.Length]]);
        using var outputs = Onnx.Run(_session, inputs, ct);
        var t = outputs.First().AsTensor<float>();
        float[] v;
        if (t.Dimensions.Length == 3)
        {
            int d = t.Dimensions[2];
            v = new float[d];
            for (int i = 0; i < d; i++) v[i] = t[0, 0, i];
        }
        else if (t.Dimensions.Length == 2)
        {
            int d = t.Dimensions[1];
            v = new float[d];
            for (int i = 0; i < d; i++) v[i] = t[0, i];
        }
        else throw new InvalidOperationException($"unexpected encoder output rank {t.Dimensions.Length}");
        Dimension = v.Length;
        VectorMath.L2NormalizeInPlace(v);
        return v;
    }

    public void Dispose() => _session.Dispose();
}

/// <summary>
/// MiniLM-style cross-encoder: pairs tokenised as [CLS] q [SEP] p [SEP] with segment ids 0/1,
/// truncated longest-first to <c>maxTokens</c>, scored in batches. Output [B,1] is the logit; for
/// a two-class head [B,2] the score is logit(relevant) − logit(not relevant).
/// </summary>
public sealed class OnnxCrossEncoder : IReranker, IDisposable
{
    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;
    private readonly int _maxTokens;
    private readonly int _batchSize;

    public OnnxCrossEncoder(string modelPath, string vocabPath, int maxTokens, int batchSize, int intraOpThreads)
    {
        _session = Onnx.CreateSession(modelPath, intraOpThreads);
        _tokenizer = Onnx.CreateTokenizer(vocabPath);
        _maxTokens = maxTokens;
        _batchSize = Math.Max(1, batchSize);
        Status = $"onnx {Path.GetFileName(modelPath)}";
    }

    public bool IsAvailable => true;
    public string Status { get; }

    public (long[] Ids, long[] TypeIds) EncodePair(IReadOnlyList<int> queryPieces, string passage)
    {
        var p = _tokenizer.EncodeToIds(passage, addSpecialTokens: false, considerPreTokenization: true, considerNormalization: true);
        var (qLen, pLen) = TruncateLongestFirst(queryPieces.Count, p.Count, _maxTokens - 3);
        var ids = new long[qLen + pLen + 3];
        var types = new long[ids.Length];
        int k = 0;
        ids[k++] = _tokenizer.ClassificationTokenId;
        for (int i = 0; i < qLen; i++) ids[k++] = queryPieces[i];
        ids[k++] = _tokenizer.SeparatorTokenId;
        for (int i = 0; i < pLen; i++) { types[k] = 1; ids[k++] = p[i]; }
        types[k] = 1;
        ids[k] = _tokenizer.SeparatorTokenId;
        return (ids, types);
    }

    /// <summary>HuggingFace "longest_first": remove one token at a time from the longer sequence.</summary>
    public static (int Q, int P) TruncateLongestFirst(int q, int p, int budget)
    {
        while (q + p > budget)
        {
            if (p >= q) p--; else q--;
        }
        return (q, p);
    }

    public float[] Score(string query, IReadOnlyList<string> passages, CancellationToken ct)
    {
        var scores = new float[passages.Count];
        var qPieces = _tokenizer.EncodeToIds(query, addSpecialTokens: false, considerPreTokenization: true, considerNormalization: true);
        for (int start = 0; start < passages.Count; start += _batchSize)
        {
            ct.ThrowIfCancellationRequested();
            int n = Math.Min(_batchSize, passages.Count - start);
            var ids = new long[n][];
            var types = new long[n][];
            for (int i = 0; i < n; i++) (ids[i], types[i]) = EncodePair(qPieces, passages[start + i]);
            var inputs = Onnx.BertInputs(_session, ids, types);
            using var outputs = Onnx.Run(_session, inputs, ct);
            var t = outputs.First().AsTensor<float>();
            for (int i = 0; i < n; i++)
            {
                scores[start + i] = t.Dimensions.Length switch
                {
                    1 => t[i],
                    2 when t.Dimensions[1] == 1 => t[i, 0],
                    2 when t.Dimensions[1] == 2 => t[i, 1] - t[i, 0],
                    _ => throw new InvalidOperationException("unexpected reranker output shape"),
                };
            }
        }
        return scores;
    }

    public void Dispose() => _session.Dispose();
}

public static class VectorMath
{
    public static void L2NormalizeInPlace(Span<float> v)
    {
        double sum = 0;
        foreach (var x in v) sum += (double)x * x;
        if (sum <= 0) return;
        float inv = (float)(1.0 / Math.Sqrt(sum));
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }
}

/// <summary>
/// Development stand-in: feature-hashes query tokens into a small L2-normalised vector — the same
/// function FakeShard uses for its passages, so dense mode returns lexically-related documents in
/// local UI work. Enabled only by explicit config (Models:QueryEncoderKind = "hashing-dev") and
/// reported as such in /readyz. It is not a model and produces no quality numbers.
/// </summary>
public sealed class HashingDevEncoder(int dimension) : IQueryEncoder
{
    public bool IsAvailable => true;
    public string Status => $"hashing-dev ({dimension}-d feature hashing; NOT a real model)";
    public int Dimension => dimension;
    public float[] Encode(string query, CancellationToken ct) => HashingEmbedding.Embed(query, dimension);
}
