namespace HybridSearch.Contracts;

/// <summary>
/// Feature-hashing "embedding" shared by FakeShard (passages) and the broker's hashing-dev
/// encoder (queries). A test/dev double only: it makes dense mode return lexically related
/// passages without a model. Never used for any reported number.
/// </summary>
public static class HashingEmbedding
{
    public static float[] Embed(string text, int dim)
    {
        var v = new float[dim];
        foreach (var tok in SimpleTokens(text))
        {
            ulong h = System.IO.Hashing.XxHash64.HashToUInt64(System.Text.Encoding.UTF8.GetBytes(tok));
            int idx = (int)(h % (ulong)dim);
            v[idx] += ((h >> 32) & 1) == 0 ? 1f : -1f;
        }
        double sum = 0;
        foreach (var x in v) sum += (double)x * x;
        if (sum > 0) { float inv = (float)(1.0 / Math.Sqrt(sum)); for (int i = 0; i < v.Length; i++) v[i] *= inv; }
        return v;
    }

    public static IEnumerable<string> SimpleTokens(string text)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var ch in text)
        {
            if (char.IsLetterOrDigit(ch)) sb.Append(char.ToLowerInvariant(ch));
            else if (sb.Length > 0) { yield return sb.ToString(); sb.Clear(); }
        }
        if (sb.Length > 0) yield return sb.ToString();
    }
}
