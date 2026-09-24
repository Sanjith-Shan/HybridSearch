using System.Text;

namespace HybridSearch.Broker.Autocomplete;

public readonly record struct Completion(string Text, long Count);

/// <summary>
/// Compact radix trie over UTF-8 bytes with the top-K completions precomputed at every node.
///
/// Layout (struct-of-arrays, no per-node objects): the distinct queries are sorted by byte
/// order and concatenated into one blob; each node's edge label is a (start, length) slice of
/// that blob; a node's children are contiguous and ordered by their first label byte, so a
/// lookup is a binary search per edge. Each node stores a slice of <c>_top</c> holding up to K
/// string indices ordered by (count desc, text asc). A lookup therefore costs O(|prefix|·log σ)
/// byte comparisons plus decoding the k answers, independent of how many queries share the prefix.
///
/// Path compression keeps the node count ≤ 2·(distinct queries); storing only min(K, subtree
/// size) completions per node keeps the precomputed lists small because most nodes are near leaves.
/// </summary>
public sealed class CompletionTrie
{
    private readonly byte[] _blob;
    private readonly int[] _strOffset;
    private readonly int[] _strLen;
    private readonly long[] _count;

    private readonly int[] _labelStart;
    private readonly int[] _labelLen;
    private readonly int[] _firstChild;
    private readonly int[] _childCount;
    private readonly int[] _topStart;
    private readonly int[] _topCount;
    private readonly int[] _top;

    public int MaxK { get; }
    public int DistinctQueries => _strLen.Length;
    public long TotalQueries { get; }
    public int NodeCount => _labelStart.Length;

    /// <summary>Approximate heap bytes held by the structure (arrays only).</summary>
    public long ApproximateBytes =>
        _blob.LongLength + 4L * (_strOffset.Length + _strLen.Length) + 8L * _count.Length +
        4L * 6 * _labelStart.Length + 4L * _top.Length;

    private CompletionTrie(byte[] blob, int[] strOffset, int[] strLen, long[] count, long total, int maxK,
        int[] labelStart, int[] labelLen, int[] firstChild, int[] childCount, int[] topStart, int[] topCount, int[] top)
    {
        _blob = blob; _strOffset = strOffset; _strLen = strLen; _count = count; TotalQueries = total; MaxK = maxK;
        _labelStart = labelStart; _labelLen = labelLen; _firstChild = firstChild; _childCount = childCount;
        _topStart = topStart; _topCount = topCount; _top = top;
    }

    /// <summary>Build from already-normalised query counts. Empty strings are ignored.</summary>
    public static CompletionTrie Build(IEnumerable<KeyValuePair<string, long>> counts, int maxK = 10)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxK);
        var entries = new List<(byte[] Bytes, long Count)>();
        long total = 0;
        foreach (var (text, c) in counts)
        {
            if (string.IsNullOrEmpty(text) || c <= 0) continue;
            entries.Add((Encoding.UTF8.GetBytes(text), c));
            total += c;
        }
        entries.Sort(static (a, b) => a.Bytes.AsSpan().SequenceCompareTo(b.Bytes));
        // Merge exact duplicates (callers normally pass a dictionary, but be robust).
        var merged = new List<(byte[] Bytes, long Count)>(entries.Count);
        foreach (var e in entries)
        {
            if (merged.Count > 0 && merged[^1].Bytes.AsSpan().SequenceEqual(e.Bytes))
                merged[^1] = (merged[^1].Bytes, merged[^1].Count + e.Count);
            else merged.Add(e);
        }

        int n = merged.Count;
        long blobLen = 0;
        foreach (var e in merged) blobLen += e.Bytes.Length;
        var blob = new byte[blobLen];
        var strOffset = new int[n];
        var strLen = new int[n];
        var count = new long[n];
        int off = 0;
        for (int i = 0; i < n; i++)
        {
            strOffset[i] = off; strLen[i] = merged[i].Bytes.Length; count[i] = merged[i].Count;
            merged[i].Bytes.CopyTo(blob, off);
            off += merged[i].Bytes.Length;
        }
        merged.Clear();

        var b = new Builder(blob, strOffset, strLen, count, maxK);
        b.BuildRoot();
        return new CompletionTrie(blob, strOffset, strLen, count, total, maxK,
            [.. b.LabelStart], [.. b.LabelLen], [.. b.FirstChild], [.. b.ChildCount], [.. b.TopStart], [.. b.TopCount], [.. b.Top]);
    }

    /// <summary>Top-k completions of an already-normalised prefix (see <see cref="QueryNormalizer.NormalizePrefix"/>).</summary>
    public List<Completion> Lookup(string normalizedPrefix, int k)
    {
        var result = new List<Completion>(Math.Min(k, MaxK));
        if (k <= 0 || DistinctQueries == 0) return result;
        int maxBytes = Encoding.UTF8.GetMaxByteCount(normalizedPrefix.Length);
        Span<byte> buf = maxBytes <= 512 ? stackalloc byte[maxBytes] : new byte[maxBytes];
        int len = Encoding.UTF8.GetBytes(normalizedPrefix, buf);
        int node = FindNode(buf[..len]);
        if (node < 0) return result;
        int take = Math.Min(k, _topCount[node]);
        for (int i = 0; i < take; i++)
        {
            int s = _top[_topStart[node] + i];
            result.Add(new Completion(Encoding.UTF8.GetString(_blob, _strOffset[s], _strLen[s]), _count[s]));
        }
        return result;
    }

    /// <summary>Count for an exact (normalised) query, 0 if absent.</summary>
    public long CountOf(string normalizedQuery)
    {
        var bytes = Encoding.UTF8.GetBytes(normalizedQuery);
        int lo = 0, hi = DistinctQueries - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            int c = _blob.AsSpan(_strOffset[mid], _strLen[mid]).SequenceCompareTo(bytes);
            if (c == 0) return _count[mid];
            if (c < 0) lo = mid + 1; else hi = mid - 1;
        }
        return 0;
    }

    private int FindNode(ReadOnlySpan<byte> prefix)
    {
        int node = 0, pos = 0;
        while (pos < prefix.Length)
        {
            int child = FindChild(node, prefix[pos]);
            if (child < 0) return -1;
            int labelLen = _labelLen[child];
            int m = Math.Min(labelLen, prefix.Length - pos);
            if (!_blob.AsSpan(_labelStart[child], m).SequenceEqual(prefix.Slice(pos, m))) return -1;
            pos += m;
            node = child; // if m < labelLen the prefix ends inside this edge: same completions as the child
        }
        return node;
    }

    private int FindChild(int node, byte first)
    {
        int lo = _firstChild[node], hi = lo + _childCount[node] - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            byte b = _blob[_labelStart[mid]];
            if (b == first) return mid;
            if (b < first) lo = mid + 1; else hi = mid - 1;
        }
        return -1;
    }

    private sealed class Builder(byte[] blob, int[] strOffset, int[] strLen, long[] count, int maxK)
    {
        public readonly List<int> LabelStart = [], LabelLen = [], FirstChild = [], ChildCount = [], TopStart = [], TopCount = [], Top = [];
        private readonly List<int> _scratch = [];

        public void BuildRoot()
        {
            int root = NewNode(0, 0);
            Build(root, 0, strLen.Length, 0);
        }

        private int NewNode(int labelStart, int labelLen)
        {
            LabelStart.Add(labelStart); LabelLen.Add(labelLen);
            FirstChild.Add(0); ChildCount.Add(0); TopStart.Add(0); TopCount.Add(0);
            return LabelStart.Count - 1;
        }

        private byte At(int s, int i) => blob[strOffset[s] + i];

        private int Lcp(int a, int b, int from)
        {
            int max = Math.Min(strLen[a], strLen[b]);
            int i = from;
            while (i < max && At(a, i) == At(b, i)) i++;
            return i;
        }

        /// <summary>
        /// Node covering sorted strings [lo, hi), all sharing their first <paramref name="depth"/> bytes.
        /// Children are allocated contiguously before recursing so each node's children are adjacent.
        /// </summary>
        private void Build(int node, int lo, int hi, int depth)
        {
            int terminal = -1;
            int i = lo;
            if (i < hi && strLen[i] == depth) { terminal = i; i++; } // sorted: the exact match comes first

            var groups = new List<(int Lo, int Hi, int Lcp)>();
            while (i < hi)
            {
                byte b = At(i, depth);
                int j = i + 1;
                while (j < hi && At(j, depth) == b) j++;
                groups.Add((i, j, Lcp(i, j - 1, depth + 1)));
                i = j;
            }

            FirstChild[node] = LabelStart.Count;
            ChildCount[node] = groups.Count;
            var children = new int[groups.Count];
            for (int g = 0; g < groups.Count; g++)
                children[g] = NewNode(strOffset[groups[g].Lo] + depth, groups[g].Lcp - depth);
            for (int g = 0; g < groups.Count; g++)
                Build(children[g], groups[g].Lo, groups[g].Hi, groups[g].Lcp);

            // Top-K of this node = best K among the terminal string and every child's top list.
            _scratch.Clear();
            if (terminal >= 0) _scratch.Add(terminal);
            foreach (var c in children)
                for (int t = 0; t < TopCount[c]; t++) _scratch.Add(Top[TopStart[c] + t]);
            _scratch.Sort((x, y) => count[x] != count[y] ? count[y].CompareTo(count[x]) : x.CompareTo(y));
            int take = Math.Min(maxK, _scratch.Count);
            TopStart[node] = Top.Count;
            TopCount[node] = take;
            for (int t = 0; t < take; t++) Top.Add(_scratch[t]);
        }
    }
}
