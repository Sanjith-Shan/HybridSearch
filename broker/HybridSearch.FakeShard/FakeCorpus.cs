using System.Text;
using HybridSearch.Contracts;

namespace HybridSearch.FakeShard;

public sealed record Passage(ulong DocId, string Text);

/// <summary>
/// Tiny in-memory corpus with textbook BM25 (k1=0.9, b=0.4) over a lowercase/alphanumeric
/// tokenizer with a small stopword list — NOT Lucene parity; a test double only.
/// Dense scores are inner products of <see cref="HashingEmbedding"/> vectors.
/// </summary>
public sealed class FakeCorpus
{
    public static readonly HashSet<string> Stopwords =
        ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not",
         "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
         "what", "how", "who", "when", "where", "which", "does", "do"];

    private readonly Dictionary<ulong, int> _byId = [];
    private readonly List<Dictionary<string, int>> _tf = [];
    private readonly List<int> _len = [];
    private readonly Dictionary<string, int> _df = [];
    private readonly List<float[]> _vectors = [];
    private readonly double _avgLen;

    public FakeCorpus(IEnumerable<Passage> passages, int dim = 64)
    {
        Dimension = dim;
        Passages = passages.OrderBy(p => p.DocId).ToList();
        foreach (var p in Passages)
        {
            _byId[p.DocId] = _byId.Count;
            var tf = new Dictionary<string, int>();
            int len = 0;
            foreach (var t in Analyze(p.Text)) { tf[t] = tf.GetValueOrDefault(t) + 1; len++; }
            _tf.Add(tf);
            _len.Add(len);
            foreach (var t in tf.Keys) _df[t] = _df.GetValueOrDefault(t) + 1;
            _vectors.Add(HashingEmbedding.Embed(p.Text, dim));
        }
        _avgLen = _len.Count == 0 ? 1 : Math.Max(1, _len.Average());
    }

    public IReadOnlyList<Passage> Passages { get; }
    public int Dimension { get; }
    public ulong FirstDocId => Passages.Count == 0 ? 0 : Passages[0].DocId;
    public ulong LastDocId => Passages.Count == 0 ? 0 : Passages[^1].DocId;

    public static IEnumerable<string> Analyze(string text) =>
        HashingEmbedding.SimpleTokens(text).Where(t => !Stopwords.Contains(t));

    public Passage? Get(ulong id) => _byId.TryGetValue(id, out var i) ? Passages[i] : null;
    public int DocLength(ulong id) => _byId.TryGetValue(id, out var i) ? _len[i] : 0;

    public sealed record LexHit(ulong DocId, float Score, List<(string Term, float Score, int Tf, int Df)> Terms);

    public List<LexHit> SearchLexical(IReadOnlyList<string> terms, int k, float k1 = 0.9f, float b = 0.4f)
    {
        int n = Passages.Count;
        var hits = new List<LexHit>();
        var distinct = terms.Distinct().ToList();
        for (int i = 0; i < n; i++)
        {
            double score = 0;
            var contrib = new List<(string, float, int, int)>();
            foreach (var t in distinct)
            {
                if (!_tf[i].TryGetValue(t, out var tf)) continue;
                int df = _df[t];
                double idf = Math.Log(1 + (n - df + 0.5) / (df + 0.5));
                double s = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * _len[i] / _avgLen));
                score += s;
                contrib.Add((t, (float)s, tf, df));
            }
            if (score > 0) hits.Add(new LexHit(Passages[i].DocId, (float)score, contrib));
        }
        return hits.OrderByDescending(h => h.Score).ThenBy(h => h.DocId).Take(k).ToList();
    }

    public List<(ulong DocId, float Score)> SearchVector(IReadOnlyList<float> q, int k)
    {
        var hits = new List<(ulong, float)>(Passages.Count);
        for (int i = 0; i < Passages.Count; i++)
        {
            float dot = 0;
            var v = _vectors[i];
            for (int j = 0; j < v.Length; j++) dot += v[j] * q[j];
            hits.Add((Passages[i].DocId, dot));
        }
        return hits.OrderByDescending(h => h.Item2).ThenBy(h => h.Item1).Take(k).ToList();
    }

    /// <summary>
    /// Best window of ~<paramref name="chars"/> characters (most highlighted tokens), snapped to
    /// token boundaries; highlights as UTF-8 byte offsets into the returned text, per shard.proto.
    /// </summary>
    public static (string Text, List<(uint Start, uint End)> Highlights, bool IsSnippet) Snippet(
        string text, IReadOnlyCollection<string> terms, int chars)
    {
        var spans = TokenSpans(text).ToList();
        var termSet = terms.ToHashSet();
        int start = 0, end = text.Length;
        bool isSnippet = false;
        if (chars > 0 && text.Length > chars)
        {
            isSnippet = true;
            int bestStart = 0, bestHits = -1;
            foreach (var (s, _, _) in spans)
            {
                int hits = spans.Count(x => x.Start >= s && x.End <= s + chars && termSet.Contains(x.Token));
                if (hits > bestHits) { bestHits = hits; bestStart = s; }
            }
            start = bestStart;
            end = Math.Min(text.Length, start + chars);
            var inside = spans.Where(x => x.Start >= start && x.End <= end).ToList();
            if (inside.Count > 0 && end < text.Length) end = inside[^1].End;
        }
        var window = text[start..end];
        var hl = new List<(uint, uint)>();
        foreach (var (s, e, tok) in spans)
        {
            if (s < start || e > end || !termSet.Contains(tok)) continue;
            uint bs = (uint)Encoding.UTF8.GetByteCount(text.AsSpan(start, s - start));
            uint be = bs + (uint)Encoding.UTF8.GetByteCount(text.AsSpan(s, e - s));
            hl.Add((bs, be));
        }
        return (window, hl, isSnippet);
    }

    private static IEnumerable<(int Start, int End, string Token)> TokenSpans(string text)
    {
        int i = 0;
        while (i < text.Length)
        {
            while (i < text.Length && !char.IsLetterOrDigit(text[i])) i++;
            int s = i;
            while (i < text.Length && char.IsLetterOrDigit(text[i])) i++;
            if (i > s) yield return (s, i, text[s..i].ToLowerInvariant());
        }
    }

    public static IEnumerable<Passage> ReadTsv(string path)
    {
        foreach (var line in File.ReadLines(path))
        {
            int tab = line.IndexOf('\t');
            if (tab <= 0 || !ulong.TryParse(line.AsSpan(0, tab), out var id)) continue;
            yield return new Passage(id, line[(tab + 1)..]);
        }
    }

    /// <summary>Contiguous-by-ID partition (as the real shards do): slice i of n.</summary>
    public static IEnumerable<Passage> Slice(IEnumerable<Passage> all, int slice, int numSlices)
    {
        var sorted = all.OrderBy(p => p.DocId).ToList();
        int per = (sorted.Count + numSlices - 1) / numSlices;
        return sorted.Skip(slice * per).Take(per);
    }

    /// <summary>Built-in demo corpus (used when no --corpus is given). IDs are arbitrary but stable.</summary>
    public static IReadOnlyList<Passage> BuiltIn { get; } =
    [
        new(1001, "The capital of Peru is Lima, a coastal city founded by Francisco Pizarro in 1535."),
        new(1002, "Machu Picchu is a 15th-century Inca citadel in the Andes mountains of Peru."),
        new(1003, "Lima is the largest city in Peru and home to about a third of the country's population."),
        new(1004, "Paris is the capital and most populous city of France, on the river Seine."),
        new(1005, "Berlin is the capital of Germany and its largest city by population."),
        new(1006, "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water."),
        new(1007, "Chlorophyll absorbs light most strongly in the blue and red portions of the spectrum."),
        new(1008, "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration."),
        new(1009, "To lose weight, eat fewer calories than you burn and exercise regularly."),
        new(1010, "A balanced diet with protein, vegetables and whole grains supports healthy weight loss."),
        new(1011, "Python is a high-level programming language known for readable syntax."),
        new(1012, "C# is a modern object-oriented programming language developed by Microsoft for .NET."),
        new(1013, "The Great Wall of China stretches thousands of kilometres across northern China."),
        new(1014, "Mount Everest is Earth's highest mountain above sea level, at 8,849 metres."),
        new(1015, "The Amazon rainforest spans nine countries, including Brazil, Peru and Colombia."),
        new(1016, "Café culture in Paris dates back to the 17th century — Le Procope opened in 1686."),
        new(1017, "Vitamin D is produced in the skin in response to sunlight exposure."),
        new(1018, "Interest rates are set by central banks to control inflation and employment."),
        new(1019, "A mortgage is a loan used to purchase a home, repaid over 15 to 30 years."),
        new(1020, "The human heart has four chambers: two atria and two ventricles."),
        new(1021, "Blood pressure is measured in millimetres of mercury, written as systolic over diastolic."),
        new(1022, "Dogs were domesticated from wolves at least 15,000 years ago."),
        new(1023, "Cats sleep between 12 and 16 hours a day on average."),
        new(1024, "The speed of light in vacuum is 299,792,458 metres per second."),
        new(1025, "Water boils at 100 degrees Celsius at sea level; at altitude it boils at a lower temperature."),
        new(1026, "The Pacific Ocean is the largest and deepest of Earth's oceans."),
        new(1027, "Shakespeare wrote Hamlet around 1600; it is his longest play."),
        new(1028, "The Eiffel Tower in Paris was completed in 1889 for the World's Fair."),
        new(1029, "Tokyo is the capital of Japan and one of the most populous metropolitan areas in the world."),
        new(1030, "Coffee contains caffeine, a stimulant that can improve alertness — naïve drinkers feel it most."),
        new(1031, "Search engines use inverted indexes to find documents containing query terms quickly."),
        new(1032, "BM25 is a ranking function that scores documents by term frequency and inverse document frequency."),
    ];
}
