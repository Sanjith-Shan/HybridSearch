using System.Text.Json;

namespace HybridSearch.Broker.Experiments;

/// <summary>A point estimate with its bootstrap interval (wire: {"value","ciLow","ciHigh"}).</summary>
public sealed record MetricValue(double Value, double CiLow, double CiHigh)
{
    public static MetricValue From(Interval i) => new(Clean(i.Estimate), Clean(i.Low), Clean(i.High));
    private static double Clean(double v) => double.IsNaN(v) || double.IsInfinity(v) ? 0 : Math.Round(v, 6);
}

/// <summary>Per-arm metrics. For interleaving experiments there is one pseudo-variant, "interleaved".</summary>
public sealed record VariantResult(string Name, int Sessions, int Queries, int Clicks, IReadOnlyDictionary<string, MetricValue> Metrics);

public sealed record InterleavingResult(
    int Wins, int Losses, int Ties, int NoClicks, MetricValue DeltaPreference, double PValue, string Convention);

public sealed record SrmCheck(int ControlSessions, int TreatmentSessions, double ChiSquare, double PValue, bool Mismatch);

/// <summary>
/// Wire shape agreed with the web UI (docs/ARCHITECTURE.md, Experiments). Extra fields (srm,
/// differences, clickModels, noClicks, note) are additive.
/// </summary>
public sealed record ExperimentResults(
    string Id, string Kind, string Status,
    bool Simulated, IReadOnlyList<string> ClickModels,
    double ConfidenceLevel, DateTimeOffset UpdatedAt,
    int Queries, int ClickEvents,
    IReadOnlyList<VariantResult> Variants,
    IReadOnlyDictionary<string, MetricValue>? Differences,
    SrmCheck? Srm,
    InterleavingResult? Interleaving,
    string Note);

/// <summary>
/// Computes experiment results from event-log records. Unit of analysis: the query impression
/// (one "served" record); unit of resampling: the session (the randomisation unit).
///
/// Clicks are joined to impressions by requestId (and must come from the same session). A/B
/// metrics per impression: CTR = clicks per impression, clicks@1 = impressions with a click at
/// rank 1, abandonment = impressions with no click, MRR of first click = 1/rank of the
/// lowest-ranked-number click (0 if none).
///
/// Interleaving: team A = control, team B = treatment; per impression the team with more
/// distinct clicked documents (server-side team attribution from the served record) wins.
/// "wins" count impressions won by the treatment. Δ = (wins − losses)/(wins + losses + ties);
/// impressions without any click carry no preference and are reported separately as noClicks.
/// </summary>
public static class ExperimentAnalyzer
{
    private sealed class Impression
    {
        public required LogRecord Served { get; init; }
        public HashSet<ulong> ClickedDocs { get; } = [];
        public int? BestClickRank { get; set; }
        public bool ClickAt1 { get; set; }
    }

    public static ExperimentResults Analyze(ExperimentDefinition exp, IEnumerable<LogRecord> records, int bootstrapIterations, DateTimeOffset now)
    {
        var impressions = new Dictionary<string, Impression>(StringComparer.Ordinal);
        var clicks = new List<LogRecord>();
        foreach (var r in records)
        {
            if (r.Kind == "served" && r.ExperimentId == exp.Id && r.RequestId is not null && r.SessionId is not null)
                impressions.TryAdd(r.RequestId, new Impression { Served = r });
            else if (r.Kind == "event" && r.Type == "click" && r.RequestId is not null && r.DocId is not null)
                clicks.Add(r);
        }

        int clickEvents = 0;
        bool simulated = false;
        var clickModels = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var c in clicks)
        {
            if (!impressions.TryGetValue(c.RequestId!, out var imp) || imp.Served.SessionId != c.SessionId) continue;
            var shown = imp.Served.Results?.FirstOrDefault(x => x.DocId == c.DocId);
            if (shown is null) continue; // a click on something that was not served cannot be attributed
            clickEvents++;
            if (c.Simulated == true) simulated = true;
            if (c.ClickModel is { Length: > 0 } cm) clickModels.Add(cm);
            if (!imp.ClickedDocs.Add(c.DocId!.Value)) continue;
            imp.BestClickRank = imp.BestClickRank is { } b ? Math.Min(b, shown.Rank) : shown.Rank;
            if (shown.Rank == 1) imp.ClickAt1 = true;
        }

        var all = impressions.Values.ToList();
        InterleavingResult? il = null;
        SrmCheck? srm = null;
        List<VariantResult> variants;
        Dictionary<string, MetricValue>? diff = null;

        if (exp.Interleaves)
        {
            il = AnalyzeInterleaving(all, bootstrapIterations);
            variants = [VariantMetrics("interleaved", all, bootstrapIterations, 21, out _)];
        }
        else
        {
            var bySession = all.GroupBy(i => i.Served.SessionId!).ToList();
            int control = bySession.Count(g => g.First().Served.Variant == "control");
            int treatment = bySession.Count(g => g.First().Served.Variant == "treatment");
            var (chi, p) = Stats.SampleRatioMismatch(control, treatment);
            srm = new SrmCheck(control, treatment, Math.Round(chi, 6), Math.Round(p, 6), p < 0.001);
            var c = VariantMetrics("control", all.Where(i => i.Served.Variant == "control").ToList(), bootstrapIterations, 11, out var cc);
            var t = VariantMetrics("treatment", all.Where(i => i.Served.Variant == "treatment").ToList(), bootstrapIterations, 12, out var tc);
            variants = [c, t];
            diff = cc.Keys.ToDictionary(k => k, k => MetricValue.From(Stats.RatioDifferenceBootstrap(cc[k], tc[k], bootstrapIterations)));
        }

        string note = simulated
            ? "SIMULATED clicks (see clickModels). Not user traffic."
            : "Computed from the broker's event log; clicks not flagged as simulated. Traffic source is whatever sent events to this broker.";
        return new ExperimentResults(exp.Id, exp.Kind, exp.Status, simulated, [.. clickModels], ConfidenceLevel, now,
            all.Count, clickEvents, variants, diff, srm, il, note);
    }

    public const double ConfidenceLevel = 0.95;

    private static VariantResult VariantMetrics(string name, List<Impression> imps, int iters, ulong seed,
        out Dictionary<string, List<(double, double)>> clusters)
    {
        var sessions = imps.GroupBy(i => i.Served.SessionId!).ToList();
        List<(double, double)> Metric(Func<Impression, double> f) => sessions.Select(g => (g.Sum(f), (double)g.Count())).ToList();
        clusters = new Dictionary<string, List<(double, double)>>
        {
            ["ctr"] = Metric(i => i.ClickedDocs.Count),
            ["clicksAt1"] = Metric(i => i.ClickAt1 ? 1 : 0),
            ["abandonment"] = Metric(i => i.ClickedDocs.Count == 0 ? 1 : 0),
            ["mrrFirstClick"] = Metric(i => i.BestClickRank is { } r ? 1.0 / r : 0),
        };
        var metrics = new Dictionary<string, MetricValue>();
        ulong k = 0;
        foreach (var (key, c) in clusters)
            metrics[key] = MetricValue.From(Stats.RatioBootstrap(c, iters, ConfidenceLevel, seed + 100 * k++));
        return new VariantResult(name, sessions.Count, imps.Count, imps.Sum(i => i.ClickedDocs.Count), metrics);
    }

    private static InterleavingResult AnalyzeInterleaving(List<Impression> all, int iters)
    {
        int wins = 0, losses = 0, ties = 0, none = 0;
        var perSession = new Dictionary<string, (int W, int L, int T)>(StringComparer.Ordinal);
        foreach (var imp in all)
        {
            var shown = (imp.Served.Results ?? [])
                .Where(r => r.Team is "A" or "B")
                .Select(r => new InterleavedDoc(r.DocId, r.Team == "A" ? Team.A : Team.B))
                .ToList();
            var outcome = InterleavingAttribution.Credit(shown, imp.ClickedDocs).Outcome;
            var s = perSession.GetValueOrDefault(imp.Served.SessionId!);
            switch (outcome)
            {
                case InterleavingOutcome.WinB: wins++; s.W++; break;
                case InterleavingOutcome.WinA: losses++; s.L++; break;
                case InterleavingOutcome.Tie: ties++; s.T++; break;
                default: none++; break;
            }
            perSession[imp.Served.SessionId!] = s;
        }
        var delta = Stats.InterleavingDeltaBootstrap(perSession.Values.Select(v => (v.W, v.L, v.T)).ToList(), iters, ConfidenceLevel);
        return new InterleavingResult(wins, losses, ties, none, MetricValue.From(delta),
            Math.Round(Stats.SignTestTwoSided(wins, losses), 8),
            "team A = control, team B = treatment; wins = impressions where the treatment's team got more clicks; " +
            "delta = (wins - losses) / (wins + losses + ties); no-click impressions excluded; sign test excludes ties");
    }

    /// <summary>Streams every record from the event-log directory (malformed lines are skipped).</summary>
    public static async IAsyncEnumerable<LogRecord> ReadLogAsync(string directory,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        if (!Directory.Exists(directory)) yield break;
        foreach (var file in Directory.EnumerateFiles(directory, "events-*.jsonl").Order(StringComparer.Ordinal))
        {
            await using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 16, useAsync: true);
            using var sr = new StreamReader(fs);
            string? line;
            while ((line = await sr.ReadLineAsync(ct).ConfigureAwait(false)) is not null)
            {
                if (line.Length == 0) continue;
                LogRecord? rec;
                try { rec = JsonSerializer.Deserialize<LogRecord>(line, LogRecord.Json); }
                catch (JsonException) { continue; }
                if (rec is not null) yield return rec;
            }
        }
    }
}
