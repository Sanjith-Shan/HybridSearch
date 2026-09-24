namespace HybridSearch.Broker.Experiments;

public readonly record struct Interval(double Estimate, double Low, double High);

/// <summary>Small, dependency-free statistics used by the experiment analyser.</summary>
public static class Stats
{
    /// <summary>
    /// Cluster (session-level) percentile bootstrap for a ratio metric Σnum / Σden. Sessions are the
    /// randomisation unit, so resampling sessions — not individual impressions — keeps the
    /// within-session correlation of queries in the interval width.
    /// </summary>
    public static Interval RatioBootstrap(
        IReadOnlyList<(double Num, double Den)> clusters, int iterations = 2000, double confidence = 0.95, ulong seed = 42)
    {
        double est = Ratio(clusters);
        if (clusters.Count == 0) return new Interval(double.NaN, double.NaN, double.NaN);
        var rng = new SplitMix64(seed);
        var samples = new double[iterations];
        for (int it = 0; it < iterations; it++)
        {
            double num = 0, den = 0;
            for (int i = 0; i < clusters.Count; i++)
            {
                var c = clusters[rng.NextInt(clusters.Count)];
                num += c.Num; den += c.Den;
            }
            samples[it] = den == 0 ? double.NaN : num / den;
        }
        return Percentiles(est, samples, confidence);
    }

    /// <summary>Bootstrap CI of ratio(treatment) − ratio(control), resampling each arm independently.</summary>
    public static Interval RatioDifferenceBootstrap(
        IReadOnlyList<(double Num, double Den)> control, IReadOnlyList<(double Num, double Den)> treatment,
        int iterations = 2000, double confidence = 0.95, ulong seed = 43)
    {
        double est = Ratio(treatment) - Ratio(control);
        if (control.Count == 0 || treatment.Count == 0) return new Interval(double.NaN, double.NaN, double.NaN);
        var rng = new SplitMix64(seed);
        var samples = new double[iterations];
        for (int it = 0; it < iterations; it++)
            samples[it] = Resample(treatment, ref rng) - Resample(control, ref rng);
        return Percentiles(est, samples, confidence);
    }

    /// <summary>
    /// Interleaving preference Δ = (wins − losses) / (wins + losses + ties), with a session-level
    /// cluster bootstrap CI. "wins" are impressions where team B (treatment) got more clicks.
    /// </summary>
    public static Interval InterleavingDeltaBootstrap(
        IReadOnlyList<(int Wins, int Losses, int Ties)> sessions, int iterations = 2000, double confidence = 0.95, ulong seed = 44)
    {
        static double Delta(double w, double l, double t) => w + l + t == 0 ? double.NaN : (w - l) / (w + l + t);
        double W = 0, L = 0, T = 0;
        foreach (var s in sessions) { W += s.Wins; L += s.Losses; T += s.Ties; }
        double est = Delta(W, L, T);
        if (sessions.Count == 0) return new Interval(double.NaN, double.NaN, double.NaN);
        var rng = new SplitMix64(seed);
        var samples = new double[iterations];
        for (int it = 0; it < iterations; it++)
        {
            double w = 0, l = 0, t = 0;
            for (int i = 0; i < sessions.Count; i++)
            {
                var s = sessions[rng.NextInt(sessions.Count)];
                w += s.Wins; l += s.Losses; t += s.Ties;
            }
            samples[it] = Delta(w, l, t);
        }
        return Percentiles(est, samples, confidence);
    }

    /// <summary>
    /// Exact two-sided sign test: under H0 each non-tied impression is a fair coin, so
    /// p = min(1, 2·P[X ≤ min(w, l)]) with X ~ Binomial(w + l, ½). Ties are excluded, as in the
    /// standard sign test. Computed in log space so it is exact-to-double for any n.
    /// </summary>
    public static double SignTestTwoSided(int wins, int losses)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(wins);
        ArgumentOutOfRangeException.ThrowIfNegative(losses);
        int n = wins + losses;
        if (n == 0) return 1.0;
        int m = Math.Min(wins, losses);
        double ln2n = n * Math.Log(2);
        // log C(n, i) built incrementally; log-sum-exp of the lower tail.
        double logC = 0, maxLog = double.NegativeInfinity;
        var terms = new double[m + 1];
        for (int i = 0; i <= m; i++)
        {
            if (i > 0) logC += Math.Log(n - i + 1) - Math.Log(i);
            terms[i] = logC - ln2n;
            maxLog = Math.Max(maxLog, terms[i]);
        }
        double sum = 0;
        foreach (var t in terms) sum += Math.Exp(t - maxLog);
        double tail = Math.Exp(maxLog + Math.Log(sum));
        return Math.Min(1.0, 2 * tail);
    }

    /// <summary>
    /// Sample-ratio-mismatch check: chi-square goodness of fit (df = 1) of observed control /
    /// treatment session counts against the designed 50/50 split. A tiny p-value (say &lt; 0.001)
    /// means assignment or logging is broken and the experiment's other numbers are not trustworthy.
    /// </summary>
    public static (double ChiSquare, double PValue) SampleRatioMismatch(long control, long treatment, double expectedControlShare = 0.5)
    {
        long n = control + treatment;
        if (n == 0) return (0, 1);
        double ec = n * expectedControlShare, et = n - ec;
        double chi = (control - ec) * (control - ec) / ec + (treatment - et) * (treatment - et) / et;
        // df = 1: P[χ² > x] = erfc(√(x/2)).
        return (chi, Erfc(Math.Sqrt(chi / 2)));
    }

    /// <summary>Complementary error function (Numerical Recipes erfcc, |relative error| &lt; 1.2e-7).</summary>
    public static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double ans = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                     t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                     t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? ans : 2.0 - ans;
    }

    private static double Ratio(IReadOnlyList<(double Num, double Den)> clusters)
    {
        double num = 0, den = 0;
        foreach (var c in clusters) { num += c.Num; den += c.Den; }
        return den == 0 ? double.NaN : num / den;
    }

    private static double Resample(IReadOnlyList<(double Num, double Den)> clusters, ref SplitMix64 rng)
    {
        double num = 0, den = 0;
        for (int i = 0; i < clusters.Count; i++)
        {
            var c = clusters[rng.NextInt(clusters.Count)];
            num += c.Num; den += c.Den;
        }
        return den == 0 ? double.NaN : num / den;
    }

    private static Interval Percentiles(double estimate, double[] samples, double confidence)
    {
        var valid = samples.Where(s => !double.IsNaN(s)).ToArray();
        if (valid.Length == 0) return new Interval(estimate, double.NaN, double.NaN);
        Array.Sort(valid);
        double a = (1 - confidence) / 2;
        return new Interval(estimate, Quantile(valid, a), Quantile(valid, 1 - a));
    }

    /// <summary>Linear-interpolated quantile of a sorted array.</summary>
    public static double Quantile(double[] sorted, double q)
    {
        if (sorted.Length == 0) return double.NaN;
        double pos = q * (sorted.Length - 1);
        int lo = (int)Math.Floor(pos), hi = (int)Math.Ceiling(pos);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }
}
