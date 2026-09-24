using System.Diagnostics;

namespace HybridSearch.Broker.Shards;

/// <summary>
/// Rolling, lock-free latency histogram.
///
/// Buckets are log-linear over microseconds (HdrHistogram-style: exact below 64 µs, then 32
/// sub-buckets per power of two, ≤ ~3% relative error, up to ~67 s). Time is split into
/// <c>slots</c> sub-windows; a recorder increments one counter with Interlocked, and the first
/// recorder to see a new sub-window claims it with a CAS on its epoch and clears it. A handful of
/// samples racing a rotation can be lost, which is fine for a p95 estimate and far cheaper than a
/// lock on the hot path. Quantile reads merge the live sub-windows; the result is cached briefly
/// because every fan-out asks for it.
/// </summary>
public sealed class LatencyTracker
{
    private const int LinearBuckets = 64;          // 0..63 µs exact
    private const int SubBucketBits = 5;           // 32 sub-buckets per octave
    private const int MaxExponent = 26;            // 2^26 µs ≈ 67 s
    public const int BucketCount = LinearBuckets + (MaxExponent - 6) * (1 << SubBucketBits);

    private sealed class Slot
    {
        public long Epoch = -1;
        public readonly long[] Counts = new long[BucketCount];
    }

    private readonly Slot[] _slots;
    private readonly long _slotTicks;
    private readonly Func<long> _clock;
    private readonly long _cacheTicks;
    private sealed record CacheEntry(long AtTicks, double Q, double Value, long Count);
    private CacheEntry? _cache; // swapped atomically (reference write), never torn

    /// <param name="window">Rolling window length.</param>
    /// <param name="slots">Number of sub-windows (granularity of expiry).</param>
    /// <param name="clock">Stopwatch-tick clock, injectable for tests.</param>
    public LatencyTracker(TimeSpan window, int slots = 6, Func<long>? clock = null, TimeSpan? cacheFor = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(slots, 2);
        _slots = new Slot[slots];
        for (int i = 0; i < slots; i++) _slots[i] = new Slot();
        _slotTicks = Math.Max(1, (long)(window.TotalSeconds * Stopwatch.Frequency / slots));
        _clock = clock ?? Stopwatch.GetTimestamp;
        _cacheTicks = (long)((cacheFor ?? TimeSpan.FromMilliseconds(100)).TotalSeconds * Stopwatch.Frequency);
    }

    public void Record(TimeSpan latency) => RecordMicros((long)latency.TotalMicroseconds);

    public void RecordMicros(long micros)
    {
        long epoch = _clock() / _slotTicks;
        var slot = _slots[(int)(epoch % _slots.Length)];
        long seen = Volatile.Read(ref slot.Epoch);
        if (seen != epoch && Interlocked.CompareExchange(ref slot.Epoch, epoch, seen) == seen)
            Array.Clear(slot.Counts);
        Interlocked.Increment(ref slot.Counts[BucketOf(micros)]);
    }

    /// <summary>Samples currently inside the window.</summary>
    public long Count
    {
        get
        {
            long total = 0;
            ForEachLiveSlot(s => { foreach (var c in s.Counts) total += c; });
            return total;
        }
    }

    /// <summary>Latency at quantile q in (0,1], or null when the window holds fewer than <paramref name="minSamples"/> samples.</summary>
    public TimeSpan? Quantile(double q, long minSamples = 1)
    {
        long now = _clock();
        var cache = Volatile.Read(ref _cache);
        if (cache is not null && cache.Q == q && now - cache.AtTicks < _cacheTicks)
            return cache.Count >= minSamples && cache.Count > 0 ? TimeSpan.FromMicroseconds(cache.Value) : null;

        var merged = new long[BucketCount];
        long total = 0;
        ForEachLiveSlot(s =>
        {
            for (int i = 0; i < BucketCount; i++)
            {
                long c = Volatile.Read(ref s.Counts[i]);
                merged[i] += c; total += c;
            }
        });
        double value = double.NaN;
        if (total > 0)
        {
            long rank = Math.Max(1, (long)Math.Ceiling(q * total));
            long cum = 0;
            for (int i = 0; i < BucketCount; i++)
            {
                cum += merged[i];
                if (cum >= rank) { value = BucketUpperMicros(i); break; }
            }
        }
        Volatile.Write(ref _cache, new CacheEntry(now, q, value, total));
        return total >= minSamples && total > 0 ? TimeSpan.FromMicroseconds(value) : null;
    }

    private void ForEachLiveSlot(Action<Slot> f)
    {
        long current = _clock() / _slotTicks;
        foreach (var s in _slots)
        {
            long e = Volatile.Read(ref s.Epoch);
            if (e >= 0 && current - e < _slots.Length) f(s);
        }
    }

    internal static int BucketOf(long micros)
    {
        if (micros < 0) micros = 0;
        if (micros < LinearBuckets) return (int)micros;
        int exp = 63 - (int)long.LeadingZeroCount(micros);                // floor(log2), ≥ 6
        if (exp >= MaxExponent) return BucketCount - 1;
        int sub = (int)((micros >> (exp - SubBucketBits)) & ((1 << SubBucketBits) - 1));
        return LinearBuckets + (exp - 6) * (1 << SubBucketBits) + sub;
    }

    /// <summary>Upper edge of a bucket (conservative: quantiles never under-report).</summary>
    internal static double BucketUpperMicros(int bucket)
    {
        if (bucket < LinearBuckets) return bucket;
        int rel = bucket - LinearBuckets;
        int exp = rel / (1 << SubBucketBits) + 6;
        int sub = rel % (1 << SubBucketBits);
        double width = Math.Pow(2, exp - SubBucketBits);
        return Math.Pow(2, exp) + (sub + 1) * width;
    }
}

/// <summary>
/// Token bucket bounding hedge load. Every primary request deposits <c>ratio</c> tokens (capped
/// at <c>burst</c>); a hedge spends one whole token. The bucket starts empty, so at every
/// instant hedges ≤ ratio × primaries — the extra load can never exceed the configured
/// fraction, even during an incident when every request is slow (exactly when unbudgeted hedging
/// would double the load on an already struggling shard).
/// </summary>
public sealed class HedgeBudget(double ratio, double burst)
{
    private const long Scale = 1_000_000;
    private readonly long _perPrimary = (long)Math.Round(ratio * Scale);
    private readonly long _cap = (long)Math.Round(burst * Scale);
    private long _tokens;

    public double Tokens => Volatile.Read(ref _tokens) / (double)Scale;

    public void OnPrimary()
    {
        while (true)
        {
            long cur = Volatile.Read(ref _tokens);
            long next = Math.Min(_cap, cur + _perPrimary);
            if (next == cur || Interlocked.CompareExchange(ref _tokens, next, cur) == cur) return;
        }
    }

    public bool TryAcquire()
    {
        while (true)
        {
            long cur = Volatile.Read(ref _tokens);
            if (cur < Scale) return false;
            if (Interlocked.CompareExchange(ref _tokens, cur - Scale, cur) == cur) return true;
        }
    }
}
