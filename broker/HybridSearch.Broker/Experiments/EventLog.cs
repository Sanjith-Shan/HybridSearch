using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using HybridSearch.Broker.Api;
using HybridSearch.Broker.Observability;
using Microsoft.Extensions.Options;

namespace HybridSearch.Broker.Experiments;

/// <summary>Incoming UI event (POST /api/events). Unknown JSON fields are ignored.</summary>
public sealed record UiEvent
{
    public string? Type { get; init; }
    public string? SessionId { get; init; }
    public string? RequestId { get; init; }
    public string? Query { get; init; }
    public long? DocId { get; init; }
    public int? Rank { get; init; }
    public long? DwellMs { get; init; }
    public string? ExperimentId { get; init; }
    public string? Variant { get; init; }
    public string? Team { get; init; }
    public string? ClientTs { get; init; }
    /// <summary>Optional: true when a click simulator (not a person) produced the event. Results report it.</summary>
    public bool? Simulated { get; init; }
    /// <summary>Optional: the click model that produced a simulated event (e.g. "pbm", "cascade", "dbn").</summary>
    public string? ClickModel { get; init; }
}

public sealed record EventBatch
{
    public List<UiEvent>? Events { get; init; }
}

public sealed record ServedResult(ulong DocId, int Rank, string? Team);

/// <summary>
/// One line of the event log. <c>Kind</c> is "event" (a validated UI event) or "served" (the
/// broker's own record of a result page it returned, so interleaving credit never depends on
/// the client echoing teams correctly). Deliberately no IP, user agent, or any header.
/// </summary>
public sealed record LogRecord
{
    public required string Kind { get; init; }
    public required DateTimeOffset ServerTs { get; init; }
    public string? Type { get; init; }
    public string? SessionId { get; init; }
    public string? RequestId { get; init; }
    public string? Query { get; init; }
    public ulong? DocId { get; init; }
    public int? Rank { get; init; }
    public long? DwellMs { get; init; }
    public string? ExperimentId { get; init; }
    public string? Variant { get; init; }
    public string? Team { get; init; }
    public DateTimeOffset? ClientTs { get; init; }
    public bool? Interleaved { get; init; }
    public bool? Simulated { get; init; }
    public string? ClickModel { get; init; }
    public List<ServedResult>? Results { get; init; }

    public static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}

public static class EventValidator
{
    public static readonly HashSet<string> Types = ["impression", "click", "dwell", "query", "abandon"];
    public const int MaxIdLength = 128;
    public const int MaxQueryLength = 512;
    public const long MaxDwellMs = 24L * 3600 * 1000;

    /// <summary>Validates one event and, if valid, returns the record to log (server-stamped).</summary>
    public static LogRecord? Validate(UiEvent e, DateTimeOffset now, out string? error)
    {
        error = Check(e, out var clientTs);
        if (error is not null) return null;
        return new LogRecord
        {
            Kind = "event",
            ServerTs = now,
            Type = e.Type,
            SessionId = e.SessionId,
            RequestId = e.RequestId,
            Query = e.Query,
            DocId = e.DocId is { } d ? (ulong)d : null,
            Rank = e.Rank,
            DwellMs = e.Type == "dwell" ? e.DwellMs : null,
            ExperimentId = e.ExperimentId,
            Variant = e.Variant,
            Team = e.Team,
            ClientTs = clientTs,
            Simulated = e.Simulated,
            ClickModel = e.ClickModel,
        };
    }

    private static string? Check(UiEvent e, out DateTimeOffset? clientTs)
    {
        clientTs = null;
        if (e.Type is null || !Types.Contains(e.Type)) return "type must be one of impression, click, dwell, query, abandon";
        if (!IsId(e.SessionId)) return "sessionId is required: 1-128 characters of [A-Za-z0-9_-]";
        if (e.RequestId is not null && !IsId(e.RequestId)) return "requestId must be 1-128 characters of [A-Za-z0-9_-]";
        if (e.Type is "click" or "dwell" or "impression")
        {
            if (e.RequestId is null) return $"requestId is required for {e.Type}";
            if (e.DocId is null or < 0) return $"docId is required for {e.Type} and must be ≥ 0";
            if (e.Rank is null or < 1 or > 1000) return $"rank is required for {e.Type} and must be in [1, 1000]";
        }
        else if (e.Rank is < 1 or > 1000) return "rank must be in [1, 1000]";
        if (e.Type == "dwell" && e.DwellMs is null or < 0 or > MaxDwellMs) return "dwellMs is required for dwell and must be in [0, 86400000]";
        if (e.Query is { Length: > MaxQueryLength }) return "query must be at most 512 characters";
        if (e.ExperimentId is { Length: > MaxIdLength } || e.Variant is { Length: > MaxIdLength }) return "experimentId / variant too long";
        if (e.Team is not null and not ("A" or "B")) return "team must be \"A\", \"B\" or null";
        if (e.ClickModel is { Length: > 32 }) return "clickModel must be at most 32 characters";
        if (e.ClientTs is not null)
        {
            if (!DateTimeOffset.TryParse(e.ClientTs, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var ts))
                return "clientTs must be an ISO-8601 timestamp";
            clientTs = ts;
        }
        return null;
    }

    private static bool IsId(string? s) => !string.IsNullOrEmpty(s) && s.Length <= MaxIdLength && s.All(SearchQueryParser.IsSessionChar);
}

/// <summary>
/// Append-only JSON Lines event log, one file per UTC hour: <c>events-yyyyMMdd-HH.jsonl</c>.
/// Producers call <see cref="TryEnqueue"/>, which never blocks: the channel is bounded and a
/// full queue drops the record and counts it (hs_events_dropped_total{reason="queue_full"}).
/// A single background reader owns the file handle and flushes after each drained batch.
/// </summary>
public sealed class EventLogWriter : BackgroundService
{
    private readonly Channel<LogRecord> _channel;
    private readonly BrokerMetrics _metrics;
    private readonly ILogger<EventLogWriter> _logger;
    private long _written;
    private long _enqueued;
    private long _processed;

    public EventLogWriter(IOptions<BrokerOptions> options, IHostEnvironment env, BrokerMetrics metrics, ILogger<EventLogWriter> logger)
    {
        Directory = PathResolver.UnderData(options.Value, env.ContentRootPath, options.Value.Events.Directory);
        _channel = Channel.CreateBounded<LogRecord>(new BoundedChannelOptions(Math.Max(1, options.Value.Events.QueueCapacity))
        {
            SingleReader = true,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait, // TryWrite then returns false instead of silently dropping
        });
        _metrics = metrics;
        _logger = logger;
    }

    public string Directory { get; }
    /// <summary>"starting", "ok", or "unwritable: …" (reported in /readyz).</summary>
    public string Status { get; private set; } = "starting";
    public long Written => Interlocked.Read(ref _written);

    public bool TryEnqueue(LogRecord record)
    {
        if (_channel.Writer.TryWrite(record)) { Interlocked.Increment(ref _enqueued); return true; }
        _metrics.EventsDropped.WithLabels("queue_full").Inc();
        return false;
    }

    public static string FileNameFor(DateTimeOffset ts) => $"events-{ts.UtcDateTime:yyyyMMdd-HH}.jsonl";

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // An unwritable event log must never take search down (the default BackgroundService
            // behaviour would stop the host). Degrade: drain and count every record as dropped.
            Status = $"unwritable: {ex.Message}";
            _logger.LogError(ex, "event log directory {Dir} is not writable; events will be dropped and counted", Directory);
            await foreach (var _ in _channel.Reader.ReadAllAsync(stoppingToken).ConfigureAwait(false))
            {
                _metrics.EventsDropped.WithLabels("write_error").Inc();
                Interlocked.Increment(ref _processed);
            }
            return;
        }
        Status = "ok";
        StreamWriter? writer = null;
        string? currentFile = null;
        var reader = _channel.Reader;
        long pendingInBatch = 0;
        try
        {
            while (await reader.WaitToReadAsync(stoppingToken).ConfigureAwait(false))
            {
                while (reader.TryRead(out var rec))
                {
                    var name = FileNameFor(rec.ServerTs);
                    if (name != currentFile)
                    {
                        if (writer is not null) await writer.DisposeAsync().ConfigureAwait(false);
                        currentFile = name;
                        writer = new StreamWriter(new FileStream(Path.Combine(Directory, name), FileMode.Append, FileAccess.Write, FileShare.Read),
                            new UTF8Encoding(false));
                    }
                    try
                    {
                        await writer!.WriteLineAsync(JsonSerializer.Serialize(rec, LogRecord.Json)).ConfigureAwait(false);
                        Interlocked.Increment(ref _written);
                        _metrics.EventsWritten.Inc();
                    }
                    catch (IOException ex)
                    {
                        _metrics.EventsDropped.WithLabels("write_error").Inc();
                        _logger.LogError(ex, "event log write failed");
                    }
                    pendingInBatch++;
                }
                if (writer is not null) await writer.FlushAsync(stoppingToken).ConfigureAwait(false);
                Interlocked.Add(ref _processed, pendingInBatch);
                pendingInBatch = 0;
            }
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
        }
        finally
        {
            // Drain what is already queued so a clean shutdown loses nothing.
            while (reader.TryRead(out var rec) && writer is not null)
                await writer.WriteLineAsync(JsonSerializer.Serialize(rec, LogRecord.Json)).ConfigureAwait(false);
            if (writer is not null) await writer.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>Waits until everything enqueued before the call is written and flushed (or the timeout passes).</summary>
    public async Task<bool> FlushAsync(TimeSpan timeout, CancellationToken ct)
    {
        long target = Interlocked.Read(ref _enqueued);
        var until = DateTime.UtcNow + timeout;
        while (Interlocked.Read(ref _processed) < target)
        {
            if (DateTime.UtcNow >= until) return false;
            await Task.Delay(2, ct).ConfigureAwait(false);
        }
        return true;
    }
}
