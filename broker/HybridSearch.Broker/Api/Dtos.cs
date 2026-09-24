using System.Text.Json.Serialization;

namespace HybridSearch.Broker.Api;

// Wire shapes from docs/ARCHITECTURE.md (Broker REST API). JSON is camelCase.

public sealed record HighlightDto(int Start, int End);

public sealed record TermDto(string Term, float Score, uint Tf, uint Df);

public sealed record ScoresDto(
    float? Bm25, int? Bm25Rank,
    float? Dense, int? DenseRank,
    double? Fused, int? FusedRank,
    float? Rerank);

public sealed record ResultDto(
    ulong DocId,
    int Rank,
    string Text,
    IReadOnlyList<HighlightDto> Highlights,
    bool IsSnippet,
    string? Team,
    ScoresDto Scores,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] IReadOnlyList<TermDto>? Terms);

public sealed record DegradationDto(
    int Level,
    IReadOnlyList<string> Steps,
    IReadOnlyList<int> PartialShards,
    IReadOnlyList<int> FailedShards,
    IReadOnlyList<int> HedgedShards);

public sealed record TimingsDto(double TotalMs, double EncodeMs, double ShardsMs, double FuseMs, double RerankMs, double FetchMs);

public sealed record ExperimentDto(string Id, string Variant, bool Interleaved);

public sealed record SearchResponseDto(
    string RequestId,
    string Query,
    string? DidYouMean,
    string Mode,
    IReadOnlyList<ResultDto> Results,
    DegradationDto Degradation,
    TimingsDto Timings,
    ExperimentDto? Experiment);

public sealed record SuggestionDto(string Text, long Count);

public sealed record SuggestResponseDto(string Prefix, IReadOnlyList<SuggestionDto> Suggestions, double Micros);

public sealed record DocResponseDto(ulong DocId, string Text, IReadOnlyList<HighlightDto> Highlights);

public sealed record EventsAcceptedDto(int Accepted, int Rejected, int Dropped, IReadOnlyList<EventErrorDto> Errors);

public sealed record EventErrorDto(int Index, string Error);

public enum SearchMode { Hybrid, Lexical, Dense }

public static class SearchModes
{
    public static string Name(SearchMode m) => m switch
    {
        SearchMode.Hybrid => "hybrid",
        SearchMode.Lexical => "lexical",
        SearchMode.Dense => "dense",
        _ => "hybrid",
    };

    public static bool TryParse(string? s, out SearchMode mode)
    {
        switch (s?.Trim().ToLowerInvariant())
        {
            case null or "" or "hybrid": mode = SearchMode.Hybrid; return true;
            case "lexical": mode = SearchMode.Lexical; return true;
            case "dense": mode = SearchMode.Dense; return true;
            default: mode = SearchMode.Hybrid; return false;
        }
    }
}

/// <summary>A validated /api/search request.</summary>
public sealed record SearchQuery(
    string Q,
    int K,
    SearchMode Mode,
    bool Rerank,
    bool Explain,
    int DeadlineMs,
    string? SessionId);
