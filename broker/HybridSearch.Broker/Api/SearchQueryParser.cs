using System.Globalization;

namespace HybridSearch.Broker.Api;

/// <summary>Validates /api/search query parameters. Errors map to a 400 ValidationProblem.</summary>
public static class SearchQueryParser
{
    public const int MaxSessionIdLength = 128;

    public static SearchQuery? Parse(IQueryCollection query, SearchOptions o, out Dictionary<string, string[]> errors)
    {
        errors = [];
        string? q = query["q"];
        if (string.IsNullOrWhiteSpace(q)) errors["q"] = ["q is required"];
        else if (q.Length > o.MaxQueryChars) errors["q"] = [$"q must be at most {o.MaxQueryChars} characters"];

        int k = ParseInt(query, "k", o.DefaultK, 1, o.MaxK, errors);
        int deadline = ParseInt(query, "deadlineMs", o.DefaultDeadlineMs, o.MinDeadlineMs, o.MaxDeadlineMs, errors);

        if (!SearchModes.TryParse(query["mode"], out var mode))
            errors["mode"] = ["mode must be one of hybrid, lexical, dense"];

        bool rerank = ParseBool(query, "rerank", true, errors);
        bool explain = ParseBool(query, "explain", false, errors);

        string? session = query["sessionId"];
        if (session is not null && (session.Length > MaxSessionIdLength || !session.All(IsSessionChar)))
            errors["sessionId"] = [$"sessionId must be 1-{MaxSessionIdLength} characters of [A-Za-z0-9_-]"];
        if (string.IsNullOrEmpty(session)) session = null;

        if (errors.Count > 0) return null;
        return new SearchQuery(q!.Trim(), k, mode, rerank, explain, deadline, session);
    }

    public static bool IsSessionChar(char c) => char.IsAsciiLetterOrDigit(c) || c is '-' or '_';

    private static int ParseInt(IQueryCollection query, string name, int def, int min, int max, Dictionary<string, string[]> errors)
    {
        string? s = query[name];
        if (string.IsNullOrEmpty(s)) return def;
        if (!int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) || v < min || v > max)
        {
            errors[name] = [$"{name} must be an integer in [{min}, {max}]"];
            return def;
        }
        return v;
    }

    private static bool ParseBool(IQueryCollection query, string name, bool def, Dictionary<string, string[]> errors)
    {
        string? s = query[name];
        if (string.IsNullOrEmpty(s)) return def;
        if (bool.TryParse(s, out var b)) return b;
        if (s == "1") return true;
        if (s == "0") return false;
        errors[name] = [$"{name} must be true or false"];
        return def;
    }
}
