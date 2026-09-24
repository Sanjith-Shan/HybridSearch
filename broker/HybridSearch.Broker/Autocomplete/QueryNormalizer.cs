using System.Text;

namespace HybridSearch.Broker.Autocomplete;

/// <summary>
/// Normalisation shared by the query log loader and the lookup path: lowercase (invariant
/// culture) and collapse every run of whitespace to one ASCII space.
/// </summary>
public static class QueryNormalizer
{
    /// <summary>Full-query form: collapsed and trimmed at both ends.</summary>
    public static string Normalize(ReadOnlySpan<char> text) => Collapse(text, keepTrailingSpace: false);

    /// <summary>
    /// Prefix form: like <see cref="Normalize"/> but one trailing space is kept if the input ended
    /// in whitespace after some content, because "how to " and "how to" are different prefixes
    /// ("how tom…" matches the second only).
    /// </summary>
    public static string NormalizePrefix(ReadOnlySpan<char> text) => Collapse(text, keepTrailingSpace: true);

    private static string Collapse(ReadOnlySpan<char> text, bool keepTrailingSpace)
    {
        var sb = new StringBuilder(text.Length);
        bool pendingSpace = false;
        foreach (var ch in text)
        {
            if (char.IsWhiteSpace(ch))
            {
                pendingSpace = sb.Length > 0;
                continue;
            }
            if (pendingSpace) { sb.Append(' '); pendingSpace = false; }
            sb.Append(char.ToLowerInvariant(ch));
        }
        if (keepTrailingSpace && pendingSpace) sb.Append(' ');
        return sb.ToString();
    }
}
