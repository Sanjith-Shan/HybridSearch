using System.Text;
using HybridSearch.Broker.Api;
using HybridSearch.Proto.V1;

namespace HybridSearch.Broker.Search;

/// <summary>
/// Shards report highlight spans as UTF-8 byte offsets (shard.proto); the REST API promises
/// UTF-16 code-unit offsets (what JavaScript string indices are). Passing bytes straight through
/// is correct only for ASCII and silently shifts every highlight after the first "é" or "—".
/// </summary>
public static class Utf8Offsets
{
    public static List<HighlightDto> ToUtf16(string text, IEnumerable<Highlight> spans)
    {
        var map = ByteToUtf16Map(text);
        var result = new List<HighlightDto>();
        foreach (var h in spans)
        {
            int start = Map(map, h.Start), end = Map(map, h.End);
            if (end > start) result.Add(new HighlightDto(start, end));
        }
        return result;
    }

    /// <summary>
    /// map[b] = UTF-16 index of the character containing byte b; map[len] = text.Length. Offsets
    /// that land inside a multi-byte sequence round down to that character's start.
    /// </summary>
    public static int[] ByteToUtf16Map(string text)
    {
        int byteLen = Encoding.UTF8.GetByteCount(text);
        var map = new int[byteLen + 1];
        int b = 0;
        for (int i = 0; i < text.Length; i++)
        {
            int units = 1, bytes;
            char c = text[i];
            if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])) { units = 2; bytes = 4; }
            else if (c < 0x80) bytes = 1;
            else if (c < 0x800) bytes = 2;
            else bytes = 3; // BMP char, or a lone surrogate (encoded as U+FFFD, 3 bytes)
            for (int k = 0; k < bytes; k++) map[b + k] = i;
            b += bytes;
            i += units - 1;
        }
        map[byteLen] = text.Length;
        return map;
    }

    private static int Map(int[] map, uint byteOffset) => byteOffset >= map.Length ? map[^1] : map[byteOffset];
}
