#pragma once
// Snippet selection: the best-matching window of a passage, not its first N characters.
//
// The passage is analyzed with the index analyzer (keeping byte offsets); a token "hits" if
// its analyzed form is one of the query terms. Every window of at most max_chars bytes that
// starts at a hit is scored by (distinct query terms covered, total hits, -span), best
// first, earliest on ties. The window is then widened symmetrically to max_chars and its
// edges moved inward to whitespace so no word (or UTF-8 sequence) is cut.
// Highlights are byte offsets into the returned window (proto Highlight is UTF-8 bytes).

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hs::lexical {

class Analyzer;

struct HighlightSpan {
  uint32_t start = 0;
  uint32_t end = 0;
};

struct Snippet {
  std::string window;
  std::vector<HighlightSpan> highlights;
  bool is_snippet = false;  // false when the whole passage fits (window == text)
  uint32_t window_start = 0; // byte offset of the window in the passage
};

// max_chars counts bytes of UTF-8 (0 = whole passage, highlights only).
Snippet select_snippet(const Analyzer& analyzer, std::string_view text, const std::vector<std::string>& query_terms,
                       uint32_t max_chars);
// ARCHITECTURE.md signature: uses the default (Lucene-parity) analyzer.
Snippet select_snippet(std::string_view text, const std::vector<std::string>& query_terms, uint32_t max_chars);

}  // namespace hs::lexical
