#include "hs/lexical/snippet.hpp"

#include <algorithm>
#include <tuple>

#include "hs/lexical/analyzer.hpp"

namespace hs::lexical {

namespace {

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
bool is_cont(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

}  // namespace

Snippet select_snippet(const Analyzer& analyzer, std::string_view text, const std::vector<std::string>& query_terms,
                       uint32_t max_chars) {
  std::vector<AnalyzedToken> toks;
  analyzer.analyze_with_offsets(text, toks);
  struct Hit {
    uint32_t start, end, term;
  };
  std::vector<Hit> hits;
  for (const auto& t : toks) {
    auto it = std::find(query_terms.begin(), query_terms.end(), t.term);
    if (it != query_terms.end()) hits.push_back({t.start, t.end, uint32_t(it - query_terms.begin())});
  }

  Snippet sn;
  const uint32_t len = uint32_t(text.size());
  uint32_t left = 0, right = len;
  if (max_chars != 0 && len > max_chars) {
    sn.is_snippet = true;
    uint32_t s = 0, e = 0;
    if (!hits.empty()) {
      // Best window starting at a hit: most distinct terms, then most hits, then tightest span.
      std::tuple<int, int, int> best{-1, -1, 0};
      size_t bi = 0, bj = 1;
      std::vector<uint32_t> seen(query_terms.size(), 0);
      for (size_t i = 0; i < hits.size(); ++i) {
        std::fill(seen.begin(), seen.end(), 0);
        int distinct = 0;
        size_t j = i;
        while (j < hits.size() && hits[j].end - hits[i].start <= max_chars) {
          if (!seen[hits[j].term]++) ++distinct;
          ++j;
        }
        if (j == i) j = i + 1;  // a single hit longer than max_chars
        std::tuple<int, int, int> score{distinct, int(j - i), -int(hits[j - 1].end - hits[i].start)};
        if (score > best) {
          best = score;
          bi = i;
          bj = j;
        }
      }
      s = hits[bi].start;
      e = std::min(len, std::max(hits[bj - 1].end, s));
      if (e - s > max_chars) e = s + max_chars;
    }
    // Widen symmetrically to max_chars, then pull edges in to whitespace.
    uint32_t slack = max_chars - (e - s);
    uint32_t grow_left = std::min(s, slack / 2);
    left = s - grow_left;
    right = std::min(len, e + (slack - grow_left));
    if (right - left < max_chars) left = right > max_chars ? right - max_chars : 0;
    if (left > 0) {
      uint32_t p = left;
      while (p < s && !is_space(text[p - 1])) ++p;
      if (p < s || (p == s && (s == 0 || is_space(text[s - 1])))) left = p;
      while (left < len && is_cont(text[left])) ++left;
    }
    if (right < len) {
      uint32_t p = right;
      while (p > e && !is_space(text[p])) --p;
      if (p > e || (p == e && (e == len || is_space(text[e])))) right = p;
      while (right > left && right < len && is_cont(text[right])) --right;
    }
    while (left < right && is_space(text[left])) ++left;
    while (right > left && is_space(text[right - 1])) --right;
  }
  sn.window_start = left;
  sn.window.assign(text.substr(left, right - left));
  for (const Hit& h : hits)
    if (h.start >= left && h.end <= right) sn.highlights.push_back({h.start - left, h.end - left});
  return sn;
}

Snippet select_snippet(std::string_view text, const std::vector<std::string>& query_terms, uint32_t max_chars) {
  static const Analyzer analyzer;
  return select_snippet(analyzer, text, query_terms, max_chars);
}

}  // namespace hs::lexical
