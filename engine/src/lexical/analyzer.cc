#include "hs/lexical/analyzer.hpp"

#include <array>

#include "porter.hpp"
#include "tokenizer.hpp"
#include "unicode_props.hpp"

namespace hs::lexical {

namespace {

// EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (Lucene 10.5.0), matched exactly (ignoreCase=false).
constexpr std::array<std::string_view, 33> kStopWords = {
    "a",    "an",   "and",   "are",  "as",   "at",   "be",    "but",  "by",   "for",  "if",
    "in",   "into", "is",    "it",   "no",   "not",  "of",    "on",   "or",   "such", "that",
    "the",  "their", "then", "there", "these", "they", "this", "to",   "was",  "will", "with"};

// UTF-8 encodings of the possessive apostrophes EnglishPossessiveFilter accepts.
constexpr std::string_view kApos1 = "'";             // U+0027
constexpr std::string_view kApos2 = "\xE2\x80\x99";  // U+2019
constexpr std::string_view kApos3 = "\xEF\xBC\x87";  // U+FF07

bool ends_with(std::string_view s, std::string_view suf) {
  return s.size() >= suf.size() && s.substr(s.size() - suf.size()) == suf;
}

// EnglishPossessiveFilter: if the token's last two UTF-16 units are (apostrophe, s|S),
// drop them. The apostrophes are BMP characters and 's' is ASCII, so checking the UTF-8
// suffix is equivalent. (A token of length >= 2 is implied by the suffix match.)
std::string_view strip_possessive(std::string_view t) {
  if (t.empty()) return t;
  char last = t.back();
  if (last != 's' && last != 'S') return t;
  std::string_view head = t.substr(0, t.size() - 1);
  for (auto a : {kApos1, kApos2, kApos3})
    if (ends_with(head, a)) return head.substr(0, head.size() - a.size());
  return t;
}

void lowercase_into(std::string_view s, std::string& out) {
  out.clear();
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const auto* e = p + s.size();
  while (p < e) {
    if (*p < 0x80) {
      unsigned char c = *p++;
      out.push_back(char((c >= 'A' && c <= 'Z') ? c + 32 : c));
      continue;
    }
    uint32_t len;
    uint32_t cp = utf8_decode(p, e, &len);
    utf8_append(uni::to_lower(cp), out);
    p += len;
  }
}

void stem_in_place(std::string& t) {
  bool ascii = true;
  for (unsigned char c : t)
    if (c >= 0x80) {
      ascii = false;
      break;
    }
  if (ascii) {
    thread_local PorterStemmer<char> ps;
    t.resize(ps.stem(t.data(), t.size()));
    return;
  }
  // Non-ASCII: run over UTF-16 code units, as Java does.
  thread_local std::vector<char16_t> u16;
  u16.clear();
  const auto* p = reinterpret_cast<const unsigned char*>(t.data());
  const auto* e = p + t.size();
  while (p < e) {
    uint32_t len;
    uint32_t cp = utf8_decode(p, e, &len);
    p += len;
    if (cp >= 0x10000) {
      cp -= 0x10000;
      u16.push_back(char16_t(0xD800 + (cp >> 10)));
      u16.push_back(char16_t(0xDC00 + (cp & 0x3FF)));
    } else {
      u16.push_back(char16_t(cp));
    }
  }
  thread_local PorterStemmer<char16_t> ps16;
  size_t n = ps16.stem(u16.data(), u16.size());
  t.clear();
  for (size_t i = 0; i < n; ++i) {
    uint32_t cp = u16[i];
    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n && u16[i + 1] >= 0xDC00 && u16[i + 1] <= 0xDFFF) {
      cp = 0x10000 + ((cp - 0xD800) << 10) + (u16[i + 1] - 0xDC00);
      ++i;
    }
    // A lone surrogate cannot arise: stemming only rewrites ASCII suffixes and cuts after
    // units that were compared equal to ASCII letters.
    utf8_append(cp, t);
  }
}

template <class Emit>
void run_chain(std::string_view text, Emit&& emit) {
  thread_local std::vector<RawToken> raw;
  thread_local std::string buf;
  raw.clear();
  StandardTokenizer::instance().tokenize(text, raw);
  for (const RawToken& rt : raw) {
    std::string_view tok = strip_possessive(text.substr(rt.start, rt.end - rt.start));
    lowercase_into(tok, buf);
    if (Analyzer::is_stopword(buf)) continue;
    stem_in_place(buf);
    emit(buf, rt);
  }
}

}  // namespace

Analyzer::Analyzer() { (void)StandardTokenizer::instance(); }

bool Analyzer::is_stopword(std::string_view w) {
  if (w.size() > 5) return false;
  for (auto s : kStopWords)
    if (s == w) return true;
  return false;
}

std::string Analyzer::porter_stem(std::string_view w) {
  std::string t(w);
  stem_in_place(t);
  return t;
}

std::string Analyzer::lowercase(std::string_view s) {
  std::string out;
  lowercase_into(s, out);
  return out;
}

void Analyzer::analyze(std::string_view text, std::vector<std::string>& out) const {
  run_chain(text, [&](const std::string& t, const RawToken&) { out.push_back(t); });
}

std::vector<std::string> Analyzer::analyze(std::string_view text) const {
  std::vector<std::string> out;
  analyze(text, out);
  return out;
}

void Analyzer::analyze_with_offsets(std::string_view text, std::vector<AnalyzedToken>& out) const {
  run_chain(text, [&](const std::string& t, const RawToken& rt) { out.push_back({t, rt.start, rt.end}); });
}

std::vector<std::string> Analyzer::tokenize(std::string_view text) const {
  std::vector<RawToken> raw;
  StandardTokenizer::instance().tokenize(text, raw);
  std::vector<std::string> out;
  out.reserve(raw.size());
  for (const auto& r : raw) out.emplace_back(text.substr(r.start, r.end - r.start));
  return out;
}

}  // namespace hs::lexical
