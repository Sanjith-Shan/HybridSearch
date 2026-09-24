#pragma once
// Lucene EnglishAnalyzer parity, as Anserini configures it for MS MARCO passage.
//
// Anserini's io.anserini.analysis.DefaultEnglishAnalyzer.newDefaultInstance() is
//   StandardTokenizer -> EnglishPossessiveFilter -> LowerCaseFilter
//     -> StopFilter(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET) -> PorterStemFilter
// (verified against anserini master, Lucene 10.5.0). This class reproduces that chain:
//
// * StandardTokenizer: the UAX#29 word-break grammar from Lucene's
//   StandardTokenizerImpl.jflex (%unicode 12.1), compiled to a DFA at startup from
//   UCD 12.1 property tables, with longest-match semantics and Lucene's 255 UTF-16 unit
//   token buffer (longer runs are split, not dropped).
// * EnglishPossessiveFilter: strips a trailing 's / ’s / ＇s (and S).
// * LowerCaseFilter: java.lang.Character.toLowerCase per code point (UnicodeData 15.0).
// * StopFilter: Lucene's 33 English stop words, exact match.
// * PorterStemFilter: Lucene's PorterStemmer (Porter's reference implementation,
//   release 3), run over UTF-16 code units exactly as Java does.
//
// Differential-tested against Lucene itself: see tools/lucene_ref and results/lexical/analyzer_parity.md.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hs::lexical {

struct AnalyzedToken {
  std::string term;    // analyzed form (what the index stores)
  uint32_t start = 0;  // byte offsets of the source token in the input (UTF-8)
  uint32_t end = 0;
};

class Analyzer {
 public:
  Analyzer();

  // Analyzed terms in order (stop words removed, so positions are not contiguous).
  std::vector<std::string> analyze(std::string_view text) const;
  void analyze(std::string_view text, std::vector<std::string>& out) const;

  // Same, keeping source byte offsets (used for snippets and highlights).
  void analyze_with_offsets(std::string_view text, std::vector<AnalyzedToken>& out) const;

  // StandardTokenizer output only (no filters), for differential debugging.
  std::vector<std::string> tokenize(std::string_view text) const;

  // Individual filter stages, exposed for tests.
  static bool is_stopword(std::string_view lowercased);
  static std::string porter_stem(std::string_view lowercased_utf8);
  static std::string lowercase(std::string_view utf8);
};

}  // namespace hs::lexical
