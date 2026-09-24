// Analyzer parity with Lucene. The golden files were produced by Lucene 10.5.0 itself
// (tools/lucene_ref/run.sh analyze), running Anserini's DefaultEnglishAnalyzer chain.

#include <gtest/gtest.h>

#include "hs/lexical/analyzer.hpp"
#include "hs/lexical/bm25.hpp"
#include "test_util.hpp"

namespace hs::lexical {
namespace {

using testing::data_dir;
using testing::read_lines;
using testing::split;

std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s.push_back('\x01');
    s += v[i];
  }
  return s;
}

void check_golden(const std::string& input, const std::string& golden, bool has_raw) {
  Analyzer an;
  auto in = read_lines(input);
  auto gold = read_lines(golden);
  ASSERT_EQ(in.size(), gold.size());
  ASSERT_GT(in.size(), 0u);
  size_t tokens = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    size_t tab = in[i].find('\t');
    std::string text = in[i].substr(tab + 1);
    auto g = split(gold[i], '\t');
    ASSERT_GE(g.size(), 2u);
    EXPECT_EQ(join(an.analyze(text)), g[1]) << "line " << i << ": " << text.substr(0, 200);
    if (has_raw) EXPECT_EQ(join(an.tokenize(text)), g[2]) << "tokenizer, line " << i;
    tokens += split(g[1], '\x01').size();
  }
  EXPECT_GT(tokens, 0u);
}

TEST(Analyzer, MatchesLuceneOnTrickyStrings) {
  check_golden(data_dir() + "/tricky.tsv", data_dir() + "/tricky.lucene.tsv", true);
}

TEST(Analyzer, MatchesLuceneOnTestCorpus) {
  check_golden(data_dir() + "/corpus.tsv", data_dir() + "/corpus.lucene_analyzed.tsv", false);
}

TEST(Analyzer, PorterExamplesFromThePaper) {
  const std::vector<std::pair<const char*, const char*>> cases = {
      {"caresses", "caress"}, {"ponies", "poni"},     {"ties", "ti"},          {"caress", "caress"},
      {"cats", "cat"},        {"feed", "feed"},       {"agreed", "agre"},      {"plastered", "plaster"},
      {"bled", "bled"},       {"motoring", "motor"},  {"sing", "sing"},        {"conflated", "conflat"},
      {"troubled", "troubl"}, {"sized", "size"},      {"hopping", "hop"},      {"tanned", "tan"},
      {"falling", "fall"},    {"hissing", "hiss"},    {"fizzed", "fizz"},      {"failing", "fail"},
      {"filing", "file"},     {"happy", "happi"},     {"sky", "sky"},          {"relational", "relat"},
      {"conditional", "condit"}, {"generalization", "gener"}, {"hopefulness", "hope"}, {"adoption", "adopt"},
  };
  for (auto [in, out] : cases) EXPECT_EQ(Analyzer::porter_stem(in), out) << in;
}

TEST(Analyzer, StopWordsAndPossessives) {
  Analyzer an;
  EXPECT_TRUE(an.analyze("the and of to a an is are was will with").empty());
  EXPECT_EQ(an.analyze("John's"), std::vector<std::string>{"john"});
  EXPECT_EQ(an.analyze("JOHN'S"), std::vector<std::string>{"john"});
  EXPECT_EQ(an.analyze("dogs'"), std::vector<std::string>{"dog"});
}

TEST(Analyzer, OffsetsPointAtSourceTokens) {
  Analyzer an;
  std::string text = "The Cats’s   runnings, café!";
  std::vector<AnalyzedToken> toks;
  an.analyze_with_offsets(text, toks);
  ASSERT_EQ(toks.size(), 3u);
  EXPECT_EQ(text.substr(toks[0].start, toks[0].end - toks[0].start), "Cats’s");
  EXPECT_EQ(toks[0].term, "cat");
  EXPECT_EQ(text.substr(toks[1].start, toks[1].end - toks[1].start), "runnings");
  EXPECT_EQ(toks[2].term, "café");
}

TEST(SmallFloat, MatchesLucene) {
  auto lines = read_lines(data_dir() + "/lucene_smallfloat.tsv");
  ASSERT_GT(lines.size(), 256u);
  size_t checked = 0;
  for (const auto& l : lines) {
    auto f = split(l, '\t');
    uint64_t a = std::stoull(f[1]), b = std::stoull(f[2]);
    if (f[0] == "b") EXPECT_EQ(smallfloat::byte4_to_int(uint8_t(a)), b) << "byte4ToInt(" << a << ")";
    else EXPECT_EQ(smallfloat::int_to_byte4(uint32_t(a)), b) << "intToByte4(" << a << ")";
    ++checked;
  }
  // Monotone and exact below the first lossy value.
  for (uint32_t i = 0; i < 40; ++i) EXPECT_EQ(smallfloat::byte4_to_int(smallfloat::int_to_byte4(i)), i);
  for (uint32_t i = 1; i < 100000; ++i) EXPECT_LE(smallfloat::int_to_byte4(i - 1), smallfloat::int_to_byte4(i));
  EXPECT_GT(checked, 256u);
}

}  // namespace
}  // namespace hs::lexical
