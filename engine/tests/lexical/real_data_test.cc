// Differential gate on real MS MARCO dev queries over the 1M-passage subset index, when the
// data exists locally (it is never committed). Skipped otherwise.
//   HS_LEX_REAL_INDEX    index dir     (default <repo>/data/indexes/lexical/1m)
//   HS_LEX_REAL_QUERIES  queries tsv   (default <repo>/data/raw/msmarco/queries.dev.small.tsv)
//   HS_LEX_REAL_LIMIT    max queries   (default 500; 0 = all)

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include "test_util.hpp"

namespace hs::lexical {
namespace {

using testing::bits;

std::string env_or(const char* k, const std::string& d) {
  const char* v = std::getenv(k);
  return v && *v ? v : d;
}

TEST(RealData, DevQueriesEveryAlgorithmAndCodecEqualsExhaustive) {
  const std::string repo = std::string(HS_TEST_DATA_DIR) + "/../../..";
  const std::string dir = env_or("HS_LEX_REAL_INDEX", repo + "/data/indexes/lexical/1m");
  const std::string qpath = env_or("HS_LEX_REAL_QUERIES", repo + "/data/raw/msmarco/queries.dev.small.tsv");
  if (!std::filesystem::exists(dir + "/meta.txt") || !std::filesystem::exists(qpath))
    GTEST_SKIP() << "no local index at " << dir << " or queries at " << qpath;
  auto ix = LexicalIndex::open(dir);
  auto lines = testing::read_lines(qpath);
  // Default: the first 500 queries keep ctest fast; HS_LEX_REAL_LIMIT=0 runs all 6,980.
  size_t limit = std::stoul(env_or("HS_LEX_REAL_LIMIT", "500"));
  if (limit && lines.size() > limit) lines.resize(limit);
  uint64_t compared = 0, mismatched = 0, scored_exh = 0, scored_bmw = 0;
  for (const auto& l : lines) {
    std::string text = l.substr(l.find('\t') + 1);
    auto terms = ix->analyzer().analyze(text);
    for (uint32_t k : {10u, 1000u}) {
      SearchOptions base;
      base.k = k;
      base.algorithm = Algorithm::Exhaustive;
      Deadline d0;
      auto ref = ix->search(terms, base, d0);
      if (k == 10) scored_exh += ref.docs_scored;
      for (int a = 1; a < kNumAlgorithms; ++a) {
        for (Codec c : {Codec::VByte, Codec::BP128}) {
          SearchOptions o = base;
          o.algorithm = Algorithm(a);
          o.codec = c;
          Deadline dl;
          auto r = ix->search(terms, o, dl);
          bool same = r.hits.size() == ref.hits.size();
          for (size_t i = 0; same && i < r.hits.size(); ++i)
            same = r.hits[i].doc == ref.hits[i].doc && bits(r.hits[i].score) == bits(ref.hits[i].score);
          EXPECT_TRUE(same) << algorithm_name(o.algorithm) << "/" << codec_name(c) << " k=" << k << " query: " << l;
          mismatched += !same;
          ++compared;
          if (k == 10 && o.algorithm == Algorithm::BMW && c == Codec::BP128) scored_bmw += r.docs_scored;
        }
      }
    }
  }
  std::printf("[real-data] %zu queries, %llu comparisons, %llu mismatches; docs scored/query at k=10: exhaustive %.0f, "
              "bmw %.0f\n",
              lines.size(), (unsigned long long)compared, (unsigned long long)mismatched,
              double(scored_exh) / double(lines.size()), double(scored_bmw) / double(lines.size()));
  EXPECT_EQ(mismatched, 0u);
}

}  // namespace
}  // namespace hs::lexical
