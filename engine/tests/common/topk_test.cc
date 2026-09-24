#include "hs/common/topk.hpp"

#include <gtest/gtest.h>

#include <random>

namespace hs {

TEST(TopK, KeepsBestAndBreaksTiesByLowerDoc) {
  TopK t(3);
  t.push(5, 1.0f);
  t.push(2, 1.0f);
  t.push(9, 3.0f);
  t.push(1, 1.0f);  // ties with 5 and 2, lower id wins
  t.push(7, 0.5f);
  auto out = t.take_sorted();
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].doc, 9u);
  EXPECT_EQ(out[1].doc, 1u);
  EXPECT_EQ(out[2].doc, 2u);
}

TEST(TopK, MatchesFullSort) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> score(0, 50);  // many ties on purpose
  std::vector<ScoredDoc> all;
  TopK t(10);
  for (uint32_t d = 0; d < 1000; ++d) {
    float s = float(score(rng));
    all.push_back({d, s});
    t.push(d, s);
  }
  std::sort(all.begin(), all.end(), ranks_before);
  auto out = t.take_sorted();
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(out[i].doc, all[i].doc);
    EXPECT_EQ(out[i].score, all[i].score);
  }
}

}  // namespace hs
