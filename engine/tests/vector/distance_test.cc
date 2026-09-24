#include "hs/vector/distance.hpp"

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "hs/vector/containers.hpp"

namespace hs::vector {

TEST(Distance, SimdMatchesScalarAcrossDims) {
  std::mt19937 rng(1);
  std::normal_distribution<float> g;
  for (size_t d : {1u, 3u, 4u, 7u, 15u, 16u, 17u, 31u, 64u, 96u, 100u, 768u, 1023u}) {
    std::vector<float> a(d), b(d);
    for (auto& x : a) x = g(rng);
    for (auto& x : b) x = g(rng);
    float tol = 1e-4f * float(d);
    EXPECT_NEAR(ip(a.data(), b.data(), d), ip_scalar(a.data(), b.data(), d), tol) << "dim " << d;
    EXPECT_NEAR(l2sq(a.data(), b.data(), d), l2sq_scalar(a.data(), b.data(), d), tol) << "dim " << d;
  }
}

TEST(Distance, L2AndInnerProductAgreeOnUnitVectors) {
  std::vector<float> a = {0.6f, 0.8f, 0.f}, b = {0.f, 1.f, 0.f};
  EXPECT_NEAR(l2sq(a.data(), b.data(), 3), 2.f - 2.f * ip(a.data(), b.data(), 3), 1e-6f);
  EXPECT_FLOAT_EQ(l2sq(a.data(), a.data(), 3), 0.f);
}

TEST(Containers, VisitedSetGrowsAndDedups) {
  VisitedSet s(2);
  for (uint32_t i = 0; i < 10000; ++i) EXPECT_TRUE(s.insert(i * 7919u));
  for (uint32_t i = 0; i < 10000; ++i) EXPECT_FALSE(s.insert(i * 7919u));
  EXPECT_EQ(s.size(), 10000u);
  EXPECT_TRUE(s.contains(7919u));
  EXPECT_FALSE(s.contains(1u));
  s.clear();
  EXPECT_TRUE(s.insert(7919u));
}

TEST(Containers, CandidateListKeepsBestLAndExpandsInOrder) {
  CandidateList c(3);
  EXPECT_TRUE(c.insert(10, 5.f));
  EXPECT_TRUE(c.insert(11, 1.f));
  EXPECT_TRUE(c.insert(12, 3.f));
  EXPECT_FALSE(c.insert(13, 9.f));  // worse than the worst of a full list
  EXPECT_EQ(c.pop_closest_unexpanded().id, 11u);
  EXPECT_TRUE(c.insert(14, 0.5f));  // better than an expanded node: cursor moves back
  EXPECT_EQ(c.pop_closest_unexpanded().id, 14u);
  EXPECT_EQ(c.pop_closest_unexpanded().id, 12u);
  EXPECT_FALSE(c.has_unexpanded());  // 10 fell off the end
  ASSERT_EQ(c.size(), 3u);
  EXPECT_EQ(c[0].id, 14u);
  EXPECT_EQ(c[2].id, 12u);
}

TEST(Containers, CandidateListTiesBreakByLowerId) {
  CandidateList c(2);
  c.insert(9, 1.f);
  c.insert(3, 1.f);
  c.insert(5, 1.f);
  EXPECT_EQ(c[0].id, 3u);
  EXPECT_EQ(c[1].id, 5u);
}

}  // namespace hs::vector
