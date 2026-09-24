#include "hs/vector/vamana.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <thread>

#include "hs/vector/bruteforce.hpp"
#include "hs/vector/distance.hpp"
#include "hs/vector/synth.hpp"

namespace hs::vector {
namespace {

struct Data {
  uint32_t n, nq, dim;
  std::vector<float> base, queries;
};

Data clustered(uint32_t n, uint32_t nq, uint32_t dim, uint64_t seed) {
  Data d{n, nq, dim, {}, {}};
  auto all = clustered_vectors(n + nq, dim, 50, 0.35f, seed);
  d.base.assign(all.begin(), all.begin() + size_t(n) * dim);
  d.queries.assign(all.begin() + size_t(n) * dim, all.end());
  return d;
}

double recall10(const VamanaIndex& ix, const Data& d, uint32_t L) {
  auto exact = exact_topk(d.base.data(), d.n, d.queries.data(), d.nq, d.dim, 10);
  GroundTruth gt{d.nq, 10, {}, {}};
  for (auto& h : exact) gt.ids.push_back(h.doc);
  std::vector<std::vector<uint32_t>> approx(d.nq);
  VectorSearchOptions o;
  o.k = 10;
  o.L = L;
  for (uint32_t q = 0; q < d.nq; ++q)
    for (auto& h : ix.search(d.queries.data() + size_t(q) * d.dim, o).hits) approx[q].push_back(h.doc);
  return recall_at_k(gt, approx, 10);
}

std::string tmp_path(const std::string& name) {
  return (std::filesystem::temp_directory_path() / ("hs_vec_test_" + name)).string();
}

}  // namespace

TEST(RobustPrune, AlphaOneKeepsOnlyUnoccludedPoints) {
  // p at origin; a at (1,0); b at (2,0) is behind a (occluded); c at (0,1) is not.
  std::vector<float> x = {0, 0, 1, 0, 2, 0, 0, 1};
  auto out = robust_prune(0, {1, 2, 3}, x.data(), 2, 1.0f, 8, 0);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], 1u);  // ties (a and c both at distance 1) break by lower id
  EXPECT_EQ(out[1], 3u);
}

TEST(RobustPrune, LargerAlphaKeepsMoreLongEdges) {
  // b at (2, 0.9): with alpha=1, d(a,b)^2 = 1.81 <= d(p,b)^2 = 4.81 so a occludes b.
  // With alpha=3 (on squared distance), 5.43 > 4.81 and b survives.
  std::vector<float> x = {0, 0, 1, 0, 2, 0.9f};
  EXPECT_EQ(robust_prune(0, {1, 2}, x.data(), 2, 1.0f, 8, 0).size(), 1u);
  EXPECT_EQ(robust_prune(0, {1, 2}, x.data(), 2, 3.0f, 8, 0).size(), 2u);
}

TEST(RobustPrune, RespectsDegreeBoundAndDropsSelfAndDuplicates) {
  auto x = random_unit_vectors(200, 16, 3);
  std::vector<uint32_t> cands;
  for (uint32_t i = 0; i < 200; ++i) cands.push_back(i), cands.push_back(i);
  auto out = robust_prune(5, cands, x.data(), 16, 100.0f, 12, 0);  // huge alpha: prunes nothing
  EXPECT_EQ(out.size(), 12u);
  for (uint32_t v : out) EXPECT_NE(v, 5u);
  std::vector<uint32_t> sorted = out;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(std::unique(sorted.begin(), sorted.end()), sorted.end());
}

TEST(Vamana, HighRecallOnClusteredData) {
  auto d = clustered(5000, 200, 32, 11);
  VamanaBuildParams p;
  p.R = 32;
  p.L = 64;
  p.alpha = 1.2f;
  VamanaBuildStats st;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p, &st);
  EXPECT_LE(st.max_degree, 32u);
  EXPECT_GT(st.avg_degree, 8.0);
  double r = recall10(ix, d, 64);
  EXPECT_GE(r, 0.97) << "recall@10 at L=64";
  EXPECT_GE(recall10(ix, d, 200), r - 1e-9);  // larger L never hurts on this data
}

TEST(Vamana, HighRecallOnUniformSphere) {
  auto base = random_unit_vectors(4000, 24, 5);
  auto qs = random_unit_vectors(100, 24, 6);
  Data d{4000, 100, 24, base, qs};
  VamanaBuildParams p;
  p.R = 32;
  p.L = 80;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p);
  EXPECT_GE(recall10(ix, d, 100), 0.95);
}

TEST(Vamana, DegreesBoundedNoSelfLoopsNoDuplicates) {
  auto d = clustered(3000, 1, 16, 2);
  VamanaBuildParams p;
  p.R = 16;
  p.L = 40;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p);
  for (uint32_t i = 0; i < d.n; ++i) {
    ASSERT_LE(ix.degree(i), 16u);
    std::vector<uint32_t> row(ix.neighbors(i), ix.neighbors(i) + ix.degree(i));
    for (uint32_t v : row) {
      ASSERT_NE(v, i);
      ASSERT_LT(v, d.n);
    }
    std::sort(row.begin(), row.end());
    ASSERT_EQ(std::unique(row.begin(), row.end()), row.end());
  }
}

TEST(Vamana, EveryNodeReachableFromMedoid) {
  auto d = clustered(3000, 1, 16, 9);
  VamanaBuildParams p;
  p.R = 16;
  p.L = 40;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p);
  std::vector<char> seen(d.n, 0);
  std::vector<uint32_t> stack = {ix.medoid()};
  seen[ix.medoid()] = 1;
  size_t count = 1;
  while (!stack.empty()) {
    uint32_t u = stack.back();
    stack.pop_back();
    for (uint32_t j = 0; j < ix.degree(u); ++j) {
      uint32_t v = ix.neighbors(u)[j];
      if (!seen[v]) seen[v] = 1, ++count, stack.push_back(v);
    }
  }
  EXPECT_EQ(count, d.n);
}

// The determinism claim: the graph is a function of (data, params, seed) only.
TEST(Vamana, RebuildIsBitIdenticalAcrossRunsAndThreadCounts) {
  auto d = clustered(4000, 1, 24, 21);
  VamanaBuildParams p;
  p.R = 24;
  p.L = 48;
  p.seed = 99;
  p.threads = 1;
  uint64_t h1 = VamanaIndex::build(d.base.data(), d.n, d.dim, p).graph_hash();
  p.threads = 8;
  uint64_t h8a = VamanaIndex::build(d.base.data(), d.n, d.dim, p).graph_hash();
  uint64_t h8b = VamanaIndex::build(d.base.data(), d.n, d.dim, p).graph_hash();
  p.threads = 3;
  uint64_t h3 = VamanaIndex::build(d.base.data(), d.n, d.dim, p).graph_hash();
  EXPECT_EQ(h1, h8a);
  EXPECT_EQ(h8a, h8b);
  EXPECT_EQ(h1, h3);
  p.seed = 100;
  EXPECT_NE(VamanaIndex::build(d.base.data(), d.n, d.dim, p).graph_hash(), h1);
}

TEST(Vamana, SaveLoadRoundTrip) {
  auto d = clustered(2000, 50, 16, 4);
  VamanaBuildParams p;
  p.R = 16;
  p.L = 32;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p);
  std::string path = tmp_path("graph.bin");
  ix.save(path);
  auto re = VamanaIndex::load(path, d.base.data(), d.n, d.dim);
  EXPECT_EQ(re.graph_hash(), ix.graph_hash());
  EXPECT_EQ(re.medoid(), ix.medoid());
  VectorSearchOptions o;
  auto a = ix.search(d.queries.data(), o).hits, b = re.search(d.queries.data(), o).hits;
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].doc, b[i].doc);
  std::remove(path.c_str());
}

TEST(Vamana, HitsSortedByInnerProductAndScoresExact) {
  auto d = clustered(2000, 20, 16, 8);
  VamanaBuildParams p;
  p.R = 16;
  p.L = 32;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p);
  VectorSearchOptions o;
  o.k = 10;
  o.L = 50;
  for (uint32_t q = 0; q < d.nq; ++q) {
    const float* qv = d.queries.data() + size_t(q) * d.dim;
    auto r = ix.search(qv, o);
    ASSERT_EQ(r.hits.size(), 10u);
    EXPECT_FALSE(r.partial);
    EXPECT_GT(r.distance_computations, 0u);
    for (size_t i = 0; i < r.hits.size(); ++i) {
      EXPECT_FLOAT_EQ(r.hits[i].score, ip(qv, ix.vector(r.hits[i].doc), d.dim));
      if (i) EXPECT_TRUE(hs::ranks_before(r.hits[i - 1], r.hits[i]));
    }
  }
}

TEST(Vamana, ExpiredDeadlineReturnsPartialBestSoFar) {
  auto d = clustered(3000, 1, 16, 12);
  VamanaBuildParams p;
  p.R = 16;
  p.L = 32;
  auto ix = VamanaIndex::build(d.base.data(), d.n, d.dim, p);
  VectorSearchOptions o;
  o.L = 200;
  hs::Deadline dl = hs::Deadline::after_us(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto r = ix.search(d.queries.data(), o, dl);
  EXPECT_TRUE(r.partial);
  EXPECT_LE(r.hops, 8u);  // stopped almost immediately
  EXPECT_FALSE(r.hits.empty());
  auto full = ix.search(d.queries.data(), o);
  EXPECT_FALSE(full.partial);
  EXPECT_GT(full.hops, r.hops);
}

TEST(BruteForce, MatchesNaiveSort) {
  auto base = random_unit_vectors(500, 8, 1);
  auto qs = random_unit_vectors(7, 8, 2);
  auto got = exact_topk(base.data(), 500, qs.data(), 7, 8, 5, 3);
  for (uint32_t q = 0; q < 7; ++q) {
    std::vector<hs::ScoredDoc> all;
    for (uint32_t i = 0; i < 500; ++i) all.push_back({i, ip(qs.data() + q * 8, base.data() + i * 8, 8)});
    std::sort(all.begin(), all.end(), hs::ranks_before);
    for (uint32_t j = 0; j < 5; ++j) {
      EXPECT_EQ(got[q * 5 + j].doc, all[j].doc);
      EXPECT_EQ(got[q * 5 + j].score, all[j].score);
    }
  }
}

TEST(BruteForce, GroundTruthFileRoundTripAndRecall) {
  auto base = random_unit_vectors(300, 8, 1);
  auto qs = random_unit_vectors(4, 8, 2);
  auto hits = exact_topk(base.data(), 300, qs.data(), 4, 8, 10);
  std::string path = tmp_path("gt.bin");
  write_groundtruth(path, hits, 4, 10);
  auto gt = read_groundtruth(path);
  EXPECT_EQ(gt.nq, 4u);
  EXPECT_EQ(gt.k, 10u);
  std::vector<std::vector<uint32_t>> perfect(4), half(4);
  for (uint32_t q = 0; q < 4; ++q)
    for (uint32_t j = 0; j < 10; ++j) {
      perfect[q].push_back(gt.row(q)[j]);
      half[q].push_back(j < 5 ? gt.row(q)[j] : 100000 + j);
    }
  EXPECT_DOUBLE_EQ(recall_at_k(gt, perfect, 10), 1.0);
  EXPECT_DOUBLE_EQ(recall_at_k(gt, half, 10), 0.5);
  std::remove(path.c_str());
}

}  // namespace hs::vector
