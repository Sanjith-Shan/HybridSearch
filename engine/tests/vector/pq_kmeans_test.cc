#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <set>

#include "hs/vector/distance.hpp"
#include "hs/vector/kmeans.hpp"
#include "hs/vector/pq.hpp"
#include "hs/vector/synth.hpp"

namespace hs::vector {

TEST(KMeans, RecoversWellSeparatedClusters) {
  // Four tight blobs at the corners of a square; every point must land with its blob.
  std::vector<float> x;
  std::vector<uint32_t> truth;
  float cx[4] = {0, 10, 0, 10}, cy[4] = {0, 0, 10, 10};
  std::mt19937 rng(3);
  std::normal_distribution<float> g(0, 0.3f);
  for (uint32_t i = 0; i < 400; ++i) {
    uint32_t c = i % 4;
    x.push_back(cx[c] + g(rng));
    x.push_back(cy[c] + g(rng));
    truth.push_back(c);
  }
  KMeansParams p;
  p.k = 4;
  p.iters = 20;
  p.seed = 5;
  std::vector<uint32_t> labels;
  auto C = kmeans(x.data(), 400, 2, p, &labels);
  // Same true cluster <=> same label.
  for (uint32_t i = 0; i < 400; ++i)
    for (uint32_t j = i + 1; j < 400; j += 37) EXPECT_EQ(truth[i] == truth[j], labels[i] == labels[j]);
}

TEST(KMeans, DeterministicAcrossThreadCounts) {
  auto x = clustered_vectors(3000, 16, 20, 0.3f, 4);
  KMeansParams p;
  p.k = 32;
  p.iters = 8;
  p.threads = 1;
  auto a = kmeans(x.data(), 3000, 16, p);
  p.threads = 7;
  auto b = kmeans(x.data(), 3000, 16, p);
  EXPECT_EQ(a, b);
}

TEST(KMeans, NearestCentroidsOrdered) {
  std::vector<float> C = {0, 0, 1, 0, 5, 0, 2, 0};
  float x[2] = {1.9f, 0};
  uint32_t out[3];
  nearest_centroids(x, C.data(), 4, 2, 3, out);
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 1u);
  EXPECT_EQ(out[2], 0u);
  EXPECT_EQ(nearest_centroid(x, C.data(), 4, 2), 3u);
}

TEST(PQ, AdcEqualsDistanceToDecodedVector) {
  auto x = clustered_vectors(4000, 32, 30, 0.3f, 7);
  auto pq = ProductQuantizer::train(x.data(), 4000, 32, 8, 10, 1);
  auto q = random_unit_vectors(1, 32, 9);
  std::vector<float> table(8 * 256), dec(32);
  pq.distance_table(q.data(), table.data());
  std::vector<uint8_t> code(8);
  for (uint32_t i = 0; i < 50; ++i) {
    pq.encode(x.data() + i * 32, code.data());
    pq.decode(code.data(), dec.data());
    EXPECT_NEAR(pq.adc(table.data(), code.data()), l2sq(q.data(), dec.data(), 32), 1e-4f);
  }
}

TEST(PQ, ReconstructionErrorSmallAndShrinksWithM) {
  auto x = clustered_vectors(5000, 64, 40, 0.3f, 8);
  auto err = [&](uint32_t M) {
    auto pq = ProductQuantizer::train(x.data(), 5000, 64, M, 10, 2);
    std::vector<uint8_t> code(M);
    std::vector<float> dec(64);
    double e = 0;
    for (uint32_t i = 0; i < 500; ++i) {
      pq.encode(x.data() + i * 64, code.data());
      pq.decode(code.data(), dec.data());
      e += l2sq(x.data() + i * 64, dec.data(), 64);
    }
    return e / 500;
  };
  double e8 = err(8), e32 = err(32);
  EXPECT_LT(e32, e8);
  EXPECT_LT(e32, 0.1);  // unit vectors: squared norm 1
}

TEST(PQ, PreservesNeighbourOrderWell) {
  // Recall@10 of PQ-only ranking inside the true top-100 is a sanity check that
  // ADC distances steer search in the right direction.
  auto x = clustered_vectors(3000, 32, 30, 0.3f, 10);
  auto pq = ProductQuantizer::train(x.data(), 3000, 32, 16, 12, 3);
  std::vector<uint8_t> codes(3000 * 16);
  pq.encode_batch(x.data(), 3000, codes.data());
  auto q = clustered_vectors(1, 32, 30, 0.3f, 10);  // same centres, first point
  std::vector<float> table(16 * 256);
  pq.distance_table(q.data(), table.data());
  std::vector<std::pair<float, uint32_t>> exact, approx;
  for (uint32_t i = 0; i < 3000; ++i) {
    exact.push_back({l2sq(q.data(), x.data() + i * 32, 32), i});
    approx.push_back({pq.adc(table.data(), codes.data() + i * 16), i});
  }
  std::sort(exact.begin(), exact.end());
  std::sort(approx.begin(), approx.end());
  std::set<uint32_t> top100;
  for (int i = 0; i < 100; ++i) top100.insert(approx[i].second);
  int hit = 0;
  for (int i = 0; i < 10; ++i) hit += top100.count(exact[i].second);
  EXPECT_GE(hit, 9);
}

TEST(PQ, SaveLoadRoundTrip) {
  auto x = random_unit_vectors(1000, 16, 2);
  auto pq = ProductQuantizer::train(x.data(), 1000, 16, 4, 5, 1);
  std::string path = ::testing::TempDir() + "/hs_pq_test.bin";
  pq.save(path);
  auto re = ProductQuantizer::load(path);
  std::vector<uint8_t> a(4), b(4);
  for (uint32_t i = 0; i < 100; ++i) {
    pq.encode(x.data() + i * 16, a.data());
    re.encode(x.data() + i * 16, b.data());
    EXPECT_EQ(a, b);
  }
  std::remove(path.c_str());
}

TEST(PQ, RejectsBadShapes) {
  auto x = random_unit_vectors(1000, 10, 2);
  EXPECT_THROW(ProductQuantizer::train(x.data(), 1000, 10, 3, 5, 1), std::invalid_argument);
  EXPECT_THROW(ProductQuantizer::train(x.data(), 100, 10, 5, 5, 1), std::invalid_argument);
}

}  // namespace hs::vector
