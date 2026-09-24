#pragma once
// Seeded Lloyd's k-means (squared L2). Deterministic regardless of thread count:
// assignment writes one label per point, and centroid sums are accumulated in a
// fixed number of chunks that are reduced in a fixed order.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hs::vector {

struct KMeansParams {
  uint32_t k = 256;
  uint32_t iters = 15;
  uint64_t seed = 20260923;
  unsigned threads = 0;
};

// Returns k*dim centroids. Init: k distinct points chosen by a seeded shuffle.
// An empty cluster is re-seeded with the point farthest from its centroid.
std::vector<float> kmeans(const float* data, size_t n, uint32_t dim, const KMeansParams& p,
                          std::vector<uint32_t>* labels = nullptr, double* inertia = nullptr);

uint32_t nearest_centroid(const float* x, const float* centroids, uint32_t k, uint32_t dim,
                          float* dist = nullptr);

// The m nearest centroids of x, closest first (ties to lower index).
void nearest_centroids(const float* x, const float* centroids, uint32_t k, uint32_t dim, uint32_t m,
                       uint32_t* out);

}  // namespace hs::vector
