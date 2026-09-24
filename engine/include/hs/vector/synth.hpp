#pragma once
// Seeded synthetic data for tests and development before real embeddings exist.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace hs::vector {

inline void normalize_rows(std::vector<float>& v, uint32_t dim) {
  for (size_t i = 0; i + dim <= v.size(); i += dim) {
    double s = 0;
    for (uint32_t d = 0; d < dim; ++d) s += double(v[i + d]) * v[i + d];
    float inv = s > 0 ? float(1.0 / std::sqrt(s)) : 0.f;
    for (uint32_t d = 0; d < dim; ++d) v[i + d] *= inv;
  }
}

// Uniform on the unit sphere.
inline std::vector<float> random_unit_vectors(uint32_t n, uint32_t dim, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> v(size_t(n) * dim);
  for (auto& x : v) x = g(rng);
  normalize_rows(v, dim);
  return v;
}

// Gaussian blobs around `clusters` random centres (spread sigma), then normalised.
// Closer to real embeddings than uniform data: dense neighbourhoods and gaps.
inline std::vector<float> clustered_vectors(uint32_t n, uint32_t dim, uint32_t clusters, float sigma,
                                            uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> centres(size_t(clusters) * dim);
  for (auto& x : centres) x = g(rng);
  normalize_rows(centres, dim);
  std::vector<float> v(size_t(n) * dim);
  std::uniform_int_distribution<uint32_t> pick(0, clusters - 1);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t c = pick(rng);
    for (uint32_t d = 0; d < dim; ++d) v[size_t(i) * dim + d] = centres[size_t(c) * dim + d] + sigma * g(rng);
  }
  normalize_rows(v, dim);
  return v;
}

}  // namespace hs::vector
