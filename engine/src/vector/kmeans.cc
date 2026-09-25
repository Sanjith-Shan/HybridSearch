#include "hs/vector/kmeans.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "hs/vector/distance.hpp"
#include "hs/vector/parallel.hpp"

namespace hs::vector {

uint32_t nearest_centroid(const float* x, const float* c, uint32_t k, uint32_t dim, float* dist) {
  uint32_t best = 0;
  float bd = std::numeric_limits<float>::infinity();
  for (uint32_t j = 0; j < k; ++j) {
    float d = l2sq(x, c + size_t(j) * dim, dim);
    if (d < bd) bd = d, best = j;
  }
  if (dist) *dist = bd;
  return best;
}

void nearest_centroids(const float* x, const float* c, uint32_t k, uint32_t dim, uint32_t m, uint32_t* out) {
  std::vector<std::pair<float, uint32_t>> d(k);
  for (uint32_t j = 0; j < k; ++j) d[j] = {l2sq(x, c + size_t(j) * dim, dim), j};
  m = std::min(m, k);
  std::partial_sort(d.begin(), d.begin() + m, d.end());
  for (uint32_t j = 0; j < m; ++j) out[j] = d[j].second;
}

std::vector<float> kmeans(const float* data, size_t n, uint32_t dim, const KMeansParams& p,
                          std::vector<uint32_t>* labels_out, double* inertia_out) {
  if (n == 0 || p.k == 0) throw std::invalid_argument("kmeans: empty input");
  const uint32_t k = p.k;
  std::vector<float> C(size_t(k) * dim);
  if (p.init == KMeansInit::PlusPlus) {
    // k-means++: first centre uniform, each next one sampled with probability proportional
    // to its squared distance from the nearest centre chosen so far.
    std::mt19937_64 rng(p.seed);
    auto unit = [&rng] { return double(rng() >> 11) * 0x1.0p-53; };  // [0,1), portable
    std::vector<double> d2(n, std::numeric_limits<double>::infinity());
    size_t first = rng() % n;
    std::copy(data + first * dim, data + (first + 1) * dim, C.begin());
    for (uint32_t j = 1; j < k; ++j) {
      const float* c = C.data() + size_t(j - 1) * dim;
      double total = 0;
      for (size_t i = 0; i < n; ++i) {
        double d = 0;
        for (uint32_t t = 0; t < dim; ++t) {
          double e = double(data[i * dim + t]) - double(c[t]);
          d += e * e;
        }
        if (d < d2[i]) d2[i] = d;
        total += d2[i];
      }
      size_t pick = n - 1;
      if (total > 0) {
        double r = unit() * total, acc = 0;
        for (size_t i = 0; i < n; ++i) {
          acc += d2[i];
          if (acc > r) { pick = i; break; }
        }
      } else {
        pick = rng() % n;  // every point coincides with a centre already
      }
      std::copy(data + pick * dim, data + (pick + 1) * dim, C.begin() + size_t(j) * dim);
    }
  } else {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t(0));
    std::mt19937_64 rng(p.seed);
    for (uint32_t j = 0; j < k; ++j) {
      size_t pick = j < n ? j + rng() % (n - j) : rng() % n;
      if (j < n) std::swap(idx[j], idx[pick]);
      size_t src = j < n ? idx[j] : pick;
      std::copy(data + src * dim, data + (src + 1) * dim, C.begin() + size_t(j) * dim);
    }
  }
  std::vector<uint32_t> labels(n);
  std::vector<float> dists(n);
  constexpr size_t kChunks = 64;
  size_t per = (n + kChunks - 1) / kChunks;
  std::vector<double> sums(kChunks * size_t(k) * dim);
  std::vector<uint64_t> counts(kChunks * size_t(k));
  double inertia = 0;
  for (uint32_t it = 0; it <= p.iters; ++it) {
    parallel_for(0, n, p.threads, [&](size_t i, unsigned) {
      labels[i] = nearest_centroid(data + i * dim, C.data(), k, dim, &dists[i]);
    }, 256);
    inertia = 0;
    for (size_t i = 0; i < n; ++i) inertia += dists[i];
    if (it == p.iters) break;  // final pass only assigns
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    parallel_for(0, kChunks, p.threads, [&](size_t c, unsigned) {
      double* S = sums.data() + c * k * dim;
      uint64_t* N = counts.data() + c * k;
      for (size_t i = c * per; i < std::min(n, (c + 1) * per); ++i) {
        uint32_t l = labels[i];
        ++N[l];
        for (uint32_t d = 0; d < dim; ++d) S[size_t(l) * dim + d] += data[i * dim + d];
      }
    });
    std::vector<size_t> empties;
    for (uint32_t j = 0; j < k; ++j) {
      uint64_t cnt = 0;
      for (size_t c = 0; c < kChunks; ++c) cnt += counts[c * k + j];
      if (cnt == 0) {
        empties.push_back(j);
        continue;
      }
      for (uint32_t d = 0; d < dim; ++d) {
        double s = 0;
        for (size_t c = 0; c < kChunks; ++c) s += sums[(c * k + j) * dim + d];
        C[size_t(j) * dim + d] = float(s / double(cnt));
      }
    }
    if (!empties.empty()) {
      // Re-seed empty clusters with the points farthest from their centroids.
      std::vector<size_t> order(n);
      std::iota(order.begin(), order.end(), size_t(0));
      size_t m = std::min(empties.size(), n);
      std::partial_sort(order.begin(), order.begin() + m, order.end(), [&](size_t a, size_t b) {
        return dists[a] > dists[b] || (dists[a] == dists[b] && a < b);
      });
      for (size_t e = 0; e < m; ++e)
        std::copy(data + order[e] * dim, data + (order[e] + 1) * dim, C.begin() + empties[e] * dim);
    }
  }
  if (labels_out) *labels_out = std::move(labels);
  if (inertia_out) *inertia_out = inertia;
  return C;
}

}  // namespace hs::vector
