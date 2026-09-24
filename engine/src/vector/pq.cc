#include "hs/vector/pq.hpp"

#include <cstdio>
#include <stdexcept>

#include "hs/vector/distance.hpp"
#include "hs/vector/kmeans.hpp"
#include "hs/vector/parallel.hpp"

namespace hs::vector {

ProductQuantizer ProductQuantizer::train(const float* data, size_t n, uint32_t dim, uint32_t M, uint32_t iters,
                                         uint64_t seed, unsigned threads) {
  if (M == 0 || dim % M != 0) throw std::invalid_argument("PQ: dim must be a multiple of M");
  if (dim / M > 1024) throw std::invalid_argument("PQ: subspace wider than 1024");
  if (n < kCentroids) throw std::invalid_argument("PQ: need at least 256 training vectors");
  ProductQuantizer pq;
  pq.dim_ = dim;
  pq.M_ = M;
  pq.dsub_ = dim / M;
  pq.mean_.assign(dim, 0.f);
  std::vector<double> mean(dim, 0.0);
  for (size_t i = 0; i < n; ++i)
    for (uint32_t d = 0; d < dim; ++d) mean[d] += data[i * dim + d];
  for (uint32_t d = 0; d < dim; ++d) pq.mean_[d] = float(mean[d] / double(n));
  pq.codebooks_.assign(size_t(M) * kCentroids * pq.dsub_, 0.f);
  // Subspaces are independent: train them in parallel, each single-threaded.
  parallel_for(0, M, threads, [&](size_t m, unsigned) {
    uint32_t ds = pq.dsub_;
    std::vector<float> sub(n * ds);
    for (size_t i = 0; i < n; ++i)
      for (uint32_t d = 0; d < ds; ++d) sub[i * ds + d] = data[i * dim + m * ds + d] - pq.mean_[m * ds + d];
    KMeansParams kp;
    kp.k = kCentroids;
    kp.iters = iters;
    kp.seed = seed + 7919 * m;
    kp.threads = 1;
    auto C = kmeans(sub.data(), n, ds, kp);
    std::copy(C.begin(), C.end(), pq.codebooks_.begin() + m * kCentroids * ds);
  });
  return pq;
}

void ProductQuantizer::encode(const float* x, uint8_t* code) const {
  float r[1024];
  for (uint32_t m = 0; m < M_; ++m) {
    for (uint32_t d = 0; d < dsub_; ++d) r[d] = x[m * dsub_ + d] - mean_[m * dsub_ + d];
    code[m] = uint8_t(nearest_centroid(r, codebooks_.data() + size_t(m) * kCentroids * dsub_, kCentroids, dsub_));
  }
}

void ProductQuantizer::encode_batch(const float* x, size_t n, uint8_t* codes, unsigned threads) const {
  parallel_for(0, n, threads, [&](size_t i, unsigned) { encode(x + i * dim_, codes + i * M_); }, 1024);
}

void ProductQuantizer::decode(const uint8_t* code, float* x) const {
  for (uint32_t m = 0; m < M_; ++m) {
    const float* c = codebooks_.data() + (size_t(m) * kCentroids + code[m]) * dsub_;
    for (uint32_t d = 0; d < dsub_; ++d) x[m * dsub_ + d] = c[d] + mean_[m * dsub_ + d];
  }
}

void ProductQuantizer::distance_table(const float* q, float* table) const {
  float r[1024];
  for (uint32_t m = 0; m < M_; ++m) {
    for (uint32_t d = 0; d < dsub_; ++d) r[d] = q[m * dsub_ + d] - mean_[m * dsub_ + d];
    const float* cb = codebooks_.data() + size_t(m) * kCentroids * dsub_;
    for (uint32_t c = 0; c < kCentroids; ++c) table[m * kCentroids + c] = l2sq(r, cb + size_t(c) * dsub_, dsub_);
  }
}

void ProductQuantizer::save(const std::string& path) const {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  uint32_t h[4] = {0x31515048u /* "HPQ1" */, dim_, M_, dsub_};
  bool ok = std::fwrite(h, 4, 4, f) == 4 && std::fwrite(mean_.data(), 4, mean_.size(), f) == mean_.size() &&
            std::fwrite(codebooks_.data(), 4, codebooks_.size(), f) == codebooks_.size();
  ok = std::fclose(f) == 0 && ok;
  if (!ok) throw std::runtime_error("short write " + path);
}

ProductQuantizer ProductQuantizer::load(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  uint32_t h[4];
  ProductQuantizer pq;
  bool ok = std::fread(h, 4, 4, f) == 4 && h[0] == 0x31515048u && h[3] <= 1024;
  if (ok) {
    pq.dim_ = h[1];
    pq.M_ = h[2];
    pq.dsub_ = h[3];
    pq.mean_.resize(pq.dim_);
    pq.codebooks_.resize(size_t(pq.M_) * kCentroids * pq.dsub_);
    ok = std::fread(pq.mean_.data(), 4, pq.mean_.size(), f) == pq.mean_.size() &&
         std::fread(pq.codebooks_.data(), 4, pq.codebooks_.size(), f) == pq.codebooks_.size();
  }
  std::fclose(f);
  if (!ok) throw std::runtime_error("bad PQ file " + path);
  return pq;
}

}  // namespace hs::vector
