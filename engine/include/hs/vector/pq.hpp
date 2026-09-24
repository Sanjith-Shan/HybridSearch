#pragma once
// Product quantization (Jegou et al. 2011) as DiskANN uses it: the data is centred
// (mean subtracted), split into M contiguous subspaces of dim/M floats, and each
// subspace is quantized to one of 256 k-means centroids, so a vector costs M bytes
// (M = 96 -> 96 bytes for 768-d, 32x smaller than fp32). A query builds an
// M x 256 table of squared L2 distances once; the asymmetric distance to any code
// is then M table lookups (ADC).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hs::vector {

class ProductQuantizer {
 public:
  static constexpr uint32_t kCentroids = 256;

  ProductQuantizer() = default;
  // Trains on `n` rows of `data` (the caller picks the sample). dim % M must be 0.
  static ProductQuantizer train(const float* data, size_t n, uint32_t dim, uint32_t M, uint32_t iters,
                                uint64_t seed, unsigned threads = 0);

  uint32_t dim() const { return dim_; }
  uint32_t M() const { return M_; }
  uint32_t dsub() const { return dsub_; }
  size_t code_bytes() const { return M_; }
  size_t pivot_bytes() const { return (codebooks_.size() + mean_.size()) * sizeof(float); }

  void encode(const float* x, uint8_t* code) const;
  void encode_batch(const float* x, size_t n, uint8_t* codes, unsigned threads = 0) const;
  void decode(const uint8_t* code, float* x) const;

  // table[m*256 + c] = || (q - mean)_m - centroid_{m,c} ||^2
  void distance_table(const float* q, float* table) const;
  float adc(const float* table, const uint8_t* code) const {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    uint32_t m = 0;
    for (; m + 4 <= M_; m += 4) {
      s0 += table[(m + 0) * kCentroids + code[m + 0]];
      s1 += table[(m + 1) * kCentroids + code[m + 1]];
      s2 += table[(m + 2) * kCentroids + code[m + 2]];
      s3 += table[(m + 3) * kCentroids + code[m + 3]];
    }
    for (; m < M_; ++m) s0 += table[m * kCentroids + code[m]];
    return (s0 + s1) + (s2 + s3);
  }

  void save(const std::string& path) const;
  static ProductQuantizer load(const std::string& path);

 private:
  uint32_t dim_ = 0, M_ = 0, dsub_ = 0;
  std::vector<float> mean_;       // dim
  std::vector<float> codebooks_;  // M * 256 * dsub
};

}  // namespace hs::vector
