#pragma once
// Exact inner-product top-k by brute force: the ground truth every recall number
// is measured against. Queries are processed in blocks so each base row is loaded
// once per block instead of once per query.

#include <cstdint>
#include <string>
#include <vector>

#include "hs/common/topk.hpp"

namespace hs::vector {

// Returns nq*k hits, row-major, best first (ties: lower ordinal first).
std::vector<hs::ScoredDoc> exact_topk(const float* base, uint32_t n, const float* queries,
                                      uint32_t nq, uint32_t dim, uint32_t k, unsigned threads = 0);

// Ground-truth file (big-ann-benchmarks layout): uint32 nq, uint32 k,
// uint32 ids[nq*k], float scores[nq*k].
struct GroundTruth {
  uint32_t nq = 0, k = 0;
  std::vector<uint32_t> ids;
  std::vector<float> scores;
  const uint32_t* row(size_t q) const { return ids.data() + q * k; }
};
void write_groundtruth(const std::string& path, const std::vector<hs::ScoredDoc>& hits, uint32_t nq,
                       uint32_t k);
GroundTruth read_groundtruth(const std::string& path);

// |approx[0:k] ∩ exact[0:k]| / k, averaged over queries. `approx` has stride `astride`.
double recall_at_k(const GroundTruth& gt, const std::vector<std::vector<uint32_t>>& approx, uint32_t k);

}  // namespace hs::vector
