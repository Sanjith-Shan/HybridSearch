#pragma once
// Vamana graph index (Subramanya et al., "DiskANN", NeurIPS 2019), in memory.
//
// Build: random R-regular init, medoid start, then two passes over a seeded
// permutation of the points (alpha = 1, then alpha = alpha_final). Each point is
// greedy-searched from the medoid with list size L; the expanded set, plus its
// current out-edges, is cut to R edges by RobustPrune(alpha); back-edges are added
// and a neighbour that overflows R is re-pruned.
//
// Determinism under threads. The paper's parallel build (and Microsoft's code)
// updates the graph under per-node locks while other threads search it, so the
// result depends on scheduling. Here each pass walks the permutation in batches
// (prefix doubling up to max_batch_fraction * n, as in ParlayANN, Manohar et al.
// PPoPP 2024) and each batch runs three barrier-separated phases:
//   1. search + prune every point of the batch against a frozen graph (read only);
//   2. install each point's new out-list (each point writes only its own row);
//   3. back-edges: (dst, src) pairs, stably sorted by dst, each dst processed by
//      exactly one task that appends/re-prunes in src-permutation order.
// No phase reads a row another task of the same phase writes, so the graph is a
// pure function of (data, params, seed), independent of thread count and timing.
// Tests assert byte equality across rebuilds and thread counts.
//
// Back-edge slack: see VamanaBuildParams::slack.
//
// alpha is applied to squared L2 distances, as in Microsoft's implementation
// (equivalent to sqrt(alpha) on Euclidean distance in the paper's notation).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hs/common/deadline.hpp"
#include "hs/vector/mmap.hpp"
#include "hs/vector/types.hpp"

namespace hs::vector {

struct VamanaBuildParams {
  uint32_t R = 64;              // max out-degree
  uint32_t L = 100;             // build search list size
  float alpha = 1.2f;           // second-pass alpha
  bool two_pass = true;         // false: single pass at alpha
  uint32_t max_candidates = 750;  // cap on RobustPrune's candidate pool (as in DiskANN)
  double max_batch_fraction = 0.02;  // largest batch as a fraction of n
  // Back-edges may grow a list to ceil(slack * R) before it is re-pruned to R (as
  // Microsoft's GRAPH_SLACK_FACTOR = 1.3). Without slack every back-edge into a full
  // node costs a RobustPrune, which dominates the build. A final pass prunes every
  // list still above R.
  float slack = 1.3f;
  uint64_t seed = 20260923;
  unsigned threads = 0;         // 0 = default_threads()
  bool verbose = false;
};

struct VamanaBuildStats {
  double seconds_total = 0, seconds_pass1 = 0, seconds_pass2 = 0;
  uint64_t batches = 0;
  double avg_degree = 0;
  uint32_t max_degree = 0;
};

// RobustPrune(p, V, alpha, R). `cands` may contain p and duplicates (both are
// dropped). Returns at most R ids, closest-first. Exposed for tests and for the
// partition merge of the disk build.
std::vector<uint32_t> robust_prune(uint32_t p, std::vector<uint32_t> cands, const float* data,
                                   uint32_t dim, float alpha, uint32_t R, uint32_t max_candidates);

class VamanaIndex {
 public:
  VamanaIndex() = default;

  // `data` must outlive the index (it is not copied).
  static VamanaIndex build(const float* data, uint32_t n, uint32_t dim, const VamanaBuildParams& p,
                           VamanaBuildStats* stats = nullptr);

  // Graph file: magic, n, dim, R, medoid, degrees[n], neighbours[n*R] (unused slots 0xFFFFFFFF).
  void save(const std::string& graph_path) const;
  // Maps `vectors_fbin` (first n rows) and loads the graph.
  static VamanaIndex load(const std::string& graph_path, const std::string& vectors_fbin);
  // Loads the graph against caller-owned vectors.
  static VamanaIndex load(const std::string& graph_path, const float* data, uint32_t n, uint32_t dim);

  VectorResult search(const float* q, const VectorSearchOptions& opts, hs::Deadline& deadline) const;
  VectorResult search(const float* q, const VectorSearchOptions& opts) const {
    hs::Deadline d;
    return search(q, opts, d);
  }

  uint32_t size() const { return n_; }
  uint32_t dim() const { return dim_; }
  uint32_t max_degree() const { return R_; }
  uint32_t medoid() const { return medoid_; }
  uint32_t degree(uint32_t i) const { return deg_[i]; }
  const uint32_t* neighbors(uint32_t i) const { return adj_.data() + size_t(i) * R_; }
  const float* vector(uint32_t i) const { return data_ + size_t(i) * dim_; }
  const float* data() const { return data_; }

  // Ordinal -> global MS MARCO passage id; identity if no docids were set.
  void set_docids(std::vector<uint64_t> ids) { docids_ = std::move(ids); }
  uint64_t global_id(uint32_t ordinal) const { return docids_.empty() ? ordinal : docids_[ordinal]; }

  // Nodes reachable from the medoid by BFS. Anything below size() means some
  // points can never be returned by search: always check it after a build.
  uint32_t reachable_from_medoid() const;

  // Hash of degrees + adjacency, for rebuild-equality checks.
  uint64_t graph_hash() const;

  // Takes ownership of the adjacency (used by the disk build's partition step).
  std::vector<uint32_t>& adjacency() { return adj_; }
  std::vector<uint32_t>& degrees() { return deg_; }

 private:
  uint32_t n_ = 0, dim_ = 0, R_ = 0, medoid_ = 0;
  const float* data_ = nullptr;
  std::shared_ptr<MappedFbin> mapped_;
  std::vector<uint32_t> deg_;
  std::vector<uint32_t> adj_;
  std::vector<uint64_t> docids_;
};

// Point closest (L2) to the mean of the data; ties to lower id.
uint32_t find_medoid(const float* data, uint32_t n, uint32_t dim, unsigned threads);

}  // namespace hs::vector
