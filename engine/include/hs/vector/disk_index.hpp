#pragma once
// DiskANN-style SSD index (Subramanya et al., NeurIPS 2019, section 3).
//
// Files in the index directory:
//   disk.index     block 0: header. Then one fixed-size node record per point:
//                  [float vec[dim]][uint32 degree][uint32 nbrs[R]], zero padded.
//                  Records never straddle a 4 KB block: small records are packed
//                  floor(4096 / record) per block, large ones take whole blocks.
//                  (768-d, R <= 255: exactly one node per 4 KB block.)
//   pq_pivots.bin  ProductQuantizer (mean + M x 256 codebooks)
//   pq_codes.bin   uint32 n, uint32 M, then n*M bytes, held in RAM at search time
//   docids.u64bin  ordinal -> global passage id (optional)
//   build.json     parameters and build statistics
//
// Search (beam search): the candidate list is ordered by PQ distance (RAM only).
// Each round takes the W closest unexpanded candidates, reads their node blocks
// in parallel (unless cached), computes the exact inner product from the
// full-precision vector that comes with the block, and pushes unvisited
// neighbours with their PQ distance. At the end the expanded nodes are re-ranked
// by their exact scores: the "full-precision rerank" is free of extra reads
// because every expanded node's vector was already fetched with its adjacency.
//
// Build (memory-limited, section 3.2 of the paper): k-means on a sample into P
// clusters; every point joins its `overlap` (=2) nearest clusters; a Vamana graph
// is built per cluster (only that cluster's vectors in RAM); per-cluster graphs
// are spilled to disk; the merge streams them in id order, takes the union of each
// node's edges, re-prunes nodes above R with RobustPrune(alpha), and writes node
// blocks. Overlap is what keeps the merged graph navigable across cluster borders.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hs/common/deadline.hpp"
#include "hs/vector/pq.hpp"
#include "hs/vector/types.hpp"
#include "hs/vector/vamana.hpp"

namespace hs::vector {

constexpr size_t kBlock = 4096;

struct DiskBuildParams {
  VamanaBuildParams vamana;          // R (<= 255 at 768-d), L, alpha, seed, threads
  uint32_t pq_M = 96;                // bytes per vector in RAM
  uint32_t pq_train_sample = 200000; // seeded uniform sample for PQ training
  uint32_t pq_iters = 15;
  uint32_t partitions = 1;           // 1 = one in-memory Vamana over everything
  double build_ram_gb = 0;           // >0: choose partitions so each fits this budget
  uint32_t overlap = 2;              // clusters per point
  uint32_t kmeans_sample = 100000;
  uint32_t kmeans_iters = 12;
  std::string prebuilt_graph;        // partitions == 1: reuse a saved VamanaIndex graph instead of building
  std::string tmp_dir;               // spill directory for partition graphs (default: out_dir)
  bool verbose = false;
};

struct DiskBuildStats {
  double seconds_total = 0, seconds_pq = 0, seconds_partition = 0, seconds_graphs = 0, seconds_merge = 0;
  uint32_t partitions = 0;
  uint64_t max_partition_size = 0, sum_partition_sizes = 0;
  uint64_t merged_nodes_pruned = 0;  // nodes whose union exceeded R
  double avg_degree = 0;
  uint64_t index_bytes = 0;
  uint64_t peak_rss_bytes = 0;
  double pq_train_seconds = 0;
};

// `data` is typically a MappedFbin (so only touched rows are resident).
void build_disk_index(const float* data, uint32_t n, uint32_t dim, const std::string& out_dir,
                      const DiskBuildParams& p, DiskBuildStats* stats = nullptr,
                      const std::vector<uint64_t>* docids = nullptr);

enum class IoMode {
  kSequential,  // one pread after another (baseline)
  kThreadPool,  // W reads issued concurrently to a pool of I/O threads
};

struct DiskOpenOptions {
  bool direct_io = true;      // cold: F_NOCACHE (macOS) / O_DIRECT (Linux). false = page cache (warm)
  uint32_t cache_nodes = 0;   // BFS-from-medoid nodes kept in RAM
  IoMode io_mode = IoMode::kThreadPool;
  unsigned io_threads = 8;    // pool size (shared by all queries on this index)
};

struct DiskMemory {
  uint64_t pq_codes = 0, pq_pivots = 0, cache = 0, docids = 0;
  uint64_t total() const { return pq_codes + pq_pivots + cache + docids; }
};

class DiskIndex {
 public:
  static std::unique_ptr<DiskIndex> open(const std::string& dir, const DiskOpenOptions& o = {});
  ~DiskIndex();

  VectorResult search(const float* q, const VectorSearchOptions& opts, hs::Deadline& deadline) const;
  VectorResult search(const float* q, const VectorSearchOptions& opts) const {
    hs::Deadline d;
    return search(q, opts, d);
  }

  uint32_t size() const { return n_; }
  uint32_t dim() const { return dim_; }
  uint32_t max_degree() const { return R_; }
  uint32_t medoid() const { return medoid_; }
  uint64_t global_id(uint32_t ordinal) const { return docids_.empty() ? ordinal : docids_[ordinal]; }
  DiskMemory memory() const;
  const ProductQuantizer& pq() const { return pq_; }

  // Reads one node straight from the file (tests, tools).
  void read_node(uint32_t id, std::vector<float>& vec, std::vector<uint32_t>& nbrs) const;

  struct Impl;

 private:
  DiskIndex() = default;
  uint32_t n_ = 0, dim_ = 0, R_ = 0, medoid_ = 0;
  uint32_t record_bytes_ = 0, nodes_per_block_ = 0, blocks_per_node_ = 0;
  int fd_ = -1;
  ProductQuantizer pq_;
  std::vector<uint8_t> codes_;
  std::vector<uint64_t> docids_;
  std::vector<uint32_t> cache_ids_;  // sorted
  std::vector<uint8_t> cache_data_;  // record_bytes_ each, same order
  std::unique_ptr<Impl> impl_;
  friend struct Impl;

  uint64_t record_offset(uint32_t id) const;  // byte offset of a record within the file
  uint64_t read_offset(uint32_t id) const;    // 4 KB aligned offset of the block(s) holding it
  uint32_t read_len() const { return uint32_t(blocks_per_node_ * kBlock); }
  const uint8_t* cached(uint32_t id) const;
};

uint64_t peak_rss_bytes();

}  // namespace hs::vector
