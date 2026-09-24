#pragma once
// Search options and results shared by VamanaIndex (in memory) and DiskIndex.

#include <cstdint>
#include <vector>

#include "hs/common/topk.hpp"

namespace hs::vector {

struct VectorSearchOptions {
  uint32_t k = 10;
  uint32_t L = 100;          // search list size (L_search); clamped up to k
  uint32_t beam_width = 4;   // DiskIndex only: nodes read from SSD per round (W)
};

struct VectorResult {
  std::vector<hs::ScoredDoc> hits;  // best first; score = inner product; doc = ordinal
  bool partial = false;             // deadline fired, hits are best-so-far
  uint64_t distance_computations = 0;  // full-precision distances
  uint64_t pq_distance_computations = 0;  // DiskIndex: PQ (ADC) distances
  uint64_t hops = 0;                // nodes expanded
  uint64_t ssd_reads = 0;           // DiskIndex: 4 KB blocks read from the file (cache misses)
  uint64_t cache_hits = 0;          // DiskIndex: node expansions served from the RAM cache
  uint64_t io_rounds = 0;           // DiskIndex: batches of parallel reads
  uint64_t io_us = 0;               // DiskIndex: wall time spent waiting on reads
};

}  // namespace hs::vector
