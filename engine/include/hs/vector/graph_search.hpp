#pragma once
// GreedySearch(s, q, L) from the DiskANN paper (Algorithm 1), over a fixed-degree
// adjacency array. Used by the Vamana build (collecting the expanded set V) and by
// in-memory search.

#include <cstdint>
#include <vector>

#include "hs/common/deadline.hpp"
#include "hs/vector/containers.hpp"
#include "hs/vector/distance.hpp"
#include "hs/vector/types.hpp"

namespace hs::vector {

struct GraphView {
  const float* data;
  uint32_t dim;
  const uint32_t* adj;  // n * R
  const uint32_t* deg;  // n
  uint32_t R;
};

struct GraphSearchScratch {
  VisitedSet visited;
  CandidateList cands;
  std::vector<uint32_t> expanded;
  std::vector<uint32_t> fresh;
};

// On return s.cands holds the best L candidates (sorted by squared L2) and, with
// collect=true, s.expanded holds every expanded node in expansion order.
// With a deadline, stops expanding once it fires (at least one node is expanded).
inline void greedy_search(const GraphView& g, const float* q, uint32_t start, uint32_t L,
                          GraphSearchScratch& s, bool collect, VectorResult* r, hs::Deadline* dl) {
  s.visited.clear();
  s.cands.reset(L);
  s.expanded.clear();
  s.visited.insert(start);
  s.cands.insert(start, l2sq(q, g.data + size_t(start) * g.dim, g.dim));
  uint64_t dists = 1, hops = 0;
  while (s.cands.has_unexpanded()) {
    if (dl && hops > 0 && dl->expired_every(4)) break;
    Neighbor cur = s.cands.pop_closest_unexpanded();
    ++hops;
    if (collect) s.expanded.push_back(cur.id);
    const uint32_t* row = g.adj + size_t(cur.id) * g.R;
    uint32_t d = g.deg[cur.id];
    s.fresh.clear();
    for (uint32_t j = 0; j < d; ++j)
      if (s.visited.insert(row[j])) s.fresh.push_back(row[j]);
    if (!s.fresh.empty()) prefetch_vector(g.data + size_t(s.fresh[0]) * g.dim, g.dim);
    for (size_t j = 0; j < s.fresh.size(); ++j) {
      if (j + 1 < s.fresh.size()) prefetch_vector(g.data + size_t(s.fresh[j + 1]) * g.dim, g.dim);
      uint32_t id = s.fresh[j];
      s.cands.insert(id, l2sq(q, g.data + size_t(id) * g.dim, g.dim));
    }
    dists += s.fresh.size();
  }
  if (r) {
    r->distance_computations += dists;
    r->hops += hops;
  }
}

}  // namespace hs::vector
