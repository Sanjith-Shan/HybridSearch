#pragma once
// Bounded top-k with a deterministic total order: higher score first, and on a
// tie the lower document ID first. Every retrieval path uses this order so that
// exhaustive and pruned evaluation can be compared ID-for-ID.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace hs {

struct ScoredDoc {
  uint32_t doc = 0;  // index-local ordinal; map through the index's docids to a global ID
  float score = 0.0f;
};

// True if a ranks strictly before b.
inline bool ranks_before(const ScoredDoc& a, const ScoredDoc& b) {
  if (a.score != b.score) return a.score > b.score;
  return a.doc < b.doc;
}

class TopK {
 public:
  explicit TopK(size_t k) : k_(k) { heap_.reserve(k + 1); }

  size_t k() const { return k_; }
  size_t size() const { return heap_.size(); }
  bool full() const { return heap_.size() >= k_; }

  // Score a candidate must beat to enter (the current k-th). -inf until full.
  float threshold() const {
    return full() ? heap_.front().score : -std::numeric_limits<float>::infinity();
  }

  // Returns true if the candidate was admitted.
  bool push(uint32_t doc, float score) {
    if (k_ == 0) return false;
    ScoredDoc c{doc, score};
    if (!full()) {
      heap_.push_back(c);
      std::push_heap(heap_.begin(), heap_.end(), ranks_before);
      return true;
    }
    // heap_.front() is the worst of the current top-k under ranks_before.
    if (!ranks_before(c, heap_.front())) return false;
    std::pop_heap(heap_.begin(), heap_.end(), ranks_before);
    heap_.back() = c;
    std::push_heap(heap_.begin(), heap_.end(), ranks_before);
    return true;
  }

  // Sorted best-first. Leaves the heap empty.
  std::vector<ScoredDoc> take_sorted() {
    std::sort(heap_.begin(), heap_.end(), ranks_before);
    return std::move(heap_);
  }

 private:
  size_t k_;
  std::vector<ScoredDoc> heap_;
};

}  // namespace hs
