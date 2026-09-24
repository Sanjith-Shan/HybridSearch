#pragma once
// Small containers shared by in-memory and on-disk graph search.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace hs::vector {

// Open-addressing set of uint32 node ids. A query visits a few thousand nodes, so
// a table sized to the visit count stays in L1/L2, unlike an n-sized bitmap.
class VisitedSet {
 public:
  explicit VisitedSet(uint32_t log2_cap = 12) { alloc(log2_cap); }

  void clear() {
    if (size_ == 0) return;
    std::fill(slots_.begin(), slots_.end(), kEmpty);
    size_ = 0;
  }
  // Returns true if id was not present (and inserts it).
  bool insert(uint32_t id) {
    if ((size_ + 1) * 2 > slots_.size()) grow();
    uint32_t h = hash(id) & mask_;
    for (;;) {
      uint32_t s = slots_[h];
      if (s == id) return false;
      if (s == kEmpty) {
        slots_[h] = id;
        ++size_;
        return true;
      }
      h = (h + 1) & mask_;
    }
  }
  bool contains(uint32_t id) const {
    uint32_t h = hash(id) & mask_;
    for (;;) {
      uint32_t s = slots_[h];
      if (s == id) return true;
      if (s == kEmpty) return false;
      h = (h + 1) & mask_;
    }
  }
  size_t size() const { return size_; }

 private:
  static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
  static uint32_t hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
  }
  void alloc(uint32_t log2_cap) {
    slots_.assign(size_t(1) << log2_cap, kEmpty);
    mask_ = uint32_t(slots_.size() - 1);
    size_ = 0;
  }
  void grow() {
    std::vector<uint32_t> old;
    old.swap(slots_);
    uint32_t lg = 0;
    while ((size_t(1) << lg) < old.size() * 2) ++lg;
    alloc(lg);
    for (uint32_t s : old)
      if (s != kEmpty) insert(s);
  }
  std::vector<uint32_t> slots_;
  uint32_t mask_ = 0;
  size_t size_ = 0;
};

struct Neighbor {
  uint32_t id;
  float dist;
  bool expanded;
};

inline bool neighbor_less(const Neighbor& a, const Neighbor& b) {
  return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

// The search list L of greedy / beam search: the best `capacity` candidates seen,
// sorted by (distance, id), with a cursor at the closest unexpanded one.
class CandidateList {
 public:
  explicit CandidateList(size_t capacity = 0) { reset(capacity); }
  void reset(size_t capacity) {
    cap_ = capacity;
    v_.clear();
    v_.reserve(capacity + 1);
    cursor_ = 0;
  }
  size_t size() const { return v_.size(); }
  size_t capacity() const { return cap_; }
  const Neighbor& operator[](size_t i) const { return v_[i]; }

  // Returns false if the candidate did not make the list.
  bool insert(uint32_t id, float dist) {
    Neighbor c{id, dist, false};
    if (v_.size() >= cap_ && !neighbor_less(c, v_.back())) return false;
    auto it = std::lower_bound(v_.begin(), v_.end(), c, neighbor_less);
    size_t pos = size_t(it - v_.begin());
    v_.insert(it, c);
    if (v_.size() > cap_) v_.pop_back();
    if (pos < cursor_) cursor_ = pos;
    return true;
  }
  bool has_unexpanded() const { return cursor_ < v_.size(); }
  // Closest unexpanded; marks it expanded and advances the cursor.
  Neighbor pop_closest_unexpanded() {
    Neighbor& n = v_[cursor_];
    n.expanded = true;
    Neighbor out = n;
    advance();
    return out;
  }

 private:
  void advance() {
    while (cursor_ < v_.size() && v_[cursor_].expanded) ++cursor_;
  }
  std::vector<Neighbor> v_;
  size_t cap_ = 0;
  size_t cursor_ = 0;
};

// splitmix64: seeded, per-index deterministic randomness.
inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

}  // namespace hs::vector
