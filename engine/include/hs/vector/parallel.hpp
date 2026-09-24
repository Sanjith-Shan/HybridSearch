#pragma once
// Minimal fork-join parallel_for. Work is handed out dynamically in grains, so
// *which* thread runs an index is nondeterministic; callers keep results
// deterministic by writing each index's result to its own slot and never
// reading state another index of the same loop writes.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

namespace hs::vector {

// HS_THREADS overrides; else hardware concurrency.
inline unsigned default_threads() {
  if (const char* e = std::getenv("HS_THREADS")) {
    int v = std::atoi(e);
    if (v > 0) return unsigned(v);
  }
  unsigned h = std::thread::hardware_concurrency();
  return h ? h : 1;
}

inline unsigned resolve_threads(unsigned t) { return t ? t : default_threads(); }

// f(i, thread_id) for i in [begin, end).
template <class F>
void parallel_for(size_t begin, size_t end, unsigned threads, F&& f, size_t grain = 1) {
  if (end <= begin) return;
  threads = resolve_threads(threads);
  size_t n = end - begin;
  grain = std::max<size_t>(1, grain);
  unsigned use = unsigned(std::min<size_t>(threads, (n + grain - 1) / grain));
  if (use <= 1) {
    for (size_t i = begin; i < end; ++i) f(i, 0u);
    return;
  }
  std::atomic<size_t> next{begin};
  auto worker = [&](unsigned tid) {
    for (;;) {
      size_t s = next.fetch_add(grain, std::memory_order_relaxed);
      if (s >= end) break;
      size_t e = std::min(end, s + grain);
      for (size_t i = s; i < e; ++i) f(i, tid);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(use - 1);
  for (unsigned t = 1; t < use; ++t) pool.emplace_back(worker, t);
  worker(0);
  for (auto& th : pool) th.join();
}

}  // namespace hs::vector
