#include "hs/vector/vamana.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>

#include "hs/vector/containers.hpp"
#include "hs/vector/distance.hpp"
#include "hs/vector/graph_search.hpp"
#include "hs/vector/parallel.hpp"

namespace hs::vector {

namespace {
constexpr uint32_t kEmptySlot = 0xFFFFFFFFu;
constexpr uint64_t kGraphMagic = 0x3148504152474d56ULL;  // "VMGRAPH1"

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

std::vector<uint32_t> robust_prune(uint32_t p, std::vector<uint32_t> cands, const float* data,
                                   uint32_t dim, float alpha, uint32_t R, uint32_t max_candidates) {
  std::sort(cands.begin(), cands.end());
  cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
  const float* xp = data + size_t(p) * dim;
  std::vector<Neighbor> pool;
  pool.reserve(cands.size());
  for (uint32_t c : cands) {
    if (c == p || c == kEmptySlot) continue;
    pool.push_back({c, l2sq(xp, data + size_t(c) * dim, dim), false});
  }
  std::sort(pool.begin(), pool.end(), neighbor_less);
  if (max_candidates && pool.size() > max_candidates) pool.resize(max_candidates);

  std::vector<uint32_t> out;
  out.reserve(R);
  // Neighbor::expanded doubles as the "pruned" flag here.
  for (size_t i = 0; i < pool.size() && out.size() < R; ++i) {
    if (pool[i].expanded) continue;
    out.push_back(pool[i].id);
    const float* xi = data + size_t(pool[i].id) * dim;
    for (size_t j = i + 1; j < pool.size(); ++j) {
      if (pool[j].expanded) continue;
      // p* = pool[i] occludes p' = pool[j] if alpha * d(p*, p') <= d(p, p').
      if (alpha * l2sq(xi, data + size_t(pool[j].id) * dim, dim) <= pool[j].dist) pool[j].expanded = true;
    }
  }
  return out;
}

uint32_t find_medoid(const float* data, uint32_t n, uint32_t dim, unsigned threads) {
  // Fixed chunking (independent of thread count) keeps the floating-point sum,
  // and therefore the medoid, deterministic.
  constexpr size_t kChunks = 256;
  std::vector<std::vector<double>> part(kChunks, std::vector<double>(dim, 0.0));
  size_t per = (size_t(n) + kChunks - 1) / kChunks;
  parallel_for(0, kChunks, threads, [&](size_t c, unsigned) {
    size_t s = c * per, e = std::min<size_t>(n, s + per);
    for (size_t i = s; i < e; ++i)
      for (uint32_t d = 0; d < dim; ++d) part[c][d] += data[i * dim + d];
  });
  std::vector<float> mean(dim, 0.f);
  for (uint32_t d = 0; d < dim; ++d) {
    double s = 0;
    for (size_t c = 0; c < kChunks; ++c) s += part[c][d];
    mean[d] = float(s / std::max<uint32_t>(1, n));
  }
  std::vector<std::pair<float, uint32_t>> best(kChunks, {std::numeric_limits<float>::infinity(), 0});
  parallel_for(0, kChunks, threads, [&](size_t c, unsigned) {
    size_t s = c * per, e = std::min<size_t>(n, s + per);
    for (size_t i = s; i < e; ++i) {
      float d = l2sq(mean.data(), data + i * dim, dim);
      if (d < best[c].first) best[c] = {d, uint32_t(i)};
    }
  });
  return std::min_element(best.begin(), best.end())->second;
}

VamanaIndex VamanaIndex::build(const float* data, uint32_t n, uint32_t dim, const VamanaBuildParams& p,
                               VamanaBuildStats* stats) {
  if (n == 0) throw std::invalid_argument("Vamana build on empty data");
  auto t_all = std::chrono::steady_clock::now();
  unsigned threads = resolve_threads(p.threads);
  VamanaIndex ix;
  ix.n_ = n;
  ix.dim_ = dim;
  ix.data_ = data;
  const uint32_t R = std::max<uint32_t>(1, std::min<uint32_t>(p.R, n > 1 ? n - 1 : 1));
  // During the build rows have capacity C >= R (back-edge slack); compacted to R at the end.
  const uint32_t C = std::max<uint32_t>(R, uint32_t(std::ceil(double(R) * std::max(1.0f, p.slack))));
  ix.R_ = C;
  ix.adj_.assign(size_t(n) * C, kEmptySlot);
  ix.deg_.assign(n, 0);

  // Random R-regular init: each point gets R distinct random out-neighbours from
  // its own seeded stream, so the init is independent of thread count.
  parallel_for(0, n, threads, [&](size_t i, unsigned) {
    uint32_t* row = ix.adj_.data() + i * C;
    if (n <= 1) return;
    if (n - 1 <= R) {
      uint32_t d = 0;
      for (uint32_t j = 0; j < n; ++j)
        if (j != i) row[d++] = j;
      ix.deg_[i] = d;
      return;
    }
    uint64_t s = splitmix64(p.seed ^ (0x51ED27A3ULL * (i + 1)));
    uint32_t d = 0;
    while (d < R) {
      s = splitmix64(s);
      uint32_t c = uint32_t(s % n);
      if (c == i || std::find(row, row + d, c) != row + d) continue;
      row[d++] = c;
    }
    ix.deg_[i] = d;
  }, 1024);

  ix.medoid_ = find_medoid(data, n, dim, threads);

  std::vector<uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 rng(p.seed);
  for (uint32_t i = n; i > 1; --i) std::swap(order[i - 1], order[rng() % i]);

  std::vector<float> alphas;
  if (p.two_pass) alphas = {1.0f, p.alpha};
  else alphas = {p.alpha};

  const size_t max_batch = std::max<size_t>(1, size_t(p.max_batch_fraction * n));
  std::vector<GraphSearchScratch> scratch(threads);
  std::vector<uint32_t> new_adj(max_batch * R);
  std::vector<uint32_t> new_deg(max_batch);
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  std::vector<size_t> group_start;
  uint64_t batches = 0;

  for (size_t pass = 0; pass < alphas.size(); ++pass) {
    auto t_pass = std::chrono::steady_clock::now();
    const float alpha = alphas[pass];
    size_t pos = 0, b = 1;
    while (pos < n) {
      size_t bs = std::min<size_t>(b, n - pos);
      ++batches;
      // Phase 1: search + prune against the frozen graph.
      parallel_for(0, bs, threads, [&](size_t j, unsigned tid) {
        uint32_t pt = order[pos + j];
        GraphSearchScratch& s = scratch[tid];
        GraphView g{data, dim, ix.adj_.data(), ix.deg_.data(), C};
        greedy_search(g, data + size_t(pt) * dim, ix.medoid_, p.L, s, /*collect=*/true, nullptr, nullptr);
        std::vector<uint32_t> cands = s.expanded;
        const uint32_t* row = ix.adj_.data() + size_t(pt) * C;
        cands.insert(cands.end(), row, row + ix.deg_[pt]);
        auto out = robust_prune(pt, std::move(cands), data, dim, alpha, R, p.max_candidates);
        std::copy(out.begin(), out.end(), new_adj.begin() + j * R);
        new_deg[j] = uint32_t(out.size());
      }, 4);
      // Phase 2: install out-lists.
      parallel_for(0, bs, threads, [&](size_t j, unsigned) {
        uint32_t pt = order[pos + j];
        uint32_t* row = ix.adj_.data() + size_t(pt) * C;
        std::copy(new_adj.begin() + j * R, new_adj.begin() + j * R + new_deg[j], row);
        std::fill(row + new_deg[j], row + C, kEmptySlot);
        ix.deg_[pt] = new_deg[j];
      }, 256);
      // Phase 3: back-edges, grouped by destination in a deterministic order.
      pairs.clear();
      for (size_t j = 0; j < bs; ++j)
        for (uint32_t e = 0; e < new_deg[j]; ++e) pairs.push_back({new_adj[j * R + e], order[pos + j]});
      std::stable_sort(pairs.begin(), pairs.end(),
                       [](const auto& a, const auto& c) { return a.first < c.first; });
      group_start.clear();
      for (size_t i = 0; i < pairs.size(); ++i)
        if (i == 0 || pairs[i].first != pairs[i - 1].first) group_start.push_back(i);
      group_start.push_back(pairs.size());
      parallel_for(0, group_start.size() - 1, threads, [&](size_t g, unsigned) {
        size_t s = group_start[g], e = group_start[g + 1];
        uint32_t dst = pairs[s].first;
        uint32_t* row = ix.adj_.data() + size_t(dst) * C;
        uint32_t d = ix.deg_[dst];
        std::vector<uint32_t> extra;
        for (size_t i = s; i < e; ++i) {
          uint32_t src = pairs[i].second;
          if (src == dst || std::find(row, row + d, src) != row + d) continue;
          if (std::find(extra.begin(), extra.end(), src) != extra.end()) continue;
          extra.push_back(src);
        }
        if (extra.empty()) return;
        if (d + extra.size() <= C) {
          std::copy(extra.begin(), extra.end(), row + d);
          ix.deg_[dst] = d + uint32_t(extra.size());
        } else {
          std::vector<uint32_t> cands(row, row + d);
          cands.insert(cands.end(), extra.begin(), extra.end());
          auto out = robust_prune(dst, std::move(cands), data, dim, alpha, R, p.max_candidates);
          std::copy(out.begin(), out.end(), row);
          std::fill(row + out.size(), row + C, kEmptySlot);
          ix.deg_[dst] = uint32_t(out.size());
        }
      }, 16);
      pos += bs;
      b = std::min(max_batch, b * 2);
      if (p.verbose && (batches % 20 == 0))
        std::fprintf(stderr, "  pass %zu: %zu / %u points, %.1fs\n", pass + 1, pos, n, seconds_since(t_pass));
    }
    double secs = seconds_since(t_pass);
    if (stats) (pass == 0 && alphas.size() == 2 ? stats->seconds_pass1 : stats->seconds_pass2) = secs;
    if (p.verbose) std::fprintf(stderr, "pass %zu (alpha=%.2f) done in %.1fs\n", pass + 1, alpha, secs);
  }
  // Final cleanup: prune lists that used the slack back to R, then compact rows
  // from capacity C to R.
  {
    const float alpha = alphas.back();
    parallel_for(0, n, threads, [&](size_t i, unsigned) {
      if (ix.deg_[i] <= R) return;
      uint32_t* row = ix.adj_.data() + i * C;
      std::vector<uint32_t> cands(row, row + ix.deg_[i]);
      auto out = robust_prune(uint32_t(i), std::move(cands), data, dim, alpha, R, p.max_candidates);
      std::copy(out.begin(), out.end(), row);
      std::fill(row + out.size(), row + C, kEmptySlot);
      ix.deg_[i] = uint32_t(out.size());
    }, 64);
    if (C != R) {
      std::vector<uint32_t> compact(size_t(n) * R, kEmptySlot);
      for (size_t i = 0; i < n; ++i) std::copy(ix.adj_.begin() + i * C, ix.adj_.begin() + i * C + R, compact.begin() + i * R);
      ix.adj_.swap(compact);
      ix.R_ = R;
    }
  }
  if (stats) {
    stats->seconds_total = seconds_since(t_all);
    stats->batches = batches;
    uint64_t sum = 0;
    uint32_t mx = 0;
    for (uint32_t d : ix.deg_) {
      sum += d;
      mx = std::max(mx, d);
    }
    stats->avg_degree = double(sum) / n;
    stats->max_degree = mx;
  }
  return ix;
}

VectorResult VamanaIndex::search(const float* q, const VectorSearchOptions& opts, hs::Deadline& deadline) const {
  thread_local GraphSearchScratch scratch;
  VectorResult r;
  uint32_t k = std::min(opts.k, n_);
  uint32_t L = std::max(opts.L, k);
  GraphView g{data_, dim_, adj_.data(), deg_.data(), R_};
  greedy_search(g, q, medoid_, L, scratch, /*collect=*/false, &r, &deadline);
  hs::TopK top(k);
  for (size_t i = 0; i < scratch.cands.size() && i < k; ++i) {
    uint32_t id = scratch.cands[i].id;
    top.push(id, ip(q, data_ + size_t(id) * dim_, dim_));
  }
  r.distance_computations += std::min<size_t>(k, scratch.cands.size());
  r.hits = top.take_sorted();
  r.partial = deadline.fired();
  return r;
}

void VamanaIndex::save(const std::string& path) const {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  uint32_t hdr[4] = {n_, dim_, R_, medoid_};
  bool ok = std::fwrite(&kGraphMagic, 8, 1, f) == 1 && std::fwrite(hdr, 4, 4, f) == 4 &&
            std::fwrite(deg_.data(), 4, deg_.size(), f) == deg_.size() &&
            std::fwrite(adj_.data(), 4, adj_.size(), f) == adj_.size();
  ok = (std::fclose(f) == 0) && ok;
  if (!ok) throw std::runtime_error("short write " + path);
}

VamanaIndex VamanaIndex::load(const std::string& path, const float* data, uint32_t n, uint32_t dim) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t magic = 0;
  uint32_t hdr[4];
  VamanaIndex ix;
  bool ok = std::fread(&magic, 8, 1, f) == 1 && magic == kGraphMagic && std::fread(hdr, 4, 4, f) == 4;
  if (ok) {
    ix.n_ = hdr[0];
    ix.dim_ = hdr[1];
    ix.R_ = hdr[2];
    ix.medoid_ = hdr[3];
    ix.deg_.resize(ix.n_);
    ix.adj_.resize(size_t(ix.n_) * ix.R_);
    ok = std::fread(ix.deg_.data(), 4, ix.n_, f) == ix.n_ &&
         std::fread(ix.adj_.data(), 4, ix.adj_.size(), f) == ix.adj_.size();
  }
  std::fclose(f);
  if (!ok) throw std::runtime_error("bad graph file " + path);
  if (ix.n_ > n || ix.dim_ != dim) throw std::runtime_error("graph does not match vectors: " + path);
  ix.data_ = data;
  return ix;
}

VamanaIndex VamanaIndex::load(const std::string& graph_path, const std::string& vectors_fbin) {
  auto m = std::make_shared<MappedFbin>(vectors_fbin);
  VamanaIndex ix = load(graph_path, m->data(), m->n(), m->dim());
  ix.mapped_ = std::move(m);
  return ix;
}

uint32_t VamanaIndex::reachable_from_medoid() const {
  if (n_ == 0) return 0;
  std::vector<char> seen(n_, 0);
  std::vector<uint32_t> stack = {medoid_};
  seen[medoid_] = 1;
  uint32_t count = 1;
  while (!stack.empty()) {
    uint32_t u = stack.back();
    stack.pop_back();
    for (uint32_t j = 0; j < deg_[u]; ++j) {
      uint32_t v = adj_[size_t(u) * R_ + j];
      if (!seen[v]) seen[v] = 1, ++count, stack.push_back(v);
    }
  }
  return count;
}

uint64_t VamanaIndex::graph_hash() const {
  uint64_t h = 1469598103934665603ULL ^ n_ ^ (uint64_t(R_) << 32) ^ (uint64_t(medoid_) << 1);
  auto mix = [&](uint32_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  };
  for (uint32_t d : deg_) mix(d);
  for (uint32_t a : adj_) mix(a);
  return h;
}

}  // namespace hs::vector
