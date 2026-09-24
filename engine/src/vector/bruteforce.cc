#include "hs/vector/bruteforce.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "hs/vector/distance.hpp"
#include "hs/vector/parallel.hpp"

namespace hs::vector {

std::vector<hs::ScoredDoc> exact_topk(const float* base, uint32_t n, const float* queries,
                                      uint32_t nq, uint32_t dim, uint32_t k, unsigned threads) {
  constexpr uint32_t kBlock = 32;
  std::vector<hs::ScoredDoc> out(size_t(nq) * k);
  uint32_t nblocks = (nq + kBlock - 1) / kBlock;
  parallel_for(0, nblocks, threads, [&](size_t b, unsigned) {
    uint32_t q0 = uint32_t(b) * kBlock, q1 = std::min(nq, q0 + kBlock);
    std::vector<hs::TopK> tops;
    tops.reserve(q1 - q0);
    for (uint32_t q = q0; q < q1; ++q) tops.emplace_back(k);
    std::vector<float> thr(q1 - q0, -std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < n; ++i) {
      const float* x = base + size_t(i) * dim;
      for (uint32_t q = q0; q < q1; ++q) {
        float s = ip(queries + size_t(q) * dim, x, dim);
        if (s >= thr[q - q0]) {
          tops[q - q0].push(i, s);
          thr[q - q0] = tops[q - q0].threshold();
        }
      }
    }
    for (uint32_t q = q0; q < q1; ++q) {
      auto v = tops[q - q0].take_sorted();
      for (uint32_t j = 0; j < k; ++j)
        out[size_t(q) * k + j] = j < v.size() ? v[j] : hs::ScoredDoc{0xFFFFFFFFu, 0.f};
    }
  });
  return out;
}

void write_groundtruth(const std::string& path, const std::vector<hs::ScoredDoc>& hits, uint32_t nq,
                       uint32_t k) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  std::vector<uint32_t> ids(size_t(nq) * k);
  std::vector<float> sc(size_t(nq) * k);
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = hits[i].doc;
    sc[i] = hits[i].score;
  }
  bool ok = std::fwrite(&nq, 4, 1, f) == 1 && std::fwrite(&k, 4, 1, f) == 1 &&
            std::fwrite(ids.data(), 4, ids.size(), f) == ids.size() &&
            std::fwrite(sc.data(), 4, sc.size(), f) == sc.size();
  std::fclose(f);
  if (!ok) throw std::runtime_error("short write " + path);
}

GroundTruth read_groundtruth(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  GroundTruth g;
  bool ok = std::fread(&g.nq, 4, 1, f) == 1 && std::fread(&g.k, 4, 1, f) == 1;
  if (ok) {
    g.ids.resize(size_t(g.nq) * g.k);
    g.scores.resize(size_t(g.nq) * g.k);
    ok = std::fread(g.ids.data(), 4, g.ids.size(), f) == g.ids.size() &&
         std::fread(g.scores.data(), 4, g.scores.size(), f) == g.scores.size();
  }
  std::fclose(f);
  if (!ok) throw std::runtime_error("truncated ground truth " + path);
  return g;
}

double recall_at_k(const GroundTruth& gt, const std::vector<std::vector<uint32_t>>& approx, uint32_t k) {
  if (k > gt.k) throw std::runtime_error("recall@k with k > ground-truth depth");
  if (approx.size() > gt.nq) throw std::runtime_error("more queries than ground truth rows");
  double total = 0;
  for (size_t q = 0; q < approx.size(); ++q) {
    std::unordered_set<uint32_t> truth(gt.row(q), gt.row(q) + k);
    uint32_t hit = 0;
    for (size_t j = 0; j < std::min<size_t>(k, approx[q].size()); ++j) hit += truth.count(approx[q][j]);
    total += double(hit) / k;
  }
  return approx.empty() ? 0.0 : total / double(approx.size());
}

}  // namespace hs::vector
