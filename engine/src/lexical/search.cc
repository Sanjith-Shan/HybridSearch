// Query processing: exhaustive TAAT, DAAT, MaxScore, WAND, block-max WAND.
//
// Score definition shared by every algorithm (so their top-k agree bit for bit):
//   contribution c_t(d) = bm25_score_scaled(w_t, tf * inv(d))           a float
//   score(d)            = (float) sum_t c_t(d), summed in a double in the query's canonical
//                         term order (first occurrence), skipping terms absent from d.
// TAAT adds each term's contributions to a double accumulator in that same term order, so
// the double additions happen in the same sequence as DAAT's.
//
// Pruning safety. Upper bounds u_t >= c_t(d) are floats computed with the same float
// operations as the contributions (score is monotone in tf * inv, and inv is monotone in the
// length), so they are true bounds, not approximations. A pruning test sums bounds in a
// double in whatever order is convenient; double rounding over n <= 2^10 terms perturbs a
// sum by a relative 2^-43 at most, far below the 2^-24 of the final float rounding, so
//   prune iff  (float)(bound_sum * (1 + 2^-40)) <= theta
// guarantees (float)(true sum) <= theta. "<=" (not "<") is safe because all DAAT algorithms
// visit candidates in increasing doc order: a later document that ties the current k-th score
// loses the tie (ranks_before prefers the lower ordinal), so it can never enter the heap.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "hs/lexical/index.hpp"

namespace hs::lexical {

namespace {

constexpr uint32_t kEnd = UINT32_MAX;
constexpr double kBoundSlack = 1.0 + 0x1p-40;
constexpr uint32_t kDeadlineEvery = 128;

struct QTerm {
  uint32_t id;
  std::string text;
  uint32_t df;
  float weight;
  float ub;  // term-level upper bound on any contribution
  uint32_t canon;
};

struct Scorer {
  Model model;
  Bm25Norms bn;
  const uint32_t* dl;
  const uint8_t* norm;
  bool exact_bounds;  // query (k1, b) equal the build parameters

  float inv(uint32_t d) const {
    return model == Model::Lucene ? bn.lucene_cache[norm[d]] : bm25_inv_norm(bn.k1, bn.b, float(dl[d]), bn.avgdl);
  }
  float contrib(float w, uint32_t d, uint32_t tf) const { return bm25_score_scaled(w, bm25_scaled_tf(float(tf), inv(d))); }
  float bound(float w, const BoundInfo& b) const {
    if (exact_bounds) {
      int m = int(model);
      return bm25_score_scaled(w, bm25_scaled_tf(float(b.best_tf[m]), bn.inv_for_len(b.best_dl[m])));
    }
    return bm25_score_scaled(w, bm25_scaled_tf(float(b.max_tf), bn.inv_for_len(b.min_dl)));
  }
};

class Cursor {
 public:
  void init(const LexicalIndex& ix, uint32_t term, Codec c, LexicalResult* stats) {
    const auto& t = ix.term(term);
    codec_ = c;
    base_ = ix.postings_base(c) + t.post_off[int(c)];
    df_ = t.df;
    nb_ = (df_ + kBlockSize - 1) / kBlockSize;
    blocks_ = nb_ > 1 ? ix.blocks(term) : nullptr;
    single_.bound = t.bound;
    single_.last_doc = 0;  // filled on first decode
    stats_ = stats;
    blk_ = 0;
    decoded_ = UINT32_MAX;
    decode(0);
    pos_ = 0;
    doc_ = docs_[0];
    if (!blocks_) single_.last_doc = docs_[n_ - 1];
  }

  uint32_t doc() const { return doc_; }
  uint32_t tf() const { return tfs_[pos_]; }

  void next() {
    if (++pos_ < n_) {
      doc_ = docs_[pos_];
    } else if (decoded_ + 1 < nb_) {
      blk_ = decoded_ + 1;
      decode(blk_);
      pos_ = 0;
      doc_ = docs_[0];
    } else {
      doc_ = kEnd;
    }
  }

  // Precondition: target is >= any target given to shallow_next() before.
  void next_geq(uint32_t target) {
    if (doc_ >= target) return;
    uint32_t b = std::max(blk_, decoded_);
    b = find_block(b, target);
    if (b >= nb_) {
      blk_ = nb_;
      doc_ = kEnd;
      return;
    }
    blk_ = b;
    if (b != decoded_) {
      decode(b);
      pos_ = 0;
    }
    while (docs_[pos_] < target) ++pos_;
    doc_ = docs_[pos_];
  }

  // Move the block pointer (not the postings) to the block that may contain target.
  void shallow_next(uint32_t target) { blk_ = find_block(blk_, target); }

  // Bound info and last doc of the block the block pointer is on (nb_ = past the end).
  bool block_valid() const { return blk_ < nb_; }
  const BoundInfo& block_bound() const { return blocks_ ? blocks_[blk_].bound : single_.bound; }
  uint32_t block_last() const { return blk_ >= nb_ ? kEnd - 1 : last_of(blk_); }

 private:
  uint32_t last_of(uint32_t b) const { return blocks_ ? blocks_[b].last_doc : single_.last_doc; }

  uint32_t find_block(uint32_t b, uint32_t target) const {
    if (!blocks_) return (b == 0 && single_.last_doc >= target) ? 0 : nb_;
    // Gallop, then binary search on last_doc.
    uint32_t step = 1, lo = b;
    while (lo < nb_ && blocks_[lo].last_doc < target) {
      uint32_t hi = std::min(nb_, lo + step);
      if (hi == nb_ || blocks_[hi - 1].last_doc >= target) {
        // answer in [lo, hi)
        uint32_t l = lo, h = hi;
        while (l < h) {
          uint32_t m = (l + h) / 2;
          if (blocks_[m].last_doc < target) l = m + 1;
          else h = m;
        }
        return l;
      }
      lo = hi;
      step *= 2;
    }
    return lo;
  }

  void decode(uint32_t b) {
    uint32_t n = std::min<uint32_t>(kBlockSize, df_ - b * kBlockSize);
    uint32_t prev = b == 0 ? UINT32_MAX : last_of(b - 1);
    uint32_t off = blocks_ ? blocks_[b].off[int(codec_)] : 0;
    decode_block(codec_, base_ + off, n, prev, docs_, tfs_);
    n_ = n;
    decoded_ = b;
    stats_->postings_decoded += n;
    stats_->blocks_decoded += 1;
  }

  Codec codec_ = Codec::VByte;
  const uint8_t* base_ = nullptr;
  uint32_t df_ = 0, nb_ = 0;
  const LexicalIndex::BlockInfo* blocks_ = nullptr;
  LexicalIndex::BlockInfo single_;
  LexicalResult* stats_ = nullptr;
  uint32_t blk_ = 0, decoded_ = UINT32_MAX, pos_ = 0, n_ = 0, doc_ = kEnd;
  alignas(16) uint32_t docs_[kBlockSize];
  alignas(16) uint32_t tfs_[kBlockSize];
};

struct Query {
  std::vector<QTerm> terms;  // canonical order
  Scorer sc;
};

inline bool cannot_beat(const TopK& top, double bound_sum) {
  return top.full() && float(bound_sum * kBoundSlack) <= top.threshold();
}

// ---------------------------------------------------------------------------------------------

void run_exhaustive(const LexicalIndex& ix, const Query& q, Codec codec, TopK& top, LexicalResult& res, Deadline& dl) {
  thread_local std::vector<double> acc;
  thread_local std::vector<uint8_t> seen;
  thread_local std::vector<uint32_t> touched;
  if (acc.size() < ix.num_docs()) {
    acc.assign(ix.num_docs(), 0.0);
    seen.assign(ix.num_docs(), 0);
  }
  touched.clear();
  std::vector<uint32_t> docs, tfs;
  docs.resize(kBlockSize);
  tfs.resize(kBlockSize);
  for (const QTerm& t : q.terms) {
    const auto& ti = ix.term(t.id);
    const uint8_t* p = ix.postings_base(codec) + ti.post_off[int(codec)];
    const uint32_t nb = (ti.df + kBlockSize - 1) / kBlockSize;
    uint32_t prev = UINT32_MAX;
    for (uint32_t j = 0; j < nb; ++j) {
      if (dl.expired_every(8)) {
        res.partial = true;
        break;
      }
      uint32_t n = std::min<uint32_t>(kBlockSize, ti.df - j * kBlockSize);
      p = decode_block(codec, p, n, prev, docs.data(), tfs.data());
      prev = docs[n - 1];
      res.postings_decoded += n;
      res.blocks_decoded += 1;
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t d = docs[i];
        if (!seen[d]) {
          seen[d] = 1;
          touched.push_back(d);
        }
        acc[d] += double(q.sc.contrib(t.weight, d, tfs[i]));
      }
    }
    if (res.partial) break;
  }
  for (uint32_t d : touched) {
    top.push(d, float(acc[d]));
    acc[d] = 0.0;
    seen[d] = 0;
  }
  res.docs_scored += touched.size();
}

void run_daat(const LexicalIndex& ix, const Query& q, Codec codec, TopK& top, LexicalResult& res, Deadline& dl) {
  const size_t n = q.terms.size();
  std::vector<Cursor> cur(n);
  for (size_t i = 0; i < n; ++i) cur[i].init(ix, q.terms[i].id, codec, &res);
  while (true) {
    uint32_t d = kEnd;
    for (auto& c : cur) d = std::min(d, c.doc());
    if (d == kEnd) break;
    if (dl.expired_every(kDeadlineEvery)) {
      res.partial = true;
      break;
    }
    double s = 0;
    for (size_t i = 0; i < n; ++i) {
      if (cur[i].doc() == d) {
        s += double(q.sc.contrib(q.terms[i].weight, d, cur[i].tf()));
        cur[i].next();
      }
    }
    top.push(d, float(s));
    ++res.docs_scored;
  }
}

// Canonical sum of per-term contributions (zeros for absent terms add exactly nothing).
inline float canonical_sum(const std::vector<float>& contrib, const std::vector<uint8_t>& hit) {
  double s = 0;
  for (size_t i = 0; i < contrib.size(); ++i)
    if (hit[i]) s += double(contrib[i]);
  return float(s);
}

void run_maxscore(const LexicalIndex& ix, const Query& q, Codec codec, TopK& top, LexicalResult& res, Deadline& dl) {
  const size_t n = q.terms.size();
  std::vector<Cursor> cur(n);
  for (size_t i = 0; i < n; ++i) cur[i].init(ix, q.terms[i].id, codec, &res);
  // Terms by increasing upper bound; prefix[i] = sum of the i+1 smallest bounds.
  std::vector<uint32_t> ord(n);
  for (size_t i = 0; i < n; ++i) ord[i] = uint32_t(i);
  std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return q.terms[a].ub < q.terms[b].ub; });
  std::vector<double> prefix(n);
  double run = 0;
  for (size_t i = 0; i < n; ++i) prefix[i] = run += double(q.terms[ord[i]].ub);
  size_t first_essential = 0;
  auto update_partition = [&] {
    while (first_essential < n && cannot_beat(top, prefix[first_essential])) ++first_essential;
  };
  std::vector<float> contrib(n, 0.0f);
  std::vector<uint8_t> hit(n, 0);
  while (first_essential < n) {
    uint32_t d = kEnd;
    for (size_t i = first_essential; i < n; ++i) d = std::min(d, cur[ord[i]].doc());
    if (d == kEnd) break;
    if (dl.expired_every(kDeadlineEvery)) {
      res.partial = true;
      break;
    }
    std::fill(hit.begin(), hit.end(), 0);
    double partial = 0;
    for (size_t i = first_essential; i < n; ++i) {
      Cursor& c = cur[ord[i]];
      if (c.doc() == d) {
        uint32_t t = ord[i];
        contrib[t] = q.sc.contrib(q.terms[t].weight, d, c.tf());
        hit[t] = 1;
        partial += double(contrib[t]);
        c.next();
      }
    }
    bool pruned = false;
    for (size_t i = first_essential; i-- > 0;) {
      if (cannot_beat(top, partial + prefix[i])) {
        pruned = true;
        break;
      }
      Cursor& c = cur[ord[i]];
      c.next_geq(d);
      if (c.doc() == d) {
        uint32_t t = ord[i];
        contrib[t] = q.sc.contrib(q.terms[t].weight, d, c.tf());
        hit[t] = 1;
        partial += double(contrib[t]);
      }
    }
    if (pruned) continue;
    ++res.docs_scored;
    if (top.push(d, canonical_sum(contrib, hit))) update_partition();
  }
}

// Keeps cursor pointers sorted by current doc (insertion sort: lists are short and nearly sorted).
inline void sort_by_doc(std::vector<Cursor*>& v, std::vector<uint32_t>& idx) {
  for (size_t i = 1; i < v.size(); ++i) {
    Cursor* c = v[i];
    uint32_t ci = idx[i];
    size_t j = i;
    while (j > 0 && v[j - 1]->doc() > c->doc()) {
      v[j] = v[j - 1];
      idx[j] = idx[j - 1];
      --j;
    }
    v[j] = c;
    idx[j] = ci;
  }
}

template <bool kBlockMax>
void run_wand(const LexicalIndex& ix, const Query& q, Codec codec, TopK& top, LexicalResult& res, Deadline& dl) {
  const size_t n = q.terms.size();
  std::vector<Cursor> store(n);
  std::vector<Cursor*> cur(n);
  std::vector<uint32_t> tix(n);  // canonical term index of cur[i]
  for (size_t i = 0; i < n; ++i) {
    store[i].init(ix, q.terms[i].id, codec, &res);
    cur[i] = &store[i];
    tix[i] = uint32_t(i);
  }
  std::vector<float> contrib(n, 0.0f);
  std::vector<uint8_t> hit(n, 0);
  sort_by_doc(cur, tix);
  while (true) {
    if (dl.expired_every(kDeadlineEvery)) {
      res.partial = true;
      break;
    }
    // Pivot: first position where the cumulative term bound could beat the threshold.
    double acc = 0;
    size_t p = n;
    for (size_t i = 0; i < n; ++i) {
      if (cur[i]->doc() == kEnd) break;
      acc += double(q.terms[tix[i]].ub);
      if (!cannot_beat(top, acc)) {
        p = i;
        break;
      }
    }
    if (p == n) break;
    const uint32_t pivot = cur[p]->doc();
    while (p + 1 < n && cur[p + 1]->doc() == pivot) ++p;

    if constexpr (kBlockMax) {
      double bsum = 0;
      for (size_t i = 0; i <= p; ++i) {
        cur[i]->shallow_next(pivot);
        if (cur[i]->block_valid()) bsum += double(q.sc.bound(q.terms[tix[i]].weight, cur[i]->block_bound()));
      }
      if (cannot_beat(top, bsum)) {
        // Skip past the smallest block boundary among the pivot lists (or the next list's doc).
        uint32_t next = kEnd;
        size_t adv = p;
        float adv_ub = q.terms[tix[p]].ub;
        for (size_t i = 0; i <= p; ++i) {
          next = std::min(next, cur[i]->block_last() + 1);
          if (q.terms[tix[i]].ub > adv_ub) {
            adv_ub = q.terms[tix[i]].ub;
            adv = i;
          }
        }
        if (p + 1 < n) next = std::min(next, cur[p + 1]->doc());
        if (next <= pivot) next = pivot + 1;
        cur[adv]->next_geq(next);
        sort_by_doc(cur, tix);
        continue;
      }
    }

    if (cur[0]->doc() == pivot) {
      std::fill(hit.begin(), hit.end(), 0);
      for (size_t i = 0; i < n && cur[i]->doc() == pivot; ++i) {
        uint32_t t = tix[i];
        contrib[t] = q.sc.contrib(q.terms[t].weight, pivot, cur[i]->tf());
        hit[t] = 1;
      }
      ++res.docs_scored;
      top.push(pivot, canonical_sum(contrib, hit));
      for (size_t i = 0; i < n && cur[i]->doc() == pivot; ++i) cur[i]->next();
    } else {
      size_t a = p;
      while (cur[a]->doc() == pivot) --a;  // the last list before the pivot's run
      cur[a]->next_geq(pivot);
    }
    sort_by_doc(cur, tix);
  }
}

}  // namespace

LexicalResult LexicalIndex::search(const std::vector<std::string>& raw_terms, const SearchOptions& opt,
                                   Deadline& deadline) const {
  LexicalResult res;
  Codec codec = opt.codec;
  if (!has_codec(codec)) codec = has_codec(Codec::BP128) ? Codec::BP128 : Codec::VByte;

  // Anserini BagOfWordsQueryGenerator: one clause per distinct term, boost = its count.
  Query q;
  {
    std::vector<std::pair<std::string, uint32_t>> uniq;
    for (const auto& t : raw_terms) {
      auto it = std::find_if(uniq.begin(), uniq.end(), [&](const auto& u) { return u.first == t; });
      if (it == uniq.end()) uniq.emplace_back(t, 1);
      else ++it->second;
    }
    q.sc.model = opt.model;
    q.sc.bn.init(opt.model, opt.k1, opt.b, avgdl_);
    q.sc.dl = doclen_.data();
    q.sc.norm = norms_.data();
    q.sc.exact_bounds = opt.k1 == build_k1_ && opt.b == build_b_;
    for (auto& [text, boost] : uniq) {
      int64_t id = term_id(text);
      if (id < 0) continue;
      QTerm qt;
      qt.id = uint32_t(id);
      qt.text = text;
      qt.df = idf_df(uint32_t(id));
      float idf = bm25_idf(qt.df, idf_n_);
      qt.weight = bm25_weight(opt.model, float(boost), idf, opt.k1);
      qt.ub = q.sc.bound(qt.weight, terms_[id].bound);
      qt.canon = uint32_t(q.terms.size());
      q.terms.push_back(std::move(qt));
    }
  }
  if (q.terms.empty() || opt.k == 0) return res;

  TopK top(opt.k);
  switch (opt.algorithm) {
    case Algorithm::Exhaustive: run_exhaustive(*this, q, codec, top, res, deadline); break;
    case Algorithm::DAAT: run_daat(*this, q, codec, top, res, deadline); break;
    case Algorithm::MaxScore: run_maxscore(*this, q, codec, top, res, deadline); break;
    case Algorithm::WAND: run_wand<false>(*this, q, codec, top, res, deadline); break;
    case Algorithm::BMW: run_wand<true>(*this, q, codec, top, res, deadline); break;
  }
  res.hits = top.take_sorted();

  if (opt.explain && !res.hits.empty()) {
    res.explain.assign(res.hits.size(), {});
    std::vector<size_t> by_doc(res.hits.size());
    for (size_t i = 0; i < by_doc.size(); ++i) by_doc[i] = i;
    std::sort(by_doc.begin(), by_doc.end(), [&](size_t a, size_t b) { return res.hits[a].doc < res.hits[b].doc; });
    LexicalResult scratch;
    for (const QTerm& t : q.terms) {
      Cursor c;
      c.init(*this, t.id, codec, &scratch);
      for (size_t hi : by_doc) {
        uint32_t d = res.hits[hi].doc;
        c.next_geq(d);
        if (c.doc() == d)
          res.explain[hi].push_back({t.text, q.sc.contrib(t.weight, d, c.tf()), c.tf(), t.df});
      }
    }
  }
  return res;
}

LexicalResult LexicalIndex::search_text(std::string_view query, const SearchOptions& opt, Deadline& deadline) const {
  return search(analyzer_.analyze(query), opt, deadline);
}

}  // namespace hs::lexical
