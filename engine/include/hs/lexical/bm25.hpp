#pragma once
// BM25, two ways.
//
// Lucene (org.apache.lucene.search.similarities.BM25Similarity, Lucene 8+ incl. 10.5):
//   idf    = (float) ln(1 + (N - df + 0.5) / (df + 0.5))            N = docs with >= 1 term
//   avgdl  = (float) (sumTotalTermFreq / (double) N)
//   norm   = SmallFloat.intToByte4(dl)                                one lossy byte per doc
//   inv[n] = 1f / (k1 * ((1 - b) + b * byte4ToInt(n) / avgdl))       256-entry cache
//   w      = boost * idf                                              boost = query term count
//   score  = w - w / (1f + tf * inv[norm])                            = w * tf / (tf + K)
// There is no (k1 + 1) numerator factor since Lucene 8. Per-document scores are the float
// contributions summed in double and rounded to float (BooleanScorer / MaxScoreBulkScorer).
//
// Textbook (Robertson et al.): the same, but with the exact document length and the
// classic (k1 + 1) factor: w = boost * idf * (k1 + 1), inv = 1 / (k1 * ((1 - b) + b * dl / avgdl)).
// The (k1 + 1) factor is a per-query constant and cannot change a ranking; the only
// ranking difference between the two models is the one-byte length quantisation.
//
// All arithmetic is IEEE single precision in Java's evaluation order, with floating-point
// contraction (FMA) disabled, so the Lucene model reproduces Lucene's scores bit for bit.

#include <cmath>
#include <cstdint>

#if defined(__clang__)
#define HS_STRICT_FP _Pragma("clang fp contract(off)")
#else
#define HS_STRICT_FP
#endif

namespace hs::lexical {

enum class Model : uint8_t { Lucene = 0, Textbook = 1 };
inline const char* model_name(Model m) { return m == Model::Lucene ? "lucene" : "textbook"; }

// Anserini's defaults for MS MARCO passage.
constexpr float kDefaultK1 = 0.9f;
constexpr float kDefaultB = 0.4f;

// org.apache.lucene.util.SmallFloat (Lucene 8+).
namespace smallfloat {
inline uint32_t long_to_int4(uint64_t i) {
  int num_bits = 64 - (i == 0 ? 64 : __builtin_clzll(i));
  if (num_bits < 4) return uint32_t(i);
  int shift = num_bits - 4;
  uint32_t encoded = uint32_t(i >> shift);
  encoded &= 0x07;
  encoded |= uint32_t(shift + 1) << 3;
  return encoded;
}
inline uint64_t int4_to_long(uint32_t i) {
  uint64_t bits = i & 0x07;
  int shift = int(i >> 3) - 1;
  return shift == -1 ? bits : (bits | 0x08) << shift;
}
constexpr uint32_t kNumFreeValues = 255 - 231;  // 255 - longToInt4(Integer.MAX_VALUE)
inline uint8_t int_to_byte4(uint32_t i) {
  if (i > 0x7FFFFFFFu) i = 0x7FFFFFFFu;
  if (i < kNumFreeValues) return uint8_t(i);
  return uint8_t(kNumFreeValues + long_to_int4(i - kNumFreeValues));
}
inline uint32_t byte4_to_int(uint8_t b) {
  if (b < kNumFreeValues) return b;
  return uint32_t(kNumFreeValues + int4_to_long(b - kNumFreeValues));
}
}  // namespace smallfloat

inline float bm25_idf(uint64_t df, uint64_t n_docs) {
  return float(std::log(1.0 + (double(int64_t(n_docs) - int64_t(df)) + 0.5) / (double(df) + 0.5)));
}

inline float bm25_avgdl(uint64_t sum_dl, uint64_t n_docs) { return float(double(sum_dl) / double(n_docs)); }

// 1 / K for a document of (possibly quantised) length len.
inline float bm25_inv_norm(float k1, float b, float len, float avgdl) {
  HS_STRICT_FP
  return 1.0f / (k1 * ((1.0f - b) + b * len / avgdl));
}

inline float bm25_weight(Model m, float boost, float idf, float k1) {
  HS_STRICT_FP
  float w = boost * idf;
  return m == Model::Lucene ? w : w * (k1 + 1.0f);
}

// tf * inv_norm: the score is monotone in this product for any positive weight (Lucene's
// argument: float * and / round monotonically, and 1 + x, 1 - 1/x preserve order).
inline float bm25_scaled_tf(float tf, float inv_norm) {
  HS_STRICT_FP
  return tf * inv_norm;
}

inline float bm25_score_scaled(float weight, float scaled_tf) {
  HS_STRICT_FP
  return weight - weight / (1.0f + scaled_tf);
}

inline float bm25_term_score(float weight, float tf, float inv_norm) {
  return bm25_score_scaled(weight, bm25_scaled_tf(tf, inv_norm));
}

// Precomputed per-(k1, b) state for one index.
struct Bm25Norms {
  Model model = Model::Lucene;
  float k1 = kDefaultK1, b = kDefaultB, avgdl = 1.0f;
  float lucene_cache[256] = {};

  void init(Model m, float k1_, float b_, float avgdl_) {
    model = m;
    k1 = k1_;
    b = b_;
    avgdl = avgdl_;
    for (int i = 0; i < 256; ++i)
      lucene_cache[i] = bm25_inv_norm(k1, b, float(smallfloat::byte4_to_int(uint8_t(i))), avgdl);
  }
  // inv norm for a document with exact length dl and Lucene norm byte `norm`.
  float inv(uint32_t dl, uint8_t norm) const {
    return model == Model::Lucene ? lucene_cache[norm] : bm25_inv_norm(k1, b, float(dl), avgdl);
  }
  float inv_for_len(uint32_t dl) const { return inv(dl, smallfloat::int_to_byte4(dl)); }
};

}  // namespace hs::lexical
