#pragma once
// Distance kernels. Embeddings are L2-normalised, so ranking by inner product and
// by squared L2 agree (||a-b||^2 = 2 - 2<a,b>). The graph code uses squared L2 so
// that RobustPrune's alpha test has its geometric meaning on any data; results are
// reported as inner product.
//
// Three paths, chosen at compile time:
//   NEON (aarch64, 4 x float32x4 accumulators, 16 floats / iteration)
//   AVX2+FMA (x86-64, 2 x __m256 accumulators, 16 floats / iteration)
//   scalar (anything else; also the reference the SIMD paths are tested against)

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define HS_SIMD_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HS_SIMD_AVX2 1
#endif

namespace hs::vector {

inline float ip_scalar(const float* a, const float* b, size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

inline float l2sq_scalar(const float* a, const float* b, size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

#if defined(HS_SIMD_NEON)
inline float ip(const float* a, const float* b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= d; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < d; ++i) s += a[i] * b[i];
  return s;
}
inline float l2sq(const float* a, const float* b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    float32x4_t t0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t t1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    float32x4_t t2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    float32x4_t t3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, t0, t0);
    s1 = vfmaq_f32(s1, t1, t1);
    s2 = vfmaq_f32(s2, t2, t2);
    s3 = vfmaq_f32(s3, t3, t3);
  }
  for (; i + 4 <= d; i += 4) {
    float32x4_t t = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, t, t);
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}
inline const char* simd_name() { return "neon"; }
#elif defined(HS_SIMD_AVX2)
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
inline float ip(const float* a, const float* b, size_t d) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  float s = hsum256(_mm256_add_ps(s0, s1));
  for (; i < d; ++i) s += a[i] * b[i];
  return s;
}
inline float l2sq(const float* a, const float* b, size_t d) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 t1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    s0 = _mm256_fmadd_ps(t0, t0, s0);
    s1 = _mm256_fmadd_ps(t1, t1, s1);
  }
  float s = hsum256(_mm256_add_ps(s0, s1));
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}
inline const char* simd_name() { return "avx2"; }
#else
inline float ip(const float* a, const float* b, size_t d) { return ip_scalar(a, b, d); }
inline float l2sq(const float* a, const float* b, size_t d) { return l2sq_scalar(a, b, d); }
inline const char* simd_name() { return "scalar"; }
#endif

// Pull a whole vector toward L1 ahead of use. Graph search touches vectors in a
// data-dependent order the hardware prefetcher cannot predict.
inline void prefetch_vector(const float* p, size_t d) {
  const char* c = reinterpret_cast<const char*>(p);
  for (size_t off = 0; off < d * sizeof(float); off += 64) __builtin_prefetch(c + off, 0, 3);
}

}  // namespace hs::vector
