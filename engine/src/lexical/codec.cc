#include "hs/lexical/codec.hpp"

#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HS_HAVE_NEON 1
#else
#define HS_HAVE_NEON 0
#endif

namespace hs::lexical {

const char* codec_name(Codec c) { return c == Codec::VByte ? "vbyte" : "bp128"; }

bool parse_codec(const std::string& s, Codec* out) {
  if (s == "vbyte") *out = Codec::VByte;
  else if (s == "bp128") *out = Codec::BP128;
  else return false;
  return true;
}

namespace {

inline void put_vbyte(uint32_t v, std::string& out) {
  while (v >= 0x80) {
    out.push_back(char((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back(char(v));
}

inline const uint8_t* get_vbyte(const uint8_t* p, uint32_t* v) {
  uint32_t x = p[0];
  if (x < 0x80) {
    *v = x;
    return p + 1;
  }
  x &= 0x7F;
  uint32_t shift = 7;
  ++p;
  while (true) {
    uint32_t b = *p++;
    x |= (b & 0x7F) << shift;
    if (b < 0x80) break;
    shift += 7;
  }
  *v = x;
  return p;
}

const uint8_t* decode_vbyte_block(const uint8_t* p, uint32_t n, uint32_t prev, uint32_t* docs, uint32_t* tfs) {
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t g;
    p = get_vbyte(p, &g);
    prev += g + 1;
    docs[i] = prev;
  }
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t t;
    p = get_vbyte(p, &t);
    tfs[i] = t + 1;
  }
  return p;
}

inline uint32_t bits_needed(uint32_t v) { return v == 0 ? 0 : 32 - uint32_t(__builtin_clz(v)); }

// Vertical 4-lane packing of 128 values into bw*4 32-bit words.
void pack128(const uint32_t* v, uint32_t bw, uint32_t* w) {
  std::memset(w, 0, bw * 16);
  if (bw == 0) return;
  for (uint32_t lane = 0; lane < 4; ++lane) {
    uint32_t bitpos = 0;
    for (uint32_t r = 0; r < 32; ++r, bitpos += bw) {
      uint32_t val = v[r * 4 + lane];
      uint32_t k = bitpos >> 5, sh = bitpos & 31;
      w[k * 4 + lane] |= val << sh;
      if (sh + bw > 32) w[(k + 1) * 4 + lane] |= val >> (32 - sh);
    }
  }
}

template <uint32_t BW>
void unpack128_scalar_t(const uint32_t* w, uint32_t* out) {
  constexpr uint32_t mask = BW == 32 ? 0xFFFFFFFFu : ((1u << BW) - 1);
  for (uint32_t r = 0; r < 32; ++r) {
    const uint32_t bitpos = r * BW, k = bitpos >> 5, sh = bitpos & 31;
    for (uint32_t lane = 0; lane < 4; ++lane) {
      uint32_t val = w[k * 4 + lane] >> sh;
      if (sh + BW > 32) val |= w[(k + 1) * 4 + lane] << (32 - sh);
      out[r * 4 + lane] = val & mask;
    }
  }
}

#if HS_HAVE_NEON
template <uint32_t BW>
void unpack128_neon_t(const uint32_t* w, uint32_t* out) {
  const uint32x4_t mask = vdupq_n_u32(BW == 32 ? 0xFFFFFFFFu : ((1u << BW) - 1));
  for (uint32_t r = 0; r < 32; ++r) {
    const uint32_t bitpos = r * BW, k = bitpos >> 5, sh = bitpos & 31;
    uint32x4_t v = vshlq_u32(vld1q_u32(w + k * 4), vdupq_n_s32(-int32_t(sh)));
    if (sh + BW > 32) v = vorrq_u32(v, vshlq_u32(vld1q_u32(w + (k + 1) * 4), vdupq_n_s32(int32_t(32 - sh))));
    vst1q_u32(out + r * 4, vandq_u32(v, mask));
  }
}
#endif

using Unpacker = void (*)(const uint32_t*, uint32_t*);

template <template <uint32_t> class F, size_t... I>
constexpr auto make_table(std::index_sequence<I...>) {
  return std::array<Unpacker, sizeof...(I)>{F<uint32_t(I)>::fn...};
}
template <uint32_t BW>
struct ScalarFn {
  static constexpr Unpacker fn = &unpack128_scalar_t<BW>;
};
#if HS_HAVE_NEON
template <uint32_t BW>
struct NeonFn {
  static constexpr Unpacker fn = &unpack128_neon_t<BW>;
};
#endif


const auto kScalarUnpack = make_table<ScalarFn>(std::make_index_sequence<33>{});
#if HS_HAVE_NEON
const auto kNeonUnpack = make_table<NeonFn>(std::make_index_sequence<33>{});
#endif

// Prefix sum of (gap_m1 + 1) from prev, in place.
inline void prefix_docs_scalar(uint32_t* d, uint32_t prev) {
  for (uint32_t i = 0; i < kBlockSize; ++i) {
    prev += d[i] + 1;
    d[i] = prev;
  }
}

#if HS_HAVE_NEON
inline void prefix_docs_neon(uint32_t* d, uint32_t prev) {
  const uint32x4_t one = vdupq_n_u32(1), zero = vdupq_n_u32(0);
  uint32x4_t carry = vdupq_n_u32(prev);
  for (uint32_t i = 0; i < kBlockSize; i += 4) {
    uint32x4_t v = vaddq_u32(vld1q_u32(d + i), one);
    v = vaddq_u32(v, vextq_u32(zero, v, 3));
    v = vaddq_u32(v, vextq_u32(zero, v, 2));
    v = vaddq_u32(v, carry);
    vst1q_u32(d + i, v);
    carry = vdupq_laneq_u32(v, 3);
  }
}
#endif

template <bool kNeon>
const uint8_t* decode_bp128(const uint8_t* p, uint32_t prev, uint32_t* docs, uint32_t* tfs) {
  uint32_t bwg = p[0], bwt = p[1];
  if (bwg > 32 || bwt > 32) throw std::runtime_error("bp128: corrupt block header");
  p += 2;
  alignas(16) uint32_t w[128];
  std::memcpy(w, p, bwg * 16);
  p += bwg * 16;
#if HS_HAVE_NEON
  if constexpr (kNeon) {
    kNeonUnpack[bwg](w, docs);
    prefix_docs_neon(docs, prev);
  } else
#endif
  {
    kScalarUnpack[bwg](w, docs);
    prefix_docs_scalar(docs, prev);
  }
  std::memcpy(w, p, bwt * 16);
  p += bwt * 16;
#if HS_HAVE_NEON
  if constexpr (kNeon) {
    kNeonUnpack[bwt](w, tfs);
  } else
#endif
  {
    kScalarUnpack[bwt](w, tfs);
  }
  for (uint32_t i = 0; i < kBlockSize; ++i) tfs[i] += 1;
  return p;
}

}  // namespace

void encode_block(Codec c, const uint32_t* gaps_m1, const uint32_t* tfs_m1, uint32_t n, std::string& out) {
  if (c == Codec::BP128 && n == kBlockSize) {
    uint32_t og = 0, ot = 0;
    for (uint32_t i = 0; i < n; ++i) {
      og |= gaps_m1[i];
      ot |= tfs_m1[i];
    }
    uint32_t bwg = bits_needed(og), bwt = bits_needed(ot);
    out.push_back(char(bwg));
    out.push_back(char(bwt));
    uint32_t w[128];
    pack128(gaps_m1, bwg, w);
    out.append(reinterpret_cast<const char*>(w), bwg * 16);
    pack128(tfs_m1, bwt, w);
    out.append(reinterpret_cast<const char*>(w), bwt * 16);
    return;
  }
  for (uint32_t i = 0; i < n; ++i) put_vbyte(gaps_m1[i], out);
  for (uint32_t i = 0; i < n; ++i) put_vbyte(tfs_m1[i], out);
}

const uint8_t* decode_block(Codec c, const uint8_t* p, uint32_t n, uint32_t prev_doc, uint32_t* docs,
                            uint32_t* tfs) {
  if (c == Codec::BP128 && n == kBlockSize) return decode_bp128<HS_HAVE_NEON != 0>(p, prev_doc, docs, tfs);
  return decode_vbyte_block(p, n, prev_doc, docs, tfs);
}

const uint8_t* decode_block_scalar(Codec c, const uint8_t* p, uint32_t n, uint32_t prev_doc, uint32_t* docs,
                                   uint32_t* tfs) {
  if (c == Codec::BP128 && n == kBlockSize) return decode_bp128<false>(p, prev_doc, docs, tfs);
  return decode_vbyte_block(p, n, prev_doc, docs, tfs);
}

const char* bp128_kernel() { return HS_HAVE_NEON ? "neon" : "scalar"; }

}  // namespace hs::lexical
