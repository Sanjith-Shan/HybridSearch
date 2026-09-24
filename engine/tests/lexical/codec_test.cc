#include <gtest/gtest.h>

#include <random>

#include "hs/lexical/codec.hpp"

namespace hs::lexical {
namespace {

void roundtrip(Codec c, const std::vector<uint32_t>& docs, const std::vector<uint32_t>& tfs, bool scalar) {
  std::string enc;
  std::vector<uint32_t> gaps(docs.size()), tfm1(docs.size());
  uint32_t prev = UINT32_MAX;
  for (size_t i = 0; i < docs.size(); ++i) {
    gaps[i] = docs[i] - prev - 1;
    prev = docs[i];
    tfm1[i] = tfs[i] - 1;
  }
  std::vector<size_t> offs;
  for (size_t lo = 0; lo < docs.size(); lo += kBlockSize) {
    offs.push_back(enc.size());
    size_t n = std::min<size_t>(kBlockSize, docs.size() - lo);
    encode_block(c, gaps.data() + lo, tfm1.data() + lo, uint32_t(n), enc);
  }
  const uint8_t* p = reinterpret_cast<const uint8_t*>(enc.data());
  prev = UINT32_MAX;
  uint32_t d[kBlockSize], t[kBlockSize];
  for (size_t b = 0, lo = 0; lo < docs.size(); ++b, lo += kBlockSize) {
    ASSERT_EQ(size_t(p - reinterpret_cast<const uint8_t*>(enc.data())), offs[b]);
    size_t n = std::min<size_t>(kBlockSize, docs.size() - lo);
    p = scalar ? decode_block_scalar(c, p, uint32_t(n), prev, d, t) : decode_block(c, p, uint32_t(n), prev, d, t);
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(d[i], docs[lo + i]) << codec_name(c) << " posting " << lo + i;
      ASSERT_EQ(t[i], tfs[lo + i]) << codec_name(c) << " posting " << lo + i;
    }
    prev = d[n - 1];
  }
  EXPECT_EQ(size_t(p - reinterpret_cast<const uint8_t*>(enc.data())), enc.size());
}

TEST(Codec, RoundTripEveryBitWidth) {
  std::mt19937 rng(42);
  for (Codec c : {Codec::VByte, Codec::BP128}) {
    for (int bw = 0; bw <= 31; ++bw) {
      for (bool scalar : {false, true}) {
        std::vector<uint32_t> docs, tfs;
        uint32_t doc = uint32_t(rng() % 5);
        const uint64_t max_gap = bw == 0 ? 0 : ((1ull << bw) - 1);
        for (int i = 0; i < 3 * 128 + 17; ++i) {
          if (i) doc += 1 + uint32_t(max_gap ? rng() % (max_gap + 1) : 0);
          docs.push_back(doc);
          tfs.push_back(1 + uint32_t(rng() % (1u << std::min(bw, 20))));
          if (doc > (UINT32_MAX >> 1)) break;
        }
        roundtrip(c, docs, tfs, scalar);
      }
    }
  }
}

TEST(Codec, FullWidthGapsAndTfs) {
  // 32-bit field widths: a gap of 2^31 and huge tfs inside a full block.
  std::vector<uint32_t> docs, tfs;
  for (uint32_t i = 0; i < 128; ++i) {
    docs.push_back(i == 127 ? 0xF0000000u : i);
    tfs.push_back(i % 2 ? 0xFFFFFFF0u : 1);
  }
  for (Codec c : {Codec::VByte, Codec::BP128}) {
    roundtrip(c, docs, tfs, false);
    roundtrip(c, docs, tfs, true);
  }
}

TEST(Codec, RandomListsNeonEqualsScalar) {
  std::mt19937 rng(7);
  for (int trial = 0; trial < 200; ++trial) {
    size_t n = 1 + rng() % 2000;
    std::vector<uint32_t> docs, tfs;
    uint32_t doc = rng() % 1000;
    for (size_t i = 0; i < n; ++i) {
      if (i) doc += 1 + (rng() % 4 == 0 ? rng() % 100000 : rng() % 30);
      docs.push_back(doc);
      tfs.push_back(1 + (rng() % 10 == 0 ? rng() % 500 : rng() % 3));
    }
    roundtrip(Codec::BP128, docs, tfs, false);
    roundtrip(Codec::BP128, docs, tfs, true);
    roundtrip(Codec::VByte, docs, tfs, false);
  }
}

}  // namespace
}  // namespace hs::lexical
