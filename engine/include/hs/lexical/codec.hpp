#pragma once
// Posting-list codecs. A posting list is cut into blocks of kBlockSize postings; every
// block stores (doc gap - 1) and (tf - 1) for its postings. The first gap of a term is
// relative to "doc -1", so it equals the first doc ID.
//
//   VByte  every block: gaps then tfs, LEB128 varints (7 bits/byte, high bit = more).
//   BP128  full blocks: [u8 bw_gap][u8 bw_tf][bw_gap*16 bytes][bw_tf*16 bytes], each field
//          bit-packed at a fixed width in the 4-lane vertical layout of SIMD-BP128
//          (Lemire & Boytsov 2015): value i lives in lane i%4, row i/4, so one 128-bit load
//          yields four values. Decoded with NEON on ARM, scalar elsewhere; D1 gaps with an
//          in-register prefix sum. The tail block (< 128 postings) uses VByte, as Lucene does.

#include <cstdint>
#include <string>

namespace hs::lexical {

constexpr uint32_t kBlockSize = 128;

enum class Codec : uint8_t { VByte = 0, BP128 = 1 };
constexpr int kNumCodecs = 2;
const char* codec_name(Codec c);
bool parse_codec(const std::string& s, Codec* out);

// Appends one encoded block. gaps_m1[i] = doc[i] - doc[i-1] - 1, tfs_m1[i] = tf[i] - 1.
void encode_block(Codec c, const uint32_t* gaps_m1, const uint32_t* tfs_m1, uint32_t n, std::string& out);

// Decodes one block of n postings starting at p. prev_doc is the doc before the block
// (UINT32_MAX for the first block of a term; arithmetic wraps). Returns the end of the block.
const uint8_t* decode_block(Codec c, const uint8_t* p, uint32_t n, uint32_t prev_doc, uint32_t* docs,
                            uint32_t* tfs);

// Which BP128 unpacker this build uses ("neon" or "scalar").
const char* bp128_kernel();

// Exposed for tests: force the scalar path even where NEON exists.
const uint8_t* decode_block_scalar(Codec c, const uint8_t* p, uint32_t n, uint32_t prev_doc, uint32_t* docs,
                                   uint32_t* tfs);

}  // namespace hs::lexical
