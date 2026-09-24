#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hs::lexical {

inline void put_varint(uint64_t v, std::string& out) {
  while (v >= 0x80) {
    out.push_back(char((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back(char(v));
}

struct VarintReader {
  const uint8_t* p;
  const uint8_t* end;
  uint64_t next() {
    uint64_t x = 0;
    int shift = 0;
    while (true) {
      if (p >= end) throw std::runtime_error("varint: truncated input");
      uint8_t b = *p++;
      x |= uint64_t(b & 0x7F) << shift;
      if (b < 0x80) return x;
      shift += 7;
      if (shift > 63) throw std::runtime_error("varint: overlong");
    }
  }
  uint32_t next32() {
    uint64_t v = next();
    if (v > 0xFFFFFFFFull) throw std::runtime_error("varint: value exceeds 32 bits");
    return uint32_t(v);
  }
};

}  // namespace hs::lexical
