#pragma once
// Lucene StandardTokenizer (UAX#29 word boundaries) as a longest-match DFA.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hs::lexical {

struct RawToken {
  uint32_t start = 0;  // byte offsets into the UTF-8 input
  uint32_t end = 0;
};

// Decode one UTF-8 code point at p (p < end). Malformed bytes decode to U+FFFD, length 1.
inline uint32_t utf8_decode(const unsigned char* p, const unsigned char* end, uint32_t* len) {
  unsigned c = p[0];
  if (c < 0x80) {
    *len = 1;
    return c;
  }
  auto cont = [&](int i) { return p + i < end && (p[i] & 0xC0) == 0x80; };
  if (c >= 0xC2 && c <= 0xDF && cont(1)) {
    *len = 2;
    return ((c & 0x1F) << 6) | (p[1] & 0x3F);
  }
  if (c >= 0xE0 && c <= 0xEF && cont(1) && cont(2)) {
    uint32_t cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    if (cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF)) {
      *len = 3;
      return cp;
    }
  }
  if (c >= 0xF0 && c <= 0xF4 && cont(1) && cont(2) && cont(3)) {
    uint32_t cp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    if (cp >= 0x10000 && cp <= 0x10FFFF) {
      *len = 4;
      return cp;
    }
  }
  *len = 1;
  return 0xFFFD;
}

inline void utf8_append(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(char(cp));
  } else if (cp < 0x800) {
    out.push_back(char(0xC0 | (cp >> 6)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(char(0xE0 | (cp >> 12)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(char(0xF0 | (cp >> 18)));
    out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(char(0x80 | (cp & 0x3F)));
  }
}

class StandardTokenizer {
 public:
  // Lucene's default maxTokenLength and JFlex %buffer: 255 UTF-16 code units.
  static constexpr uint32_t kMaxTokenUnits = 255;

  static const StandardTokenizer& instance();

  void tokenize(std::string_view text, std::vector<RawToken>& out) const;

  int num_classes() const { return nclasses_; }
  int num_states() const { return int(accept_.size()); }

 private:
  StandardTokenizer();
  uint8_t cls(uint32_t cp) const {
    return cp < 128 ? ascii_cls_[cp] : stage2_[size_t(stage1_[cp >> 7]) * 128 + (cp & 127)];
  }

  int nclasses_ = 0;
  uint8_t ascii_cls_[128] = {};
  std::vector<uint16_t> stage1_;
  std::vector<uint8_t> stage2_;
  std::vector<int16_t> trans_;  // state * nclasses_ + class -> next state, -1 = dead
  std::vector<uint8_t> accept_;
};

}  // namespace hs::lexical
