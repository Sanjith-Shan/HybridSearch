#pragma once
// Unicode properties used by the StandardTokenizer grammar, plus Java's simple lowercase
// mapping. Tables are generated from the UCD by py/hybridsearch/lexical/gen_unicode_tables.py.

#include <cstdint>

namespace hs::lexical::uni {

struct PropRange {
  uint32_t lo, hi, mask;
};
struct LowerPair {
  uint32_t from, to;
};

// Must match BITS in gen_unicode_tables.py.
enum Prop : uint32_t {
  WB_Format = 1u << 0,
  WB_Extend = 1u << 1,
  WB_ZWJ = 1u << 2,
  WB_Regional_Indicator = 1u << 3,
  WB_ALetter = 1u << 4,
  WB_Hebrew_Letter = 1u << 5,
  WB_Numeric = 1u << 6,
  WB_Katakana = 1u << 7,
  WB_MidLetter = 1u << 8,
  WB_MidNumLet = 1u << 9,
  WB_Single_Quote = 1u << 10,
  WB_MidNum = 1u << 11,
  WB_ExtendNumLet = 1u << 12,
  WB_Double_Quote = 1u << 13,
  SC_Hangul = 1u << 14,
  SC_Han = 1u << 15,
  SC_Hiragana = 1u << 16,
  LB_SA = 1u << 17,
  Emoji = 1u << 18,
  Emoji_Modifier = 1u << 19,
  Emoji_Modifier_Base = 1u << 20,
  Extended_Pictographic = 1u << 21,
  KeyCapBase = 1u << 22,
  AccidentalEmoji = 1u << 23,
  VS15 = 1u << 24,
  VS16 = 1u << 25,
  KeyCap = 1u << 26,
  TagSpec = 1u << 27,
  TagTerm = 1u << 28,
};

// Property bit mask of a code point (0 for anything the grammar does not name).
uint32_t prop_mask(uint32_t cp);

// java.lang.Character.toLowerCase(int) for Unicode 15.0.
uint32_t to_lower(uint32_t cp);

// Every distinct mask in the table (plus 0), for building the tokenizer alphabet.
void all_masks(uint32_t* out, int* n, int cap);

}  // namespace hs::lexical::uni
