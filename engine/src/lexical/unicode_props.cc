#include "unicode_props.hpp"

#include <algorithm>
#include <iterator>

namespace hs::lexical::uni {

namespace {
#include "unicode_data.inc"
}  // namespace

uint32_t prop_mask(uint32_t cp) {
  const PropRange* b = std::begin(kPropRanges);
  const PropRange* e = std::end(kPropRanges);
  const PropRange* it = std::upper_bound(b, e, cp, [](uint32_t c, const PropRange& r) { return c < r.lo; });
  if (it == b) return 0;
  --it;
  return cp <= it->hi ? it->mask : 0;
}

uint32_t to_lower(uint32_t cp) {
  if (cp < 128) return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
  const LowerPair* b = std::begin(kLowerPairs);
  const LowerPair* e = std::end(kLowerPairs);
  const LowerPair* it = std::lower_bound(b, e, cp, [](const LowerPair& p, uint32_t c) { return p.from < c; });
  return (it != e && it->from == cp) ? it->to : cp;
}

void all_masks(uint32_t* out, int* n, int cap) {
  int k = 0;
  out[k++] = 0;
  for (const auto& r : kPropRanges) {
    if (std::find(out, out + k, r.mask) == out + k && k < cap) out[k++] = r.mask;
  }
  *n = k;
}

}  // namespace hs::lexical::uni
