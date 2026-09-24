// StandardTokenizer: Lucene 10's StandardTokenizerImpl.jflex grammar, transcribed rule by rule
// into a small regular-expression AST, compiled with Thompson's construction and the subset
// construction into a DFA over property equivalence classes, and run with JFlex's
// longest-match semantics.
//
// Only the length of the longest match matters for analysis (token types are not used by
// any downstream filter), so every token rule is unioned into one automaton; a position
// where no token rule matches is skipped one code point at a time (the grammar's `[^]` rule).

#include "tokenizer.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "unicode_props.hpp"

namespace hs::lexical {

namespace {

using namespace uni;

// A character-set predicate: (mask & all) == all && (any == 0 || mask & any) && !(mask & none).
struct Pred {
  uint32_t all = 0, any = 0, none = 0;
  bool matches(uint32_t m) const {
    return (m & all) == all && (any == 0 || (m & any) != 0) && (m & none) == 0;
  }
};

struct Node;
using N = std::shared_ptr<const Node>;
struct Node {
  enum Kind { Sym, Cat, Alt, Star, Plus, Opt } kind;
  Pred pred;
  std::vector<N> kids;
};

N sym(Pred p) { return std::make_shared<Node>(Node{Node::Sym, p, {}}); }
N any_of(uint32_t any, uint32_t none = 0) { return sym(Pred{0, any, none}); }
template <class... T>
N cat(T... k) { return std::make_shared<Node>(Node{Node::Cat, {}, {k...}}); }
template <class... T>
N alt(T... k) { return std::make_shared<Node>(Node{Node::Alt, {}, {k...}}); }
N star(N k) { return std::make_shared<Node>(Node{Node::Star, {}, {k}}); }
N plus(N k) { return std::make_shared<Node>(Node{Node::Plus, {}, {k}}); }
N opt(N k) { return std::make_shared<Node>(Node{Node::Opt, {}, {k}}); }

// The grammar. Names follow StandardTokenizerImpl.jflex.
N build_grammar() {
  // UAX#29 WB4: X (Extend | Format | ZWJ)* --> X
  N ExtFmtZwj = star(any_of(WB_Format | WB_Extend | WB_ZWJ));
  auto Ex = [&](N x) { return cat(x, ExtFmtZwj); };

  // --- Emoji macros ---
  N ExtFmtZwjSansPresSel = star(any_of(WB_Format | WB_Extend | WB_ZWJ, VS15 | VS16));
  N KeyCapBaseCharEx = cat(any_of(KeyCapBase), ExtFmtZwjSansPresSel);
  N KeyCapEx = cat(any_of(KeyCap), ExtFmtZwjSansPresSel);
  const uint32_t EmojiRKAM = WB_Regional_Indicator | KeyCapBase | AccidentalEmoji | Emoji_Modifier;
  N EmojiChar = alt(any_of(Extended_Pictographic), any_of(Emoji, EmojiRKAM));
  N EmojiCharEx = cat(EmojiChar, ExtFmtZwjSansPresSel);
  N EmojiModifierBaseEx = cat(any_of(Emoji_Modifier_Base), ExtFmtZwjSansPresSel);
  N EmojiModifierEx = cat(any_of(Emoji_Modifier), ExtFmtZwjSansPresSel);
  N EmojiPresentationSelector = any_of(VS16);
  N ZWJ = any_of(WB_ZWJ);
  auto EmojiCharOrPresSeqOrModSeq = [&] {
    return alt(cat(star(ZWJ), EmojiCharEx, opt(EmojiPresentationSelector)),
               cat(opt(cat(star(ZWJ), EmojiModifierBaseEx)), EmojiModifierEx));
  };
  N RegionalIndicatorEx = Ex(any_of(WB_Regional_Indicator));
  N emoji = alt(cat(EmojiCharOrPresSeqOrModSeq(),
                    alt(star(cat(ZWJ, EmojiCharOrPresSeqOrModSeq())), cat(plus(any_of(TagSpec)), any_of(TagTerm)))),
                cat(KeyCapBaseCharEx, opt(EmojiPresentationSelector), KeyCapEx),
                cat(RegionalIndicatorEx, RegionalIndicatorEx));

  // --- Word macros ---
  N HangulEx = Ex(sym(Pred{SC_Hangul, WB_ALetter | WB_Hebrew_Letter, 0}));
  N AHLetterEx = Ex(any_of(WB_ALetter | WB_Hebrew_Letter));
  N NumericEx = Ex(any_of(WB_Numeric));
  N KatakanaEx = Ex(any_of(WB_Katakana));
  N MidLetterEx = Ex(any_of(WB_MidLetter | WB_MidNumLet | WB_Single_Quote));
  N MidNumericEx = Ex(any_of(WB_MidNum | WB_MidNumLet | WB_Single_Quote));
  N ExtendNumLetEx = Ex(any_of(WB_ExtendNumLet));
  N HanEx = Ex(any_of(SC_Han));
  N HiraganaEx = Ex(any_of(SC_Hiragana));
  N SingleQuoteEx = Ex(any_of(WB_Single_Quote));
  N DoubleQuoteEx = Ex(any_of(WB_Double_Quote));
  N HebrewLetterEx = Ex(any_of(WB_Hebrew_Letter));
  N ComplexContextEx = Ex(any_of(LB_SA));

  // WB8, WB11, WB12, WB13a, WB13b
  N numeric = cat(star(ExtendNumLetEx), NumericEx,
                  star(cat(alt(star(ExtendNumLetEx), MidNumericEx), NumericEx)), star(ExtendNumLetEx));
  N hangul = plus(HangulEx);
  N katakana = plus(KatakanaEx);

  // WB5-WB7c, WB9, WB10, WB13, WB13a, WB13b
  auto inner = [&] {
    return alt(cat(KatakanaEx, star(cat(star(ExtendNumLetEx), KatakanaEx))),
               plus(alt(cat(HebrewLetterEx, alt(SingleQuoteEx, cat(DoubleQuoteEx, HebrewLetterEx))),
                        cat(NumericEx, star(cat(alt(star(ExtendNumLetEx), MidNumericEx), NumericEx))),
                        cat(AHLetterEx, star(cat(alt(star(ExtendNumLetEx), MidLetterEx), AHLetterEx))))));
  };
  N word = cat(star(ExtendNumLetEx), inner(), star(cat(plus(ExtendNumLetEx), inner())), star(ExtendNumLetEx));

  N south_east_asian = plus(ComplexContextEx);
  return alt(emoji, numeric, hangul, katakana, word, south_east_asian, HanEx, HiraganaEx);
}

struct Nfa {
  struct State {
    int pred = -1;  // index into preds, or -1 for an epsilon-only state
    int next = -1;
    std::vector<int> eps;
  };
  std::vector<State> st;
  std::vector<Pred> preds;
  int add() {
    st.emplace_back();
    return int(st.size()) - 1;
  }
  // Returns (start, end); end has no outgoing edges yet.
  std::pair<int, int> compile(const Node& n) {
    switch (n.kind) {
      case Node::Sym: {
        int s = add(), e = add();
        preds.push_back(n.pred);
        st[s].pred = int(preds.size()) - 1;
        st[s].next = e;
        return {s, e};
      }
      case Node::Cat: {
        auto [s, e] = compile(*n.kids[0]);
        for (size_t i = 1; i < n.kids.size(); ++i) {
          auto [s2, e2] = compile(*n.kids[i]);
          st[e].eps.push_back(s2);
          e = e2;
        }
        return {s, e};
      }
      case Node::Alt: {
        int s = add(), e = add();
        for (const auto& k : n.kids) {
          auto [ks, ke] = compile(*k);
          st[s].eps.push_back(ks);
          st[ke].eps.push_back(e);
        }
        return {s, e};
      }
      case Node::Star:
      case Node::Plus:
      case Node::Opt: {
        int s = add(), e = add();
        auto [ks, ke] = compile(*n.kids[0]);
        st[s].eps.push_back(ks);
        if (n.kind != Node::Plus) st[s].eps.push_back(e);
        st[ke].eps.push_back(e);
        if (n.kind != Node::Opt) st[ke].eps.push_back(ks);
        return {s, e};
      }
    }
    throw std::logic_error("bad node");
  }
  void closure(std::vector<int>& set) const {
    std::vector<char> seen(st.size(), 0);
    std::vector<int> stack(set.begin(), set.end());
    for (int s : set) seen[s] = 1;
    while (!stack.empty()) {
      int s = stack.back();
      stack.pop_back();
      for (int t : st[s].eps)
        if (!seen[t]) {
          seen[t] = 1;
          set.push_back(t);
          stack.push_back(t);
        }
    }
    std::sort(set.begin(), set.end());
  }
};

}  // namespace

const StandardTokenizer& StandardTokenizer::instance() {
  static const StandardTokenizer t;
  return t;
}

StandardTokenizer::StandardTokenizer() {
  // Alphabet: one class per distinct property mask.
  uint32_t masks[256];
  int nm = 0;
  uni::all_masks(masks, &nm, 256);
  if (nm >= 250) throw std::runtime_error("tokenizer: too many property classes");
  nclasses_ = nm;
  std::unordered_map<uint32_t, uint8_t> class_of;
  for (int i = 0; i < nm; ++i) class_of[masks[i]] = uint8_t(i);

  // Two-stage code point -> class table with deduplicated 128-entry blocks.
  const uint32_t kBlocks = 0x110000 / 128;
  stage1_.resize(kBlocks);
  std::map<std::vector<uint8_t>, uint16_t> dedupe;
  std::vector<uint8_t> block(128);
  for (uint32_t bi = 0; bi < kBlocks; ++bi) {
    for (uint32_t j = 0; j < 128; ++j) block[j] = class_of.at(uni::prop_mask(bi * 128 + j));
    auto it = dedupe.find(block);
    if (it == dedupe.end()) {
      uint16_t id = uint16_t(stage2_.size() / 128);
      stage2_.insert(stage2_.end(), block.begin(), block.end());
      it = dedupe.emplace(block, id).first;
    }
    stage1_[bi] = it->second;
  }
  for (uint32_t c = 0; c < 128; ++c) ascii_cls_[c] = class_of.at(uni::prop_mask(c));

  // Grammar -> NFA -> DFA.
  Nfa nfa;
  N g = build_grammar();
  auto [start, accept] = nfa.compile(*g);
  std::vector<std::vector<uint8_t>> pred_ok(nfa.preds.size(), std::vector<uint8_t>(nm));
  for (size_t p = 0; p < nfa.preds.size(); ++p)
    for (int c = 0; c < nm; ++c) pred_ok[p][c] = nfa.preds[p].matches(masks[c]);

  std::map<std::vector<int>, int> ids;
  std::vector<std::vector<int>> sets;
  std::vector<int> s0{start};
  nfa.closure(s0);
  ids[s0] = 0;
  sets.push_back(s0);
  for (size_t d = 0; d < sets.size(); ++d) {
    const std::vector<int> cur = sets[d];
    accept_.push_back(std::binary_search(cur.begin(), cur.end(), accept) ? 1 : 0);
    trans_.resize(sets.size() * nm + nm, -1);
    for (int c = 0; c < nm; ++c) {
      std::vector<int> nxt;
      for (int s : cur) {
        const auto& ns = nfa.st[s];
        if (ns.pred >= 0 && pred_ok[ns.pred][c]) nxt.push_back(ns.next);
      }
      if (nxt.empty()) continue;
      std::sort(nxt.begin(), nxt.end());
      nxt.erase(std::unique(nxt.begin(), nxt.end()), nxt.end());
      nfa.closure(nxt);
      auto it = ids.find(nxt);
      int id;
      if (it == ids.end()) {
        id = int(sets.size());
        if (id > 32000) throw std::runtime_error("tokenizer: DFA too large");
        ids.emplace(nxt, id);
        sets.push_back(nxt);
      } else {
        id = it->second;
      }
      trans_[d * nm + c] = int16_t(id);
    }
  }
  trans_.resize(sets.size() * nm, -1);
  if (accept_[0]) throw std::logic_error("tokenizer grammar accepts the empty string");
}

void StandardTokenizer::tokenize(std::string_view text, std::vector<RawToken>& out) const {
  const auto* base = reinterpret_cast<const unsigned char*>(text.data());
  const auto* end = base + text.size();
  const int16_t* trans = trans_.data();
  const uint8_t* accept = accept_.data();
  const int nc = nclasses_;
  const unsigned char* pos = base;
  while (pos < end) {
    int state = 0;
    const unsigned char* p = pos;
    const unsigned char* last = nullptr;
    uint32_t units = 0;
    while (p < end) {
      uint32_t cp, len;
      if (*p < 0x80) {
        cp = *p;
        len = 1;
      } else {
        cp = utf8_decode(p, end, &len);
      }
      uint32_t u16 = cp >= 0x10000 ? 2 : 1;
      if (units + u16 > kMaxTokenUnits) break;  // JFlex buffer full: match ends here
      int ns = trans[state * nc + cls(cp)];
      if (ns < 0) break;
      state = ns;
      p += len;
      units += u16;
      if (accept[state]) last = p;
    }
    if (last) {
      out.push_back({uint32_t(pos - base), uint32_t(last - base)});
      pos = last;
    } else {
      uint32_t len = 1;
      if (*pos >= 0x80) utf8_decode(pos, end, &len);
      pos += len;
    }
  }
}

}  // namespace hs::lexical
