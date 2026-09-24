#include <gtest/gtest.h>

#include "hs/lexical/index.hpp"

namespace hs::lexical {
namespace {

std::string spans(const Snippet& s) {
  std::string out;
  for (const auto& h : s.highlights) out += "[" + s.window.substr(h.start, h.end - h.start) + "]";
  return out;
}

TEST(Snippet, ShortPassageIsReturnedWholeWithHighlights) {
  Analyzer an;
  auto s = select_snippet(an, "The Capital of Peru is Lima.", an.analyze("capital peru"), 200);
  EXPECT_FALSE(s.is_snippet);
  EXPECT_EQ(s.window, "The Capital of Peru is Lima.");
  EXPECT_EQ(spans(s), "[Capital][Peru]");
}

TEST(Snippet, PicksTheBestWindowNotTheFirst) {
  Analyzer an;
  std::string filler;
  for (int i = 0; i < 40; ++i) filler += "lorem ipsum dolor sit amet ";
  std::string text = "Peru appears once here. " + filler + "The capital of Peru is Lima, and Lima's capital district. " + filler;
  auto q = an.analyze("capital of peru lima");
  auto s = select_snippet(an, text, q, 120);
  EXPECT_TRUE(s.is_snippet);
  EXPECT_LE(s.window.size(), 120u);
  EXPECT_NE(s.window.find("capital of Peru is Lima"), std::string::npos) << s.window;
  EXPECT_EQ(text.substr(s.window_start, s.window.size()), s.window);
  // Highlights land on the analyzed hits and never cut a word.
  for (const auto& h : s.highlights) {
    std::string w = s.window.substr(h.start, h.end - h.start);
    auto a = an.analyze(w);
    ASSERT_EQ(a.size(), 1u) << w;
    EXPECT_NE(std::find(q.begin(), q.end(), a[0]), q.end());
  }
  EXPECT_NE(s.window.front(), ' ');
  EXPECT_NE(s.window.back(), ' ');
}

TEST(Snippet, NoHitsFallsBackToAPrefixCutAtAWordBoundary) {
  Analyzer an;
  std::string text;
  for (int i = 0; i < 30; ++i) text += "alpha beta gamma ";
  auto s = select_snippet(an, text, an.analyze("zeta"), 50);
  EXPECT_TRUE(s.is_snippet);
  EXPECT_LE(s.window.size(), 50u);
  EXPECT_TRUE(s.highlights.empty());
  EXPECT_EQ(text.substr(0, s.window.size()), s.window);
}

TEST(Snippet, NeverSplitsUtf8) {
  Analyzer an;
  std::string text;
  for (int i = 0; i < 50; ++i) text += "café naïve Zürich ";
  auto s = select_snippet(an, text, an.analyze("zürich"), 41);
  EXPECT_LE(s.window.size(), 41u);
  auto valid = [](const std::string& w) {
    for (size_t i = 0; i < w.size();) {
      unsigned char c = w[i];
      size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
      if (c >= 0x80 && c < 0xC0) return false;
      i += n;
      if (i > w.size()) return false;
    }
    return true;
  };
  EXPECT_TRUE(valid(s.window)) << s.window;
  EXPECT_FALSE(s.highlights.empty());
}

TEST(Snippet, ArchitectureSignature) {
  Analyzer an;
  auto s = select_snippet("Where is Lima? Lima is in Peru.", an.analyze("lima"), 0);
  EXPECT_FALSE(s.is_snippet);
  EXPECT_EQ(spans(s), "[Lima][Lima]");
}

}  // namespace
}  // namespace hs::lexical
