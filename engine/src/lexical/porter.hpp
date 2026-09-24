#pragma once
// The Porter stemmer (Porter 1980, "An algorithm for suffix stripping"), following Martin
// Porter's reference implementation (release 3), which is what Lucene's PorterStemmer is.
// Two departures from the 1980 paper are part of that reference and therefore of Lucene:
// step 2 maps -bli -> -ble (not -abli -> -able) and adds -logi -> -log.
//
// Java runs it over UTF-16 code units, so every non-ASCII unit (including each half of a
// surrogate pair) is a consonant and length checks count UTF-16 units. `Unit` is char for
// pure-ASCII words and char16_t otherwise so that behaviour is reproduced exactly.

#include <cstddef>
#include <vector>

namespace hs::lexical {

template <class Unit>
class PorterStemmer {
 public:
  // Stems b[0..len) in place; returns the new length.
  size_t stem(Unit* buf, size_t len) {
    b = buf;
    k = int(len) - 1;
    k0 = 0;
    if (k > k0 + 1) {
      step1();
      step2();
      step3();
      step4();
      step5();
      step6();
    }
    return size_t(k + 1);
  }

 private:
  Unit* b = nullptr;
  int j = 0, k = 0, k0 = 0;

  bool cons(int i) const {
    switch (b[i]) {
      case 'a': case 'e': case 'i': case 'o': case 'u':
        return false;
      case 'y':
        return (i == k0) ? true : !cons(i - 1);
      default:
        return true;
    }
  }

  // Number of consonant sequences between k0 and j.
  int m() const {
    int n = 0;
    int i = k0;
    while (true) {
      if (i > j) return n;
      if (!cons(i)) break;
      i++;
    }
    i++;
    while (true) {
      while (true) {
        if (i > j) return n;
        if (cons(i)) break;
        i++;
      }
      i++;
      n++;
      while (true) {
        if (i > j) return n;
        if (!cons(i)) break;
        i++;
      }
      i++;
    }
  }

  bool vowelinstem() const {
    for (int i = k0; i <= j; i++)
      if (!cons(i)) return true;
    return false;
  }

  bool doublec(int jj) const {
    if (jj < k0 + 1) return false;
    if (b[jj] != b[jj - 1]) return false;
    return cons(jj);
  }

  bool cvc(int i) const {
    if (i < k0 + 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
    auto ch = b[i];
    return !(ch == 'w' || ch == 'x' || ch == 'y');
  }

  template <size_t L>
  bool ends(const char (&s)[L]) {
    constexpr int l = int(L) - 1;
    int o = k - l + 1;
    if (o < k0) return false;
    for (int i = 0; i < l; i++)
      if (b[o + i] != Unit(s[i])) return false;
    j = k - l;
    return true;
  }

  template <size_t L>
  void setto(const char (&s)[L]) {
    constexpr int l = int(L) - 1;
    int o = j + 1;
    for (int i = 0; i < l; i++) b[o + i] = Unit(s[i]);
    k = j + l;
  }

  template <size_t L>
  void r(const char (&s)[L]) {
    if (m() > 0) setto(s);
  }

  void step1() {
    if (b[k] == 's') {
      if (ends("sses")) k -= 2;
      else if (ends("ies")) setto("i");
      else if (b[k - 1] != 's') k--;
    }
    if (ends("eed")) {
      if (m() > 0) k--;
    } else if ((ends("ed") || ends("ing")) && vowelinstem()) {
      k = j;
      if (ends("at")) setto("ate");
      else if (ends("bl")) setto("ble");
      else if (ends("iz")) setto("ize");
      else if (doublec(k)) {
        auto ch = b[k--];
        if (ch == 'l' || ch == 's' || ch == 'z') k++;
      } else if (m() == 1 && cvc(k)) setto("e");
    }
  }

  void step2() {
    if (ends("y") && vowelinstem()) b[k] = 'i';
  }

  void step3() {
    if (k == k0) return;
    switch (b[k - 1]) {
      case 'a':
        if (ends("ational")) { r("ate"); break; }
        if (ends("tional")) { r("tion"); break; }
        break;
      case 'c':
        if (ends("enci")) { r("ence"); break; }
        if (ends("anci")) { r("ance"); break; }
        break;
      case 'e':
        if (ends("izer")) { r("ize"); break; }
        break;
      case 'l':
        if (ends("bli")) { r("ble"); break; }
        if (ends("alli")) { r("al"); break; }
        if (ends("entli")) { r("ent"); break; }
        if (ends("eli")) { r("e"); break; }
        if (ends("ousli")) { r("ous"); break; }
        break;
      case 'o':
        if (ends("ization")) { r("ize"); break; }
        if (ends("ation")) { r("ate"); break; }
        if (ends("ator")) { r("ate"); break; }
        break;
      case 's':
        if (ends("alism")) { r("al"); break; }
        if (ends("iveness")) { r("ive"); break; }
        if (ends("fulness")) { r("ful"); break; }
        if (ends("ousness")) { r("ous"); break; }
        break;
      case 't':
        if (ends("aliti")) { r("al"); break; }
        if (ends("iviti")) { r("ive"); break; }
        if (ends("biliti")) { r("ble"); break; }
        break;
      case 'g':
        if (ends("logi")) { r("log"); break; }
        break;
      default:
        break;
    }
  }

  void step4() {
    switch (b[k]) {
      case 'e':
        if (ends("icate")) { r("ic"); break; }
        if (ends("ative")) { r(""); break; }
        if (ends("alize")) { r("al"); break; }
        break;
      case 'i':
        if (ends("iciti")) { r("ic"); break; }
        break;
      case 'l':
        if (ends("ical")) { r("ic"); break; }
        if (ends("ful")) { r(""); break; }
        break;
      case 's':
        if (ends("ness")) { r(""); break; }
        break;
      default:
        break;
    }
  }

  void step5() {
    if (k == k0) return;
    switch (b[k - 1]) {
      case 'a':
        if (ends("al")) break;
        return;
      case 'c':
        if (ends("ance")) break;
        if (ends("ence")) break;
        return;
      case 'e':
        if (ends("er")) break;
        return;
      case 'i':
        if (ends("ic")) break;
        return;
      case 'l':
        if (ends("able")) break;
        if (ends("ible")) break;
        return;
      case 'n':
        if (ends("ant")) break;
        if (ends("ement")) break;
        if (ends("ment")) break;
        if (ends("ent")) break;
        return;
      case 'o':
        if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
        if (ends("ou")) break;
        return;
      case 's':
        if (ends("ism")) break;
        return;
      case 't':
        if (ends("ate")) break;
        if (ends("iti")) break;
        return;
      case 'u':
        if (ends("ous")) break;
        return;
      case 'v':
        if (ends("ive")) break;
        return;
      case 'z':
        if (ends("ize")) break;
        return;
      default:
        return;
    }
    if (m() > 1) k = j;
  }

  void step6() {
    j = k;
    if (b[k] == 'e') {
      int a = m();
      if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
    }
    if (b[k] == 'l' && doublec(k) && m() > 1) k--;
  }
};

}  // namespace hs::lexical
