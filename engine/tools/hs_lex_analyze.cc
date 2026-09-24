// hs_lex_analyze: run the analyzer over "id\ttext" lines from stdin.
// Output matches tools/lucene_ref (mode analyze): "id\tanalyzed\traw" with tokens
// separated by U+0001, so the two can be diffed line by line.
//   hs_lex_analyze < passages.tsv > ours.tsv

#include <cstdio>
#include <iostream>
#include <string>

#include "hs/lexical/analyzer.hpp"

int main() {
  std::ios::sync_with_stdio(false);
  hs::lexical::Analyzer an;
  std::string line, out;
  std::vector<std::string> toks;
  while (std::getline(std::cin, line)) {
    size_t tab = line.find('\t');
    std::string_view id = tab == std::string::npos ? std::string_view() : std::string_view(line).substr(0, tab);
    std::string_view text = tab == std::string::npos ? std::string_view(line) : std::string_view(line).substr(tab + 1);
    out.assign(id);
    out.push_back('\t');
    toks.clear();
    an.analyze(text, toks);
    for (size_t i = 0; i < toks.size(); ++i) {
      if (i) out.push_back('\x01');
      out += toks[i];
    }
    out.push_back('\t');
    auto raw = an.tokenize(text);
    for (size_t i = 0; i < raw.size(); ++i) {
      if (i) out.push_back('\x01');
      out += raw[i];
    }
    out.push_back('\n');
    std::cout << out;
  }
  return 0;
}
