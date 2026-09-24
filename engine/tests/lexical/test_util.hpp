#pragma once
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hs/lexical/index.hpp"
#include "hs/lexical/index_builder.hpp"

namespace hs::lexical::testing {

inline std::string data_dir() { return std::string(HS_TEST_DATA_DIR) + "/lexical"; }

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t p = s.find(sep, start);
    out.push_back(s.substr(start, p == std::string::npos ? std::string::npos : p - start));
    if (p == std::string::npos) break;
    start = p + 1;
  }
  return out;
}

inline std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  std::string l;
  while (std::getline(in, l)) lines.push_back(l);
  return lines;
}

inline std::string temp_dir(const std::string& name) {
  auto p = std::filesystem::temp_directory_path() / ("hs_lex_" + name + "_" + std::to_string(::getpid()));
  std::filesystem::remove_all(p);
  std::filesystem::create_directories(p);
  return p.string();
}

// Builds (once per process) the index of the checked-in test corpus.
inline const LexicalIndex& corpus_index() {
  static std::unique_ptr<LexicalIndex> ix = [] {
    std::string dir = temp_dir("corpus");
    BuildOptions o;
    o.input_tsv = data_dir() + "/corpus.tsv";
    o.out_dir = dir;
    o.threads = 2;
    o.mem_budget_mb = 1;  // forces several runs + the k-way merge
    o.verbose = false;
    o.min_free_gb = 0.1;
    build_index(o);
    return LexicalIndex::open(dir);
  }();
  return *ix;
}

inline uint32_t bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

}  // namespace hs::lexical::testing
