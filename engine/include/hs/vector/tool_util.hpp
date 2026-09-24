#pragma once
// Helpers shared by the hs_vec_* command-line tools: flag parsing, latency
// statistics, TREC run output. Header-only; not part of the search library API.

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hs/common/topk.hpp"

namespace hs::vector::tool {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument " + a);
      a = a.substr(2);
      auto eq = a.find('=');
      if (eq != std::string::npos) kv_[a.substr(0, eq)] = a.substr(eq + 1);
      else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv_[a] = argv[++i];
      else kv_[a] = "1";
      if (i < argc) cmd_ += std::string(" ") + argv[i];
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def = "") const {
    auto it = kv_.find(k);
    if (it != kv_.end()) return it->second;
    if (def.empty() && !optional_) throw std::invalid_argument("missing --" + k);
    return def;
  }
  std::string opt(const std::string& k, const std::string& def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  double num(const std::string& k, double def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::stod(it->second);
  }
  std::vector<uint32_t> list(const std::string& k, const std::string& def) const {
    std::vector<uint32_t> out;
    std::stringstream ss(opt(k, def));
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(uint32_t(std::stoul(t)));
    return out;
  }

 private:
  std::map<std::string, std::string> kv_;
  std::string cmd_;
  bool optional_ = false;
};

inline std::string command_line(int argc, char** argv) {
  std::string s;
  for (int i = 0; i < argc; ++i) s += (i ? " " : "") + std::string(argv[i]);
  return s;
}

struct LatencyStats {
  double mean_us = 0, p50_us = 0, p90_us = 0, p99_us = 0, max_us = 0;
};

inline LatencyStats latency_stats(std::vector<double> us) {
  LatencyStats s;
  if (us.empty()) return s;
  std::sort(us.begin(), us.end());
  double sum = 0;
  for (double v : us) sum += v;
  auto pct = [&](double p) { return us[std::min(us.size() - 1, size_t(p * double(us.size() - 1) + 0.5))]; };
  s.mean_us = sum / double(us.size());
  s.p50_us = pct(0.50);
  s.p90_us = pct(0.90);
  s.p99_us = pct(0.99);
  s.max_us = us.back();
  return s;
}

inline std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) out.push_back(line);
  }
  return out;
}

// TREC run: qid Q0 docid rank score tag.
inline void write_trec(const std::string& path, const std::vector<std::string>& qids,
                       const std::vector<std::vector<hs::ScoredDoc>>& hits, const std::vector<uint64_t>& docids,
                       const std::string& tag) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + path);
  for (size_t q = 0; q < hits.size(); ++q)
    for (size_t r = 0; r < hits[q].size(); ++r) {
      uint64_t id = docids.empty() ? hits[q][r].doc : docids[hits[q][r].doc];
      std::fprintf(f, "%s Q0 %llu %zu %.6f %s\n", qids[q].c_str(), (unsigned long long)id, r + 1,
                   double(hits[q][r].score), tag.c_str());
    }
  std::fclose(f);
}

// CPU time consumed by the calling thread. Under a loaded, shared machine this is a
// far steadier cost measure than wall time (it excludes time spent descheduled).
inline double thread_cpu_s() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace hs::vector::tool
