// hs_lex_search: queries.tsv -> TREC run file.
//
//   hs_lex_search --index DIR --queries queries.tsv --out run.trec
//                 [--algo exhaustive|daat|maxscore|wand|bmw] [--codec vbyte|bp128]
//                 [--model lucene|textbook] [--k1 0.9] [--b 0.4] [--k 1000] [--threads 4]
//                 [--tag NAME] [--deadline-us N] [--raw-out FILE] [--no-tie-adjust]
//
// Run-file scores follow Anserini's ScoreTiesAdjusterReranker (round to 1e-4, then subtract
// 1e-6 per tied rank) so that trec_eval / ir_measures, which re-sort by score and break ties
// by docno, evaluate exactly the ranking we produced. --raw-out writes the unrounded float
// scores ("qid docid rank score float_bits_hex") for exact comparisons.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "hs/lexical/index.hpp"

using namespace hs::lexical;

struct Q {
  std::string qid, text;
};

std::vector<Q> read_queries(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<Q> qs;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    size_t tab = line.find('\t');
    if (tab == std::string::npos) throw std::runtime_error("bad query line: " + line);
    qs.push_back({line.substr(0, tab), line.substr(tab + 1)});
  }
  return qs;
}

// io.anserini.rerank.lib.ScoreTiesAdjusterReranker, transcribed.
void adjust_ties(std::vector<float>& s) {
  int dup = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = float(std::floor(double(s[i]) * 1e4 + 0.5) / 1e4);
    if (i == 0 || s[i - 1] - s[i] > 1e-4f) {
      dup = 0;
    } else {
      dup++;
      s[i] -= 1e-6f * float(dup);
    }
  }
}

int main(int argc, char** argv) {
  std::string index_dir, queries, out, tag = "hybridsearch", raw_out;
  SearchOptions opt;
  opt.k = 1000;
  int threads = 4;
  uint64_t deadline_us = 0;
  bool tie_adjust = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      return argv[++i];
    };
    if (a == "--index") index_dir = val();
    else if (a == "--queries") queries = val();
    else if (a == "--out") out = val();
    else if (a == "--raw-out") raw_out = val();
    else if (a == "--tag") tag = val();
    else if (a == "--k") opt.k = uint32_t(std::stoul(val()));
    else if (a == "--k1") opt.k1 = std::stof(val());
    else if (a == "--b") opt.b = std::stof(val());
    else if (a == "--threads") threads = std::stoi(val());
    else if (a == "--deadline-us") deadline_us = std::stoull(val());
    else if (a == "--no-tie-adjust") tie_adjust = false;
    else if (a == "--algo") {
      if (!parse_algorithm(val(), &opt.algorithm)) throw std::invalid_argument("unknown algorithm");
    } else if (a == "--codec") {
      if (!parse_codec(val(), &opt.codec)) throw std::invalid_argument("unknown codec");
    } else if (a == "--model") {
      std::string m = val();
      if (m == "lucene") opt.model = Model::Lucene;
      else if (m == "textbook") opt.model = Model::Textbook;
      else throw std::invalid_argument("unknown model " + m);
    } else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (index_dir.empty() || queries.empty() || out.empty()) {
    std::fprintf(stderr, "usage: hs_lex_search --index DIR --queries TSV --out RUN [options]\n");
    return 2;
  }
  auto ix = LexicalIndex::open(index_dir);
  auto qs = read_queries(queries);
  std::vector<LexicalResult> results(qs.size());
  std::atomic<size_t> next{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, threads); ++t)
    pool.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < qs.size();) {
        hs::Deadline dl = deadline_us ? hs::Deadline::after_us(deadline_us) : hs::Deadline();
        results[i] = ix->search_text(qs[i].text, opt, dl);
      }
    });
  for (auto& th : pool) th.join();
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::FILE* f = std::fopen(out.c_str(), "w");
  std::FILE* raw = raw_out.empty() ? nullptr : std::fopen(raw_out.c_str(), "w");
  if (!f || (!raw_out.empty() && !raw)) throw std::runtime_error("cannot write output");
  uint64_t scored = 0, decoded = 0, partial = 0;
  for (size_t i = 0; i < qs.size(); ++i) {
    const auto& r = results[i];
    scored += r.docs_scored;
    decoded += r.postings_decoded;
    partial += r.partial;
    std::vector<float> s(r.hits.size());
    for (size_t j = 0; j < s.size(); ++j) s[j] = r.hits[j].score;
    if (tie_adjust) adjust_ties(s);
    for (size_t j = 0; j < r.hits.size(); ++j) {
      unsigned long long gid = (unsigned long long)ix->global_id(r.hits[j].doc);
      if (tie_adjust) std::fprintf(f, "%s Q0 %llu %zu %.6f %s\n", qs[i].qid.c_str(), gid, j + 1, double(s[j]), tag.c_str());
      else std::fprintf(f, "%s Q0 %llu %zu %.9g %s\n", qs[i].qid.c_str(), gid, j + 1, double(s[j]), tag.c_str());
      if (raw) {
        uint32_t bits;
        std::memcpy(&bits, &r.hits[j].score, 4);
        std::fprintf(raw, "%s\t%llu\t%zu\t%.9g\t%08x\n", qs[i].qid.c_str(), gid, j + 1, double(r.hits[j].score), bits);
      }
    }
  }
  std::fclose(f);
  if (raw) std::fclose(raw);
  std::fprintf(stderr,
               "[search] %zu queries, algo=%s codec=%s model=%s k1=%g b=%g k=%u threads=%d: %.2fs (%.1f q/s), "
               "docs_scored/q=%.0f postings_decoded/q=%.0f partial=%llu\n",
               qs.size(), algorithm_name(opt.algorithm), codec_name(opt.codec), model_name(opt.model), double(opt.k1),
               double(opt.b), opt.k, threads, secs, double(qs.size()) / secs, double(scored) / double(qs.size()),
               double(decoded) / double(qs.size()), (unsigned long long)partial);
  return 0;
}
