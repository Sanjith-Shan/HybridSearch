// hs_lex_bench: per-query latency distribution for each algorithm x codec.
//
//   hs_lex_bench --index DIR --queries TSV --out results/lexical/bench.json
//                [--algos exhaustive,daat,maxscore,wand,bmw] [--codecs vbyte,bp128] [--k 10]
//                [--model lucene] [--k1 0.9] [--b 0.4] [--limit N] [--warmup 1] [--reps 1] [--label NAME]
//
// Single-threaded, one query at a time (a latency measurement, not a throughput one). Queries
// are analyzed once up front; the timed region is LexicalIndex::search only. Every query of
// every rep is kept, and percentiles are exact nearest-rank percentiles of the sorted sample.
// A warm-up pass over all queries precedes measurement, so numbers are warm-cache. The
// .meta.json sidecar records hardware, flags, git SHA and the command; on macOS the label is
// dev-signal-only (threads cannot be pinned).

#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "hs/lexical/index.hpp"
#include "lex_bench_util.hpp"

using namespace hs::lexical;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  std::string index_dir, queries, out, label = "lexical";
  std::string algos = "exhaustive,daat,maxscore,wand,bmw", codecs = "vbyte,bp128";
  SearchOptions base;
  base.k = 10;
  size_t limit = 0;
  int warmup = 1, reps = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      return argv[++i];
    };
    if (a == "--index") index_dir = val();
    else if (a == "--queries") queries = val();
    else if (a == "--out") out = val();
    else if (a == "--algos") algos = val();
    else if (a == "--codecs") codecs = val();
    else if (a == "--k") base.k = uint32_t(std::stoul(val()));
    else if (a == "--k1") base.k1 = std::stof(val());
    else if (a == "--b") base.b = std::stof(val());
    else if (a == "--limit") limit = std::stoul(val());
    else if (a == "--warmup") warmup = std::stoi(val());
    else if (a == "--reps") reps = std::stoi(val());
    else if (a == "--label") label = val();
    else if (a == "--model") base.model = val() == "textbook" ? Model::Textbook : Model::Lucene;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (index_dir.empty() || queries.empty() || out.empty()) {
    std::fprintf(stderr, "usage: hs_lex_bench --index DIR --queries TSV --out FILE.json [options]\n");
    return 2;
  }
  auto ix = LexicalIndex::open(index_dir);
  std::vector<std::string> qids;
  std::vector<std::vector<std::string>> qterms;
  {
    std::ifstream in(queries);
    std::string line;
    auto t0 = Clock::now();
    while (std::getline(in, line)) {
      size_t tab = line.find('\t');
      if (tab == std::string::npos) continue;
      qids.push_back(line.substr(0, tab));
      qterms.push_back(ix->analyzer().analyze(std::string_view(line).substr(tab + 1)));
      if (limit && qids.size() >= limit) break;
    }
    double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    std::fprintf(stderr, "[bench] %zu queries analyzed in %.0f us (%.2f us/query)\n", qids.size(), us,
                 us / double(qids.size()));
  }
  auto split = [](const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(x);
    return v;
  };

  std::ostringstream js;
  js << "{\n  \"label\": \"" << bench::json_escape(label) << "\",\n  \"index\": \"" << bench::json_escape(index_dir)
     << "\",\n  \"queries\": \"" << bench::json_escape(queries) << "\",\n  \"num_queries\": " << qids.size()
     << ",\n  \"k\": " << base.k << ",\n  \"model\": \"" << model_name(base.model) << "\",\n  \"k1\": " << base.k1
     << ",\n  \"b\": " << base.b << ",\n  \"warmup_passes\": " << warmup << ",\n  \"reps\": " << reps
     << ",\n  \"threads\": 1,\n  \"timed_region\": \"LexicalIndex::search (analysis excluded)\",\n"
     << "  \"index_num_docs\": " << ix->num_docs() << ",\n  \"index_num_postings\": " << ix->stats().num_postings
     << ",\n  \"bp128_kernel\": \"" << bp128_kernel() << "\",\n  \"results\": [\n";
  bool first = true;
  for (const auto& cs : split(codecs)) {
    Codec c;
    if (!parse_codec(cs, &c) || !ix->has_codec(c)) {
      std::fprintf(stderr, "skipping codec %s (not in index)\n", cs.c_str());
      continue;
    }
    for (const auto& as : split(algos)) {
      Algorithm a;
      if (!parse_algorithm(as, &a)) throw std::invalid_argument("unknown algorithm " + as);
      SearchOptions o = base;
      o.algorithm = a;
      o.codec = c;
      for (int w = 0; w < warmup; ++w)
        for (const auto& t : qterms) {
          hs::Deadline dl;
          (void)ix->search(t, o, dl);
        }
      std::vector<double> lat;
      lat.reserve(qterms.size() * size_t(reps));
      double scored = 0, decoded = 0, blocks = 0, hits = 0;
      for (int r = 0; r < reps; ++r)
        for (const auto& t : qterms) {
          hs::Deadline dl;
          auto t0 = Clock::now();
          auto res = ix->search(t, o, dl);
          lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
          scored += double(res.docs_scored);
          decoded += double(res.postings_decoded);
          blocks += double(res.blocks_decoded);
          hits += double(res.hits.size());
        }
      std::vector<double> sorted = lat;
      std::sort(sorted.begin(), sorted.end());
      double n = double(lat.size()), mean = 0;
      for (double x : lat) mean += x;
      mean /= n;
      std::fprintf(stderr,
                   "[bench] %-10s %-6s n=%zu mean=%.1fus p50=%.1fus p90=%.1fus p99=%.1fus max=%.1fus "
                   "docs_scored/q=%.0f postings/q=%.0f\n",
                   algorithm_name(a), codec_name(c), lat.size(), mean, bench::percentile(sorted, 50),
                   bench::percentile(sorted, 90), bench::percentile(sorted, 99), sorted.back(), scored / n,
                   decoded / n);
      js << (first ? "" : ",\n") << "    {\"algorithm\": \"" << algorithm_name(a) << "\", \"codec\": \""
         << codec_name(c) << "\", \"n\": " << lat.size() << ", \"mean_us\": " << mean
         << ", \"p50_us\": " << bench::percentile(sorted, 50) << ", \"p90_us\": " << bench::percentile(sorted, 90)
         << ", \"p95_us\": " << bench::percentile(sorted, 95) << ", \"p99_us\": " << bench::percentile(sorted, 99)
         << ", \"p999_us\": " << bench::percentile(sorted, 99.9) << ", \"max_us\": " << sorted.back()
         << ", \"docs_scored_per_query\": " << scored / n << ", \"postings_decoded_per_query\": " << decoded / n
         << ", \"blocks_decoded_per_query\": " << blocks / n << ", \"hits_per_query\": " << hits / n << "}";
      first = false;
    }
  }
  js << "\n  ]\n}\n";
  std::ofstream(out) << js.str();
  bench::write_meta(out, argc, argv, "tools/hs_lex_bench.cc", "warm (1+ untimed pass over all queries first)");
  std::fprintf(stderr, "[bench] wrote %s\n", out.c_str());
  return 0;
}
