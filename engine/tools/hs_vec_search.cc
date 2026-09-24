// Sweep L_search over an in-memory Vamana index: recall@k vs QPS.
//   hs_vec_search --base passages.fbin [--max-n N] --graph graph.bin --queries q.fbin --gt gt.bin
//                 [--k 10] [--L 10,20,40,80,160] [--query-threads 1] [--max-queries N]
//                 [--out results.jsonl] [--label name]
//                 [--trec-prefix run] [--qids q.qids.txt] [--docids docids.u64bin]
// One JSON line per L. Single query thread = latency-bound QPS (ann-benchmarks style).
#include <cstdio>

#include "hs/common/fbin.hpp"
#include "hs/vector/bruteforce.hpp"
#include "hs/vector/parallel.hpp"
#include "hs/vector/tool_util.hpp"
#include "hs/vector/vamana.hpp"

using namespace hs::vector;

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  auto base = std::make_shared<MappedFbin>(a.str("base"), uint32_t(a.num("max-n", 0)), uint32_t(a.num("row-begin", 0)));
  auto ix = VamanaIndex::load(a.str("graph"), base->data(), base->n(), base->dim());
  auto qs = hs::read_fbin(a.str("queries"), uint32_t(a.num("max-queries", 0)));
  GroundTruth gt = read_groundtruth(a.str("gt"));
  uint32_t k = uint32_t(a.num("k", 10));
  unsigned qt = unsigned(a.num("query-threads", 1));
  std::vector<uint64_t> docids;
  if (a.has("docids")) docids = hs::read_u64bin(a.str("docids"));
  std::vector<std::string> qids;
  if (a.has("qids")) qids = tool::read_lines(a.str("qids"));
  std::string label = a.opt("label", "vamana");

  // Warm-up: touch the graph and vectors once so every L sees the same memory state.
  {
    VectorSearchOptions o;
    o.k = k;
    o.L = 50;
    for (uint32_t q = 0; q < std::min<uint32_t>(qs.n, 500); ++q) ix.search(qs.row(q), o);
  }
  for (uint32_t L : a.list("L", "10,20,30,50,75,100,150,200,300")) {
    VectorSearchOptions o;
    o.k = k;
    o.L = std::max(L, k);
    std::vector<std::vector<hs::ScoredDoc>> hits(qs.n);
    std::vector<double> lat(qs.n), cpu(qs.n);
    std::vector<uint64_t> dist(qs.n), hops(qs.n);
    double t0 = tool::now_s();
    parallel_for(0, qs.n, qt, [&](size_t q, unsigned) {
      double s = tool::now_s(), c = tool::thread_cpu_s();
      auto r = ix.search(qs.row(q), o);
      lat[q] = (tool::now_s() - s) * 1e6;
          cpu[q] = (tool::thread_cpu_s() - c) * 1e6;
      hits[q] = std::move(r.hits);
      dist[q] = r.distance_computations;
      hops[q] = r.hops;
    });
    double wall = tool::now_s() - t0;
    std::vector<std::vector<uint32_t>> ids(qs.n);
    double sd = 0, sh = 0;
    for (uint32_t q = 0; q < qs.n; ++q) {
      for (auto& h : hits[q]) ids[q].push_back(h.doc);
      sd += double(dist[q]);
      sh += double(hops[q]);
    }
    double rec = recall_at_k(gt, ids, k);
    auto ls = tool::latency_stats(lat);
      double cpu_sum = 0;
      for (double v : cpu) cpu_sum += v;
      double cpu_us = cpu_sum / qs.n;
    char line[1024];
    std::snprintf(line, sizeof line,
                  "{\"system\": \"%s\", \"n\": %u, \"nq\": %u, \"k\": %u, \"L\": %u, \"query_threads\": %u, "
                  "\"recall\": %.5f, \"qps\": %.1f, \"cpu_us_per_query\": %.1f, \"cpu_qps\": %.1f, \"mean_us\": %.1f, \"p50_us\": %.1f, \"p99_us\": %.1f, "
                  "\"dist_per_query\": %.1f, \"hops_per_query\": %.1f}",
                  label.c_str(), ix.size(), qs.n, k, o.L, qt, rec, qs.n / wall, cpu_us, 1e6 / cpu_us, ls.mean_us, ls.p50_us, ls.p99_us,
                  sd / qs.n, sh / qs.n);
    std::printf("%s\n", line);
    std::fflush(stdout);
    if (a.has("out")) {
      std::FILE* f = std::fopen(a.str("out").c_str(), "a");
      if (f) std::fprintf(f, "%s\n", line), std::fclose(f);
    }
    if (a.has("trec-prefix")) {
      if (qids.size() < qs.n) throw std::runtime_error("--qids has fewer lines than queries");
      tool::write_trec(a.str("trec-prefix") + ".L" + std::to_string(o.L) + ".trec", qids, hits, docids,
                       label + "-L" + std::to_string(o.L));
    }
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_search: %s\n", e.what());
  return 1;
}
