// Sweep the DiskANN-style index: recall@k, QPS, SSD reads/query, RAM.
//   hs_vec_disk_search --index dir --queries q.fbin --gt gt.bin [--k 10] [--L 10,20,50,100]
//                      [--W 4] [--cache 0] [--mode cold|warm] [--io pool|seq] [--io-threads 8]
//                      [--query-threads 1] [--max-queries N] [--out results.jsonl] [--label name]
//                      [--trec-prefix run --qids q.qids.txt]
// cold: F_NOCACHE (macOS) / O_DIRECT (Linux); every node read goes to the device.
// warm: page cache; the query set is run once unmeasured before each measured pass.
#include <cstdio>

#include "hs/common/fbin.hpp"
#include "hs/vector/bruteforce.hpp"
#include "hs/vector/disk_index.hpp"
#include "hs/vector/parallel.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  std::string mode = a.opt("mode", "cold");
  DiskOpenOptions oo;
  oo.direct_io = mode == "cold";
  oo.cache_nodes = uint32_t(a.num("cache", 0));
  oo.io_mode = a.opt("io", "pool") == "seq" ? IoMode::kSequential : IoMode::kThreadPool;
  oo.io_threads = unsigned(a.num("io-threads", 8));
  double t_open = tool::now_s();
  auto ix = DiskIndex::open(a.str("index"), oo);
  t_open = tool::now_s() - t_open;
  auto qs = hs::read_fbin(a.str("queries"), uint32_t(a.num("max-queries", 0)));
  GroundTruth gt = read_groundtruth(a.str("gt"));
  uint32_t k = uint32_t(a.num("k", 10));
  unsigned qt = unsigned(a.num("query-threads", 1));
  std::vector<std::string> qids;
  if (a.has("qids")) qids = tool::read_lines(a.str("qids"));
  std::string label = a.opt("label", "diskann");
  auto mem = ix->memory();

  for (uint32_t W : a.list("W", "4"))
    for (uint32_t L : a.list("L", "10,20,30,50,75,100,150,200")) {
      VectorSearchOptions o;
      o.k = k;
      o.L = std::max(L, k);
      o.beam_width = W;
      std::vector<std::vector<hs::ScoredDoc>> hits(qs.n);
      std::vector<double> lat(qs.n), cpu(qs.n);
      std::vector<VectorResult> res(qs.n);
      auto run = [&] {
        double t0 = tool::now_s();
        parallel_for(0, qs.n, qt, [&](size_t q, unsigned) {
          double s = tool::now_s(), c = tool::thread_cpu_s();
          auto r = ix->search(qs.row(q), o);
          lat[q] = (tool::now_s() - s) * 1e6;
          cpu[q] = (tool::thread_cpu_s() - c) * 1e6;
          hits[q] = r.hits;
          res[q] = std::move(r);
        });
        return tool::now_s() - t0;
      };
      if (mode == "warm") run();
      double wall = run();
      std::vector<std::vector<uint32_t>> ids(qs.n);
      double reads = 0, hops = 0, pqd = 0, io_us = 0, rounds = 0, chits = 0;
      for (uint32_t q = 0; q < qs.n; ++q) {
        for (auto& h : hits[q]) ids[q].push_back(h.doc);
        reads += double(res[q].ssd_reads);
        hops += double(res[q].hops);
        pqd += double(res[q].pq_distance_computations);
        io_us += double(res[q].io_us);
        rounds += double(res[q].io_rounds);
        chits += double(res[q].cache_hits);
      }
      double rec = recall_at_k(gt, ids, k);
      auto ls = tool::latency_stats(lat);
      double cpu_sum = 0;
      for (double v : cpu) cpu_sum += v;
      double cpu_us = cpu_sum / qs.n;
      char line[2048];
      std::snprintf(line, sizeof line,
                    "{\"system\": \"%s\", \"n\": %u, \"nq\": %u, \"k\": %u, \"L\": %u, \"W\": %u, \"cache_nodes\": %u, "
                    "\"mode\": \"%s\", \"io\": \"%s\", \"io_threads\": %u, \"query_threads\": %u, "
                    "\"recall\": %.5f, \"qps\": %.1f, \"cpu_us_per_query\": %.1f, \"cpu_qps\": %.1f, \"mean_us\": %.1f, \"p50_us\": %.1f, \"p99_us\": %.1f, "
                    "\"ssd_reads_per_query\": %.2f, \"hops_per_query\": %.2f, \"cache_hits_per_query\": %.2f, "
                    "\"io_rounds_per_query\": %.2f, \"io_us_per_query\": %.1f, \"us_per_io_round\": %.1f, "
                    "\"pq_dist_per_query\": %.1f, "
                    "\"ram_bytes\": {\"pq_codes\": %llu, \"pq_pivots\": %llu, \"cache\": %llu, \"docids\": %llu}, "
                    "\"open_seconds\": %.2f}",
                    label.c_str(), ix->size(), qs.n, k, o.L, W, oo.cache_nodes, mode.c_str(),
                    oo.io_mode == IoMode::kSequential ? "seq" : "pool", oo.io_threads, qt, rec, qs.n / wall, cpu_us, 1e6 / cpu_us,
                    ls.mean_us, ls.p50_us, ls.p99_us, reads / qs.n, hops / qs.n, chits / qs.n, rounds / qs.n,
                    io_us / qs.n, rounds > 0 ? io_us / rounds : 0.0, pqd / qs.n,
                    (unsigned long long)mem.pq_codes, (unsigned long long)mem.pq_pivots,
                    (unsigned long long)mem.cache, (unsigned long long)mem.docids, t_open);
      std::printf("%s\n", line);
      std::fflush(stdout);
      if (a.has("out")) {
        std::FILE* f = std::fopen(a.str("out").c_str(), "a");
        if (f) std::fprintf(f, "%s\n", line), std::fclose(f);
      }
      if (a.has("trec-prefix")) {
        if (qids.size() < qs.n) throw std::runtime_error("--qids has fewer lines than queries");
        std::vector<uint64_t> map(ix->size());
        for (uint32_t i = 0; i < ix->size(); ++i) map[i] = ix->global_id(i);
        tool::write_trec(a.str("trec-prefix") + ".W" + std::to_string(W) + ".L" + std::to_string(o.L) + ".trec",
                         qids, hits, map, label + "-W" + std::to_string(W) + "-L" + std::to_string(o.L));
      }
    }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_disk_search: %s\n", e.what());
  return 1;
}
