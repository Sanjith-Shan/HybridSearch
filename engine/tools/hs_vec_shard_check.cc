// Verify a document-sharded vector deployment and measure it as the broker sees it.
//   hs_vec_shard_check --shards d0,d1,d2,d3 --base passages.fbin --docids docids.u64bin
//                      --queries q.fbin --gt gt_global.bin [--k 10] [--L 20,50,100] [--W 4]
//                      [--mode warm|cold] [--cache 0] [--no-exact] [--max-queries N]
//                      [--out results.jsonl] [--trec-prefix run --qids q.qids.txt] [--label name]
// 1. Exact: brute-force top-k inside each shard's row slice, merged by score, must equal
//    the brute-force top-k over all rows (--gt), query by query (ids compared as global pids).
// 2. Approximate: each shard's DiskIndex searched with the same options, per-shard top-k
//    merged by score; recall@k of the merged list against the global ground truth,
//    SSD reads summed over shards, CPU time summed (shards searched one after another on
//    one thread: the broker fans out in parallel, so wall latency there is the max).
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <unordered_set>

#include "hs/common/fbin.hpp"
#include "hs/vector/bruteforce.hpp"
#include "hs/vector/disk_index.hpp"
#include "hs/vector/mmap.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

struct Hit {
  uint64_t pid;
  float score;
};
static bool hit_before(const Hit& a, const Hit& b) {
  return a.score != b.score ? a.score > b.score : a.pid < b.pid;
}

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  std::vector<std::string> dirs;
  {
    std::stringstream ss(a.str("shards"));
    std::string t;
    while (std::getline(ss, t, ',')) dirs.push_back(t);
  }
  auto docids = hs::read_u64bin(a.str("docids"));
  auto qs = hs::read_fbin(a.str("queries"), uint32_t(a.num("max-queries", 0)));
  GroundTruth gt = read_groundtruth(a.str("gt"));
  const uint32_t k = uint32_t(a.num("k", 10));
  std::string label = a.opt("label", "sharded");
  // Global truth as pids.
  std::vector<std::vector<uint64_t>> truth(qs.n);
  for (uint32_t q = 0; q < qs.n; ++q)
    for (uint32_t j = 0; j < k; ++j) truth[q].push_back(docids[gt.row(q)[j]]);

  DiskOpenOptions oo;
  oo.direct_io = a.opt("mode", "warm") == "cold";
  oo.cache_nodes = uint32_t(a.num("cache", 0));
  std::vector<std::unique_ptr<DiskIndex>> shards;
  std::vector<uint32_t> begin;
  for (auto& d : dirs) {
    shards.push_back(DiskIndex::open(d, oo));
    // Row slice of this shard in the global file: docids are sorted by global id.
    uint64_t first = shards.back()->global_id(0);
    auto it = std::lower_bound(docids.begin(), docids.end(), first);
    if (it == docids.end() || *it != first) throw std::runtime_error("shard docids not in global docids: " + d);
    begin.push_back(uint32_t(it - docids.begin()));
    for (uint32_t i = 0; i < shards.back()->size(); i += 9973)
      if (shards.back()->global_id(i) != docids[begin.back() + i]) throw std::runtime_error("shard not a contiguous slice: " + d);
  }
  uint64_t rows = 0;
  for (auto& s : shards) rows += s->size();
  std::printf("{\"shards\": %zu, \"rows\": %llu, \"global_rows\": %zu, \"slices\": [", shards.size(),
              (unsigned long long)rows, docids.size());
  for (size_t i = 0; i < shards.size(); ++i)
    std::printf("%s[%u, %u)", i ? ", " : "", begin[i], begin[i] + shards[i]->size());
  std::printf("]}\n");

  if (!a.has("no-exact")) {
    std::vector<std::vector<Hit>> merged(qs.n);
    for (size_t s = 0; s < shards.size(); ++s) {
      MappedFbin slice(a.str("base"), shards[s]->size(), begin[s]);
      auto hits = exact_topk(slice.data(), slice.n(), qs.data.data(), qs.n, slice.dim(), k);
      for (uint32_t q = 0; q < qs.n; ++q)
        for (uint32_t j = 0; j < k; ++j) {
          const auto& h = hits[size_t(q) * k + j];
          merged[q].push_back({docids[begin[s] + h.doc], h.score});
        }
    }
    uint32_t identical = 0, set_equal = 0;
    for (uint32_t q = 0; q < qs.n; ++q) {
      std::sort(merged[q].begin(), merged[q].end(), hit_before);
      merged[q].resize(k);
      bool same = true;
      std::unordered_set<uint64_t> tset(truth[q].begin(), truth[q].end());
      uint32_t inter = 0;
      for (uint32_t j = 0; j < k; ++j) {
        same &= merged[q][j].pid == truth[q][j];
        inter += tset.count(merged[q][j].pid);
      }
      identical += same;
      set_equal += inter == k;
    }
    char line[512];
    std::snprintf(line, sizeof line,
                  "{\"check\": \"exact_merged_vs_global\", \"nq\": %u, \"k\": %u, \"identical_ranked_lists\": %u, "
                  "\"identical_sets\": %u}",
                  qs.n, k, identical, set_equal);
    std::printf("%s\n", line);
    if (a.has("out")) {
      std::FILE* f = std::fopen(a.str("out").c_str(), "a");
      if (f) std::fprintf(f, "%s\n", line), std::fclose(f);
    }
  }

  std::vector<std::string> qids;
  if (a.has("qids")) qids = tool::read_lines(a.str("qids"));
  for (uint32_t W : a.list("W", "4"))
    for (uint32_t L : a.list("L", "20,50,100")) {
      VectorSearchOptions o;
      o.k = k;
      o.L = std::max(L, k);
      o.beam_width = W;
      double hits = 0, reads = 0, cpu = 0, wall = 0, maxshard_wall = 0;
      std::vector<std::vector<Hit>> out(qs.n);
      for (uint32_t q = 0; q < qs.n; ++q) {
        double c0 = tool::thread_cpu_s(), w0 = tool::now_s(), mx = 0;
        for (size_t s = 0; s < shards.size(); ++s) {
          double ws = tool::now_s();
          auto r = shards[s]->search(qs.row(q), o);
          mx = std::max(mx, tool::now_s() - ws);
          reads += double(r.ssd_reads);
          for (auto& h : r.hits) out[q].push_back({shards[s]->global_id(h.doc), h.score});
        }
        std::sort(out[q].begin(), out[q].end(), hit_before);
        if (out[q].size() > k) out[q].resize(k);
        cpu += tool::thread_cpu_s() - c0;
        wall += tool::now_s() - w0;
        maxshard_wall += mx;
        std::unordered_set<uint64_t> tset(truth[q].begin(), truth[q].end());
        for (auto& h : out[q]) hits += tset.count(h.pid);
      }
      char line[1024];
      std::snprintf(line, sizeof line,
                    "{\"system\": \"%s\", \"shards\": %zu, \"n\": %llu, \"nq\": %u, \"k\": %u, \"L\": %u, \"W\": %u, "
                    "\"mode\": \"%s\", \"cache_nodes\": %u, \"recall\": %.5f, \"ssd_reads_per_query_all_shards\": %.1f, "
                    "\"cpu_us_per_query_all_shards\": %.1f, \"wall_us_per_query_sequential\": %.1f, "
                    "\"wall_us_per_query_max_shard\": %.1f}",
                    label.c_str(), shards.size(), (unsigned long long)rows, qs.n, k, o.L, W, a.opt("mode", "warm").c_str(),
                    oo.cache_nodes, hits / (double(qs.n) * k), reads / qs.n, cpu / qs.n * 1e6, wall / qs.n * 1e6,
                    maxshard_wall / qs.n * 1e6);
      std::printf("%s\n", line);
      std::fflush(stdout);
      if (a.has("out")) {
        std::FILE* f = std::fopen(a.str("out").c_str(), "a");
        if (f) std::fprintf(f, "%s\n", line), std::fclose(f);
      }
      if (a.has("trec-prefix")) {
        std::string path = a.str("trec-prefix") + ".W" + std::to_string(W) + ".L" + std::to_string(o.L) + ".trec";
        std::FILE* f = std::fopen(path.c_str(), "w");
        for (uint32_t q = 0; q < qs.n; ++q)
          for (size_t r = 0; r < out[q].size(); ++r)
            std::fprintf(f, "%s Q0 %llu %zu %.6f %s-W%u-L%u\n", qids.at(q).c_str(), (unsigned long long)out[q][r].pid,
                         r + 1, double(out[q][r].score), label.c_str(), W, o.L);
        std::fclose(f);
      }
    }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_shard_check: %s\n", e.what());
  return 1;
}
