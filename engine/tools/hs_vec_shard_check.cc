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
  std::vector<std::vector<uint32_t>> rows_of;  // shard ordinal -> global row
  for (auto& d : dirs) {
    shards.push_back(DiskIndex::open(d, oo));
    // Map every shard ordinal to its global row (docids.u64bin is sorted by global id).
    std::vector<uint32_t> rws(shards.back()->size());
    for (uint32_t i = 0; i < rws.size(); ++i) {
      uint64_t pid = shards.back()->global_id(i);
      auto it = std::lower_bound(docids.begin(), docids.end(), pid);
      if (it == docids.end() || *it != pid) throw std::runtime_error("shard docid not in global docids: " + d);
      rws[i] = uint32_t(it - docids.begin());
    }
    rows_of.push_back(std::move(rws));
  }
  uint64_t rows = 0;
  {  // Partition check: every global row in exactly one shard.
    std::vector<uint8_t> seen(docids.size(), 0);
    uint64_t dup = 0;
    for (auto& r : rows_of)
      for (uint32_t g : r) dup += seen[g]++ ? 1 : 0;
    for (auto& s : shards) rows += s->size();
    std::printf("{\"shards\": %zu, \"rows\": %llu, \"global_rows\": %zu, \"duplicate_rows\": %llu, \"missing_rows\": %llu}\n",
                shards.size(), (unsigned long long)rows, docids.size(), (unsigned long long)dup,
                (unsigned long long)(docids.size() - (rows - dup)));
  }
  if (!a.has("no-exact")) {
    std::vector<std::vector<Hit>> merged(qs.n);
    MappedFbin all(a.str("base"));
    for (size_t s = 0; s < shards.size(); ++s) {
      std::vector<float> vec(size_t(rows_of[s].size()) * all.dim());
      for (size_t i = 0; i < rows_of[s].size(); ++i)
        std::copy(all.row(rows_of[s][i]), all.row(rows_of[s][i]) + all.dim(), vec.begin() + i * all.dim());
      auto hits = exact_topk(vec.data(), uint32_t(rows_of[s].size()), qs.data.data(), qs.n, all.dim(), k);
      for (uint32_t q = 0; q < qs.n; ++q)
        for (uint32_t j = 0; j < k; ++j) {
          const auto& h = hits[size_t(q) * k + j];
          merged[q].push_back({shards[s]->global_id(h.doc), h.score});
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
