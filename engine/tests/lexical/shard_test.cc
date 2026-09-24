// Sharding with global BM25 statistics: the per-shard top-k lists, merged by ranks_before on
// (score, global ID), must equal the single index's top-k (global IDs and float score bits),
// for every algorithm, N in {2, 3, 4}. A control shows that shard-local statistics break it.
// A second test checks the same on real data when data/indexes/lexical/1m-4shards exists.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>

#include "test_util.hpp"

namespace hs::lexical {
namespace {

using testing::bits;

struct GHit {
  uint64_t gid;
  float score;
};

std::vector<GHit> single(const LexicalIndex& ix, const std::vector<std::string>& q, const SearchOptions& o) {
  Deadline dl;
  auto r = ix.search(q, o, dl);
  std::vector<GHit> out;
  for (const auto& h : r.hits) out.push_back({ix.global_id(h.doc), h.score});
  return out;
}

std::vector<GHit> merged(const std::vector<std::unique_ptr<LexicalIndex>>& shards, const std::vector<std::string>& q,
                         const SearchOptions& o) {
  std::vector<GHit> all;
  for (const auto& s : shards) {
    auto part = single(*s, q, o);
    all.insert(all.end(), part.begin(), part.end());
  }
  std::sort(all.begin(), all.end(), [](const GHit& a, const GHit& b) {
    return a.score != b.score ? a.score > b.score : a.gid < b.gid;
  });
  if (all.size() > o.k) all.resize(o.k);
  return all;
}

bool same(const std::vector<GHit>& a, const std::vector<GHit>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].gid != b[i].gid || bits(a[i].score) != bits(b[i].score)) return false;
  return true;
}

std::vector<std::unique_ptr<LexicalIndex>> build_shards(uint32_t n, bool global, const std::string& tag) {
  const auto& full = testing::corpus_index();
  const std::string input = testing::data_dir() + "/corpus.tsv";
  const uint64_t lines = count_lines(input);
  std::vector<std::unique_ptr<LexicalIndex>> shards;
  for (uint32_t i = 0; i < n; ++i) {
    BuildOptions o;
    o.input_tsv = input;
    o.out_dir = testing::temp_dir("shard_" + tag + std::to_string(n) + "_" + std::to_string(i));
    o.skip_docs = lines * i / n;
    o.max_docs = lines * (i + 1) / n - o.skip_docs;
    if (global) o.global_stats_dir = full.dir();
    o.codecs = {Codec::BP128};
    o.threads = 2;
    o.mem_budget_mb = 1;
    o.verbose = false;
    o.min_free_gb = 0.1;
    build_index(o);
    shards.push_back(LexicalIndex::open(o.out_dir));
  }
  return shards;
}

std::vector<std::vector<std::string>> corpus_queries(const Analyzer& an) {
  std::vector<std::vector<std::string>> qs;
  for (const auto& l : testing::read_lines(testing::data_dir() + "/queries.tsv")) qs.push_back(an.analyze(l.substr(l.find('\t') + 1)));
  return qs;
}

TEST(Sharding, GlobalStatsMergedTopKEqualsSingleIndex) {
  const auto& full = testing::corpus_index();
  const auto qs = corpus_queries(full.analyzer());
  uint64_t checks = 0;
  for (uint32_t n : {2u, 3u, 4u}) {
    auto shards = build_shards(n, true, "g");
    uint64_t docs = 0;
    for (const auto& s : shards) {
      EXPECT_TRUE(s->has_global_stats());
      EXPECT_EQ(s->idf_n(), full.idf_n());
      EXPECT_EQ(bits(s->avgdl()), bits(full.avgdl()));
      docs += s->num_docs();
    }
    EXPECT_EQ(docs, full.num_docs());
    for (Model m : {Model::Lucene, Model::Textbook})
      for (uint32_t k : {1u, 10u, 100u})
        for (int a = 0; a < kNumAlgorithms; ++a)
          for (const auto& q : qs) {
            SearchOptions o;
            o.k = k;
            o.model = m;
            o.algorithm = Algorithm(a);
            ASSERT_TRUE(same(single(full, q, o), merged(shards, q, o)))
                << "N=" << n << " " << algorithm_name(o.algorithm) << " " << model_name(m) << " k=" << k;
            ++checks;
          }
  }
  EXPECT_GT(checks, 25000u);
}

TEST(Sharding, ShardLocalStatsDoNotMatch) {
  // Control: without global statistics the merged ranking drifts, so the gate above has teeth.
  const auto& full = testing::corpus_index();
  const auto qs = corpus_queries(full.analyzer());
  auto shards = build_shards(3, false, "l");
  EXPECT_FALSE(shards[0]->has_global_stats());
  size_t differ = 0, nonempty = 0;
  for (const auto& q : qs) {
    SearchOptions o;
    o.k = 10;
    auto a = single(full, q, o);
    nonempty += !a.empty();
    differ += !same(a, merged(shards, q, o));
  }
  EXPECT_GT(nonempty, 0u);
  EXPECT_GT(differ, nonempty / 2);
}

TEST(Sharding, RealData1mFourShards) {
  const std::string repo = std::string(HS_TEST_DATA_DIR) + "/../../..";
  const char* env = std::getenv("HS_LEX_SHARDS");
  const std::string base = env && *env ? env : repo + "/data/indexes/lexical/1m-4shards";
  const std::string single_dir = repo + "/data/indexes/lexical/1m";
  const std::string qpath = repo + "/data/raw/msmarco/queries.dev.small.tsv";
  if (!std::filesystem::exists(base + "/shard0/meta.txt") || !std::filesystem::exists(single_dir + "/meta.txt") ||
      !std::filesystem::exists(qpath))
    GTEST_SKIP() << "no local 1M shards at " << base;
  auto full = LexicalIndex::open(single_dir);
  std::vector<std::unique_ptr<LexicalIndex>> shards;
  for (int i = 0; std::filesystem::exists(base + "/shard" + std::to_string(i) + "/meta.txt"); ++i)
    shards.push_back(LexicalIndex::open(base + "/shard" + std::to_string(i)));
  auto lines = testing::read_lines(qpath);
  if (lines.size() > 500) lines.resize(500);
  uint64_t checks = 0, bad = 0;
  for (const auto& l : lines) {
    auto q = full->analyzer().analyze(l.substr(l.find('\t') + 1));
    for (Algorithm a : {Algorithm::Exhaustive, Algorithm::MaxScore, Algorithm::WAND, Algorithm::BMW}) {
      SearchOptions o;
      o.k = 100;
      o.algorithm = a;
      bool ok = same(single(*full, q, o), merged(shards, q, o));
      EXPECT_TRUE(ok) << algorithm_name(a) << " " << l;
      bad += !ok;
      ++checks;
    }
  }
  std::printf("[shards] %zu shards, %zu queries, %llu comparisons, %llu mismatches\n", shards.size(), lines.size(),
              (unsigned long long)checks, (unsigned long long)bad);
}

}  // namespace
}  // namespace hs::lexical
