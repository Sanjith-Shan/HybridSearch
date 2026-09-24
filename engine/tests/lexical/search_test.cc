// The differential gate: every pruning algorithm x codec returns the same top-k (ordinals AND
// float score bits, in ranks_before order) as exhaustive term-at-a-time evaluation, and the
// Lucene-model scores equal Lucene 10.5.0's own BM25 scores bit for bit.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <set>
#include <thread>

#include "test_util.hpp"

namespace hs::lexical {
namespace {

using testing::bits;
using testing::corpus_index;
using testing::data_dir;
using testing::read_lines;
using testing::split;

struct Query {
  std::string qid, text;
};

std::vector<Query> corpus_queries() {
  std::vector<Query> qs;
  for (const auto& l : read_lines(data_dir() + "/queries.tsv")) {
    size_t t = l.find('\t');
    qs.push_back({l.substr(0, t), l.substr(t + 1)});
  }
  return qs;
}

LexicalResult run(const LexicalIndex& ix, const std::string& q, SearchOptions o) {
  Deadline dl;
  return ix.search_text(q, o, dl);
}

::testing::AssertionResult same_topk(const LexicalResult& a, const LexicalResult& b) {
  if (a.hits.size() != b.hits.size())
    return ::testing::AssertionFailure() << "size " << a.hits.size() << " vs " << b.hits.size();
  for (size_t i = 0; i < a.hits.size(); ++i) {
    if (a.hits[i].doc != b.hits[i].doc || bits(a.hits[i].score) != bits(b.hits[i].score))
      return ::testing::AssertionFailure() << "rank " << i << ": (" << a.hits[i].doc << ", " << a.hits[i].score
                                           << ") vs (" << b.hits[i].doc << ", " << b.hits[i].score << ")";
  }
  return ::testing::AssertionSuccess();
}

TEST(LexicalIndex, OpensTestCorpus) {
  const auto& ix = corpus_index();
  auto lines = read_lines(data_dir() + "/corpus.tsv");
  EXPECT_EQ(ix.num_docs(), lines.size());
  EXPECT_LT(ix.stats().docs_with_terms, ix.num_docs());  // the corpus has stop-word-only passages
  EXPECT_GT(ix.stats().num_blocks, 0u);
  for (uint32_t o = 0; o < ix.num_docs(); ++o) {
    uint64_t gid = std::stoull(lines[o].substr(0, lines[o].find('\t')));
    ASSERT_EQ(ix.global_id(o), gid);
    ASSERT_EQ(ix.ordinal_of(gid), int64_t(o));
  }
  EXPECT_EQ(ix.ordinal_of(999999999), -1);
}

TEST(LexicalIndex, PostingsIdenticalAcrossCodecs) {
  const auto& ix = corpus_index();
  ASSERT_TRUE(ix.has_codec(Codec::VByte));
  ASSERT_TRUE(ix.has_codec(Codec::BP128));
  uint64_t total = 0;
  std::vector<uint32_t> d1, t1, d2, t2;
  for (uint32_t t = 0; t < ix.stats().num_terms; ++t) {
    ix.postings(t, Codec::VByte, d1, t1);
    ix.postings(t, Codec::BP128, d2, t2);
    ASSERT_EQ(d1, d2);
    ASSERT_EQ(t1, t2);
    ASSERT_EQ(d1.size(), ix.df(t));
    ASSERT_TRUE(std::is_sorted(d1.begin(), d1.end()));
    total += d1.size();
    if (t) ASSERT_LT(ix.term_string(t - 1), ix.term_string(t));
    ASSERT_EQ(ix.term_id(ix.term_string(t)), int64_t(t));
  }
  EXPECT_EQ(total, ix.stats().num_postings);
}

TEST(LexicalIndex, DocStoreRoundTrip) {
  const auto& ix = corpus_index();
  auto ds = DocStore::open(ix.dir());
  auto lines = read_lines(data_dir() + "/corpus.tsv");
  ASSERT_EQ(ds->num_docs(), lines.size());
  for (uint32_t o = 0; o < lines.size(); o += 1) ASSERT_EQ(ds->text(o), lines[o].substr(lines[o].find('\t') + 1));
  EXPECT_EQ(ds->text(7), lines[7].substr(lines[7].find('\t') + 1));  // re-fetch after other blocks
}

// Lucene's golden output: qid docid rank score floatbits (k = 20, k1 = 0.9, b = 0.4).
TEST(Bm25, LuceneModelMatchesLuceneBitForBit) {
  const auto& ix = corpus_index();
  std::map<std::string, std::vector<std::pair<uint64_t, uint32_t>>> gold;
  for (const auto& l : read_lines(data_dir() + "/lucene_bm25_k20.tsv")) {
    auto f = split(l, '\t');
    gold[f[0]].push_back({std::stoull(f[1]), uint32_t(std::stoul(f[4], nullptr, 16))});
  }
  size_t checked = 0, empty = 0;
  for (const auto& q : corpus_queries()) {
    SearchOptions o;
    o.k = 20;
    o.algorithm = Algorithm::Exhaustive;
    auto r = run(ix, q.text, o);
    const auto& g = gold[q.qid];
    ASSERT_EQ(r.hits.size(), g.size()) << q.qid;
    if (g.empty()) {
      ++empty;
      continue;
    }
    // Same score at every rank, bit for bit. Documents within an exact-score tie may be in
    // another order (Lucene/Anserini break ties by docid *string*, we by numeric ID), and
    // the tie group cut by the k boundary may pick different members; all other score
    // groups must hold the same documents.
    std::multiset<std::pair<uint32_t, uint64_t>> ours, theirs;
    uint32_t last = g.back().second;
    for (size_t i = 0; i < g.size(); ++i) {
      ASSERT_EQ(bits(r.hits[i].score), g[i].second) << q.qid << " rank " << i + 1;
      if (g[i].second != last) {
        theirs.insert({g[i].second, g[i].first});
        ours.insert({bits(r.hits[i].score), ix.global_id(r.hits[i].doc)});
      }
    }
    EXPECT_EQ(ours, theirs) << q.qid;
    ++checked;
  }
  EXPECT_GT(checked, 250u);
  EXPECT_LT(empty, 30u);
}

TEST(Bm25, TextbookDiffersFromLuceneOnlyThroughLengths) {
  const auto& ix = corpus_index();
  size_t differs = 0;
  for (const auto& q : corpus_queries()) {
    SearchOptions l, t;
    l.k = t.k = 20;
    l.algorithm = t.algorithm = Algorithm::Exhaustive;
    t.model = Model::Textbook;
    auto a = run(ix, q.text, l), b = run(ix, q.text, t);
    ASSERT_EQ(a.hits.size(), b.hits.size());
    bool d = false;
    for (size_t i = 0; i < a.hits.size(); ++i) d |= a.hits[i].doc != b.hits[i].doc;
    differs += d;
  }
  // Documents longer than 40 tokens have lossy norms, so some rankings must change.
  EXPECT_GT(differs, 0u);
}

struct Params {
  float k1, b;
};

TEST(DifferentialGate, EveryAlgorithmAndCodecEqualsExhaustive) {
  const auto& ix = corpus_index();
  const auto qs = corpus_queries();
  const Params params[] = {{0.9f, 0.4f}, {1.2f, 0.75f}, {0.5f, 1.0f}, {2.0f, 0.0f}};
  uint64_t searches = 0, scored_exh = 0, scored_bmw = 0;
  for (Model m : {Model::Lucene, Model::Textbook}) {
    for (const Params& p : params) {
      for (uint32_t k : {1u, 3u, 10u, 20u, 100u, 1000u}) {
        for (const auto& q : qs) {
          SearchOptions base;
          base.k = k;
          base.model = m;
          base.k1 = p.k1;
          base.b = p.b;
          base.algorithm = Algorithm::Exhaustive;
          base.codec = Codec::VByte;
          auto ref = run(ix, q.text, base);
          if (k == 10) scored_exh += ref.docs_scored;
          for (int a = 0; a < kNumAlgorithms; ++a) {
            for (Codec c : {Codec::VByte, Codec::BP128}) {
              SearchOptions o = base;
              o.algorithm = Algorithm(a);
              o.codec = c;
              auto r = run(ix, q.text, o);
              ASSERT_TRUE(same_topk(ref, r)) << algorithm_name(o.algorithm) << "/" << codec_name(c) << " model="
                                             << model_name(m) << " k1=" << p.k1 << " b=" << p.b << " k=" << k
                                             << " query " << q.qid << " '" << q.text << "'";
              EXPECT_FALSE(r.partial);
              if (k == 10 && o.algorithm == Algorithm::BMW) scored_bmw += r.docs_scored;
              ++searches;
            }
          }
        }
      }
    }
  }
  EXPECT_GT(searches, 100000u);
  // Pruning has to actually prune (per codec, BMW is counted twice).
  EXPECT_LT(scored_bmw / 2, scored_exh);
}

TEST(Explain, ContributionsSumToTheScore) {
  const auto& ix = corpus_index();
  for (const auto& q : corpus_queries()) {
    SearchOptions o;
    o.k = 10;
    o.explain = true;
    auto r = run(ix, q.text, o);
    ASSERT_EQ(r.explain.size(), r.hits.size());
    for (size_t i = 0; i < r.hits.size(); ++i) {
      double s = 0;
      for (const auto& c : r.explain[i]) {
        EXPECT_GT(c.tf, 0u);
        EXPECT_GE(c.df, 1u);
        EXPECT_GT(c.score, 0.0f);
        s += double(c.score);
      }
      EXPECT_EQ(bits(float(s)), bits(r.hits[i].score)) << q.qid;
    }
  }
}

TEST(Deadline, ExpiredDeadlineReturnsPartial) {
  const auto& ix = corpus_index();
  for (int a = 0; a < kNumAlgorithms; ++a) {
    SearchOptions o;
    o.k = 10;
    o.algorithm = Algorithm(a);
    Deadline dl = Deadline::after_us(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto r = ix.search_text("costful bookly studentable climate write", o, dl);
    EXPECT_TRUE(r.partial) << algorithm_name(o.algorithm);
    EXPECT_LE(r.hits.size(), 10u);
    Deadline none;
    auto full = ix.search_text("costful bookly studentable climate write", o, none);
    EXPECT_FALSE(full.partial);
    EXPECT_EQ(full.hits.size(), 10u);
  }
}

TEST(Search, EmptyAndOutOfVocabularyQueries) {
  const auto& ix = corpus_index();
  SearchOptions o;
  Deadline dl;
  EXPECT_TRUE(ix.search_text("", o, dl).hits.empty());
  EXPECT_TRUE(ix.search_text("the of and", o, dl).hits.empty());
  EXPECT_TRUE(ix.search_text("zzqxnotaterm", o, dl).hits.empty());
  o.k = 0;
  EXPECT_TRUE(ix.search_text("costful", o, dl).hits.empty());
}

TEST(Search, RepeatedQueryTermIsBoosted) {
  const auto& ix = corpus_index();
  SearchOptions o;
  o.k = 5;
  o.explain = true;
  Deadline dl;
  auto once = ix.search_text("climate", o, dl);
  auto twice = ix.search_text("climate climate", o, dl);
  ASSERT_FALSE(once.hits.empty());
  ASSERT_EQ(once.hits.size(), twice.hits.size());
  for (size_t i = 0; i < once.hits.size(); ++i) {
    EXPECT_EQ(once.hits[i].doc, twice.hits[i].doc);
    EXPECT_GT(twice.hits[i].score, once.hits[i].score);
  }
}

}  // namespace
}  // namespace hs::lexical
