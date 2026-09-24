// Property tests on random corpora: (1) exhaustive search equals a brute-force scorer that
// never touches the index (tf counted from the raw token lists), and (2) every pruning
// algorithm x codec equals exhaustive, over random corpora shapes, (k1, b), k and queries.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <fstream>
#include <random>

#include "test_util.hpp"

namespace hs::lexical {
namespace {

using testing::bits;

struct Corpus {
  std::vector<std::vector<std::string>> docs;  // analyzed tokens (terms are "w<n>", analyzer-stable)
};

// "w12" survives the analyzer unchanged (alphanumeric, lowercase, no Porter suffix rule fires).
std::string word(uint32_t i) { return "w" + std::to_string(i); }

LexicalResult brute_force(const Corpus& c, const std::vector<std::string>& q, const SearchOptions& o) {
  // Collection stats exactly as the index defines them.
  uint64_t n_with = 0, sum = 0;
  std::map<std::string, uint32_t> df;
  for (const auto& d : c.docs) {
    sum += d.size();
    n_with += !d.empty();
    std::set<std::string> u(d.begin(), d.end());
    for (const auto& t : u) ++df[t];
  }
  float avgdl = bm25_avgdl(sum, n_with);
  Bm25Norms bn;
  bn.init(o.model, o.k1, o.b, avgdl);
  std::vector<std::pair<std::string, uint32_t>> uniq;
  for (const auto& t : q) {
    auto it = std::find_if(uniq.begin(), uniq.end(), [&](auto& u) { return u.first == t; });
    if (it == uniq.end()) uniq.emplace_back(t, 1);
    else ++it->second;
  }
  TopK top(o.k);
  for (uint32_t d = 0; d < c.docs.size(); ++d) {
    double s = 0;
    bool any = false;
    for (const auto& [t, boost] : uniq) {
      if (!df.count(t)) continue;
      uint32_t tf = uint32_t(std::count(c.docs[d].begin(), c.docs[d].end(), t));
      if (!tf) continue;
      any = true;
      float w = bm25_weight(o.model, float(boost), bm25_idf(df[t], n_with), o.k1);
      uint32_t dl = uint32_t(c.docs[d].size());
      s += double(bm25_term_score(w, float(tf), bn.inv(dl, smallfloat::int_to_byte4(dl))));
    }
    if (any) top.push(d, float(s));
  }
  LexicalResult r;
  r.hits = top.take_sorted();
  return r;
}

TEST(RandomCorpora, AllAlgorithmsEqualBruteForce) {
  std::mt19937 rng(20260923);
  uint64_t checks = 0;
  for (int trial = 0; trial < 24; ++trial) {
    const uint32_t vocab = 3 + rng() % 400;
    const uint32_t ndocs = 1 + rng() % 3000;
    const double skew = 0.5 + (rng() % 100) / 50.0;
    std::vector<double> w(vocab);
    for (uint32_t i = 0; i < vocab; ++i) w[i] = 1.0 / std::pow(double(i + 1), skew);
    std::discrete_distribution<uint32_t> pick(w.begin(), w.end());
    Corpus c;
    const std::string dir = testing::temp_dir("rand" + std::to_string(trial));
    const std::string tsv = dir + "/c.tsv";
    {
      std::ofstream out(tsv);
      for (uint32_t d = 0; d < ndocs; ++d) {
        uint32_t len = rng() % 20 == 0 ? 0 : (rng() % 10 == 0 ? 100 + rng() % 900 : 1 + rng() % 60);
        std::vector<std::string> toks;
        std::string text;
        for (uint32_t i = 0; i < len; ++i) {
          toks.push_back(word(pick(rng)));
          text += (i ? (rng() % 7 == 0 ? " the " : " ") : "") + toks.back();
        }
        if (len == 0) text = "the of and";
        c.docs.push_back(toks);
        out << (d * 3 + 1) << '\t' << text << '\n';
      }
    }
    BuildOptions bo;
    bo.input_tsv = tsv;
    bo.out_dir = dir + "/ix";
    bo.threads = 1 + rng() % 3;
    bo.mem_budget_mb = 1;
    bo.docstore = trial % 3 == 0;
    bo.verbose = false;
    bo.min_free_gb = 0.1;
    build_index(bo);
    auto ix = LexicalIndex::open(bo.out_dir);
    ASSERT_EQ(ix->num_docs(), ndocs);

    for (int qi = 0; qi < 40; ++qi) {
      std::vector<std::string> q;
      uint32_t nq = 1 + rng() % 9;
      for (uint32_t i = 0; i < nq; ++i) {
        if (rng() % 10 == 0) q.push_back("zz" + std::to_string(rng() % 5));
        else if (rng() % 8 == 0 && !q.empty()) q.push_back(q.back());
        else q.push_back(word(rng() % 3 == 0 ? rng() % vocab : pick(rng)));
      }
      SearchOptions o;
      o.k = std::vector<uint32_t>{1, 2, 5, 10, 50, 1000}[rng() % 6];
      o.model = rng() % 2 ? Model::Lucene : Model::Textbook;
      if (rng() % 2) {
        o.k1 = 0.1f + float(rng() % 30) / 10.0f;
        o.b = float(rng() % 11) / 10.0f;
      }
      auto want = brute_force(c, q, o);
      for (int a = 0; a < kNumAlgorithms; ++a) {
        for (Codec cc : {Codec::VByte, Codec::BP128}) {
          o.algorithm = Algorithm(a);
          o.codec = cc;
          Deadline dl;
          auto got = ix->search(q, o, dl);
          ASSERT_EQ(got.hits.size(), want.hits.size()) << "trial " << trial << " " << algorithm_name(o.algorithm);
          for (size_t i = 0; i < got.hits.size(); ++i) {
            ASSERT_EQ(got.hits[i].doc, want.hits[i].doc)
                << "trial " << trial << " q" << qi << " " << algorithm_name(o.algorithm) << "/" << codec_name(cc)
                << " rank " << i;
            ASSERT_EQ(bits(got.hits[i].score), bits(want.hits[i].score));
          }
          ++checks;
        }
      }
    }
    ix.reset();
    std::filesystem::remove_all(dir);
  }
  EXPECT_GT(checks, 9000u);
}

}  // namespace
}  // namespace hs::lexical
