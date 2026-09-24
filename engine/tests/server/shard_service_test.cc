// End-to-end tests of the gRPC shard: a real lexical index and a real Vamana graph
// built into a temp directory, served in-process, called through a real channel.

#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

#include "hs/common/fbin.hpp"
#include "hs/lexical/index_builder.hpp"
#include "hs/server/shard_service.hpp"
#include "hs/vector/vamana.hpp"

namespace fs = std::filesystem;
namespace pb = hybridsearch::v1;
using namespace hs::server;

namespace {

constexpr uint64_t kIdBase = 7000000;  // global IDs are not ordinals: catch any mix-up
constexpr uint32_t kDocs = 400, kDim = 16;

const char* kWords[] = {"peru", "capital", "lima", "river", "amazon", "mountain", "andes", "coffee",
                        "export", "history", "inca", "empire", "city", "population", "ocean", "pacific",
                        "desert", "nazca", "lines", "gold", "silver", "copper", "fishing", "port"};

struct Fixture {
  fs::path dir;
  std::vector<std::string> texts;
  std::vector<float> vecs;
  std::vector<uint64_t> ids;

  Fixture() {
    dir = fs::temp_directory_path() / ("hs_shard_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> word(0, int(std::size(kWords)) - 1), len(5, 30);
    std::normal_distribution<float> g(0, 1);
    std::ofstream tsv(dir / "c.tsv");
    for (uint32_t i = 0; i < kDocs; ++i) {
      std::string t;
      int n = len(rng);
      for (int j = 0; j < n; ++j) t += (j ? " " : "") + std::string(kWords[word(rng)]);
      if (i % 50 == 0) t += " The Capital's story, told again.";
      texts.push_back(t);
      ids.push_back(kIdBase + 3 * i);
      tsv << ids.back() << '\t' << t << '\n';
      float norm = 0;
      std::vector<float> v(kDim);
      for (auto& x : v) norm += (x = g(rng)) * x;
      for (auto& x : v) vecs.push_back(x / std::sqrt(norm));
    }
    tsv.close();

    hs::lexical::BuildOptions o;
    o.input_tsv = (dir / "c.tsv").string();
    o.out_dir = (dir / "lex").string();
    o.threads = 1;
    o.mem_budget_mb = 64;
    o.verbose = false;
    o.min_free_gb = 0;
    hs::lexical::build_index(o);

    hs::write_fbin((dir / "v.fbin").string(), vecs.data(), kDocs, kDim);
    hs::write_u64bin((dir / "ids.u64bin").string(), ids);
    hs::vector::VamanaBuildParams p;
    p.R = 16;
    p.L = 32;
    p.threads = 1;
    auto idx = hs::vector::VamanaIndex::build(vecs.data(), kDocs, kDim, p);
    idx.save((dir / "g.graph").string());
  }
  ~Fixture() { fs::remove_all(dir); }
};

Fixture& fixture() {
  static Fixture f;
  return f;
}

struct Running {
  std::unique_ptr<ShardService> service;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<pb::Shard::Stub> stub;

  explicit Running(std::shared_ptr<Chaos> chaos = nullptr, bool with_vector = true) {
    auto& f = fixture();
    ShardConfig cfg;
    cfg.shard_id = 3;
    auto lex = hs::lexical::LexicalIndex::open((f.dir / "lex").string());
    auto docs = hs::lexical::DocStore::open((f.dir / "lex").string());
    std::unique_ptr<VectorBackend> vec;
    if (with_vector)
      vec = open_memory_backend((f.dir / "g.graph").string(), (f.dir / "v.fbin").string(),
                                (f.dir / "ids.u64bin").string());
    service = std::make_unique<ShardService>(cfg, std::move(lex), std::move(docs), std::move(vec), chaos);
    grpc::ServerBuilder b;
    int port = 0;
    b.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    b.RegisterService(service.get());
    server = b.BuildAndStart();
    stub = pb::Shard::NewStub(grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                                  grpc::InsecureChannelCredentials()));
  }
  ~Running() { server->Shutdown(); }
};

pb::SearchRequest lexical_request(const std::string& q, uint32_t k = 10) {
  pb::SearchRequest r;
  r.set_query(q);
  r.mutable_lexical()->set_enabled(true);
  r.mutable_lexical()->set_k(k);
  return r;
}

}  // namespace

TEST(ShardServer, LexicalMatchesLibraryWithGlobalIds) {
  Running s;
  auto& f = fixture();
  auto lex = hs::lexical::LexicalIndex::open((f.dir / "lex").string());
  for (const char* q : {"capital of peru", "inca empire gold", "pacific ocean fishing port", "nazca"}) {
    for (auto algo : {pb::PRUNING_EXHAUSTIVE, pb::PRUNING_MAXSCORE, pb::PRUNING_WAND, pb::PRUNING_BMW}) {
      auto req = lexical_request(q, 20);
      req.mutable_lexical()->set_algorithm(algo);
      pb::SearchResponse resp;
      grpc::ClientContext ctx;
      ASSERT_TRUE(s.stub->Search(&ctx, req, &resp).ok());

      hs::Deadline d;
      hs::lexical::SearchOptions o;
      o.k = 20;
      o.algorithm = to_algorithm(algo);
      auto direct = lex->search_text(q, o, d);
      ASSERT_EQ(size_t(resp.lexical_hits_size()), direct.hits.size()) << q;
      for (size_t i = 0; i < direct.hits.size(); ++i) {
        EXPECT_EQ(resp.lexical_hits(int(i)).doc_id(), lex->global_id(direct.hits[i].doc));
        EXPECT_EQ(resp.lexical_hits(int(i)).score(), direct.hits[i].score);
        EXPECT_GE(resp.lexical_hits(int(i)).doc_id(), kIdBase);
      }
      EXPECT_FALSE(resp.partial());
      EXPECT_EQ(resp.shard_id(), 3u);
      EXPECT_GT(resp.analyzed_query_terms_size(), 0);
    }
  }
}

TEST(ShardServer, ExplainReturnsPerTermContributionsThatSumToScore) {
  Running s;
  auto req = lexical_request("capital peru lima", 5);
  req.mutable_lexical()->set_explain(true);
  pb::SearchResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(s.stub->Search(&ctx, req, &resp).ok());
  ASSERT_GT(resp.lexical_hits_size(), 0);
  for (const auto& h : resp.lexical_hits()) {
    ASSERT_GT(h.terms_size(), 0);
    double sum = 0;
    for (const auto& t : h.terms()) {
      EXPECT_GT(t.tf(), 0u);
      EXPECT_GT(t.df(), 0u);
      sum += t.score();
    }
    EXPECT_NEAR(sum, h.score(), 1e-4 * std::max(1.0f, h.score()));
  }
}

TEST(ShardServer, VectorMatchesLibraryAndRunsAlongsideLexical) {
  Running s;
  auto& f = fixture();
  auto idx = hs::vector::VamanaIndex::load((f.dir / "g.graph").string(), (f.dir / "v.fbin").string());
  for (uint32_t qi : {0u, 17u, 399u}) {
    pb::SearchRequest req = lexical_request("river andes", 10);
    auto* v = req.mutable_vector();
    v->set_enabled(true);
    v->set_k(10);
    v->set_beam_width(64);
    for (uint32_t j = 0; j < kDim; ++j) v->add_query_embedding(f.vecs[qi * kDim + j]);
    pb::SearchResponse resp;
    grpc::ClientContext ctx;
    ASSERT_TRUE(s.stub->Search(&ctx, req, &resp).ok());

    hs::vector::VectorSearchOptions o;
    o.k = 10;
    o.L = 64;
    auto direct = idx.search(f.vecs.data() + qi * kDim, o);
    ASSERT_EQ(size_t(resp.vector_hits_size()), direct.hits.size());
    for (size_t i = 0; i < direct.hits.size(); ++i) {
      EXPECT_EQ(resp.vector_hits(int(i)).doc_id(), f.ids[direct.hits[i].doc]);
      EXPECT_FLOAT_EQ(resp.vector_hits(int(i)).score(), direct.hits[i].score);
    }
    // A stored vector is its own nearest neighbour.
    EXPECT_EQ(resp.vector_hits(0).doc_id(), f.ids[qi]);
    EXPECT_GT(resp.lexical_hits_size(), 0);
  }
}

TEST(ShardServer, RejectsWrongEmbeddingDimension) {
  Running s;
  pb::SearchRequest req;
  req.set_query("x");
  req.mutable_vector()->set_enabled(true);
  req.mutable_vector()->add_query_embedding(1.0f);
  pb::SearchResponse resp;
  grpc::ClientContext ctx;
  EXPECT_EQ(s.stub->Search(&ctx, req, &resp).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(ShardServer, FetchReturnsSnippetsWithHighlightsAndOmitsForeignIds) {
  Running s;
  auto& f = fixture();
  pb::FetchRequest req;
  req.add_doc_ids(f.ids[0]);
  req.add_doc_ids(123);  // not in this shard
  req.add_doc_ids(f.ids[50]);
  req.add_highlight_terms("capit");  // analyzed form of "Capital's"
  req.set_snippet_chars(60);
  pb::FetchResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(s.stub->Fetch(&ctx, req, &resp).ok());
  ASSERT_EQ(resp.documents_size(), 2);
  EXPECT_EQ(resp.documents(0).doc_id(), f.ids[0]);
  EXPECT_EQ(resp.documents(1).doc_id(), f.ids[50]);
  for (const auto& d : resp.documents()) {
    EXPECT_LE(d.text().size(), 60u + 8u);  // edges move to whitespace, never far past the budget
    EXPECT_GT(d.doc_length(), 0u);
    ASSERT_GT(d.highlights_size(), 0);
    for (const auto& h : d.highlights()) {
      ASSERT_LE(h.end(), d.text().size());
      EXPECT_EQ(d.text().substr(h.start(), h.end() - h.start()).rfind("Capital", 0), 0u)
          << d.text().substr(h.start(), h.end() - h.start());
    }
  }

  pb::FetchRequest full;
  full.add_doc_ids(f.ids[7]);
  pb::FetchResponse fresp;
  grpc::ClientContext ctx2;
  ASSERT_TRUE(s.stub->Fetch(&ctx2, full, &fresp).ok());
  ASSERT_EQ(fresp.documents_size(), 1);
  EXPECT_EQ(fresp.documents(0).text(), f.texts[7]);
  EXPECT_FALSE(fresp.documents(0).is_snippet());
}

TEST(ShardServer, HealthReportsSliceAndIndexes) {
  Running s;
  pb::HealthResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(s.stub->Health(&ctx, pb::HealthRequest(), &resp).ok());
  EXPECT_EQ(resp.shard_id(), 3u);
  EXPECT_EQ(resp.first_doc_id(), fixture().ids.front());
  EXPECT_EQ(resp.last_doc_id(), fixture().ids.back());
  EXPECT_EQ(resp.num_docs(), kDocs);
  EXPECT_TRUE(resp.lexical_ready());
  EXPECT_TRUE(resp.vector_ready());
  EXPECT_EQ(resp.vector_dim(), kDim);
}

TEST(ShardServer, LexicalOnlyShardIgnoresVectorRequest) {
  Running s(nullptr, /*with_vector=*/false);
  auto req = lexical_request("peru");
  req.mutable_vector()->set_enabled(true);
  req.mutable_vector()->add_query_embedding(1.0f);
  pb::SearchResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(s.stub->Search(&ctx, req, &resp).ok());
  EXPECT_EQ(resp.vector_hits_size(), 0);
  EXPECT_GT(resp.lexical_hits_size(), 0);
}

class ChaosFile {
 public:
  ChaosFile() : path_(fixture().dir / ("chaos_" + std::to_string(counter_++))) {}
  void write(const std::string& s) {
    std::ofstream(path_) << s;
    // The shard re-reads at most every 250 ms, keyed on mtime.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }
  std::string path() const { return path_.string(); }

 private:
  static inline int counter_ = 0;
  fs::path path_;
};

TEST(ShardServer, ChaosFailRateAndRecoveryAtRuntime) {
  ChaosFile cf;
  auto chaos = std::make_shared<Chaos>(cf.path());
  Running s(chaos);
  auto search = [&] {
    pb::SearchResponse resp;
    grpc::ClientContext ctx;
    return s.stub->Search(&ctx, lexical_request("peru"), &resp).error_code();
  };
  EXPECT_EQ(search(), grpc::StatusCode::OK);
  cf.write("fail_rate=1\n");
  EXPECT_EQ(search(), grpc::StatusCode::UNAVAILABLE);
  cf.write("fail_rate=0\n");
  EXPECT_EQ(search(), grpc::StatusCode::OK);
  EXPECT_GE(s.service->injected_failures(), 1u);
}

TEST(ShardServer, ChaosLatencyIsAddedAndBlackholeHoldsToDeadline) {
  ChaosFile cf;
  auto chaos = std::make_shared<Chaos>(cf.path());
  Running s(chaos);
  cf.write("latency_ms=80\n");
  auto t0 = std::chrono::steady_clock::now();
  pb::SearchResponse resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(s.stub->Search(&ctx, lexical_request("peru"), &resp).ok());
  EXPECT_GE(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(80));

  cf.write("blackhole=1\n");
  auto req = lexical_request("peru");
  req.set_deadline_budget_us(50'000);
  t0 = std::chrono::steady_clock::now();
  grpc::ClientContext ctx2;
  EXPECT_EQ(s.stub->Search(&ctx2, req, &resp).error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
  EXPECT_GE(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(50));
}

TEST(ShardServer, GrpcDeadlineTightensTheBudget) {
  // Injected latency consumes most of the gRPC deadline; whatever is left is what the
  // retrievers see, and a response that arrives in time must still be well-formed.
  ChaosFile cf;
  auto chaos = std::make_shared<Chaos>(cf.path());
  Running s(chaos);
  cf.write("latency_ms=150\n");
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(100));
  pb::SearchResponse resp;
  EXPECT_EQ(s.stub->Search(&ctx, lexical_request("peru"), &resp).error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
}

TEST(Chaos, ParsesAndClampsAndIgnoresGarbage) {
  auto s = parse_chaos("latency_ms = 50\njitter_ms=5 # comment\nfail_rate=7\nblackhole=true\nbogus=1\nlatency_ms=x\n");
  EXPECT_EQ(s.latency_ms, 50u);
  EXPECT_EQ(s.jitter_ms, 5u);
  EXPECT_EQ(s.fail_rate, 1.0);
  EXPECT_TRUE(s.blackhole);
  EXPECT_FALSE(parse_chaos("").any());
}

TEST(Trace, ParsesW3cTraceparent) {
  auto t = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
  ASSERT_TRUE(t.has_value());
  EXPECT_EQ(t->trace_id, "4bf92f3577b34da6a3ce929d0e0e4736");
  EXPECT_EQ(t->parent_id, "00f067aa0ba902b7");
  EXPECT_TRUE(t->sampled);
  EXPECT_FALSE(parse_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01"));
  EXPECT_FALSE(parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01"));
  EXPECT_FALSE(parse_traceparent("ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"));
  EXPECT_FALSE(parse_traceparent("00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01"));
  EXPECT_FALSE(parse_traceparent("garbage"));
  EXPECT_EQ(random_span_id().size(), 16u);
}

TEST(Trace, OtlpJsonCarriesIdsAndEscapes) {
  Span s;
  s.trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
  s.parent_id = "00f067aa0ba902b7";
  s.span_id = "1111111111111111";
  s.name = "shard.search";
  s.start_unix_ns = 1;
  s.end_unix_ns = 2;
  s.str_attrs = {{"q", "a \"quoted\"\nline"}};
  s.int_attrs = {{"shard.id", 3}};
  std::string j = spans_to_otlp_json({s}, "hs-shard-3");
  EXPECT_NE(j.find(R"("traceId":"4bf92f3577b34da6a3ce929d0e0e4736")"), std::string::npos);
  EXPECT_NE(j.find(R"("parentSpanId":"00f067aa0ba902b7")"), std::string::npos);
  EXPECT_NE(j.find(R"(a \"quoted\"\nline)"), std::string::npos);
  EXPECT_NE(j.find(R"("intValue":"3")"), std::string::npos);
}

TEST(Trace, ExporterDropsWhenCollectorIsDown) {
  SpanExporter ex("http://127.0.0.1:1", "t");  // nothing listens on port 1
  Span s;
  s.trace_id = std::string(32, 'a');
  s.span_id = std::string(16, 'b');
  ex.record(s);
  for (int i = 0; i < 50 && ex.dropped() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(ex.dropped(), 1u);
  EXPECT_EQ(ex.exported(), 0u);
}
