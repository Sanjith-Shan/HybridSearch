#include "hs/server/shard_service.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <thread>

#include "hs/common/fbin.hpp"
#include "hs/lexical/snippet.hpp"
#include "hs/vector/disk_index.hpp"
#include "hs/vector/vamana.hpp"

namespace hs::server {

namespace pb = hybridsearch::v1;
using hs::lexical::Algorithm;

namespace {

class DiskBackend final : public VectorBackend {
 public:
  explicit DiskBackend(std::unique_ptr<hs::vector::DiskIndex> idx) : idx_(std::move(idx)) {}
  hs::vector::VectorResult search(const float* q, const hs::vector::VectorSearchOptions& o,
                                  hs::Deadline& d) const override {
    return idx_->search(q, o, d);
  }
  uint64_t global_id(uint32_t i) const override { return idx_->global_id(i); }
  uint32_t dim() const override { return idx_->dim(); }
  uint32_t size() const override { return idx_->size(); }
  std::string describe() const override {
    return "diskann n=" + std::to_string(idx_->size()) + " R=" + std::to_string(idx_->max_degree()) +
           " pq_M=" + std::to_string(idx_->pq().M());
  }

 private:
  std::unique_ptr<hs::vector::DiskIndex> idx_;
};

class MemoryBackend final : public VectorBackend {
 public:
  explicit MemoryBackend(hs::vector::VamanaIndex idx) : idx_(std::move(idx)) {}
  hs::vector::VectorResult search(const float* q, const hs::vector::VectorSearchOptions& o,
                                  hs::Deadline& d) const override {
    return idx_.search(q, o, d);
  }
  uint64_t global_id(uint32_t i) const override { return idx_.global_id(i); }
  uint32_t dim() const override { return idx_.dim(); }
  uint32_t size() const override { return idx_.size(); }
  std::string describe() const override {
    return "vamana-memory n=" + std::to_string(idx_.size()) + " R=" + std::to_string(idx_.max_degree());
  }

 private:
  hs::vector::VamanaIndex idx_;
};

void sleep_us(uint64_t us) {
  if (us) std::this_thread::sleep_for(std::chrono::microseconds(us));
}

}  // namespace

std::unique_ptr<VectorBackend> open_disk_backend(const std::string& dir, bool direct_io, uint32_t cache_nodes,
                                                 unsigned io_threads) {
  hs::vector::DiskOpenOptions o;
  o.direct_io = direct_io;
  o.cache_nodes = cache_nodes;
  o.io_threads = io_threads;
  return std::make_unique<DiskBackend>(hs::vector::DiskIndex::open(dir, o));
}

std::unique_ptr<VectorBackend> open_memory_backend(const std::string& graph, const std::string& vectors_fbin,
                                                   const std::string& docids_u64bin) {
  auto idx = hs::vector::VamanaIndex::load(graph, vectors_fbin);
  if (!docids_u64bin.empty()) idx.set_docids(hs::read_u64bin(docids_u64bin));
  return std::make_unique<MemoryBackend>(std::move(idx));
}

uint64_t effective_budget_us(uint64_t request_budget_us, const grpc::ServerContext* ctx) {
  uint64_t budget = request_budget_us;
  auto deadline = ctx->deadline();
  // gRPC reports "no deadline" as a time point far in the future.
  auto now = std::chrono::system_clock::now();
  if (deadline < now + std::chrono::hours(24 * 365)) {
    int64_t left = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
    uint64_t grpc_budget = left > 0 ? uint64_t(left) : 1;  // 1 us: already expired, but never "unlimited"
    budget = budget == 0 ? grpc_budget : std::min(budget, grpc_budget);
  }
  return budget;
}

Algorithm to_algorithm(pb::PruningAlgorithm a) {
  switch (a) {
    case pb::PRUNING_EXHAUSTIVE: return Algorithm::Exhaustive;
    case pb::PRUNING_DAAT: return Algorithm::DAAT;
    case pb::PRUNING_MAXSCORE: return Algorithm::MaxScore;
    case pb::PRUNING_WAND: return Algorithm::WAND;
    case pb::PRUNING_BMW:
    case pb::PRUNING_DEFAULT:
    default: return Algorithm::BMW;
  }
}

ShardService::ShardService(ShardConfig cfg, std::unique_ptr<hs::lexical::LexicalIndex> lexical,
                           std::unique_ptr<hs::lexical::DocStore> docstore, std::unique_ptr<VectorBackend> vector,
                           std::shared_ptr<Chaos> chaos, std::shared_ptr<SpanExporter> spans)
    : cfg_(std::move(cfg)),
      lex_(std::move(lexical)),
      docs_(std::move(docstore)),
      vec_(std::move(vector)),
      chaos_(std::move(chaos)),
      spans_(std::move(spans)) {
  if (lex_ && lex_->num_docs() > 0) {
    first_doc_ = UINT64_MAX;
    for (uint64_t i = 0; i < lex_->num_docs(); ++i) {
      first_doc_ = std::min(first_doc_, lex_->global_id(uint32_t(i)));
      last_doc_ = std::max(last_doc_, lex_->global_id(uint32_t(i)));
    }
  }
}

grpc::Status ShardService::Search(grpc::ServerContext* ctx, const pb::SearchRequest* req, pb::SearchResponse* resp) {
  searches_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t start_ns = unix_now_ns();
  const auto t0 = std::chrono::steady_clock::now();

  // Trace context from the broker, if any.
  std::optional<TraceContext> trace;
  if (spans_) {
    auto md = ctx->client_metadata().find("traceparent");
    if (md != ctx->client_metadata().end()) trace = parse_traceparent(std::string(md->second.data(), md->second.size()));
  }
  auto finish_span = [&](bool error, const pb::SearchResponse* r) {
    if (!spans_ || !trace) return;
    Span s;
    s.trace_id = trace->trace_id;
    s.parent_id = trace->parent_id;
    s.span_id = random_span_id();
    s.name = "shard.search";
    s.start_unix_ns = start_ns;
    s.end_unix_ns = unix_now_ns();
    s.error = error;
    s.int_attrs = {{"shard.id", cfg_.shard_id}};
    s.str_attrs = {{"request.id", req->request_id()}};
    if (r) {
      s.int_attrs.push_back({"docs_scored", int64_t(r->stats().docs_scored())});
      s.int_attrs.push_back({"ssd_reads", int64_t(r->stats().ssd_reads())});
      s.int_attrs.push_back({"lexical_hits", r->lexical_hits_size()});
      s.int_attrs.push_back({"vector_hits", r->vector_hits_size()});
      s.int_attrs.push_back({"partial", r->partial() ? 1 : 0});
    }
    spans_->record(std::move(s));
  };

  const uint64_t budget_us = effective_budget_us(req->deadline_budget_us(), ctx);

  // Fault injection happens first so injected latency eats into the same budget
  // a real slow shard would.
  if (chaos_) {
    ChaosSettings cs = chaos_->current();
    if (cs.any()) {
      if (cs.blackhole) {
        sleep_us(budget_us ? budget_us : 30'000'000);
        injected_failures_.fetch_add(1, std::memory_order_relaxed);
        finish_span(true, nullptr);
        return {grpc::StatusCode::DEADLINE_EXCEEDED, "chaos: blackhole"};
      }
      sleep_us(chaos_->sample_delay_us(cs));
      if (chaos_->sample_failure(cs)) {
        injected_failures_.fetch_add(1, std::memory_order_relaxed);
        finish_span(true, nullptr);
        return {grpc::StatusCode::UNAVAILABLE, "chaos: injected failure"};
      }
    }
  }
  // The budget is re-measured after injected delay: what is left is what the retrievers get.
  uint64_t elapsed_us =
      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
  uint64_t remaining_us = budget_us == 0 ? 0 : (budget_us > elapsed_us ? budget_us - elapsed_us : 1);

  const auto& lp = req->lexical();
  const auto& vp = req->vector();
  const bool want_lex = lp.enabled() && lex_;
  const bool want_vec = vp.enabled() && vec_;
  if (vp.enabled() && vec_ && uint32_t(vp.query_embedding_size()) != vec_->dim()) {
    finish_span(true, nullptr);
    return {grpc::StatusCode::INVALID_ARGUMENT,
            "query_embedding has " + std::to_string(vp.query_embedding_size()) + " dims, index has " +
                std::to_string(vec_->dim())};
  }

  // Analyze once; the terms are returned so the broker can highlight without re-analysis.
  std::vector<std::string> terms;
  if (lex_) {
    lex_->analyzer().analyze(req->query(), terms);
    for (auto& t : terms) resp->add_analyzed_query_terms(t);
  }

  auto clamp_k = [&](uint32_t k) { return std::min(k == 0 ? cfg_.default_k : k, cfg_.max_k); };

  // Vector retrieval, possibly on its own thread. Each retriever gets its own
  // Deadline object over the same budget: the clock is shared, the state is not.
  hs::vector::VectorResult vres;
  uint64_t vec_us = 0;
  auto run_vector = [&] {
    auto v0 = std::chrono::steady_clock::now();
    hs::Deadline d = hs::Deadline::after_us(remaining_us);
    hs::vector::VectorSearchOptions o;
    o.k = clamp_k(vp.k());
    o.L = std::max(vp.beam_width() ? vp.beam_width() : cfg_.default_L, o.k);
    o.beam_width = vp.io_width() ? vp.io_width() : cfg_.default_io_width;
    vres = vec_->search(vp.query_embedding().data(), o, d);
    vec_us = uint64_t(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - v0).count());
  };
  std::future<void> vec_future;
  if (want_vec && want_lex && cfg_.parallel_retrieval) vec_future = std::async(std::launch::async, run_vector);

  hs::lexical::LexicalResult lres;
  uint64_t lex_us = 0;
  if (want_lex) {
    auto l0 = std::chrono::steady_clock::now();
    hs::Deadline d = hs::Deadline::after_us(remaining_us);
    hs::lexical::SearchOptions o;
    o.k = clamp_k(lp.k());
    o.algorithm = to_algorithm(lp.algorithm());
    if (lp.k1() > 0) o.k1 = lp.k1();
    if (lp.b() > 0) o.b = lp.b();
    o.explain = lp.explain();
    lres = lex_->search(terms, o, d);
    lex_us = uint64_t(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - l0).count());
  }
  if (vec_future.valid()) {
    vec_future.get();
  } else if (want_vec) {
    run_vector();
  }

  for (size_t i = 0; i < lres.hits.size(); ++i) {
    auto* h = resp->add_lexical_hits();
    h->set_doc_id(lex_->global_id(lres.hits[i].doc));
    h->set_score(lres.hits[i].score);
    if (i < lres.explain.size()) {
      for (const auto& tc : lres.explain[i]) {
        auto* t = h->add_terms();
        t->set_term(tc.term);
        t->set_score(tc.score);
        t->set_tf(tc.tf);
        t->set_df(tc.df);
      }
    }
  }
  for (const auto& hit : vres.hits) {
    auto* h = resp->add_vector_hits();
    h->set_doc_id(vec_->global_id(hit.doc));
    h->set_score(hit.score);
  }

  const bool partial = lres.partial || vres.partial;
  resp->set_partial(partial);
  if (partial) partials_.fetch_add(1, std::memory_order_relaxed);
  resp->set_shard_id(cfg_.shard_id);
  auto* st = resp->mutable_stats();
  st->set_docs_scored(lres.docs_scored);
  st->set_postings_decoded(lres.postings_decoded);
  st->set_vector_distance_computations(vres.distance_computations + vres.pq_distance_computations);
  st->set_ssd_reads(vres.ssd_reads);
  st->set_lexical_us(lex_us);
  st->set_vector_us(vec_us);

  finish_span(false, resp);
  return grpc::Status::OK;
}

grpc::Status ShardService::Fetch(grpc::ServerContext*, const pb::FetchRequest* req, pb::FetchResponse* resp) {
  if (!lex_ || !docs_) return {grpc::StatusCode::FAILED_PRECONDITION, "shard has no doc store"};
  std::vector<std::string> terms(req->highlight_terms().begin(), req->highlight_terms().end());
  for (uint64_t id : req->doc_ids()) {
    int64_t ord = lex_->ordinal_of(id);
    if (ord < 0) continue;  // not this shard's document: omitted, per the contract
    std::string text = docs_->text(uint32_t(ord));
    auto* d = resp->add_documents();
    d->set_doc_id(id);
    d->set_doc_length(lex_->doc_length(uint32_t(ord)));
    if (terms.empty() && req->snippet_chars() == 0) {
      d->set_text(std::move(text));
      continue;
    }
    hs::lexical::Snippet sn = hs::lexical::select_snippet(lex_->analyzer(), text, terms, req->snippet_chars());
    d->set_text(sn.window);
    d->set_is_snippet(sn.is_snippet);
    for (const auto& hl : sn.highlights) {
      auto* h = d->add_highlights();
      h->set_start(hl.start);
      h->set_end(hl.end);
    }
  }
  return grpc::Status::OK;
}

grpc::Status ShardService::Health(grpc::ServerContext*, const pb::HealthRequest*, pb::HealthResponse* resp) {
  resp->set_shard_id(cfg_.shard_id);
  resp->set_first_doc_id(first_doc_);
  resp->set_last_doc_id(last_doc_);
  resp->set_num_docs(lex_ ? lex_->num_docs() : (vec_ ? vec_->size() : 0));
  resp->set_lexical_ready(lex_ != nullptr);
  resp->set_vector_ready(vec_ != nullptr);
  resp->set_vector_dim(vec_ ? vec_->dim() : 0);
  std::string info = cfg_.build_info;
  if (vec_) info += (info.empty() ? "" : "; ") + vec_->describe();
  resp->set_build_info(info);
  return grpc::Status::OK;
}

}  // namespace hs::server
