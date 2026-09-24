#pragma once
// The gRPC shard server: one slice of the collection, both indexes, one Search RPC.
//
// A shard owns a lexical index (with its doc store) and optionally a vector index
// for the same slice. Search runs the two retrievers concurrently under one
// deadline: the tighter of the broker's propagated budget and the gRPC deadline.
// When the deadline fires, each retriever returns its best partial top-k and the
// response says partial=true. Document IDs on the wire are always global.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "hs/lexical/index.hpp"
#include "hs/server/chaos.hpp"
#include "hs/server/trace.hpp"
#include "hs/vector/types.hpp"
#include "hybridsearch/v1/shard.grpc.pb.h"

namespace hs::server {

// Either vector index type behind one interface.
class VectorBackend {
 public:
  virtual ~VectorBackend() = default;
  virtual hs::vector::VectorResult search(const float* q, const hs::vector::VectorSearchOptions& opts,
                                          hs::Deadline& deadline) const = 0;
  virtual uint64_t global_id(uint32_t ordinal) const = 0;
  virtual uint32_t dim() const = 0;
  virtual uint32_t size() const = 0;
  virtual std::string describe() const = 0;
};

// DiskANN-style SSD index directory (hs_vec_disk_build output).
std::unique_ptr<VectorBackend> open_disk_backend(const std::string& dir, bool direct_io, uint32_t cache_nodes,
                                                 unsigned io_threads);
// In-memory Vamana graph over an .fbin, with an optional docids.u64bin.
std::unique_ptr<VectorBackend> open_memory_backend(const std::string& graph, const std::string& vectors_fbin,
                                                   const std::string& docids_u64bin);

struct ShardConfig {
  uint32_t shard_id = 0;
  uint32_t default_k = 10;
  uint32_t max_k = 1000;
  uint32_t default_L = 100;      // vector L_search when the request leaves it 0
  uint32_t default_io_width = 4; // vector beam width W when the request leaves it 0
  bool parallel_retrieval = true;
  std::string build_info;
};

class ShardService final : public hybridsearch::v1::Shard::Service {
 public:
  ShardService(ShardConfig cfg, std::unique_ptr<hs::lexical::LexicalIndex> lexical,
               std::unique_ptr<hs::lexical::DocStore> docstore, std::unique_ptr<VectorBackend> vector,
               std::shared_ptr<Chaos> chaos = nullptr, std::shared_ptr<SpanExporter> spans = nullptr);

  grpc::Status Search(grpc::ServerContext* ctx, const hybridsearch::v1::SearchRequest* req,
                      hybridsearch::v1::SearchResponse* resp) override;
  grpc::Status Fetch(grpc::ServerContext* ctx, const hybridsearch::v1::FetchRequest* req,
                     hybridsearch::v1::FetchResponse* resp) override;
  grpc::Status Health(grpc::ServerContext* ctx, const hybridsearch::v1::HealthRequest* req,
                      hybridsearch::v1::HealthResponse* resp) override;

  // Counters for logs and tests.
  uint64_t searches() const { return searches_.load(); }
  uint64_t partials() const { return partials_.load(); }
  uint64_t injected_failures() const { return injected_failures_.load(); }

 private:
  ShardConfig cfg_;
  std::unique_ptr<hs::lexical::LexicalIndex> lex_;
  std::unique_ptr<hs::lexical::DocStore> docs_;
  std::unique_ptr<VectorBackend> vec_;
  std::shared_ptr<Chaos> chaos_;
  std::shared_ptr<SpanExporter> spans_;
  uint64_t first_doc_ = 0, last_doc_ = 0;
  std::atomic<uint64_t> searches_{0}, partials_{0}, injected_failures_{0};
};

// Microseconds left under the tighter of the request budget and the gRPC deadline;
// 0 means unlimited.
uint64_t effective_budget_us(uint64_t request_budget_us, const grpc::ServerContext* ctx);

hs::lexical::Algorithm to_algorithm(hybridsearch::v1::PruningAlgorithm a);

}  // namespace hs::server
