// hs_shard: serve one slice of the collection over gRPC (proto/hybridsearch/v1/shard.proto).
//
//   hs_shard --port 50051 --shard-id 0 --lexical data/indexes/lexical/1m-4shards/shard0
//            [--vector-disk DIR [--cold] [--cache-nodes N] [--io-threads 8]]
//            [--vamana-graph G --vectors F.fbin [--docids D.u64bin]]
//            [--threads 8] [--chaos-file /tmp/hs-shard0.chaos]
//            [--otlp http://localhost:4318] [--sequential-retrieval]
//
// --threads caps concurrent RPCs. --chaos-file is re-read at runtime (see chaos.hpp).
// --cold opens the SSD index with the page cache bypassed (F_NOCACHE / O_DIRECT);
// the default is warm, and every latency number says which one it was.

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/resource_quota.h>

#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "hs/server/shard_service.hpp"

using namespace hs::server;

namespace {

std::unique_ptr<grpc::Server>* g_server = nullptr;

void on_signal(int) {
  // Shutdown from a signal handler must not block; hand it to a thread.
  if (g_server && *g_server) std::thread([] { (*g_server)->Shutdown(); }).detach();
}

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: hs_shard --port P --shard-id I --lexical DIR\n"
               "                [--vector-disk DIR [--cold] [--cache-nodes N] [--io-threads N]]\n"
               "                [--vamana-graph G --vectors F.fbin [--docids D.u64bin]]\n"
               "                [--threads N] [--chaos-file PATH] [--otlp URL] [--sequential-retrieval]\n"
               "                [--default-L N] [--default-io-width W]\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  int port = 50051;
  ShardConfig cfg;
  std::string lexical_dir, disk_dir, graph, vectors, docids, chaos_file, otlp;
  bool cold = false;
  uint32_t cache_nodes = 0;
  unsigned io_threads = 8, threads = 8;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--port") port = std::stoi(val());
    else if (a == "--shard-id") cfg.shard_id = uint32_t(std::stoul(val()));
    else if (a == "--lexical") lexical_dir = val();
    else if (a == "--vector-disk") disk_dir = val();
    else if (a == "--cold") cold = true;
    else if (a == "--cache-nodes") cache_nodes = uint32_t(std::stoul(val()));
    else if (a == "--io-threads") io_threads = unsigned(std::stoul(val()));
    else if (a == "--vamana-graph") graph = val();
    else if (a == "--vectors") vectors = val();
    else if (a == "--docids") docids = val();
    else if (a == "--threads") threads = unsigned(std::stoul(val()));
    else if (a == "--chaos-file") chaos_file = val();
    else if (a == "--otlp") otlp = val();
    else if (a == "--sequential-retrieval") cfg.parallel_retrieval = false;
    else if (a == "--default-L") cfg.default_L = uint32_t(std::stoul(val()));
    else if (a == "--default-io-width") cfg.default_io_width = uint32_t(std::stoul(val()));
    else usage();
  }
  if (lexical_dir.empty() && disk_dir.empty() && graph.empty()) usage();
  if (!graph.empty() && vectors.empty()) usage();

  std::unique_ptr<hs::lexical::LexicalIndex> lex;
  std::unique_ptr<hs::lexical::DocStore> docs;
  std::unique_ptr<VectorBackend> vec;
  try {
    if (!lexical_dir.empty()) {
      lex = hs::lexical::LexicalIndex::open(lexical_dir);
      docs = hs::lexical::DocStore::open(lexical_dir);
      std::fprintf(stderr, "[shard %u] lexical: %s, %llu docs\n", cfg.shard_id, lexical_dir.c_str(),
                   static_cast<unsigned long long>(lex->num_docs()));
    }
    if (!disk_dir.empty()) {
      vec = open_disk_backend(disk_dir, cold, cache_nodes, io_threads);
    } else if (!graph.empty()) {
      vec = open_memory_backend(graph, vectors, docids);
    }
    if (vec) std::fprintf(stderr, "[shard %u] vector: %s (%s)\n", cfg.shard_id, vec->describe().c_str(),
                          disk_dir.empty() ? "memory" : (cold ? "cold" : "warm"));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[shard %u] failed to open indexes: %s\n", cfg.shard_id, e.what());
    return 1;
  }

  cfg.build_info = "hs_shard lexical=" + (lexical_dir.empty() ? std::string("none") : lexical_dir);
  std::shared_ptr<Chaos> chaos = chaos_file.empty() ? nullptr : std::make_shared<Chaos>(chaos_file);
  std::shared_ptr<SpanExporter> spans =
      otlp.empty() ? nullptr : std::make_shared<SpanExporter>(otlp, "hs-shard-" + std::to_string(cfg.shard_id));

  ShardService service(cfg, std::move(lex), std::move(docs), std::move(vec), chaos, spans);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  std::string addr = "0.0.0.0:" + std::to_string(port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  grpc::ResourceQuota quota("hs-shard");
  quota.SetMaxThreads(int(threads) + 2);  // the sync server needs a couple of its own
  builder.SetResourceQuota(quota);

  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (!server) {
    std::fprintf(stderr, "[shard %u] could not listen on %s\n", cfg.shard_id, addr.c_str());
    return 1;
  }
  g_server = &server;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::fprintf(stderr, "[shard %u] listening on %s%s%s\n", cfg.shard_id, addr.c_str(),
               chaos ? ", chaos file " : "", chaos ? chaos_file.c_str() : "");
  server->Wait();
  std::fprintf(stderr, "[shard %u] stopped after %llu searches (%llu partial, %llu injected failures)\n",
               cfg.shard_id, static_cast<unsigned long long>(service.searches()),
               static_cast<unsigned long long>(service.partials()),
               static_cast<unsigned long long>(service.injected_failures()));
  return 0;
}
