#pragma once
// Just enough OpenTelemetry for the shard hop: W3C traceparent parsing and an
// OTLP/HTTP JSON span exporter with no dependencies beyond POSIX sockets.
//
// The broker forwards the browser's traceparent in gRPC metadata; the shard
// records a child span per Search and ships spans in batches from a background
// thread, so exporting never sits on the request path. If the collector is down,
// spans are dropped and counted, never retried in a way that could build memory.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace hs::server {

struct TraceContext {
  std::string trace_id;   // 32 lowercase hex
  std::string parent_id;  // 16 lowercase hex
  bool sampled = false;
};

// "00-<32 hex>-<16 hex>-<2 hex>"; rejects all-zero IDs and unknown formats.
std::optional<TraceContext> parse_traceparent(const std::string& header);
std::string random_span_id();

struct Span {
  std::string trace_id, span_id, parent_id, name;
  uint64_t start_unix_ns = 0, end_unix_ns = 0;
  bool error = false;
  std::vector<std::pair<std::string, std::string>> str_attrs;
  std::vector<std::pair<std::string, int64_t>> int_attrs;
};

// OTLP/HTTP JSON body for one batch of spans (exposed for tests).
std::string spans_to_otlp_json(const std::vector<Span>& spans, const std::string& service_name);

class SpanExporter {
 public:
  // endpoint like "http://localhost:4318" (the /v1/traces path is appended).
  SpanExporter(std::string endpoint, std::string service_name, size_t max_queue = 4096);
  ~SpanExporter();

  void record(Span s);
  uint64_t exported() const { return exported_.load(); }
  uint64_t dropped() const { return dropped_.load(); }

 private:
  void run();
  bool post(const std::string& body);

  std::string host_, port_, path_, service_;
  size_t max_queue_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Span> queue_;
  bool stop_ = false;
  std::atomic<uint64_t> exported_{0}, dropped_{0};
  std::thread worker_;
};

uint64_t unix_now_ns();

}  // namespace hs::server
