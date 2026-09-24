#include "hs/server/trace.hpp"

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <sstream>

namespace hs::server {

namespace {

bool is_lower_hex(const std::string& s) {
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

bool all_zero(const std::string& s) { return s.find_first_not_of('0') == std::string::npos; }

std::string hex(uint64_t v, int digits) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%0*llx", digits, static_cast<unsigned long long>(v));
  return buf;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += char(c);
        }
    }
  }
  return out;
}

}  // namespace

std::optional<TraceContext> parse_traceparent(const std::string& h) {
  // version(2) - trace-id(32) - parent-id(16) - flags(2)
  if (h.size() < 55 || h[2] != '-' || h[35] != '-' || h[52] != '-') return std::nullopt;
  std::string version = h.substr(0, 2), trace = h.substr(3, 32), parent = h.substr(36, 16), flags = h.substr(53, 2);
  if (!is_lower_hex(version) || version == "ff") return std::nullopt;
  if (version == "00" && h.size() != 55) return std::nullopt;
  if (!is_lower_hex(trace) || !is_lower_hex(parent) || !is_lower_hex(flags)) return std::nullopt;
  if (all_zero(trace) || all_zero(parent)) return std::nullopt;
  TraceContext ctx{trace, parent, (std::stoi(flags, nullptr, 16) & 1) != 0};
  return ctx;
}

std::string random_span_id() {
  thread_local std::mt19937_64 rng{std::random_device{}()};
  uint64_t v = 0;
  while (v == 0) v = rng();
  return hex(v, 16);
}

uint64_t unix_now_ns() {
  return uint64_t(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string spans_to_otlp_json(const std::vector<Span>& spans, const std::string& service_name) {
  std::ostringstream o;
  o << R"({"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":")"
    << json_escape(service_name) << R"("}}]},"scopeSpans":[{"scope":{"name":"hybridsearch.shard"},"spans":[)";
  for (size_t i = 0; i < spans.size(); ++i) {
    const Span& s = spans[i];
    if (i) o << ',';
    o << R"({"traceId":")" << s.trace_id << R"(","spanId":")" << s.span_id << '"';
    if (!s.parent_id.empty()) o << R"(,"parentSpanId":")" << s.parent_id << '"';
    // kind 2 = SERVER. Timestamps are strings in OTLP/JSON (uint64 as decimal).
    o << R"(,"name":")" << json_escape(s.name) << R"(","kind":2,"startTimeUnixNano":")" << s.start_unix_ns
      << R"(","endTimeUnixNano":")" << s.end_unix_ns << R"(","attributes":[)";
    bool first = true;
    for (auto& [k, v] : s.str_attrs) {
      o << (first ? "" : ",") << R"({"key":")" << json_escape(k) << R"(","value":{"stringValue":")" << json_escape(v)
        << R"("}})";
      first = false;
    }
    for (auto& [k, v] : s.int_attrs) {
      o << (first ? "" : ",") << R"({"key":")" << json_escape(k) << R"(","value":{"intValue":")" << v << R"("}})";
      first = false;
    }
    o << "]";
    if (s.error) o << R"(,"status":{"code":2})";
    o << '}';
  }
  o << "]}]}]}";
  return o.str();
}

SpanExporter::SpanExporter(std::string endpoint, std::string service_name, size_t max_queue)
    : service_(std::move(service_name)), max_queue_(max_queue) {
  // Accepts "http://host:port" or "host:port"; TLS is out of scope for a local collector.
  std::string rest = endpoint;
  if (rest.rfind("http://", 0) == 0) rest = rest.substr(7);
  auto slash = rest.find('/');
  path_ = "/v1/traces";
  if (slash != std::string::npos) rest = rest.substr(0, slash);
  auto colon = rest.rfind(':');
  host_ = colon == std::string::npos ? rest : rest.substr(0, colon);
  port_ = colon == std::string::npos ? "4318" : rest.substr(colon + 1);
  worker_ = std::thread([this] { run(); });
}

SpanExporter::~SpanExporter() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  worker_.join();
}

void SpanExporter::record(Span s) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.size() >= max_queue_) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    queue_.push_back(std::move(s));
  }
  cv_.notify_one();
}

void SpanExporter::run() {
  std::unique_lock<std::mutex> lock(mu_);
  while (true) {
    cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stop_ || queue_.size() >= 256; });
    if (queue_.empty()) {
      if (stop_) return;
      continue;
    }
    std::vector<Span> batch(std::make_move_iterator(queue_.begin()), std::make_move_iterator(queue_.end()));
    queue_.clear();
    bool stopping = stop_;
    lock.unlock();
    if (post(spans_to_otlp_json(batch, service_))) {
      exported_.fetch_add(batch.size(), std::memory_order_relaxed);
    } else {
      dropped_.fetch_add(batch.size(), std::memory_order_relaxed);
    }
    lock.lock();
    if (stopping && queue_.empty()) return;
  }
}

bool SpanExporter::post(const std::string& body) {
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host_.c_str(), port_.c_str(), &hints, &res) != 0) return false;
  int fd = -1;
  for (addrinfo* p = res; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    timeval tv{2, 0};  // a stuck collector must not stall the exporter forever
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(res);
  if (fd < 0) return false;

  std::ostringstream req;
  req << "POST " << path_ << " HTTP/1.1\r\nHost: " << host_ << ':' << port_
      << "\r\nContent-Type: application/json\r\nContent-Length: " << body.size() << "\r\nConnection: close\r\n\r\n"
      << body;
  std::string data = req.str();
  size_t sent = 0;
  while (sent < data.size()) {
    ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
    if (n <= 0) {
      ::close(fd);
      return false;
    }
    sent += size_t(n);
  }
  char buf[64] = {0};
  ssize_t n = ::recv(fd, buf, sizeof buf - 1, 0);
  ::close(fd);
  // "HTTP/1.1 200 OK"
  return n >= 12 && buf[9] == '2';
}

}  // namespace hs::server
