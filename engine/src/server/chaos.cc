#include "hs/server/chaos.hpp"

#include <sys/stat.h>

#include <fstream>
#include <sstream>

namespace hs::server {

ChaosSettings parse_chaos(const std::string& text) {
  ChaosSettings s;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    auto hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    auto trim = [](std::string v) {
      size_t a = v.find_first_not_of(" \t\r");
      size_t b = v.find_last_not_of(" \t\r");
      return a == std::string::npos ? std::string() : v.substr(a, b - a + 1);
    };
    std::string key = trim(line.substr(0, eq)), val = trim(line.substr(eq + 1));
    try {
      if (key == "latency_ms") s.latency_ms = uint32_t(std::stoul(val));
      else if (key == "jitter_ms") s.jitter_ms = uint32_t(std::stoul(val));
      else if (key == "fail_rate") s.fail_rate = std::stod(val);
      else if (key == "blackhole") s.blackhole = val == "1" || val == "true";
    } catch (const std::exception&) {
      // A half-written control file must never crash a shard; ignore the line.
    }
  }
  if (s.fail_rate < 0) s.fail_rate = 0;
  if (s.fail_rate > 1) s.fail_rate = 1;
  return s;
}

Chaos::Chaos(std::string control_file, uint64_t seed) : path_(std::move(control_file)), rng_(seed) {}

ChaosSettings Chaos::current() {
  std::lock_guard<std::mutex> lock(mu_);
  if (path_.empty()) return settings_;
  auto now = std::chrono::steady_clock::now();
  if (last_mtime_ns_ != -1 && now - last_check_ < std::chrono::milliseconds(250)) return settings_;
  last_check_ = now;

  struct stat st {};
  if (::stat(path_.c_str(), &st) != 0) {
    settings_ = {};
    last_mtime_ns_ = 0;
    return settings_;
  }
#ifdef __APPLE__
  int64_t mtime = int64_t(st.st_mtimespec.tv_sec) * 1000000000 + st.st_mtimespec.tv_nsec;
#else
  int64_t mtime = int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
#endif
  if (mtime != last_mtime_ns_) {
    std::ifstream f(path_);
    std::stringstream buf;
    buf << f.rdbuf();
    settings_ = parse_chaos(buf.str());
    last_mtime_ns_ = mtime;
  }
  return settings_;
}

uint64_t Chaos::sample_delay_us(const ChaosSettings& s) {
  uint64_t us = uint64_t(s.latency_ms) * 1000;
  if (s.jitter_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    us += std::uniform_int_distribution<uint64_t>(0, uint64_t(s.jitter_ms) * 1000 - 1)(rng_);
  }
  return us;
}

bool Chaos::sample_failure(const ChaosSettings& s) {
  if (s.fail_rate <= 0) return false;
  std::lock_guard<std::mutex> lock(mu_);
  return std::uniform_real_distribution<double>(0, 1)(rng_) < s.fail_rate;
}

}  // namespace hs::server
