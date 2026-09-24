#pragma once
// Fault injection for chaos experiments, controlled at runtime through a file.
//
// The shard re-reads the control file (key=value lines) when its mtime changes,
// at most every 250 ms, so an experiment can inject and remove a fault without
// restarting the process (a restart would also empty the page cache and change
// what is being measured). Keys:
//   latency_ms=50     added before every Search reply
//   jitter_ms=10      uniform extra latency in [0, jitter_ms)
//   fail_rate=0.1     fraction of Search calls answered UNAVAILABLE
//   blackhole=1       hold every Search until its deadline, then fail
// A missing or empty file means no faults. CPU starvation is injected from
// outside the process (taskpolicy / cgroups), since that is what it models.

#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>

namespace hs::server {

struct ChaosSettings {
  uint32_t latency_ms = 0;
  uint32_t jitter_ms = 0;
  double fail_rate = 0;
  bool blackhole = false;
  bool any() const { return latency_ms || jitter_ms || fail_rate > 0 || blackhole; }
};

ChaosSettings parse_chaos(const std::string& text);

class Chaos {
 public:
  explicit Chaos(std::string control_file, uint64_t seed = 20260923);

  // Current settings (re-reading the file if it changed).
  ChaosSettings current();
  // Delay to inject for one request, in microseconds.
  uint64_t sample_delay_us(const ChaosSettings& s);
  bool sample_failure(const ChaosSettings& s);

 private:
  std::string path_;
  std::mutex mu_;
  ChaosSettings settings_;
  int64_t last_mtime_ns_ = -1;
  std::chrono::steady_clock::time_point last_check_{};
  std::mt19937_64 rng_;
};

}  // namespace hs::server
