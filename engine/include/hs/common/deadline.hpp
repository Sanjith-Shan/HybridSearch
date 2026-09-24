#pragma once
// A request budget carried from the broker down into the scoring loops.
//
// Checking the clock is not free, so hot loops call expired_every(n) which only
// reads the clock once every n calls. A Deadline with no budget never expires.

#include <chrono>
#include <cstdint>

namespace hs {

class Deadline {
 public:
  using Clock = std::chrono::steady_clock;

  Deadline() = default;  // unlimited
  static Deadline after_us(uint64_t budget_us) {
    Deadline d;
    if (budget_us > 0) {
      d.limited_ = true;
      d.at_ = Clock::now() + std::chrono::microseconds(budget_us);
    }
    return d;
  }

  bool limited() const { return limited_; }

  bool expired() {
    if (!limited_) return false;
    if (!expired_ && Clock::now() >= at_) expired_ = true;
    return expired_;
  }

  // Amortised check for inner loops. Returns true once the deadline has passed.
  bool expired_every(uint32_t n) {
    if (!limited_) return false;
    if (++ticks_ % n != 0) return expired_;
    return expired();
  }

  // Set once the deadline actually cut work short; surfaced as partial=true.
  bool fired() const { return expired_; }

  uint64_t remaining_us() const {
    if (!limited_) return UINT64_MAX;
    auto now = Clock::now();
    if (now >= at_) return 0;
    return std::chrono::duration_cast<std::chrono::microseconds>(at_ - now).count();
  }

 private:
  bool limited_ = false;
  bool expired_ = false;
  uint32_t ticks_ = 0;
  Clock::time_point at_{};
};

}  // namespace hs
