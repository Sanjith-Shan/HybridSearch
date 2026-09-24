// Microbenchmark: SIMD vs scalar inner product / squared L2 at 768-d, and PQ ADC.
//   bench_vec_distance [--iters N]
// Reports ns per call, working set in L1 (one pair of vectors reused) and out of
// cache (streaming over 64 MB of vectors). Single thread; macOS: dev-signal-only.
#include <chrono>
#include <cstdio>
#include <vector>

#include "hs/vector/containers.hpp"
#include "hs/vector/distance.hpp"
#include "hs/vector/pq.hpp"
#include "hs/vector/synth.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

template <class F>
double ns_per(F&& f, size_t iters) {
  auto t0 = std::chrono::steady_clock::now();
  volatile float sink = 0;
  for (size_t i = 0; i < iters; ++i) sink = sink + f(i);
  return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(iters);
}

int main(int argc, char** argv) {
  tool::Args a(argc, argv);
  size_t iters = size_t(a.num("iters", 2e6));
  const uint32_t d = 768;
  const uint32_t nvec = 64u * 1024 * 1024 / (d * 4);  // 64 MB
  auto x = random_unit_vectors(nvec, d, 1);
  auto q = random_unit_vectors(1, d, 2);
  std::printf("{\"simd\": \"%s\", \"dim\": %u", simd_name(), d);
  std::printf(", \"ip_simd_l1_ns\": %.2f", ns_per([&](size_t) { return ip(q.data(), x.data(), d); }, iters));
  std::printf(", \"ip_scalar_l1_ns\": %.2f", ns_per([&](size_t) { return ip_scalar(q.data(), x.data(), d); }, iters));
  std::printf(", \"l2_simd_l1_ns\": %.2f", ns_per([&](size_t) { return l2sq(q.data(), x.data(), d); }, iters));
  std::printf(", \"l2_scalar_l1_ns\": %.2f", ns_per([&](size_t) { return l2sq_scalar(q.data(), x.data(), d); }, iters));
  // Random rows: the access pattern of graph search (no prefetch).
  std::vector<uint32_t> perm(iters);
  uint64_t s = 7;
  for (auto& p : perm) p = uint32_t((s = splitmix64(s)) % nvec);
  std::printf(", \"l2_simd_random_row_ns\": %.2f",
              ns_per([&](size_t i) { return l2sq(q.data(), x.data() + size_t(perm[i]) * d, d); }, iters));
  auto pq = ProductQuantizer::train(x.data(), 20000, d, 96, 8, 3);
  std::vector<uint8_t> codes(size_t(nvec) * 96);
  pq.encode_batch(x.data(), nvec, codes.data());
  std::vector<float> table(96 * 256);
  std::printf(", \"pq_table_build_ns\": %.1f", ns_per([&](size_t) { pq.distance_table(q.data(), table.data()); return table[0]; }, 2000));
  std::printf(", \"pq_adc_M96_random_code_ns\": %.2f",
              ns_per([&](size_t i) { return pq.adc(table.data(), codes.data() + size_t(perm[i]) * 96); }, iters));
  std::printf("}\n");
  return 0;
}
