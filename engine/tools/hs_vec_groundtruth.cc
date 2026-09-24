// Exact inner-product top-k by brute force: the reference every recall number uses.
//   hs_vec_groundtruth --base passages.fbin --queries q.fbin --k 100 --out gt.bin
//                      [--max-base N] [--max-queries N] [--threads T]
#include <cstdio>

#include "hs/common/fbin.hpp"
#include "hs/vector/bruteforce.hpp"
#include "hs/vector/mmap.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  MappedFbin base(a.str("base"), uint32_t(a.num("max-base", 0)));
  auto qs = hs::read_fbin(a.str("queries"), uint32_t(a.num("max-queries", 0)));
  if (qs.dim != base.dim()) throw std::runtime_error("dimension mismatch");
  uint32_t k = uint32_t(a.num("k", 100));
  double t0 = tool::now_s();
  auto hits = exact_topk(base.data(), base.n(), qs.data.data(), qs.n, base.dim(), k, unsigned(a.num("threads", 0)));
  double secs = tool::now_s() - t0;
  write_groundtruth(a.str("out"), hits, qs.n, k);
  std::printf("{\"n\": %u, \"nq\": %u, \"dim\": %u, \"k\": %u, \"seconds\": %.2f}\n", base.n(), qs.n, base.dim(), k, secs);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_groundtruth: %s\n", e.what());
  return 1;
}
