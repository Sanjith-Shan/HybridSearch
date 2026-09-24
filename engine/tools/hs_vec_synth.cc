// Synthetic data for development: clustered (default) or uniform unit vectors.
//   hs_vec_synth --n 100000 --nq 1000 --dim 768 --clusters 1000 --sigma 0.35 --seed 1
//                --out-base base.fbin --out-queries q.fbin [--uniform]
#include <cstdio>

#include "hs/common/fbin.hpp"
#include "hs/vector/synth.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  uint32_t n = uint32_t(a.num("n", 100000)), nq = uint32_t(a.num("nq", 1000)), dim = uint32_t(a.num("dim", 768));
  uint64_t seed = uint64_t(a.num("seed", 1));
  std::vector<float> all = a.has("uniform")
                               ? random_unit_vectors(n + nq, dim, seed)
                               : clustered_vectors(n + nq, dim, uint32_t(a.num("clusters", 1000)),
                                                   float(a.num("sigma", 0.35)), seed);
  hs::write_fbin(a.str("out-base"), all.data(), n, dim);
  hs::write_fbin(a.str("out-queries"), all.data() + size_t(n) * dim, nq, dim);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_synth: %s\n", e.what());
  return 1;
}
