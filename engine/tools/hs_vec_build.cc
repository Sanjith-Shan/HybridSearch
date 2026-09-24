// Build an in-memory Vamana graph and save it.
//   hs_vec_build --base passages.fbin [--max-n N] --out graph.bin
//                [--R 64] [--L 100] [--alpha 1.2] [--single-pass] [--seed S] [--threads T]
// Prints one JSON line of build statistics (also appended to --stats if given).
#include <cstdio>

#include "hs/vector/distance.hpp"
#include "hs/vector/parallel.hpp"
#include "hs/vector/mmap.hpp"
#include "hs/vector/tool_util.hpp"
#include "hs/vector/vamana.hpp"

using namespace hs::vector;

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  MappedFbin base(a.str("base"), uint32_t(a.num("max-n", 0)));
  VamanaBuildParams p;
  p.R = uint32_t(a.num("R", 64));
  p.L = uint32_t(a.num("L", 100));
  p.alpha = float(a.num("alpha", 1.2));
  p.two_pass = !a.has("single-pass");
  p.slack = float(a.num("slack", 1.3));
  p.seed = uint64_t(a.num("seed", 20260923));
  p.threads = unsigned(a.num("threads", 0));
  p.verbose = a.has("verbose");
  VamanaBuildStats st;
  auto ix = VamanaIndex::build(base.data(), base.n(), base.dim(), p, &st);
  ix.save(a.str("out"));
  char line[1024];
  std::snprintf(line, sizeof line,
                "{\"n\": %u, \"dim\": %u, \"R\": %u, \"L_build\": %u, \"alpha\": %.3f, \"two_pass\": %s, "
                "\"slack\": %.2f, \"seed\": %llu, \"threads\": %u, \"build_seconds\": %.2f, \"pass1_seconds\": %.2f, "
                "\"pass2_seconds\": %.2f, \"batches\": %llu, \"avg_degree\": %.3f, \"max_degree\": %u, "
                "\"reachable_from_medoid\": %u, \"graph_hash\": \"%016llx\", \"simd\": \"%s\", \"graph\": \"%s\"}",
                base.n(), base.dim(), p.R, p.L, p.alpha, p.two_pass ? "true" : "false", p.slack,
                (unsigned long long)p.seed, resolve_threads(p.threads), st.seconds_total, st.seconds_pass1,
                st.seconds_pass2, (unsigned long long)st.batches, st.avg_degree, st.max_degree,
                ix.reachable_from_medoid(), (unsigned long long)ix.graph_hash(), simd_name(), a.str("out").c_str());
  std::printf("%s\n", line);
  if (a.has("stats")) {
    std::FILE* f = std::fopen(a.str("stats").c_str(), "a");
    if (f) std::fprintf(f, "%s\n", line), std::fclose(f);
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_build: %s\n", e.what());
  return 1;
}
