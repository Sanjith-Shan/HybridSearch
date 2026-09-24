// Build the DiskANN-style SSD index.
//   hs_vec_disk_build --base passages.fbin [--max-n N] --out dir [--docids docids.u64bin]
//                     [--R 64] [--L 100] [--alpha 1.2] [--pq-M 96] [--pq-sample 200000]
//                     [--partitions P | --ram-gb G] [--overlap 2] [--seed S] [--threads T]
//                     [--min-free-gb 3] [--graph saved_vamana.graph  (reuse; single partition)]
// Refuses to start if free disk after the estimated index size would drop below --min-free-gb.
#include <sys/statvfs.h>

#include <cstdio>
#include <filesystem>

#include "hs/common/fbin.hpp"
#include "hs/vector/disk_format.hpp"
#include "hs/vector/disk_index.hpp"
#include "hs/vector/mmap.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

int main(int argc, char** argv) try {
  tool::Args a(argc, argv);
  MappedFbin base(a.str("base"), uint32_t(a.num("max-n", 0)));
  DiskBuildParams p;
  p.vamana.R = uint32_t(a.num("R", 64));
  p.vamana.L = uint32_t(a.num("L", 100));
  p.vamana.alpha = float(a.num("alpha", 1.2));
  p.vamana.seed = uint64_t(a.num("seed", 20260923));
  p.vamana.threads = unsigned(a.num("threads", 0));
  p.pq_M = uint32_t(a.num("pq-M", 96));
  p.pq_train_sample = uint32_t(a.num("pq-sample", 200000));
  p.pq_iters = uint32_t(a.num("pq-iters", 15));
  p.partitions = uint32_t(a.num("partitions", 1));
  p.build_ram_gb = a.num("ram-gb", 0);
  p.overlap = uint32_t(a.num("overlap", 2));
  p.kmeans_sample = uint32_t(a.num("kmeans-sample", 100000));
  p.tmp_dir = a.opt("tmp", "");
  p.prebuilt_graph = a.opt("graph", "");
  p.verbose = true;

  std::string out = a.str("out");
  std::filesystem::create_directories(out);
  uint32_t rb, npb, bpn;
  disk_layout(base.dim(), p.vamana.R, &rb, &npb, &bpn);
  double est = double(base.n()) / npb * bpn * kBlock + double(base.n()) * (p.pq_M + 8) +
               (p.partitions > 1 || p.build_ram_gb > 0 ? 2.0 * base.n() * (p.vamana.R + 1) * 4 : 0);
  struct statvfs sv {};
  if (statvfs(out.c_str(), &sv) == 0) {
    double free_b = double(sv.f_bavail) * sv.f_frsize;
    double min_free = a.num("min-free-gb", 3) * 1e9;
    std::fprintf(stderr, "estimated index + spill %.2f GB, free %.2f GB\n", est / 1e9, free_b / 1e9);
    if (free_b - est < min_free) {
      std::fprintf(stderr, "hs_vec_disk_build: refusing: would leave < %.1f GB free\n", min_free / 1e9);
      return 2;
    }
  }
  std::vector<uint64_t> docids;
  if (a.has("docids")) {
    docids = hs::read_u64bin(a.str("docids"));
    docids.resize(base.n());
  }
  DiskBuildStats st;
  build_disk_index(base.data(), base.n(), base.dim(), out, p, &st, a.has("docids") ? &docids : nullptr);
  std::printf("{\"n\": %u, \"partitions\": %u, \"build_seconds\": %.1f, \"index_bytes\": %llu, "
              "\"peak_rss_bytes\": %llu, \"avg_degree\": %.2f}\n",
              base.n(), st.partitions, st.seconds_total, (unsigned long long)st.index_bytes,
              (unsigned long long)st.peak_rss_bytes, st.avg_degree);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "hs_vec_disk_build: %s\n", e.what());
  return 1;
}
