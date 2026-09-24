#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>

#include "hs/common/fbin.hpp"
#include "hs/vector/disk_format.hpp"
#include "hs/vector/disk_index.hpp"
#include "hs/vector/kmeans.hpp"
#include "hs/vector/parallel.hpp"

namespace hs::vector {

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Seeded uniform sample of m ids out of n, returned sorted (sequential access).
std::vector<uint32_t> sample_ids(uint32_t n, uint32_t m, uint64_t seed) {
  m = std::min(m, n);
  std::vector<uint32_t> ids(n);
  for (uint32_t i = 0; i < n; ++i) ids[i] = i;
  std::mt19937_64 rng(seed);
  for (uint32_t i = 0; i < m; ++i) std::swap(ids[i], ids[i + rng() % (n - i)]);
  ids.resize(m);
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<float> gather(const float* data, uint32_t dim, const std::vector<uint32_t>& ids) {
  std::vector<float> out(ids.size() * size_t(dim));
  for (size_t i = 0; i < ids.size(); ++i)
    std::memcpy(out.data() + i * dim, data + size_t(ids[i]) * dim, size_t(dim) * 4);
  return out;
}

void pwrite_full(int fd, const void* buf, size_t len, uint64_t off) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  while (len > 0) {
    ssize_t w = ::pwrite(fd, p, len, off_t(off));
    if (w <= 0) throw std::runtime_error("pwrite failed");
    p += w;
    len -= size_t(w);
    off += uint64_t(w);
  }
}

// Streams node records into disk.index in id order, one chunk of blocks at a time.
class NodeWriter {
 public:
  NodeWriter(const std::string& path, uint32_t n, uint32_t dim, uint32_t R, uint32_t medoid) : n_(n), dim_(dim) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) throw std::runtime_error("cannot create " + path);
#if defined(__APPLE__)
    // Keep freshly written blocks out of the page cache, so a later "cold" run
    // really reads from the SSD instead of hitting pages the build left behind.
    ::fcntl(fd_, F_NOCACHE, 1);
#endif
    disk_layout(dim, R, &h_.record_bytes, &h_.nodes_per_block, &h_.blocks_per_node);
    h_.n = n;
    h_.dim = dim;
    h_.R = R;
    h_.medoid = medoid;
    std::vector<uint8_t> hb(kBlock, 0);
    std::memcpy(hb.data(), &h_, sizeof(h_));
    pwrite_full(fd_, hb.data(), kBlock, 0);
  }
  ~NodeWriter() {
    if (fd_ >= 0) ::close(fd_);
  }
  uint32_t chunk_nodes() const { return h_.nodes_per_block * 8192; }

  // Nodes [s, e): s must be a multiple of chunk_nodes(). nbrs(i) -> (ptr, deg).
  template <class F>
  void write(const float* data, uint32_t s, uint32_t e, F&& nbrs) {
    uint64_t first = disk_record_offset(s, h_.record_bytes, h_.nodes_per_block, h_.blocks_per_node);
    uint64_t last = disk_record_offset(e - 1, h_.record_bytes, h_.nodes_per_block, h_.blocks_per_node);
    uint64_t end = (last + h_.record_bytes + kBlock - 1) / kBlock * kBlock;
    buf_.assign(end - first, 0);
    for (uint32_t i = s; i < e; ++i) {
      uint8_t* rec = buf_.data() + (disk_record_offset(i, h_.record_bytes, h_.nodes_per_block, h_.blocks_per_node) - first);
      std::memcpy(rec, data + size_t(i) * dim_, size_t(dim_) * 4);
      auto [ptr, deg] = nbrs(i);
      std::memcpy(rec + size_t(dim_) * 4, &deg, 4);
      std::memcpy(rec + size_t(dim_) * 4 + 4, ptr, size_t(deg) * 4);
      degree_sum_ += deg;
    }
    pwrite_full(fd_, buf_.data(), buf_.size(), first);
    bytes_ = std::max(bytes_, first + buf_.size());
  }
  void finish() {
    if (::fsync(fd_) != 0) throw std::runtime_error("fsync failed");
#if defined(POSIX_FADV_DONTNEED)
    ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
#endif
    ::close(fd_);
    fd_ = -1;
  }
  uint64_t bytes() const { return bytes_; }
  double avg_degree() const { return double(degree_sum_) / std::max<uint32_t>(1, n_); }

 private:
  int fd_ = -1;
  uint32_t n_, dim_;
  DiskHeader h_;
  std::vector<uint8_t> buf_;
  uint64_t bytes_ = 0, degree_sum_ = 0;
};

}  // namespace

void build_disk_index(const float* data, uint32_t n, uint32_t dim, const std::string& out_dir,
                      const DiskBuildParams& p, DiskBuildStats* stats_out, const std::vector<uint64_t>* docids) {
  namespace fs = std::filesystem;
  fs::create_directories(out_dir);
  DiskBuildStats st;
  auto t_all = Clock::now();
  const auto& vp = p.vamana;
  const uint32_t R = vp.R;
  const unsigned threads = resolve_threads(vp.threads);
  auto log = [&](const char* fmt, auto... a) {
    if (p.verbose) {
      std::fprintf(stderr, "[disk-build %.1fs] ", secs(t_all));
      std::fprintf(stderr, fmt, a...);
      std::fprintf(stderr, "\n");
    }
  };

  // 1. PQ: train on a seeded uniform sample, encode everything.
  auto t = Clock::now();
  {
    auto ids = sample_ids(n, p.pq_train_sample, vp.seed ^ 0x5051ULL);
    auto sample = gather(data, dim, ids);
    auto t_train = Clock::now();
    auto pq = ProductQuantizer::train(sample.data(), ids.size(), dim, p.pq_M, p.pq_iters, vp.seed, threads);
    st.pq_train_seconds = secs(t_train);
    pq.save(out_dir + "/pq_pivots.bin");
    std::vector<uint8_t> codes(size_t(n) * p.pq_M);
    pq.encode_batch(data, n, codes.data(), threads);
    std::FILE* f = std::fopen((out_dir + "/pq_codes.bin").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write pq_codes.bin");
    uint32_t hdr[2] = {n, p.pq_M};
    bool ok = std::fwrite(hdr, 4, 2, f) == 2 && std::fwrite(codes.data(), 1, codes.size(), f) == codes.size();
    ok = std::fclose(f) == 0 && ok;
    if (!ok) throw std::runtime_error("short write pq_codes.bin");
    log("PQ trained on %zu vectors (M=%u) and %u vectors encoded", ids.size(), p.pq_M, n);
  }
  st.seconds_pq = secs(t);

  // 2. How many partitions.
  uint32_t P = std::max<uint32_t>(1, p.partitions);
  if (!p.prebuilt_graph.empty() && (P != 1 || p.build_ram_gb > 0))
    throw std::invalid_argument("prebuilt_graph requires a single partition");
  if (p.build_ram_gb > 0) {
    double per_point = double(dim) * 4 + double(R) * 4 * 2 + 64;  // vectors + graph + scratch
    double cap = p.build_ram_gb * 1e9 / per_point;
    P = cap >= n ? 1 : uint32_t(std::ceil(double(p.overlap) * n / cap));
  }
  const uint32_t overlap = std::min(p.overlap, P);
  st.partitions = P;

  NodeWriter* writer_ptr = nullptr;
  std::string index_path = out_dir + "/disk.index";

  if (P == 1) {
    t = Clock::now();
    VamanaBuildParams bp = vp;
    VamanaBuildStats vs;
    auto ix = p.prebuilt_graph.empty() ? VamanaIndex::build(data, n, dim, bp, &vs)
                                       : VamanaIndex::load(p.prebuilt_graph, data, n, dim);
    if (ix.max_degree() != R) throw std::runtime_error("prebuilt graph has a different R");
    st.seconds_graphs = secs(t);
    st.max_partition_size = st.sum_partition_sizes = n;
    log("single Vamana graph built in %.1fs", st.seconds_graphs);
    t = Clock::now();
    NodeWriter w(index_path, n, dim, R, ix.medoid());
    for (uint32_t s = 0; s < n; s += w.chunk_nodes())
      w.write(data, s, std::min(n, s + w.chunk_nodes()),
              [&](uint32_t i) { return std::pair<const uint32_t*, uint32_t>(ix.neighbors(i), ix.degree(i)); });
    w.finish();
    st.index_bytes = w.bytes();
    st.avg_degree = w.avg_degree();
    st.seconds_merge = secs(t);
    (void)writer_ptr;
  } else {
    // 3a. k-means on a sample; each point joins its `overlap` nearest clusters.
    t = Clock::now();
    auto kids = sample_ids(n, p.kmeans_sample, vp.seed ^ 0x4b4dULL);
    auto ksample = gather(data, dim, kids);
    KMeansParams kp;
    kp.k = P;
    kp.iters = p.kmeans_iters;
    kp.seed = vp.seed;
    kp.threads = threads;
    auto centroids = kmeans(ksample.data(), kids.size(), dim, kp);
    std::vector<float>().swap(ksample);
    std::vector<uint32_t> labels(size_t(n) * overlap);
    parallel_for(0, n, threads, [&](size_t i, unsigned) {
      nearest_centroids(data + i * dim, centroids.data(), P, dim, overlap, labels.data() + i * overlap);
    }, 1024);
    std::vector<std::vector<uint32_t>> members(P);
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t o = 0; o < overlap; ++o) members[labels[size_t(i) * overlap + o]].push_back(i);
    std::vector<uint32_t>().swap(labels);
    st.seconds_partition = secs(t);
    for (auto& m : members) {
      st.max_partition_size = std::max<uint64_t>(st.max_partition_size, m.size());
      st.sum_partition_sizes += m.size();
    }
    log("%u partitions, overlap %u, largest %llu points", P, overlap, (unsigned long long)st.max_partition_size);

    // 3b. One Vamana graph per partition, only that partition's vectors in RAM.
    t = Clock::now();
    std::string tmp = p.tmp_dir.empty() ? out_dir : p.tmp_dir;
    fs::create_directories(tmp);
    std::vector<std::string> spill(P);
    for (uint32_t c = 0; c < P; ++c) {
      spill[c] = tmp + "/partition_" + std::to_string(c) + ".graph";
      const auto& ids = members[c];
      std::FILE* f = std::fopen(spill[c].c_str(), "wb");
      if (!f) throw std::runtime_error("cannot write " + spill[c]);
      if (ids.size() >= 2) {
        auto vecs = gather(data, dim, ids);
        VamanaBuildParams bp = vp;
        bp.seed = vp.seed + 1000003ULL * (c + 1);
        bp.verbose = false;
        auto ix = VamanaIndex::build(vecs.data(), uint32_t(ids.size()), dim, bp);
        std::vector<uint32_t> rec(size_t(R) + 1);
        for (uint32_t i = 0; i < ids.size(); ++i) {
          uint32_t d = ix.degree(i);
          rec[0] = d;
          for (uint32_t j = 0; j < d; ++j) rec[1 + j] = ids[ix.neighbors(i)[j]];
          if (std::fwrite(rec.data(), 4, size_t(d) + 1, f) != size_t(d) + 1) throw std::runtime_error("spill write");
        }
      } else {
        for (size_t i = 0; i < ids.size(); ++i) {
          uint32_t zero = 0;
          std::fwrite(&zero, 4, 1, f);
        }
      }
      if (std::fclose(f) != 0) throw std::runtime_error("spill close");
      log("partition %u/%u (%zu points) built", c + 1, P, ids.size());
    }
    st.seconds_graphs = secs(t);

    // 3c. Merge: stream the spilled graphs in id order; union edges; re-prune.
    t = Clock::now();
    uint32_t medoid = find_medoid(data, n, dim, threads);
    NodeWriter w(index_path, n, dim, R, medoid);
    std::vector<std::FILE*> files(P);
    std::vector<size_t> cursor(P, 0);
    for (uint32_t c = 0; c < P; ++c) {
      files[c] = std::fopen(spill[c].c_str(), "rb");
      if (!files[c]) throw std::runtime_error("cannot reopen " + spill[c]);
    }
    const uint32_t CH = w.chunk_nodes();
    std::vector<std::vector<uint32_t>> merged(CH), out(CH);
    std::atomic<uint64_t> pruned{0};
    for (uint32_t s = 0; s < n; s += CH) {
      uint32_t e = std::min(n, s + CH);
      for (uint32_t i = 0; i < e - s; ++i) merged[i].clear();
      for (uint32_t c = 0; c < P; ++c) {
        const auto& ids = members[c];
        while (cursor[c] < ids.size() && ids[cursor[c]] < e) {
          uint32_t g = ids[cursor[c]++];
          uint32_t d;
          if (std::fread(&d, 4, 1, files[c]) != 1) throw std::runtime_error("spill read");
          size_t old = merged[g - s].size();
          merged[g - s].resize(old + d);
          if (d && std::fread(merged[g - s].data() + old, 4, d, files[c]) != d) throw std::runtime_error("spill read");
        }
      }
      parallel_for(s, e, threads, [&](size_t gi, unsigned) {
        uint32_t g = uint32_t(gi);
        auto& m = merged[g - s];
        // Deterministic dedupe that keeps first-seen order.
        std::vector<uint32_t> uniq;
        uniq.reserve(m.size());
        for (uint32_t v : m)
          if (v != g && std::find(uniq.begin(), uniq.end(), v) == uniq.end()) uniq.push_back(v);
        if (uniq.size() > R) {
          out[g - s] = robust_prune(g, std::move(uniq), data, dim, vp.alpha, R, vp.max_candidates);
          pruned.fetch_add(1, std::memory_order_relaxed);
        } else {
          out[g - s] = std::move(uniq);
        }
      }, 64);
      w.write(data, s, e, [&](uint32_t i) {
        return std::pair<const uint32_t*, uint32_t>(out[i - s].data(), uint32_t(out[i - s].size()));
      });
    }
    for (auto* f : files) std::fclose(f);
    for (auto& path : spill) fs::remove(path);
    w.finish();
    st.index_bytes = w.bytes();
    st.avg_degree = w.avg_degree();
    st.merged_nodes_pruned = pruned.load();
    st.seconds_merge = secs(t);
    log("merge done: %llu nodes re-pruned, avg degree %.1f", (unsigned long long)st.merged_nodes_pruned,
        st.avg_degree);
  }
  if (docids) hs::write_u64bin(out_dir + "/docids.u64bin", *docids);
  st.seconds_total = secs(t_all);
  st.peak_rss_bytes = peak_rss_bytes();

  std::FILE* f = std::fopen((out_dir + "/build.json").c_str(), "w");
  if (f) {
    std::fprintf(f,
                 "{\n  \"n\": %u, \"dim\": %u, \"R\": %u, \"L_build\": %u, \"alpha\": %.3f, \"two_pass\": %s,\n"
                 "  \"seed\": %llu, \"threads\": %u, \"pq_M\": %u, \"pq_train_sample\": %u, \"pq_iters\": %u,\n"
                 "  \"partitions\": %u, \"overlap\": %u, \"build_ram_gb\": %.3f, \"prebuilt_graph\": \"%s\",\n"
                 "  \"max_partition_size\": %llu, \"sum_partition_sizes\": %llu, \"merged_nodes_pruned\": %llu,\n"
                 "  \"avg_degree\": %.3f, \"index_bytes\": %llu, \"peak_rss_bytes\": %llu,\n"
                 "  \"seconds\": {\"total\": %.2f, \"pq\": %.2f, \"pq_train\": %.2f, \"partition\": %.2f, "
                 "\"graphs\": %.2f, \"merge_write\": %.2f}\n}\n",
                 n, dim, R, vp.L, vp.alpha, vp.two_pass ? "true" : "false", (unsigned long long)vp.seed, threads,
                 p.pq_M, std::min(n, p.pq_train_sample), p.pq_iters, P, overlap, p.build_ram_gb, p.prebuilt_graph.c_str(),
                 (unsigned long long)st.max_partition_size, (unsigned long long)st.sum_partition_sizes,
                 (unsigned long long)st.merged_nodes_pruned, st.avg_degree, (unsigned long long)st.index_bytes,
                 (unsigned long long)st.peak_rss_bytes, st.seconds_total, st.seconds_pq, st.pq_train_seconds,
                 st.seconds_partition, st.seconds_graphs, st.seconds_merge);
    std::fclose(f);
  }
  if (stats_out) *stats_out = st;
}

}  // namespace hs::vector
