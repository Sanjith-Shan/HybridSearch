#include "hs/vector/disk_index.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <thread>
#include <unistd.h>

#include "hs/vector/bruteforce.hpp"
#include "hs/vector/disk_format.hpp"
#include "hs/vector/distance.hpp"
#include "hs/vector/synth.hpp"

namespace hs::vector {
namespace {
namespace fs = std::filesystem;

struct Fixture {
  uint32_t n = 6000, nq = 100, dim = 32;
  std::vector<float> base, queries;
  GroundTruth gt;
  Fixture() {
    auto all = clustered_vectors(n + nq, dim, 60, 0.35f, 31);
    base.assign(all.begin(), all.begin() + size_t(n) * dim);
    queries.assign(all.begin() + size_t(n) * dim, all.end());
    auto hits = exact_topk(base.data(), n, queries.data(), nq, dim, 10);
    gt.nq = nq;
    gt.k = 10;
    for (auto& h : hits) gt.ids.push_back(h.doc);
  }
};

const Fixture& fx() {
  static Fixture f;
  return f;
}

std::string tmpdir(const std::string& name) {
  auto p = fs::temp_directory_path() / ("hs_disk_test_" + name + "_" + std::to_string(::getpid()));
  fs::remove_all(p);
  return p.string();
}

// Page-cache reads keep the suite fast; the cold (F_NOCACHE / O_DIRECT) path is
// exercised explicitly where it matters.
DiskOpenOptions warm() {
  DiskOpenOptions o;
  o.direct_io = false;
  return o;
}

DiskBuildParams small_params(uint32_t partitions) {
  DiskBuildParams p;
  p.vamana.R = 24;
  p.vamana.L = 50;
  p.vamana.alpha = 1.2f;
  p.pq_M = 16;
  p.pq_train_sample = 3000;
  p.pq_iters = 8;
  p.partitions = partitions;
  p.kmeans_sample = 3000;
  return p;
}

double recall(const DiskIndex& ix, uint32_t L, uint32_t W, VectorResult* sum = nullptr) {
  const auto& f = fx();
  std::vector<std::vector<uint32_t>> approx(f.nq);
  VectorSearchOptions o;
  o.k = 10;
  o.L = L;
  o.beam_width = W;
  for (uint32_t q = 0; q < f.nq; ++q) {
    auto r = ix.search(f.queries.data() + size_t(q) * f.dim, o);
    for (auto& h : r.hits) approx[q].push_back(h.doc);
    if (sum) {
      sum->ssd_reads += r.ssd_reads;
      sum->hops += r.hops;
      sum->cache_hits += r.cache_hits;
    }
  }
  return recall_at_k(f.gt, approx, 10);
}

}  // namespace

TEST(DiskLayout, RecordsNeverStraddleBlocks) {
  uint32_t rb, npb, bpn;
  disk_layout(768, 64, &rb, &npb, &bpn);
  EXPECT_EQ(rb, 768u * 4 + 4 + 64 * 4);
  EXPECT_EQ(npb, 1u);
  EXPECT_EQ(bpn, 1u);
  disk_layout(768, 255, &rb, &npb, &bpn);
  EXPECT_EQ(bpn, 1u);  // 4096 bytes exactly
  disk_layout(768, 256, &rb, &npb, &bpn);
  EXPECT_EQ(bpn, 2u);
  disk_layout(32, 24, &rb, &npb, &bpn);
  EXPECT_EQ(npb, 4096u / rb);
  for (uint32_t id = 0; id < 1000; ++id) {
    uint64_t off = disk_record_offset(id, rb, npb, bpn);
    EXPECT_EQ(off / 4096, (off + rb - 1) / 4096) << id;
    EXPECT_GE(off, 4096u);
  }
}

TEST(DiskIndex, SinglePartitionMatchesVectorsAndHasHighRecall) {
  const auto& f = fx();
  std::string dir = tmpdir("single");
  DiskBuildStats st;
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1), &st);
  auto ix = DiskIndex::open(dir, {});
  EXPECT_EQ(ix->size(), f.n);
  // Records on disk hold exactly the input vectors.
  std::vector<float> v;
  std::vector<uint32_t> nb;
  for (uint32_t id : {0u, 1u, 777u, f.n - 1}) {
    ix->read_node(id, v, nb);
    ASSERT_EQ(v.size(), f.dim);
    for (uint32_t d = 0; d < f.dim; ++d) EXPECT_EQ(v[d], f.base[size_t(id) * f.dim + d]);
    EXPECT_LE(nb.size(), 24u);
  }
  VectorResult sum;
  double r = recall(*ix, 64, 4, &sum);
  EXPECT_GE(r, 0.95);
  EXPECT_GT(sum.ssd_reads, 0u);
  EXPECT_EQ(sum.ssd_reads, sum.hops);  // no cache: one 4 KB read per expanded node
  fs::remove_all(dir);
}

TEST(DiskIndex, PartitionedBuildWithOverlapKeepsRecall) {
  const auto& f = fx();
  std::string dir = tmpdir("parts");
  DiskBuildStats st;
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(4), &st);
  EXPECT_EQ(st.partitions, 4u);
  EXPECT_EQ(st.sum_partition_sizes, 2ull * f.n);  // every point in exactly two partitions
  EXPECT_LT(st.max_partition_size, uint64_t(f.n));
  auto ix = DiskIndex::open(dir, warm());
  EXPECT_GE(recall(*ix, 64, 4), 0.93);
  // No partition spill files left behind.
  for (auto& e : fs::directory_iterator(dir)) EXPECT_EQ(e.path().string().find("partition_"), std::string::npos);
  fs::remove_all(dir);
}

TEST(DiskIndex, PartitionedBuildWithoutOverlapIsWorse) {
  // The reason for overlap: with each point in one cluster the merged graph is a
  // union of disconnected islands and search cannot cross cluster borders.
  const auto& f = fx();
  std::string d1 = tmpdir("ov1"), d2 = tmpdir("ov2");
  auto p1 = small_params(6);
  p1.overlap = 1;
  auto p2 = small_params(6);
  build_disk_index(f.base.data(), f.n, f.dim, d1, p1);
  build_disk_index(f.base.data(), f.n, f.dim, d2, p2);
  auto a = DiskIndex::open(d1, warm()), b = DiskIndex::open(d2, warm());
  double r1 = recall(*a, 40, 4), r2 = recall(*b, 40, 4);
  EXPECT_LT(r1 + 0.05, r2) << "overlap=1 " << r1 << " overlap=2 " << r2;
  fs::remove_all(d1);
  fs::remove_all(d2);
}

TEST(DiskIndex, BuildIsDeterministic) {
  const auto& f = fx();
  std::string d1 = tmpdir("det1"), d2 = tmpdir("det2");
  auto p = small_params(3);
  p.vamana.threads = 2;
  build_disk_index(f.base.data(), f.n, f.dim, d1, p);
  p.vamana.threads = 6;
  build_disk_index(f.base.data(), f.n, f.dim, d2, p);
  for (const char* file : {"disk.index", "pq_codes.bin", "pq_pivots.bin"}) {
    auto read = [](const std::string& path) {
      std::FILE* fp = std::fopen(path.c_str(), "rb");
      std::vector<char> buf;
      char tmp[65536];
      size_t r;
      while ((r = std::fread(tmp, 1, sizeof tmp, fp)) > 0) buf.insert(buf.end(), tmp, tmp + r);
      std::fclose(fp);
      return buf;
    };
    EXPECT_TRUE(read(d1 + "/" + file) == read(d2 + "/" + file)) << file;
  }
  fs::remove_all(d1);
  fs::remove_all(d2);
}

TEST(DiskIndex, CacheServesHopsAndDoesNotChangeResults) {
  const auto& f = fx();
  std::string dir = tmpdir("cache");
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1));
  auto plain = DiskIndex::open(dir, warm());
  DiskOpenOptions o = warm();
  o.cache_nodes = 500;
  auto cached = DiskIndex::open(dir, o);
  EXPECT_EQ(cached->memory().cache, 500ull * (f.dim * 4 + 4 + 24 * 4) + 500 * 4);
  VectorSearchOptions so;
  so.L = 64;
  uint64_t reads_plain = 0, reads_cached = 0, hits = 0;
  for (uint32_t q = 0; q < f.nq; ++q) {
    const float* qv = f.queries.data() + size_t(q) * f.dim;
    auto a = plain->search(qv, so), b = cached->search(qv, so);
    ASSERT_EQ(a.hits.size(), b.hits.size());
    for (size_t i = 0; i < a.hits.size(); ++i) EXPECT_EQ(a.hits[i].doc, b.hits[i].doc);
    reads_plain += a.ssd_reads;
    reads_cached += b.ssd_reads;
    hits += b.cache_hits;
    EXPECT_EQ(b.ssd_reads + b.cache_hits, a.ssd_reads);
  }
  EXPECT_GT(hits, 0u);
  EXPECT_LT(reads_cached, reads_plain);
  fs::remove_all(dir);
}

TEST(DiskIndex, IoModesAndCacheModesGiveIdenticalResults) {
  const auto& f = fx();
  std::string dir = tmpdir("io");
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1));
  DiskOpenOptions seq;
  seq.io_mode = IoMode::kSequential;
  DiskOpenOptions warm;
  warm.direct_io = false;
  auto a = DiskIndex::open(dir, seq), b = DiskIndex::open(dir, {}), c = DiskIndex::open(dir, warm);
  VectorSearchOptions so;
  so.L = 50;
  so.beam_width = 8;
  for (uint32_t q = 0; q < 30; ++q) {
    const float* qv = f.queries.data() + size_t(q) * f.dim;
    auto ra = a->search(qv, so), rb = b->search(qv, so), rc = c->search(qv, so);
    ASSERT_EQ(ra.hits.size(), rb.hits.size());
    for (size_t i = 0; i < ra.hits.size(); ++i) {
      EXPECT_EQ(ra.hits[i].doc, rb.hits[i].doc);
      EXPECT_EQ(ra.hits[i].doc, rc.hits[i].doc);
      EXPECT_FLOAT_EQ(ra.hits[i].score, ip(qv, f.base.data() + size_t(ra.hits[i].doc) * f.dim, f.dim));
    }
    EXPECT_EQ(ra.ssd_reads, rb.ssd_reads);
  }
  fs::remove_all(dir);
}

TEST(DiskIndex, ConcurrentQueriesAgreeWithSerial) {
  const auto& f = fx();
  std::string dir = tmpdir("conc");
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1));
  DiskOpenOptions o = warm();
  o.io_threads = 4;
  auto ix = DiskIndex::open(dir, o);
  VectorSearchOptions so;
  std::vector<std::vector<uint32_t>> serial(f.nq), par(f.nq);
  for (uint32_t q = 0; q < f.nq; ++q)
    for (auto& h : ix->search(f.queries.data() + size_t(q) * f.dim, so).hits) serial[q].push_back(h.doc);
  std::vector<std::thread> ts;
  for (int t = 0; t < 6; ++t)
    ts.emplace_back([&, t] {
      for (uint32_t q = t; q < f.nq; q += 6)
        for (auto& h : ix->search(f.queries.data() + size_t(q) * f.dim, so).hits) par[q].push_back(h.doc);
    });
  for (auto& th : ts) th.join();
  EXPECT_EQ(serial, par);
  fs::remove_all(dir);
}

TEST(DiskIndex, BeamWidthCutsRoundsNotResultsMuch) {
  const auto& f = fx();
  std::string dir = tmpdir("beam");
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1));
  auto ix = DiskIndex::open(dir, warm());
  VectorSearchOptions so;
  so.L = 64;
  uint64_t rounds1 = 0, rounds8 = 0;
  for (uint32_t q = 0; q < 50; ++q) {
    const float* qv = f.queries.data() + size_t(q) * f.dim;
    so.beam_width = 1;
    rounds1 += ix->search(qv, so).io_rounds;
    so.beam_width = 8;
    rounds8 += ix->search(qv, so).io_rounds;
  }
  EXPECT_LT(rounds8 * 3, rounds1);
  EXPECT_GE(recall(*ix, 64, 8), 0.95);
}

TEST(DiskIndex, DeadlineReturnsPartialBestSoFar) {
  const auto& f = fx();
  std::string dir = tmpdir("deadline");
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1));
  auto ix = DiskIndex::open(dir, warm());
  VectorSearchOptions so;
  so.L = 200;
  so.beam_width = 2;
  hs::Deadline dl = hs::Deadline::after_us(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto r = ix->search(f.queries.data(), so, dl);
  EXPECT_TRUE(r.partial);
  EXPECT_EQ(r.io_rounds, 1u);  // exactly one round completes, then it stops
  EXPECT_FALSE(r.hits.empty());
  for (size_t i = 1; i < r.hits.size(); ++i) EXPECT_TRUE(hs::ranks_before(r.hits[i - 1], r.hits[i]));
  auto full = ix->search(f.queries.data(), so);
  EXPECT_FALSE(full.partial);
  EXPECT_GT(full.hops, r.hops);
  fs::remove_all(dir);
}

TEST(DiskIndex, DocidsMapOrdinals) {
  const auto& f = fx();
  std::string dir = tmpdir("docids");
  std::vector<uint64_t> ids(f.n);
  for (uint32_t i = 0; i < f.n; ++i) ids[i] = 1000000007ull + 3 * i;
  build_disk_index(f.base.data(), f.n, f.dim, dir, small_params(1), nullptr, &ids);
  auto ix = DiskIndex::open(dir, warm());
  EXPECT_EQ(ix->global_id(5), 1000000022ull);
  EXPECT_EQ(ix->memory().docids, 8ull * f.n);
  EXPECT_EQ(ix->memory().pq_codes, 16ull * f.n);
  fs::remove_all(dir);
}

}  // namespace hs::vector

namespace hs::vector {
TEST(DiskIndex, PrebuiltGraphGivesSameIndexAsBuildingIt) {
  const auto& f = fx();
  std::string d1 = tmpdir("pre1"), d2 = tmpdir("pre2");
  auto p = small_params(1);
  build_disk_index(f.base.data(), f.n, f.dim, d1, p);
  auto g = VamanaIndex::build(f.base.data(), f.n, f.dim, p.vamana);
  std::string gp = tmpdir("pre_graph");
  g.save(gp);
  p.prebuilt_graph = gp;
  build_disk_index(f.base.data(), f.n, f.dim, d2, p);
  auto a = DiskIndex::open(d1, warm()), b = DiskIndex::open(d2, warm());
  std::vector<float> va, vb;
  std::vector<uint32_t> na, nb;
  for (uint32_t id = 0; id < f.n; id += 97) {
    a->read_node(id, va, na);
    b->read_node(id, vb, nb);
    EXPECT_EQ(na, nb);
  }
  EXPECT_EQ(a->medoid(), b->medoid());
  fs::remove_all(d1);
  fs::remove_all(d2);
  fs::remove(gp);
}
}  // namespace hs::vector
