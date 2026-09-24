#include "hs/vector/disk_index.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "hs/common/fbin.hpp"
#include "hs/vector/containers.hpp"
#include "hs/vector/disk_format.hpp"
#include "hs/vector/distance.hpp"

namespace hs::vector {

uint64_t peak_rss_bytes() {
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return uint64_t(ru.ru_maxrss);
#else
  return uint64_t(ru.ru_maxrss) * 1024;
#endif
}

void pread_full(int fd, void* buf, size_t len, uint64_t off) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (len > 0) {
    ssize_t r = ::pread(fd, p, len, off_t(off));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
    p += r;
    len -= size_t(r);
    off += uint64_t(r);
  }
}

namespace {

struct IoRequest {
  int fd;
  void* buf;
  size_t len;
  uint64_t off;
  std::atomic<int>* remaining;
  std::atomic<int>* errors;
};

// A fixed pool of threads that execute preads. A query hands it W-1 requests,
// performs one read itself, then waits for the rest: W reads in flight at once,
// which is what an SSD needs to approach its IOPS (one read at a time leaves it
// idle between requests).
class IoPool {
 public:
  explicit IoPool(unsigned threads) {
    for (unsigned t = 0; t < threads; ++t) workers_.emplace_back([this] { run(); });
  }
  ~IoPool() {
    {
      std::lock_guard<std::mutex> g(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }
  void submit(const IoRequest* reqs, size_t n) {
    {
      std::lock_guard<std::mutex> g(m_);
      for (size_t i = 0; i < n; ++i) q_.push_back(reqs[i]);
    }
    if (n == 1) cv_.notify_one();
    else cv_.notify_all();
  }

 private:
  void run() {
    for (;;) {
      IoRequest r;
      {
        std::unique_lock<std::mutex> g(m_);
        cv_.wait(g, [this] { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return;
        r = q_.front();
        q_.pop_front();
      }
      execute(r);
    }
  }
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<IoRequest> q_;
  std::vector<std::thread> workers_;
  bool stop_ = false;

 public:
  static void execute(const IoRequest& r) {
    try {
      pread_full(r.fd, r.buf, r.len, r.off);
    } catch (...) {
      r.errors->fetch_add(1);
    }
    if (r.remaining->fetch_sub(1) == 1) r.remaining->notify_all();
  }
};

struct AlignedBuf {
  uint8_t* p = nullptr;
  size_t cap = 0;
  ~AlignedBuf() { std::free(p); }
  uint8_t* get(size_t bytes) {
    if (bytes > cap) {
      std::free(p);
      void* q = nullptr;
      if (posix_memalign(&q, kBlock, bytes) != 0) throw std::bad_alloc();
      p = static_cast<uint8_t*>(q);
      cap = bytes;
    }
    return p;
  }
};

struct DiskScratch {
  VisitedSet visited;
  CandidateList cands;
  std::vector<float> table;
  AlignedBuf buf;
  std::vector<uint32_t> frontier;
  std::vector<const uint8_t*> ptrs;
  std::vector<IoRequest> reqs;
  std::vector<hs::ScoredDoc> full;
  // Completion counters live here, not on the stack: an I/O worker may still be
  // inside notify_all() after the waiting query has seen zero and moved on, so
  // the atomic must outlive the query (a stale wake-up is harmless, the waiter
  // re-checks the count).
  std::atomic<int> remaining{0}, errors{0};
};

}  // namespace

struct DiskIndex::Impl {
  std::unique_ptr<IoPool> pool;
  IoMode mode = IoMode::kSequential;
};

DiskIndex::~DiskIndex() {
  impl_.reset();
  if (fd_ >= 0) ::close(fd_);
}

uint64_t DiskIndex::record_offset(uint32_t id) const {
  return disk_record_offset(id, record_bytes_, nodes_per_block_, blocks_per_node_);
}
uint64_t DiskIndex::read_offset(uint32_t id) const {
  return record_offset(id) / kBlock * kBlock;
}

const uint8_t* DiskIndex::cached(uint32_t id) const {
  if (cache_ids_.empty()) return nullptr;
  auto it = std::lower_bound(cache_ids_.begin(), cache_ids_.end(), id);
  if (it == cache_ids_.end() || *it != id) return nullptr;
  return cache_data_.data() + size_t(it - cache_ids_.begin()) * record_bytes_;
}

std::unique_ptr<DiskIndex> DiskIndex::open(const std::string& dir, const DiskOpenOptions& o) {
  std::unique_ptr<DiskIndex> ix(new DiskIndex());
  std::string path = dir + "/disk.index";
  int flags = O_RDONLY;
#if defined(O_DIRECT)
  if (o.direct_io) flags |= O_DIRECT;
#endif
  ix->fd_ = ::open(path.c_str(), flags);
  if (ix->fd_ < 0) throw std::runtime_error("cannot open " + path);
#if defined(__APPLE__)
  // F_NOCACHE: reads do not populate the unified buffer cache. They ARE still
  // served from it when the page is already resident (measured: 2 us vs 270 us,
  // bench_vec_coldread), so a cold run after a warm one would silently read RAM.
  // Evict the file's cached pages first: msync(MS_INVALIDATE) over a mapping.
  if (o.direct_io) {
    ::fcntl(ix->fd_, F_NOCACHE, 1);
    struct stat st {};
    if (::fstat(ix->fd_, &st) == 0 && st.st_size > 0) {
      void* m = ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, ix->fd_, 0);
      if (m != MAP_FAILED) {
        ::msync(m, size_t(st.st_size), MS_INVALIDATE);
        ::munmap(m, size_t(st.st_size));
      }
    }
  }
#elif defined(POSIX_FADV_DONTNEED)
  if (o.direct_io) ::posix_fadvise(ix->fd_, 0, 0, POSIX_FADV_DONTNEED);
#endif
  AlignedBuf hb;
  uint8_t* h = hb.get(kBlock);
  pread_full(ix->fd_, h, kBlock, 0);
  DiskHeader hdr;
  std::memcpy(&hdr, h, sizeof(hdr));
  if (hdr.magic != kDiskMagic) throw std::runtime_error("not a disk index: " + path);
  ix->n_ = hdr.n;
  ix->dim_ = hdr.dim;
  ix->R_ = hdr.R;
  ix->medoid_ = hdr.medoid;
  ix->record_bytes_ = hdr.record_bytes;
  ix->nodes_per_block_ = hdr.nodes_per_block;
  ix->blocks_per_node_ = hdr.blocks_per_node;

  ix->pq_ = ProductQuantizer::load(dir + "/pq_pivots.bin");
  {
    std::FILE* f = std::fopen((dir + "/pq_codes.bin").c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open pq_codes.bin in " + dir);
    uint32_t cn = 0, cm = 0;
    bool ok = std::fread(&cn, 4, 1, f) == 1 && std::fread(&cm, 4, 1, f) == 1 && cn == ix->n_ &&
              cm == ix->pq_.M();
    if (ok) {
      ix->codes_.resize(size_t(cn) * cm);
      ok = std::fread(ix->codes_.data(), 1, ix->codes_.size(), f) == ix->codes_.size();
    }
    std::fclose(f);
    if (!ok) throw std::runtime_error("bad pq_codes.bin in " + dir);
  }
  if (std::filesystem::exists(dir + "/docids.u64bin")) ix->docids_ = hs::read_u64bin(dir + "/docids.u64bin");

  ix->impl_ = std::make_unique<Impl>();
  ix->impl_->mode = o.io_mode;
  if (o.io_mode == IoMode::kThreadPool && o.io_threads > 0)
    ix->impl_->pool = std::make_unique<IoPool>(o.io_threads);
  else
    ix->impl_->mode = IoMode::kSequential;

  if (o.cache_nodes > 0) {
    // Cache the BFS neighbourhood of the medoid: every search starts there, so
    // the first hops are always served from RAM.
    uint32_t want = std::min(o.cache_nodes, ix->n_);
    std::vector<uint32_t> order;
    VisitedSet seen;
    order.push_back(ix->medoid_);
    seen.insert(ix->medoid_);
    std::vector<float> v;
    std::vector<uint32_t> nb;
    std::vector<std::vector<uint8_t>> recs;
    for (size_t head = 0; head < order.size() && recs.size() < want; ++head) {
      AlignedBuf& b = hb;
      uint8_t* p = b.get(ix->read_len());
      pread_full(ix->fd_, p, ix->read_len(), ix->read_offset(order[head]));
      const uint8_t* rec = p + (ix->record_offset(order[head]) - ix->read_offset(order[head]));
      recs.emplace_back(rec, rec + ix->record_bytes_);
      uint32_t deg;
      std::memcpy(&deg, rec + size_t(ix->dim_) * 4, 4);
      const uint32_t* nbrs = reinterpret_cast<const uint32_t*>(rec + size_t(ix->dim_) * 4 + 4);
      for (uint32_t j = 0; j < deg && order.size() < want; ++j)
        if (seen.insert(nbrs[j])) order.push_back(nbrs[j]);
    }
    order.resize(recs.size());
    std::vector<size_t> perm(order.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) { return order[a] < order[b]; });
    ix->cache_ids_.resize(perm.size());
    ix->cache_data_.resize(perm.size() * ix->record_bytes_);
    for (size_t i = 0; i < perm.size(); ++i) {
      ix->cache_ids_[i] = order[perm[i]];
      std::memcpy(ix->cache_data_.data() + i * ix->record_bytes_, recs[perm[i]].data(), ix->record_bytes_);
    }
  }
  return ix;
}

DiskMemory DiskIndex::memory() const {
  DiskMemory m;
  m.pq_codes = codes_.size();
  m.pq_pivots = pq_.pivot_bytes();
  m.cache = cache_data_.size() + cache_ids_.size() * 4;
  m.docids = docids_.size() * 8;
  return m;
}

void DiskIndex::read_node(uint32_t id, std::vector<float>& vec, std::vector<uint32_t>& nbrs) const {
  AlignedBuf b;
  uint8_t* p = b.get(read_len());
  pread_full(fd_, p, read_len(), read_offset(id));
  const uint8_t* rec = p + (record_offset(id) - read_offset(id));
  vec.assign(reinterpret_cast<const float*>(rec), reinterpret_cast<const float*>(rec) + dim_);
  uint32_t deg;
  std::memcpy(&deg, rec + size_t(dim_) * 4, 4);
  const uint32_t* nb = reinterpret_cast<const uint32_t*>(rec + size_t(dim_) * 4 + 4);
  nbrs.assign(nb, nb + deg);
}

VectorResult DiskIndex::search(const float* q, const VectorSearchOptions& opts, hs::Deadline& deadline) const {
  thread_local DiskScratch s;
  VectorResult r;
  const uint32_t k = std::min(opts.k, n_);
  const uint32_t L = std::max(opts.L, k);
  const uint32_t W = std::max<uint32_t>(1, opts.beam_width);
  const uint32_t M = pq_.M();
  s.table.resize(size_t(M) * ProductQuantizer::kCentroids);
  pq_.distance_table(q, s.table.data());
  s.visited.clear();
  s.cands.reset(L);
  s.full.clear();
  uint8_t* buf = s.buf.get(size_t(W) * read_len());

  s.visited.insert(medoid_);
  s.cands.insert(medoid_, pq_.adc(s.table.data(), codes_.data() + size_t(medoid_) * M));
  r.pq_distance_computations = 1;

  while (s.cands.has_unexpanded()) {
    s.frontier.clear();
    while (s.frontier.size() < W && s.cands.has_unexpanded()) s.frontier.push_back(s.cands.pop_closest_unexpanded().id);
    s.ptrs.assign(s.frontier.size(), nullptr);
    s.reqs.clear();
    std::atomic<int>& remaining = s.remaining;
    std::atomic<int>& errors = s.errors;
    errors.store(0);
    for (size_t f = 0; f < s.frontier.size(); ++f) {
      uint32_t id = s.frontier[f];
      if (const uint8_t* c = cached(id)) {
        s.ptrs[f] = c;
        ++r.cache_hits;
        continue;
      }
      uint8_t* slot = buf + s.reqs.size() * read_len();
      s.reqs.push_back({fd_, slot, read_len(), read_offset(id), &remaining, &errors});
      s.ptrs[f] = slot + (record_offset(id) - read_offset(id));
    }
    if (!s.reqs.empty()) {
      auto t0 = std::chrono::steady_clock::now();
      remaining.store(int(s.reqs.size()));
      if (impl_->mode == IoMode::kThreadPool && s.reqs.size() > 1) {
        impl_->pool->submit(s.reqs.data() + 1, s.reqs.size() - 1);
        IoPool::execute(s.reqs[0]);
        for (int v; (v = remaining.load(std::memory_order_acquire)) != 0;) remaining.wait(v);
      } else {
        for (auto& rq : s.reqs) IoPool::execute(rq);
      }
      if (errors.load()) throw std::runtime_error("disk index read failed");
      r.io_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - t0).count());
      r.ssd_reads += s.reqs.size() * blocks_per_node_;
      ++r.io_rounds;
    }
    for (size_t f = 0; f < s.frontier.size(); ++f) {
      const uint8_t* rec = s.ptrs[f];
      const float* vec = reinterpret_cast<const float*>(rec);
      uint32_t deg;
      std::memcpy(&deg, rec + size_t(dim_) * 4, 4);
      const uint32_t* nbrs = reinterpret_cast<const uint32_t*>(rec + size_t(dim_) * 4 + 4);
      s.full.push_back({s.frontier[f], ip(q, vec, dim_)});
      for (uint32_t j = 0; j < deg; ++j) {
        uint32_t nb = nbrs[j];
        if (!s.visited.insert(nb)) continue;
        s.cands.insert(nb, pq_.adc(s.table.data(), codes_.data() + size_t(nb) * M));
        ++r.pq_distance_computations;
      }
    }
    r.distance_computations += s.frontier.size();
    r.hops += s.frontier.size();
    if (deadline.expired()) break;  // checked once per round; a round is >= one SSD read
  }
  hs::TopK top(k);
  for (auto& h : s.full) top.push(h.doc, h.score);
  r.hits = top.take_sorted();
  r.partial = deadline.fired();
  return r;
}

}  // namespace hs::vector
