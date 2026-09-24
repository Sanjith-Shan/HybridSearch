#include "hs/lexical/index_builder.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>

#include "hs/common/fbin.hpp"
#include "hs/lexical/analyzer.hpp"
#include "hs/lexical/bm25.hpp"
#include "hs/lexical/docstore.hpp"
#include "hs/lexical/index.hpp"
#include "varint.hpp"

namespace hs::lexical {

uint64_t count_lines(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> b(1 << 22);
  uint64_t n = 0;
  size_t got;
  char last = '\n';
  while ((got = std::fread(b.data(), 1, b.size(), f)) > 0) {
    n += uint64_t(std::count(b.data(), b.data() + got, '\n'));
    last = b[got - 1];
  }
  std::fclose(f);
  return n + (last != '\n');
}

uint64_t free_disk_bytes(const std::string& path) {
  struct statvfs s {};
  if (statvfs(path.c_str(), &s) != 0) return 0;
  return uint64_t(s.f_bavail) * s.f_frsize;
}

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

inline uint64_t hash_bytes(std::string_view s) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ (s.size() * 0xC2B2AE3D27D4EB4Full);
  const char* p = s.data();
  size_t n = s.size();
  while (n >= 8) {
    uint64_t k;
    std::memcpy(&k, p, 8);
    h = (h ^ k) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 32;
    p += 8;
    n -= 8;
  }
  uint64_t k = 0;
  std::memcpy(&k, p, n);
  h = (h ^ k) * 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 29;
  return h;
}

// Term -> dense ID, open addressing over an append-only string arena.
class Vocab {
 public:
  Vocab() : table_(1 << 20, kEmpty), mask_((1 << 20) - 1) { off_.push_back(0); }
  uint32_t size() const { return uint32_t(off_.size() - 1); }
  std::string_view str(uint32_t id) const { return {arena_.data() + off_[id], size_t(off_[id + 1] - off_[id])}; }
  uint32_t get_or_add(std::string_view s) {
    uint64_t h = hash_bytes(s);
    size_t i = h & mask_;
    while (true) {
      uint32_t id = table_[i];
      if (id == kEmpty) break;
      if (str(id) == s) return id;
      i = (i + 1) & mask_;
    }
    uint32_t id = size();
    arena_.insert(arena_.end(), s.begin(), s.end());
    off_.push_back(arena_.size());
    table_[i] = id;
    if (uint64_t(size()) * 10 > table_.size() * 7) grow();
    return id;
  }

 private:
  static constexpr uint32_t kEmpty = UINT32_MAX;
  void grow() {
    std::vector<uint32_t> t(table_.size() * 2, kEmpty);
    size_t m = t.size() - 1;
    for (uint32_t id = 0; id < size(); ++id) {
      size_t i = hash_bytes(str(id)) & m;
      while (t[i] != kEmpty) i = (i + 1) & m;
      t[i] = id;
    }
    table_.swap(t);
    mask_ = m;
  }
  std::vector<char> arena_;
  std::vector<uint64_t> off_;
  std::vector<uint32_t> table_;
  size_t mask_;
};

// Output of one analysis job (a chunk of consecutive lines).
struct Batch {
  std::vector<uint64_t> pids;
  std::vector<uint32_t> dls;
  std::vector<uint32_t> nterms;    // unique terms per doc
  std::string arena;               // term bytes
  std::vector<uint32_t> term_len;  // per (doc, unique term)
  std::vector<uint32_t> tfs;
  std::vector<std::string> ds_blocks;
  std::vector<uint32_t> ds_block_docs;  // docs per docstore block
  std::string error;
};

Batch analyze_chunk(const std::shared_ptr<std::string>& chunk, uint64_t first_line, bool docstore, int level,
                    uint32_t mod_n, uint32_t mod_i) {
  thread_local Analyzer analyzer;
  Batch b;
  std::vector<std::string> toks;
  std::vector<std::string_view> ds_texts;
  size_t ds_bytes = 0;
  auto flush_ds = [&] {
    if (ds_texts.empty()) return;
    b.ds_blocks.push_back(DocStoreWriter::compress_block(ds_texts, level));
    b.ds_block_docs.push_back(uint32_t(ds_texts.size()));
    ds_texts.clear();
    ds_bytes = 0;
  };
  std::string_view all(*chunk);
  size_t pos = 0;
  uint64_t line_no = first_line;
  while (pos < all.size()) {
    size_t nl = all.find('\n', pos);
    if (nl == std::string_view::npos) nl = all.size();
    std::string_view line = all.substr(pos, nl - pos);
    pos = nl + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    size_t tab = line.find('\t');
    if (tab == std::string_view::npos || tab == 0) {
      b.error = "line " + std::to_string(line_no) + ": expected 'pid<TAB>text'";
      return b;
    }
    uint64_t pid = 0;
    for (char c : line.substr(0, tab)) {
      if (c < '0' || c > '9') {
        b.error = "line " + std::to_string(line_no) + ": non-numeric passage id";
        return b;
      }
      pid = pid * 10 + uint64_t(c - '0');
    }
    if (mod_n && pid % mod_n != mod_i) continue;  // hash (mod) partitioning: not this shard's doc
    std::string_view text = line.substr(tab + 1);
    toks.clear();
    analyzer.analyze(text, toks);
    std::sort(toks.begin(), toks.end());
    uint32_t uniq = 0;
    for (size_t i = 0; i < toks.size();) {
      size_t j = i + 1;
      while (j < toks.size() && toks[j] == toks[i]) ++j;
      b.arena += toks[i];
      b.term_len.push_back(uint32_t(toks[i].size()));
      b.tfs.push_back(uint32_t(j - i));
      ++uniq;
      i = j;
    }
    b.pids.push_back(pid);
    b.dls.push_back(uint32_t(toks.size()));
    b.nterms.push_back(uniq);
    if (docstore) {
      ds_texts.push_back(text);
      ds_bytes += text.size();
      if (ds_bytes >= DocStoreWriter::kTargetBlockBytes) flush_ds();
    }
  }
  flush_ds();
  return b;
}

struct Run {
  std::string path;
  uint32_t vocab = 0;             // terms known when the run was written
  std::vector<uint64_t> offsets;  // vocab + 1 byte offsets
  uint64_t bytes = 0;
};

class Mmap {
 public:
  explicit Mmap(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    fstat(fd, &st);
    len_ = size_t(st.st_size);
    if (len_ > 0) {
      base_ = mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd, 0);
      if (base_ == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed: " + path);
      }
      madvise(base_, len_, MADV_SEQUENTIAL);
    }
    ::close(fd);
  }
  ~Mmap() {
    if (base_ && base_ != MAP_FAILED) munmap(base_, len_);
  }
  const uint8_t* data() const { return static_cast<const uint8_t*>(base_); }
  size_t size() const { return len_; }

 private:
  void* base_ = nullptr;
  size_t len_ = 0;
};

class Writer {
 public:
  explicit Writer(const std::string& path) : path_(path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot create " + path);
    std::setvbuf(f_, nullptr, _IOFBF, 1 << 22);
  }
  ~Writer() {
    if (f_) std::fclose(f_);
  }
  void write(const void* p, size_t n) {
    if (n && std::fwrite(p, 1, n, f_) != n) throw std::runtime_error("short write: " + path_ + " (disk full?)");
    bytes_ += n;
  }
  void write(const std::string& s) { write(s.data(), s.size()); }
  void close() {
    if (f_ && std::fclose(f_) != 0) throw std::runtime_error("close failed: " + path_);
    f_ = nullptr;
  }
  uint64_t bytes() const { return bytes_; }

 private:
  std::string path_;
  std::FILE* f_ = nullptr;
  uint64_t bytes_ = 0;
};

void check_disk(const std::string& dir, double min_free_gb, uint64_t still_needed, const char* stage) {
  uint64_t free_b = free_disk_bytes(dir);
  double after = (double(free_b) - double(still_needed)) / 1e9;
  if (after < min_free_gb) {
    char msg[256];
    std::snprintf(msg, sizeof msg, "refusing to continue (%s): %.2f GB free, ~%.2f GB still to write, need %.1f GB headroom",
                  stage, double(free_b) / 1e9, double(still_needed) / 1e9, min_free_gb);
    throw std::runtime_error(msg);
  }
}

}  // namespace

BuildReport build_index(const BuildOptions& opt) {
  const auto t_start = Clock::now();
  BuildReport rep;
  if (opt.codecs.empty()) throw std::invalid_argument("build_index: no codecs");
  fs::create_directories(opt.out_dir);
  const std::string tmp = opt.tmp_dir.empty() ? opt.out_dir + "/tmp" : opt.tmp_dir;
  fs::create_directories(tmp);
  const uint64_t input_bytes = fs::file_size(opt.input_tsv);
  // Rough upper estimate of what the build writes (runs + postings for each codec + doc store),
  // measured on MS MARCO: ~0.25 B/B of input per codec, ~0.35 for runs, ~0.35 for the doc store.
  const uint64_t estimate =
      uint64_t(double(input_bytes) * (0.35 + 0.25 * double(opt.codecs.size()) + (opt.docstore ? 0.35 : 0.0)));
  check_disk(opt.out_dir, opt.min_free_gb, estimate, "start");

  std::FILE* in = std::fopen(opt.input_tsv.c_str(), "rb");
  if (!in) throw std::runtime_error("cannot open " + opt.input_tsv);
  if (opt.skip_docs) {
    std::vector<char> b(1 << 22);
    uint64_t seen = 0, off = 0;
    bool done = false;
    while (!done) {
      size_t got = std::fread(b.data(), 1, b.size(), in);
      if (got == 0) break;
      for (size_t i = 0; i < got; ++i)
        if (b[i] == '\n' && ++seen == opt.skip_docs) {
          off += i + 1;
          done = true;
          break;
        }
      if (!done) off += got;
    }
    if (!done) throw std::runtime_error("skip_docs beyond end of input");
    std::fseek(in, long(off), SEEK_SET);
  }
  std::unique_ptr<LexicalIndex> global;
  if (!opt.global_stats_dir.empty()) global = LexicalIndex::open(opt.global_stats_dir);

  Vocab vocab;
  std::vector<uint64_t> pids;
  std::vector<uint32_t> dls;
  const uint64_t cap = std::max<uint64_t>(1 << 16, opt.mem_budget_mb * (1ull << 20) / 20);
  std::vector<uint32_t> r_term, r_doc, r_tf;
  r_term.reserve(cap);
  r_doc.reserve(cap);
  r_tf.reserve(cap);
  std::vector<Run> runs;
  std::unique_ptr<DocStoreWriter> ds;
  if (opt.docstore) ds = std::make_unique<DocStoreWriter>(opt.out_dir, opt.zstd_level);

  auto flush_run = [&] {
    if (r_term.empty()) return;
    check_disk(opt.out_dir, opt.min_free_gb, estimate / 2, "run spill");
    const auto t0 = Clock::now();
    Run run;
    run.path = tmp + "/run" + std::to_string(runs.size()) + ".bin";
    run.vocab = vocab.size();
    const size_t n = r_term.size();
    std::vector<uint64_t> start(size_t(run.vocab) + 1, 0);
    for (size_t i = 0; i < n; ++i) ++start[size_t(r_term[i]) + 1];
    for (size_t t = 0; t < run.vocab; ++t) start[t + 1] += start[t];
    std::vector<uint32_t> s_doc(n), s_tf(n);
    {
      std::vector<uint64_t> pos(start.begin(), start.end() - 1);
      for (size_t i = 0; i < n; ++i) {
        uint64_t p = pos[r_term[i]]++;
        s_doc[p] = r_doc[i];
        s_tf[p] = r_tf[i];
      }
    }
    Writer w(run.path);
    run.offsets.resize(size_t(run.vocab) + 1);
    std::string buf;
    uint64_t off = 0;
    for (uint32_t t = 0; t < run.vocab; ++t) {
      run.offsets[t] = off + buf.size();
      uint32_t prev = UINT32_MAX;
      for (uint64_t i = start[t]; i < start[t + 1]; ++i) {
        put_varint(uint32_t(s_doc[i] - prev - 1), buf);
        put_varint(s_tf[i], buf);
        prev = s_doc[i];
      }
      if (buf.size() > (32u << 20)) {
        w.write(buf);
        off += buf.size();
        buf.clear();
      }
    }
    w.write(buf);
    off += buf.size();
    run.offsets[run.vocab] = off;
    w.close();
    run.bytes = off;
    rep.peak_tmp_bytes += off;
    if (opt.verbose)
      std::fprintf(stderr, "[build] run %zu: %zu postings, %u terms, %.1f MB, %.1fs\n", runs.size(), n, run.vocab,
                   double(off) / 1e6, secs(t0));
    runs.push_back(std::move(run));
    r_term.clear();
    r_doc.clear();
    r_tf.clear();
  };

  uint32_t ds_ordinal = 0;
  auto consume = [&](Batch&& b) {
    if (!b.error.empty()) throw std::runtime_error(opt.input_tsv + ": " + b.error);
    size_t ti = 0;
    const char* ap = b.arena.data();
    for (size_t d = 0; d < b.pids.size(); ++d) {
      if (pids.size() >= 0xFFFFFFF0ull) throw std::runtime_error("too many documents for 32-bit ordinals");
      const uint32_t ord = uint32_t(pids.size());
      pids.push_back(b.pids[d]);
      dls.push_back(b.dls[d]);
      for (uint32_t j = 0; j < b.nterms[d]; ++j, ++ti) {
        std::string_view term(ap, b.term_len[ti]);
        ap += b.term_len[ti];
        r_term.push_back(vocab.get_or_add(term));
        r_doc.push_back(ord);
        r_tf.push_back(b.tfs[ti]);
        if (r_term.size() >= cap) flush_run();
      }
    }
    if (ds) {
      for (size_t i = 0; i < b.ds_blocks.size(); ++i) {
        ds->append_block(b.ds_blocks[i], ds_ordinal);
        ds_ordinal += b.ds_block_docs[i];
      }
    }
  };

  // Reader + ordered pipeline of analysis jobs.
  const int workers = std::max(1, opt.threads - 1);
  const size_t kChunk = 8u << 20;
  std::deque<std::future<Batch>> inflight;
  std::string carry;
  uint64_t lines_read = 0;
  bool eof = false;
  std::vector<char> rbuf(kChunk);
  const auto t_inv = Clock::now();
  uint64_t last_report = 0;
  while (!eof || !inflight.empty()) {
    while (!eof && int(inflight.size()) < workers) {
      size_t got = std::fread(rbuf.data(), 1, rbuf.size(), in);
      auto chunk = std::make_shared<std::string>(std::move(carry));
      carry.clear();
      chunk->append(rbuf.data(), got);
      if (got < rbuf.size()) {
        eof = true;
        if (!chunk->empty() && chunk->back() != '\n') chunk->push_back('\n');
      } else {
        size_t last_nl = chunk->rfind('\n');
        if (last_nl == std::string::npos) {
          carry = std::move(*chunk);
          continue;
        }
        carry.assign(*chunk, last_nl + 1, std::string::npos);
        chunk->resize(last_nl + 1);
      }
      // Drop blank lines at the very end; honour max_docs.
      uint64_t n_lines = uint64_t(std::count(chunk->begin(), chunk->end(), '\n'));
      if (opt.max_docs && lines_read + n_lines >= opt.max_docs) {
        uint64_t keep = opt.max_docs - lines_read;
        size_t p = 0;
        for (uint64_t i = 0; i < keep; ++i) p = chunk->find('\n', p) + 1;
        chunk->resize(p);
        n_lines = keep;
        eof = true;
      }
      while (chunk->size() >= 2 && (*chunk)[chunk->size() - 1] == '\n' && (*chunk)[chunk->size() - 2] == '\n') {
        chunk->pop_back();
        --n_lines;
      }
      if (chunk->size() == 1 && (*chunk)[0] == '\n') chunk->clear(), n_lines = 0;
      if (chunk->empty()) continue;
      uint64_t first = lines_read;
      lines_read += n_lines;
      inflight.push_back(std::async(std::launch::async, analyze_chunk, chunk, first, opt.docstore, opt.zstd_level,
                                        opt.mod_n, opt.mod_i));
    }
    if (inflight.empty()) break;
    consume(inflight.front().get());
    inflight.pop_front();
    if (opt.verbose && pids.size() - last_report >= 1000000) {
      last_report = pids.size();
      std::fprintf(stderr, "[build] %zu docs, %u terms, %.0fs\n", pids.size(), vocab.size(), secs(t_inv));
    }
  }
  std::fclose(in);
  flush_run();
  std::vector<uint32_t>().swap(r_term);
  std::vector<uint32_t>().swap(r_doc);
  std::vector<uint32_t>().swap(r_tf);
  if (ds) {
    ds->finish();
    rep.bytes_docstore = ds->bytes_written();
  }
  rep.seconds_invert = secs(t_inv);
  rep.num_runs = uint32_t(runs.size());

  // Collection statistics (Lucene: docCount = documents with at least one indexed term).
  const uint64_t num_docs = pids.size();
  uint64_t sum_dl = 0, with_terms = 0;
  for (uint32_t d : dls) {
    sum_dl += d;
    with_terms += d > 0;
  }
  rep.num_docs = num_docs;
  rep.docs_with_terms = with_terms;
  rep.sum_dl = sum_dl;
  // Scoring statistics: the global ones when building a shard.
  const uint64_t score_n = global ? global->idf_n() : with_terms;
  const uint64_t score_sum_dl = global ? global->idf_sum_dl() : sum_dl;
  const float avgdl = score_n ? bm25_avgdl(score_sum_dl, score_n) : 1.0f;
  std::vector<uint8_t> norms(num_docs);
  for (size_t i = 0; i < num_docs; ++i) norms[i] = smallfloat::int_to_byte4(dls[i]);
  Bm25Norms bn[2];
  bn[0].init(Model::Lucene, kDefaultK1, kDefaultB, avgdl);
  bn[1].init(Model::Textbook, kDefaultK1, kDefaultB, avgdl);

  // Lexicographic term order.
  const auto t_merge = Clock::now();
  check_disk(opt.out_dir, opt.min_free_gb, estimate / 2, "merge");
  const uint32_t V = vocab.size();
  std::vector<uint32_t> order(V);
  for (uint32_t i = 0; i < V; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return vocab.str(a) < vocab.str(b); });

  std::vector<std::unique_ptr<Mmap>> maps;
  for (const auto& r : runs) maps.push_back(std::make_unique<Mmap>(r.path));

  bool has[kNumCodecs] = {false, false};
  for (Codec c : opt.codecs) has[int(c)] = true;
  std::unique_ptr<Writer> post[kNumCodecs];
  for (int c = 0; c < kNumCodecs; ++c)
    if (has[c]) post[c] = std::make_unique<Writer>(opt.out_dir + "/postings." + codec_name(Codec(c)));
  Writer terms_w(opt.out_dir + "/terms.bin"), lex_w(opt.out_dir + "/lexicon.bin"), blk_w(opt.out_dir + "/blocks.bin");
  std::string lex_buf, blk_buf, term_buf, enc;
  std::vector<uint32_t> docs, tfs, gaps, tfm1;
  uint64_t num_postings = 0, num_blocks = 0;

  auto bound_of = [&](size_t lo, size_t hi, BoundInfo& bi, float best_x[2]) {
    bi.max_tf = 0;
    bi.min_dl = UINT32_MAX;
    best_x[0] = best_x[1] = -1.0f;
    for (size_t i = lo; i < hi; ++i) {
      uint32_t d = docs[i], tf = tfs[i], dl = dls[d];
      bi.max_tf = std::max(bi.max_tf, tf);
      bi.min_dl = std::min(bi.min_dl, dl);
      for (int m = 0; m < 2; ++m) {
        float x = bm25_scaled_tf(float(tf), bn[m].inv(dl, norms[d]));
        if (x > best_x[m]) {
          best_x[m] = x;
          bi.best_tf[m] = tf;
          bi.best_dl[m] = dl;
        }
      }
    }
  };

  for (uint32_t oi = 0; oi < V; ++oi) {
    const uint32_t t = order[oi];
    docs.clear();
    tfs.clear();
    for (size_t r = 0; r < runs.size(); ++r) {
      if (t >= runs[r].vocab) continue;
      VarintReader vr{maps[r]->data() + runs[r].offsets[t], maps[r]->data() + runs[r].offsets[t + 1]};
      uint32_t prev = UINT32_MAX;
      while (vr.p < vr.end) {
        prev = prev + vr.next32() + 1;
        docs.push_back(prev);
        tfs.push_back(vr.next32());
      }
    }
    const size_t n = docs.size();
    if (n == 0) throw std::logic_error("term with no postings");
    uint64_t cf = 0;
    gaps.resize(n);
    tfm1.resize(n);
    uint32_t prev = UINT32_MAX;
    for (size_t i = 0; i < n; ++i) {
      if (i && docs[i] <= docs[i - 1]) throw std::logic_error("postings out of order");
      cf += tfs[i];
      gaps[i] = docs[i] - prev - 1;
      prev = docs[i];
      tfm1[i] = tfs[i] - 1;
    }
    const size_t nb = (n + kBlockSize - 1) / kBlockSize;
    BoundInfo term_bi;
    term_bi.min_dl = UINT32_MAX;
    float term_x[2] = {-1.0f, -1.0f};
    std::vector<BoundInfo> bis(nb);
    for (size_t j = 0; j < nb; ++j) {
      float bx[2];
      bound_of(j * kBlockSize, std::min(n, (j + 1) * kBlockSize), bis[j], bx);
      term_bi.max_tf = std::max(term_bi.max_tf, bis[j].max_tf);
      term_bi.min_dl = std::min(term_bi.min_dl, bis[j].min_dl);
      for (int m = 0; m < 2; ++m)
        if (bx[m] > term_x[m]) {
          term_x[m] = bx[m];
          term_bi.best_tf[m] = bis[j].best_tf[m];
          term_bi.best_dl[m] = bis[j].best_dl[m];
        }
    }
    uint64_t term_bytes[kNumCodecs] = {0, 0};
    std::vector<uint32_t> boff[kNumCodecs];
    for (int c = 0; c < kNumCodecs; ++c) {
      if (!has[c]) continue;
      enc.clear();
      boff[c].resize(nb);
      for (size_t j = 0; j < nb; ++j) {
        boff[c][j] = uint32_t(enc.size());
        size_t lo = j * kBlockSize, cnt = std::min(n, lo + kBlockSize) - lo;
        encode_block(Codec(c), gaps.data() + lo, tfm1.data() + lo, uint32_t(cnt), enc);
      }
      term_bytes[c] = enc.size();
      post[c]->write(enc);
    }
    std::string_view s = vocab.str(t);
    term_buf.append(s);
    put_varint(s.size(), lex_buf);
    put_varint(n, lex_buf);
    put_varint(cf - n, lex_buf);
    for (int c = 0; c < kNumCodecs; ++c)
      if (has[c]) put_varint(term_bytes[c], lex_buf);
    auto put_bound = [](const BoundInfo& b, std::string& out) {
      put_varint(b.max_tf, out);
      put_varint(b.min_dl, out);
      for (int m = 0; m < 2; ++m) {
        put_varint(b.best_tf[m], out);
        put_varint(b.best_dl[m], out);
      }
    };
    put_bound(term_bi, lex_buf);
    if (nb > 1) {
      uint32_t prev_last = 0;
      for (size_t j = 0; j < nb; ++j) {
        uint32_t last = docs[std::min(n, (j + 1) * kBlockSize) - 1];
        put_varint(last - prev_last, blk_buf);
        prev_last = last;
        for (int c = 0; c < kNumCodecs; ++c)
          if (has[c]) put_varint(boff[c][j] - (j ? boff[c][j - 1] : 0), blk_buf);
        put_bound(bis[j], blk_buf);
      }
      num_blocks += nb;
    }
    num_postings += n;
    if (lex_buf.size() > (8u << 20)) lex_w.write(lex_buf), lex_buf.clear();
    if (blk_buf.size() > (8u << 20)) blk_w.write(blk_buf), blk_buf.clear();
    if (term_buf.size() > (8u << 20)) terms_w.write(term_buf), term_buf.clear();
  }
  lex_w.write(lex_buf);
  blk_w.write(blk_buf);
  terms_w.write(term_buf);
  lex_w.close();
  blk_w.close();
  terms_w.close();
  for (int c = 0; c < kNumCodecs; ++c)
    if (has[c]) {
      rep.bytes_postings[c] = post[c]->bytes();
      post[c]->close();
    }
  maps.clear();
  for (const auto& r : runs) fs::remove(r.path);
  std::error_code ec;
  if (opt.tmp_dir.empty()) fs::remove(tmp, ec);

  if (global) {
    std::vector<uint32_t> gdf(V);
    for (uint32_t oi = 0; oi < V; ++oi) {
      int64_t g = global->term_id(vocab.str(order[oi]));
      if (g < 0)
        throw std::runtime_error("global stats index lacks term '" + std::string(vocab.str(order[oi])) +
                                 "': it must cover this shard's documents");
      gdf[oi] = global->idf_df(uint32_t(g));
    }
    Writer w(opt.out_dir + "/global_df.u32");
    w.write(gdf.data(), gdf.size() * 4);
    w.close();
  }
  write_u64bin(opt.out_dir + "/docids.u64bin", pids);
  {
    Writer w(opt.out_dir + "/doclen.u32");
    w.write(dls.data(), dls.size() * 4);
    w.close();
    Writer wn(opt.out_dir + "/norms.u8");
    wn.write(norms.data(), norms.size());
    wn.close();
  }
  {
    std::string codecs;
    for (int c = 0; c < kNumCodecs; ++c)
      if (has[c]) codecs += std::string(codecs.empty() ? "" : ",") + codec_name(Codec(c));
    std::FILE* f = std::fopen((opt.out_dir + "/meta.txt").c_str(), "w");
    if (!f) throw std::runtime_error("cannot write meta.txt");
    uint32_t avg_bits;
    std::memcpy(&avg_bits, &avgdl, 4);
    std::fprintf(f,
                 "format=hs-lexical-1\nnum_docs=%llu\ndocs_with_terms=%llu\nsum_dl=%llu\nnum_terms=%u\n"
                 "num_postings=%llu\nnum_blocks=%llu\ncodecs=%s\nblock_size=%u\nbuild_k1=%.9g\nbuild_b=%.9g\n"
                 "avgdl=%.9g\navgdl_bits=%08x\nanalyzer=lucene-10.5.0-anserini-default-english\ninput=%s\n",
                 (unsigned long long)num_docs, (unsigned long long)with_terms, (unsigned long long)sum_dl, V,
                 (unsigned long long)num_postings, (unsigned long long)num_blocks, codecs.c_str(), kBlockSize,
                 double(kDefaultK1), double(kDefaultB), double(avgdl), avg_bits, opt.input_tsv.c_str());
    if (global)
      std::fprintf(f, "global_docs_with_terms=%llu\nglobal_sum_dl=%llu\nglobal_stats_from=%s\nshard=%s\n",
                   (unsigned long long)score_n, (unsigned long long)score_sum_dl, opt.global_stats_dir.c_str(),
                   opt.shard_label.c_str());
    std::fclose(f);
  }
  rep.num_terms = V;
  rep.num_postings = num_postings;
  rep.num_blocks = num_blocks;
  rep.bytes_lexicon = lex_w.bytes();
  rep.bytes_blocks = blk_w.bytes();
  rep.bytes_terms = terms_w.bytes();
  rep.bytes_docmeta = num_docs * (8 + 4 + 1);
  rep.seconds_merge = secs(t_merge);
  rep.seconds_total = secs(t_start);
  return rep;
}

}  // namespace hs::lexical
