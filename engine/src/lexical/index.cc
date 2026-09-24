#include "hs/lexical/index.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "hs/common/fbin.hpp"
#include "varint.hpp"

namespace hs::lexical {

const char* algorithm_name(Algorithm a) {
  switch (a) {
    case Algorithm::Exhaustive: return "exhaustive";
    case Algorithm::DAAT: return "daat";
    case Algorithm::MaxScore: return "maxscore";
    case Algorithm::WAND: return "wand";
    case Algorithm::BMW: return "bmw";
  }
  return "?";
}

bool parse_algorithm(const std::string& s, Algorithm* out) {
  for (int i = 0; i < kNumAlgorithms; ++i)
    if (s == algorithm_name(Algorithm(i))) {
      *out = Algorithm(i);
      return true;
    }
  if (s == "taat") {
    *out = Algorithm::Exhaustive;
    return true;
  }
  return false;
}

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::map<std::string, std::string> read_meta(const std::string& path) {
  std::map<std::string, std::string> m;
  std::istringstream in(read_file(path));
  std::string line;
  while (std::getline(in, line)) {
    size_t eq = line.find('=');
    if (eq != std::string::npos) m[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return m;
}

uint64_t meta_u64(const std::map<std::string, std::string>& m, const char* k) {
  auto it = m.find(k);
  if (it == m.end()) throw std::runtime_error(std::string("meta.txt: missing ") + k);
  return std::stoull(it->second);
}

}  // namespace

std::unique_ptr<LexicalIndex> LexicalIndex::open(const std::string& dir) {
  std::unique_ptr<LexicalIndex> ix(new LexicalIndex());
  ix->dir_ = dir;
  auto meta = read_meta(dir + "/meta.txt");
  if (meta["format"] != "hs-lexical-1") throw std::runtime_error(dir + ": unknown index format");
  IndexStats& st = ix->stats_;
  st.num_docs = meta_u64(meta, "num_docs");
  st.docs_with_terms = meta_u64(meta, "docs_with_terms");
  st.sum_dl = meta_u64(meta, "sum_dl");
  st.num_terms = meta_u64(meta, "num_terms");
  st.num_postings = meta_u64(meta, "num_postings");
  st.num_blocks = meta_u64(meta, "num_blocks");
  ix->build_k1_ = std::stof(meta["build_k1"]);
  ix->build_b_ = std::stof(meta["build_b"]);
  if (meta_u64(meta, "block_size") != kBlockSize) throw std::runtime_error("index block size mismatch");
  ix->idf_n_ = meta.count("global_docs_with_terms") ? meta_u64(meta, "global_docs_with_terms") : st.docs_with_terms;
  ix->idf_sum_dl_ = meta.count("global_sum_dl") ? meta_u64(meta, "global_sum_dl") : st.sum_dl;
  ix->avgdl_ = ix->idf_n_ ? bm25_avgdl(ix->idf_sum_dl_, ix->idf_n_) : 1.0f;

  bool has[kNumCodecs] = {false, false};
  {
    std::istringstream cs(meta["codecs"]);
    std::string c;
    while (std::getline(cs, c, ',')) {
      Codec cc;
      if (!parse_codec(c, &cc)) throw std::runtime_error("meta.txt: unknown codec " + c);
      has[int(cc)] = true;
    }
  }

  ix->docids_ = read_u64bin(dir + "/docids.u64bin");
  if (ix->docids_.size() != st.num_docs) throw std::runtime_error("docids.u64bin size mismatch");
  ix->docids_sorted_ = std::is_sorted(ix->docids_.begin(), ix->docids_.end());
  {
    std::string dl = read_file(dir + "/doclen.u32");
    if (dl.size() != st.num_docs * 4) throw std::runtime_error("doclen.u32 size mismatch");
    ix->doclen_.resize(st.num_docs);
    std::memcpy(ix->doclen_.data(), dl.data(), dl.size());
    std::string nm = read_file(dir + "/norms.u8");
    if (nm.size() != st.num_docs) throw std::runtime_error("norms.u8 size mismatch");
    ix->norms_.assign(nm.begin(), nm.end());
  }
  ix->term_strings_ = read_file(dir + "/terms.bin");
  if (meta.count("global_docs_with_terms")) {
    std::string g = read_file(dir + "/global_df.u32");
    if (g.size() != st.num_terms * 4) throw std::runtime_error("global_df.u32 size mismatch");
    ix->global_df_.resize(st.num_terms);
    std::memcpy(ix->global_df_.data(), g.data(), g.size());
  }

  {
    std::string lex = read_file(dir + "/lexicon.bin");
    VarintReader r{reinterpret_cast<const uint8_t*>(lex.data()), reinterpret_cast<const uint8_t*>(lex.data()) + lex.size()};
    ix->terms_.resize(st.num_terms);
    uint64_t str_off = 0, post_off[kNumCodecs] = {0, 0};
    uint32_t next_block = 0;
    auto read_bound = [&](BoundInfo& b) {
      b.max_tf = r.next32();
      b.min_dl = r.next32();
      for (int m = 0; m < 2; ++m) {
        b.best_tf[m] = r.next32();
        b.best_dl[m] = r.next32();
      }
    };
    for (auto& t : ix->terms_) {
      t.str_off = str_off;
      t.str_len = r.next32();
      str_off += t.str_len;
      t.df = r.next32();
      t.cf = r.next() + t.df;
      for (int c = 0; c < kNumCodecs; ++c) {
        if (!has[c]) continue;
        t.post_off[c] = post_off[c];
        post_off[c] += r.next();
      }
      read_bound(t.bound);
      t.first_block = next_block;
      if (t.df > kBlockSize) next_block += (t.df + kBlockSize - 1) / kBlockSize;
    }
    if (r.p != r.end || str_off != ix->term_strings_.size()) throw std::runtime_error("lexicon.bin: inconsistent");
    if (next_block != st.num_blocks) throw std::runtime_error("lexicon/blocks count mismatch");

    std::string blk = read_file(dir + "/blocks.bin");
    VarintReader br{reinterpret_cast<const uint8_t*>(blk.data()), reinterpret_cast<const uint8_t*>(blk.data()) + blk.size()};
    ix->blocks_.resize(st.num_blocks);
    for (const auto& t : ix->terms_) {
      if (t.df <= kBlockSize) continue;
      uint32_t nb = (t.df + kBlockSize - 1) / kBlockSize;
      uint32_t last = 0, off[kNumCodecs] = {0, 0};
      for (uint32_t j = 0; j < nb; ++j) {
        BlockInfo& b = ix->blocks_[t.first_block + j];
        last += br.next32();
        b.last_doc = last;
        for (int c = 0; c < kNumCodecs; ++c) {
          if (!has[c]) continue;
          off[c] += br.next32();
          b.off[c] = off[c];
        }
        b.bound.max_tf = br.next32();
        b.bound.min_dl = br.next32();
        for (int m = 0; m < 2; ++m) {
          b.bound.best_tf[m] = br.next32();
          b.bound.best_dl[m] = br.next32();
        }
      }
    }
    if (br.p != br.end) throw std::runtime_error("blocks.bin: trailing bytes");

    for (int c = 0; c < kNumCodecs; ++c) {
      if (!has[c]) continue;
      std::string path = dir + "/postings." + codec_name(Codec(c));
      int fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0) throw std::runtime_error("cannot open " + path);
      struct stat sb {};
      fstat(fd, &sb);
      Mapped& m = ix->postings_[c];
      m.size = uint64_t(sb.st_size);
      if (m.size != post_off[c]) {
        ::close(fd);
        throw std::runtime_error(path + ": size does not match lexicon");
      }
      // Decoders never read past a block's end, so an exact-length mapping is enough.
      static const uint8_t kEmpty[1] = {0};
      m.map_len = size_t(m.size);
      if (m.map_len == 0) {
        m.data = kEmpty;
        ::close(fd);
        continue;
      }
      m.base = mmap(nullptr, m.map_len, PROT_READ, MAP_SHARED, fd, 0);
      ::close(fd);
      if (m.base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
      m.data = static_cast<const uint8_t*>(m.base);
    }
  }
  return ix;
}

LexicalIndex::~LexicalIndex() {
  for (auto& m : postings_)
    if (m.base && m.base != MAP_FAILED) munmap(m.base, m.map_len);
}

int64_t LexicalIndex::ordinal_of(uint64_t gid) const {
  if (docids_sorted_) {
    auto it = std::lower_bound(docids_.begin(), docids_.end(), gid);
    if (it == docids_.end() || *it != gid) return -1;
    return int64_t(it - docids_.begin());
  }
  for (size_t i = 0; i < docids_.size(); ++i)
    if (docids_[i] == gid) return int64_t(i);
  return -1;
}

std::string_view LexicalIndex::term_string(uint32_t id) const {
  const TermInfo& t = terms_[id];
  return std::string_view(term_strings_).substr(t.str_off, t.str_len);
}

uint32_t LexicalIndex::df(uint32_t id) const { return terms_[id].df; }

int64_t LexicalIndex::term_id(std::string_view term) const {
  size_t lo = 0, hi = terms_.size();
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    int c = term_string(uint32_t(mid)).compare(term);
    if (c == 0) return int64_t(mid);
    if (c < 0) lo = mid + 1;
    else hi = mid;
  }
  return -1;
}

void LexicalIndex::postings(uint32_t id, Codec c, std::vector<uint32_t>& docs, std::vector<uint32_t>& tfs) const {
  const TermInfo& t = terms_[id];
  if (!has_codec(c)) throw std::invalid_argument("index has no postings for this codec");
  uint32_t nb = (t.df + kBlockSize - 1) / kBlockSize;
  docs.resize(size_t(nb) * kBlockSize);
  tfs.resize(size_t(nb) * kBlockSize);
  const uint8_t* p = postings_[int(c)].data + t.post_off[int(c)];
  uint32_t prev = UINT32_MAX;
  for (uint32_t j = 0; j < nb; ++j) {
    uint32_t n = std::min<uint32_t>(kBlockSize, t.df - j * kBlockSize);
    p = decode_block(c, p, n, prev, docs.data() + size_t(j) * kBlockSize, tfs.data() + size_t(j) * kBlockSize);
    prev = docs[size_t(j) * kBlockSize + n - 1];
  }
  docs.resize(t.df);
  tfs.resize(t.df);
}

}  // namespace hs::lexical
