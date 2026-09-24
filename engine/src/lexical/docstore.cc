#include "hs/lexical/docstore.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>

#if HS_HAVE_ZSTD
#include <zstd.h>
#endif

namespace hs::lexical {

namespace {
constexpr uint32_t kMagicZstd = 0x5A445348;  // "HSDZ"
constexpr uint32_t kMagicRaw = 0x52445348;   // "HSDR"
std::atomic<uint64_t> g_store_ids{1};
}  // namespace

DocStoreWriter::DocStoreWriter(const std::string& dir, int level) : dir_(dir), level_(level) {
  bin_ = std::fopen((dir + "/docstore.bin").c_str(), "wb");
  if (!bin_) throw std::runtime_error("cannot create " + dir + "/docstore.bin");
#if HS_HAVE_ZSTD
  uint32_t magic = kMagicZstd;
#else
  uint32_t magic = kMagicRaw;
#endif
  std::fwrite(&magic, 4, 1, bin_);
  offset_ = 4;
}

DocStoreWriter::~DocStoreWriter() {
  if (bin_) std::fclose(bin_);
}

std::string DocStoreWriter::compress_block(const std::vector<std::string_view>& texts, int level) {
  std::string raw;
  uint32_t n = uint32_t(texts.size());
  raw.append(reinterpret_cast<const char*>(&n), 4);
  uint32_t end = 0;
  for (auto t : texts) {
    end += uint32_t(t.size());
    raw.append(reinterpret_cast<const char*>(&end), 4);
  }
  for (auto t : texts) raw.append(t);
#if HS_HAVE_ZSTD
  std::string out(ZSTD_compressBound(raw.size()), '\0');
  size_t r = ZSTD_compress(out.data(), out.size(), raw.data(), raw.size(), level);
  if (ZSTD_isError(r)) throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(r));
  out.resize(r);
  return out;
#else
  (void)level;
  return raw;
#endif
}

void DocStoreWriter::append_block(const std::string& c, uint32_t first_ordinal) {
  if (std::fwrite(c.data(), 1, c.size(), bin_) != c.size()) throw std::runtime_error("docstore: short write");
  idx_.push_back({offset_, uint32_t(c.size()), first_ordinal});
  offset_ += c.size();
}

void DocStoreWriter::finish() {
  if (std::fclose(bin_) != 0) throw std::runtime_error("docstore: close failed");
  bin_ = nullptr;
  std::FILE* f = std::fopen((dir_ + "/docstore.idx").c_str(), "wb");
  if (!f) throw std::runtime_error("cannot create docstore.idx");
  uint64_t n = idx_.size();
  std::fwrite(&n, 8, 1, f);
  for (const auto& e : idx_) {
    std::fwrite(&e.off, 8, 1, f);
    std::fwrite(&e.len, 4, 1, f);
    std::fwrite(&e.first, 4, 1, f);
  }
  std::fclose(f);
}

std::unique_ptr<DocStore> DocStore::open(const std::string& dir) {
  std::unique_ptr<DocStore> s(new DocStore());
  s->id_ = g_store_ids.fetch_add(1);
  std::FILE* f = std::fopen((dir + "/docstore.idx").c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + dir + "/docstore.idx");
  uint64_t n = 0;
  if (std::fread(&n, 8, 1, f) != 1) throw std::runtime_error("docstore.idx: short read");
  s->idx_.resize(n);
  for (auto& e : s->idx_) {
    if (std::fread(&e.off, 8, 1, f) != 1 || std::fread(&e.len, 4, 1, f) != 1 || std::fread(&e.first, 4, 1, f) != 1)
      throw std::runtime_error("docstore.idx: short read");
  }
  std::fclose(f);
  int fd = ::open((dir + "/docstore.bin").c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open docstore.bin");
  struct stat st {};
  fstat(fd, &st);
  s->map_len_ = size_t(st.st_size);
  s->map_ = mmap(nullptr, s->map_len_, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (s->map_ == MAP_FAILED) throw std::runtime_error("mmap docstore.bin failed");
  s->data_ = static_cast<const uint8_t*>(s->map_);
  uint32_t magic;
  std::memcpy(&magic, s->data_, 4);
  if (magic != kMagicZstd && magic != kMagicRaw) throw std::runtime_error("docstore.bin: bad magic");
  s->compressed_ = magic == kMagicZstd;
#if !HS_HAVE_ZSTD
  if (s->compressed_) throw std::runtime_error("docstore is zstd-compressed but this build has no zstd");
#endif
  // Doc count = first ordinal of the last block + the count in that block's header.
  if (!s->idx_.empty()) {
    const auto& e = s->idx_.back();
    std::string block;
#if HS_HAVE_ZSTD
    if (s->compressed_) {
      unsigned long long sz = ZSTD_getFrameContentSize(s->data_ + e.off, e.len);
      if (sz == ZSTD_CONTENTSIZE_ERROR || sz == ZSTD_CONTENTSIZE_UNKNOWN) throw std::runtime_error("docstore: bad frame");
      block.resize(size_t(sz));
      if (ZSTD_isError(ZSTD_decompress(block.data(), block.size(), s->data_ + e.off, e.len)))
        throw std::runtime_error("docstore: decompress failed");
    } else
#endif
      block.assign(reinterpret_cast<const char*>(s->data_ + e.off), e.len);
    uint32_t cnt;
    std::memcpy(&cnt, block.data(), 4);
    s->num_docs_ = uint64_t(e.first) + cnt;
  }
  return s;
}

DocStore::~DocStore() {
  if (map_ && map_ != MAP_FAILED) munmap(map_, map_len_);
}

std::string DocStore::text(uint32_t ordinal) const {
  if (ordinal >= num_docs_) throw std::out_of_range("docstore: ordinal out of range");
  auto it = std::upper_bound(idx_.begin(), idx_.end(), ordinal, [](uint32_t o, const Entry& e) { return o < e.first; });
  --it;
  size_t bi = size_t(it - idx_.begin());
  thread_local uint64_t cached_store = 0;
  thread_local size_t cached_block = SIZE_MAX;
  thread_local std::string block;
  if (cached_store != id_ || cached_block != bi) {
    const Entry& e = *it;
#if HS_HAVE_ZSTD
    if (compressed_) {
      unsigned long long sz = ZSTD_getFrameContentSize(data_ + e.off, e.len);
      if (sz == ZSTD_CONTENTSIZE_ERROR || sz == ZSTD_CONTENTSIZE_UNKNOWN) throw std::runtime_error("docstore: bad frame");
      block.resize(size_t(sz));
      size_t r = ZSTD_decompress(block.data(), block.size(), data_ + e.off, e.len);
      if (ZSTD_isError(r)) throw std::runtime_error("docstore: decompress failed");
    } else
#endif
      block.assign(reinterpret_cast<const char*>(data_ + e.off), e.len);
    cached_store = id_;
    cached_block = bi;
  }
  uint32_t n;
  std::memcpy(&n, block.data(), 4);
  uint32_t i = ordinal - it->first;
  uint32_t end, begin = 0;
  std::memcpy(&end, block.data() + 4 + 4 * i, 4);
  if (i > 0) std::memcpy(&begin, block.data() + 4 + 4 * (i - 1), 4);
  size_t base = 4 + 4 * size_t(n);
  return block.substr(base + begin, end - begin);
}

}  // namespace hs::lexical
