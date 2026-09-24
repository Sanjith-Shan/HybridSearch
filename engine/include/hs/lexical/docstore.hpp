#pragma once
// Compressed passage store for Fetch and snippets.
//
// docstore.bin is a sequence of independently compressed blocks; each block holds the texts
// of consecutive ordinals: [u32 count][u32 end offsets x count][texts]. Blocks close once
// they reach kTargetBlockBytes of raw text, so a fetch decompresses ~16 KB. docstore.idx is
// [u64 nblocks] then per block {u64 file offset, u32 compressed bytes, u32 first ordinal}.
// Compression is zstd when the build found libzstd (HS_HAVE_ZSTD); otherwise blocks are
// stored raw and the header records that.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hs::lexical {

class DocStoreWriter {
 public:
  static constexpr size_t kTargetBlockBytes = 16 * 1024;
  DocStoreWriter(const std::string& dir, int level);
  ~DocStoreWriter();

  // Compress one block of consecutive documents (thread-safe, no shared state).
  static std::string compress_block(const std::vector<std::string_view>& texts, int level);
  // Append an already-compressed block holding `count` docs starting at `first_ordinal`.
  void append_block(const std::string& compressed, uint32_t first_ordinal);
  void finish();
  uint64_t bytes_written() const { return offset_; }
  int level() const { return level_; }

 private:
  std::string dir_;
  int level_;  // recorded for reports; blocks are compressed by the caller
  std::FILE* bin_ = nullptr;
  struct Entry {
    uint64_t off;
    uint32_t len;
    uint32_t first;
  };
  std::vector<Entry> idx_;
  uint64_t offset_ = 0;
};

class DocStore {
 public:
  static std::unique_ptr<DocStore> open(const std::string& dir);
  ~DocStore();
  uint64_t num_docs() const { return num_docs_; }
  // Full passage text of an ordinal (decompresses its block; cached per thread).
  std::string text(uint32_t ordinal) const;
  bool compressed() const { return compressed_; }

 private:
  DocStore() = default;
  struct Entry {
    uint64_t off;
    uint32_t len;
    uint32_t first;
  };
  std::vector<Entry> idx_;
  uint64_t num_docs_ = 0;
  bool compressed_ = true;
  const uint8_t* data_ = nullptr;
  void* map_ = nullptr;
  size_t map_len_ = 0;
  uint64_t id_ = 0;  // distinguishes stores in the per-thread cache
};

}  // namespace hs::lexical
