#pragma once
// Read-only memory map of a file, and of a .fbin vector file in particular.
// Mapping lets a 3 GB embedding file back an index without a heap copy and lets
// the memory-limited disk build touch only the rows a partition needs.

#include <cstddef>
#include <cstdint>
#include <string>

namespace hs::vector {

class MappedFile {
 public:
  MappedFile() = default;
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept { *this = std::move(o); }
  MappedFile& operator=(MappedFile&& o) noexcept;

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  // madvise(MADV_DONTNEED)-style hint that the given range will not be reused soon.
  void drop(size_t offset, size_t len) const;

 private:
  uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

// A .fbin (uint32 n, uint32 dim, float rows) viewed in place.
class MappedFbin {
 public:
  MappedFbin() = default;
  explicit MappedFbin(const std::string& path, uint32_t max_rows = 0);
  uint32_t n() const { return n_; }
  uint32_t dim() const { return dim_; }
  const float* data() const { return rows_; }
  const float* row(size_t i) const { return rows_ + i * dim_; }
  const MappedFile& file() const { return file_; }

 private:
  MappedFile file_;
  uint32_t n_ = 0, dim_ = 0;
  const float* rows_ = nullptr;
};

}  // namespace hs::vector
