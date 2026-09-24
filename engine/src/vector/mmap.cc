#include "hs/vector/mmap.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>
#include <utility>

namespace hs::vector {

MappedFile::MappedFile(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = size_t(st.st_size);
  if (size_ > 0) {
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("cannot mmap " + path);
    }
    data_ = static_cast<uint8_t*>(p);
  }
  ::close(fd);
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(data_, size_);
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    if (data_) ::munmap(data_, size_);
    data_ = std::exchange(o.data_, nullptr);
    size_ = std::exchange(o.size_, 0);
  }
  return *this;
}

void MappedFile::drop(size_t offset, size_t len) const {
  if (!data_ || offset >= size_) return;
  size_t page = size_t(::getpagesize());
  size_t start = offset / page * page;
  size_t end = std::min(size_, offset + len);
  if (end > start) ::madvise(data_ + start, end - start, MADV_DONTNEED);
}

MappedFbin::MappedFbin(const std::string& path, uint32_t max_rows) : file_(path) {
  if (file_.size() < 8) throw std::runtime_error("truncated fbin " + path);
  const uint32_t* h = reinterpret_cast<const uint32_t*>(file_.data());
  n_ = h[0];
  dim_ = h[1];
  if (file_.size() < 8 + size_t(n_) * dim_ * 4) throw std::runtime_error("truncated fbin " + path);
  if (max_rows && max_rows < n_) n_ = max_rows;
  rows_ = reinterpret_cast<const float*>(file_.data() + 8);
}

}  // namespace hs::vector
