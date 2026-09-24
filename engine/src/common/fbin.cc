#include "hs/common/fbin.hpp"

#include <cstdio>
#include <stdexcept>

namespace hs {

namespace {
struct File {
  std::FILE* f;
  File(const std::string& path, const char* mode) : f(std::fopen(path.c_str(), mode)) {
    if (!f) throw std::runtime_error("cannot open " + path);
  }
  ~File() { std::fclose(f); }
  void read(void* p, size_t bytes, const std::string& what) {
    if (bytes && std::fread(p, 1, bytes, f) != bytes) throw std::runtime_error("short read: " + what);
  }
  void write(const void* p, size_t bytes, const std::string& what) {
    if (bytes && std::fwrite(p, 1, bytes, f) != bytes) throw std::runtime_error("short write: " + what);
  }
};
}  // namespace

FloatMatrix read_fbin(const std::string& path, uint32_t max_rows) {
  File file(path, "rb");
  FloatMatrix m;
  file.read(&m.n, 4, path);
  file.read(&m.dim, 4, path);
  if (max_rows && max_rows < m.n) m.n = max_rows;
  m.data.resize(size_t(m.n) * m.dim);
  file.read(m.data.data(), m.data.size() * sizeof(float), path);
  return m;
}

void write_fbin(const std::string& path, const float* data, uint32_t n, uint32_t dim) {
  File file(path, "wb");
  file.write(&n, 4, path);
  file.write(&dim, 4, path);
  file.write(data, size_t(n) * dim * sizeof(float), path);
}

std::vector<uint64_t> read_u64bin(const std::string& path) {
  File file(path, "rb");
  uint64_t n = 0;
  file.read(&n, 8, path);
  std::vector<uint64_t> v(n);
  file.read(v.data(), n * 8, path);
  return v;
}

void write_u64bin(const std::string& path, const std::vector<uint64_t>& v) {
  File file(path, "wb");
  uint64_t n = v.size();
  file.write(&n, 8, path);
  file.write(v.data(), n * 8, path);
}

}  // namespace hs
