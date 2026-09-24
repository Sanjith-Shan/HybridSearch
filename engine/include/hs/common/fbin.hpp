#pragma once
// big-ann-benchmarks .fbin / .u64bin readers and writers.
//
// .fbin layout: uint32 n, uint32 dim, then n*dim float32 row-major.
// .u64bin layout: uint64 n, then n uint64 values (used for docid maps).

#include <cstdint>
#include <string>
#include <vector>

namespace hs {

struct FloatMatrix {
  uint32_t n = 0;
  uint32_t dim = 0;
  std::vector<float> data;
  const float* row(size_t i) const { return data.data() + i * dim; }
};

FloatMatrix read_fbin(const std::string& path, uint32_t max_rows = 0);
void write_fbin(const std::string& path, const float* data, uint32_t n, uint32_t dim);

std::vector<uint64_t> read_u64bin(const std::string& path);
void write_u64bin(const std::string& path, const std::vector<uint64_t>& v);

}  // namespace hs
