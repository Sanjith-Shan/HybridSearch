#pragma once
// On-disk layout constants for disk.index (see disk_index.hpp).

#include <cstddef>
#include <cstdint>

namespace hs::vector {

constexpr uint64_t kDiskMagic = 0x31584e4944534b48ULL;  // "HKSDINX1"

struct DiskHeader {
  uint64_t magic = kDiskMagic;
  uint32_t version = 1;
  uint32_t n = 0, dim = 0, R = 0, medoid = 0;
  uint32_t record_bytes = 0, nodes_per_block = 0, blocks_per_node = 0;
};

// Byte offset of node `id`'s record. Block 0 is the header.
inline uint64_t disk_record_offset(uint32_t id, uint32_t record_bytes, uint32_t nodes_per_block,
                                   uint32_t blocks_per_node) {
  constexpr uint64_t B = 4096;
  if (nodes_per_block > 1)
    return B + uint64_t(id / nodes_per_block) * B + uint64_t(id % nodes_per_block) * record_bytes;
  return B + uint64_t(id) * blocks_per_node * B;
}

inline void disk_layout(uint32_t dim, uint32_t R, uint32_t* record_bytes, uint32_t* nodes_per_block,
                        uint32_t* blocks_per_node) {
  constexpr uint32_t B = 4096;
  *record_bytes = dim * 4 + 4 + R * 4;
  if (*record_bytes <= B) {
    *nodes_per_block = B / *record_bytes;
    *blocks_per_node = 1;
  } else {
    *nodes_per_block = 1;
    *blocks_per_node = (*record_bytes + B - 1) / B;
  }
}

}  // namespace hs::vector
