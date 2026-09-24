// Is a "cold" read really cold? Random 4 KB pread latency on a file, measured
//   (a) through the page cache, (b) with F_NOCACHE / O_DIRECT, before and after
//   warming the file, and after an eviction attempt (mmap + msync(MS_INVALIDATE)).
// If (b) after warming is as fast as (a), F_NOCACHE is being served from cache and
// "cold" numbers taken after a warm pass would be lies.
//   bench_vec_coldread --file big.bin [--reads 2000] [--warm-mb 512]
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "hs/vector/containers.hpp"
#include "hs/vector/tool_util.hpp"

using namespace hs::vector;

static double probe(const char* path, bool nocache, const std::vector<off_t>& offs) {
  int flags = O_RDONLY;
#if defined(O_DIRECT)
  if (nocache) flags |= O_DIRECT;
#endif
  int fd = ::open(path, flags);
#if defined(__APPLE__)
  if (nocache) ::fcntl(fd, F_NOCACHE, 1);
#endif
  void* buf = nullptr;
  if (posix_memalign(&buf, 4096, 4096) != 0) return -1;
  std::vector<double> us;
  for (off_t o : offs) {
    auto t = std::chrono::steady_clock::now();
    if (::pread(fd, buf, 4096, o) != 4096) { std::perror("pread"); break; }
    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count());
  }
  ::close(fd);
  std::free(buf);
  std::sort(us.begin(), us.end());
  return us.empty() ? -1 : us[us.size() / 2];
}

int main(int argc, char** argv) {
  tool::Args a(argc, argv);
  std::string path = a.str("file");
  size_t reads = size_t(a.num("reads", 2000));
  size_t warm = size_t(a.num("warm-mb", 512)) << 20;
  struct stat st {};
  ::stat(path.c_str(), &st);
  size_t span = std::min<size_t>(size_t(st.st_size), warm) / 4096;
  std::vector<off_t> offs(reads);
  uint64_t s = 11;
  for (auto& o : offs) o = off_t(((s = splitmix64(s)) % span) * 4096);
  std::printf("{\"file_bytes\": %lld, \"span_bytes\": %zu", (long long)st.st_size, span * 4096);
  std::printf(", \"nocache_first_p50_us\": %.1f", probe(path.c_str(), true, offs));
  std::printf(", \"cached_after_warming_p50_us\": %.1f", (probe(path.c_str(), false, offs), probe(path.c_str(), false, offs)));
  std::printf(", \"nocache_after_warming_p50_us\": %.1f", probe(path.c_str(), true, offs));
  int fd = ::open(path.c_str(), O_RDONLY);
  void* m = ::mmap(nullptr, span * 4096, PROT_READ, MAP_SHARED, fd, 0);
  int rc = ::msync(m, span * 4096, MS_INVALIDATE);
  ::munmap(m, span * 4096);
  ::close(fd);
  std::printf(", \"msync_invalidate_rc\": %d", rc);
  std::printf(", \"nocache_after_invalidate_p50_us\": %.1f}\n", probe(path.c_str(), true, offs));
  return 0;
}
