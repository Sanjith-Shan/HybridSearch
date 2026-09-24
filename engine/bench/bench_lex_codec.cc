// bench_lex_codec: bits/posting and decode speed of each posting codec on a real index.
//
//   bench_lex_codec --index DIR --out results/lexical/codec.json [--reps 5] [--min-df 0]
//
// Decodes every block of every term with df >= min-df (docs + tfs, gaps prefix-summed to
// doc IDs), sequentially from the page cache (one untimed pass first), and reports the median
// of `reps` passes. For BP128 the NEON kernel and the scalar fallback are timed separately.
// The tail blocks of BP128 (< 128 postings) are VByte, as in Lucene; they are included.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../tools/lex_bench_util.hpp"
#include "hs/lexical/index.hpp"

using namespace hs::lexical;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  std::string index_dir, out;
  int reps = 5;
  uint32_t min_df = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--index" && i + 1 < argc) index_dir = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
    else if (a == "--min-df" && i + 1 < argc) min_df = uint32_t(std::stoul(argv[++i]));
    else {
      std::fprintf(stderr, "usage: bench_lex_codec --index DIR --out FILE.json [--reps N] [--min-df N]\n");
      return 2;
    }
  }
  if (index_dir.empty() || out.empty()) return 2;
  auto ix = LexicalIndex::open(index_dir);
  const uint32_t nt = uint32_t(ix->stats().num_terms);

  struct Variant {
    Codec codec;
    bool scalar;
    const char* name;
  };
  std::vector<Variant> variants;
  if (ix->has_codec(Codec::VByte)) variants.push_back({Codec::VByte, false, "vbyte"});
  if (ix->has_codec(Codec::BP128)) {
    variants.push_back({Codec::BP128, false, std::string(bp128_kernel()) == "neon" ? "bp128-neon" : "bp128-scalar"});
    if (std::string(bp128_kernel()) == "neon") variants.push_back({Codec::BP128, true, "bp128-scalar"});
  }

  std::ostringstream js;
  js << "{\n  \"index\": \"" << bench::json_escape(index_dir) << "\",\n  \"num_postings\": " << ix->stats().num_postings
     << ",\n  \"num_terms\": " << nt << ",\n  \"min_df\": " << min_df << ",\n  \"reps\": " << reps
     << ",\n  \"results\": [\n";
  alignas(16) uint32_t docs[kBlockSize], tfs[kBlockSize];
  bool first = true;
  for (const Variant& v : variants) {
    uint64_t postings = 0, full_postings = 0, bytes = 0;
    std::vector<double> secs;
    uint64_t checksum = 0;
    for (int r = -1; r < reps; ++r) {  // r = -1: untimed warm-up pass
      postings = 0;
      full_postings = 0;
      bytes = 0;
      auto t0 = Clock::now();
      for (uint32_t t = 0; t < nt; ++t) {
        const auto& ti = ix->term(t);
        if (ti.df < min_df) continue;
        const uint8_t* p = ix->postings_base(v.codec) + ti.post_off[int(v.codec)];
        const uint8_t* start = p;
        uint32_t prev = UINT32_MAX;
        const uint32_t nb = (ti.df + kBlockSize - 1) / kBlockSize;
        for (uint32_t j = 0; j < nb; ++j) {
          uint32_t n = std::min<uint32_t>(kBlockSize, ti.df - j * kBlockSize);
          p = v.scalar ? decode_block_scalar(v.codec, p, n, prev, docs, tfs) : decode_block(v.codec, p, n, prev, docs, tfs);
          prev = docs[n - 1];
          checksum += tfs[n - 1];
          if (n == kBlockSize) full_postings += n;
        }
        postings += ti.df;
        bytes += uint64_t(p - start);
      }
      double s = std::chrono::duration<double>(Clock::now() - t0).count();
      if (r >= 0) secs.push_back(s);
    }
    std::sort(secs.begin(), secs.end());
    double med = secs[secs.size() / 2];
    double ns_per = med * 1e9 / double(postings);
    double bits = 8.0 * double(bytes) / double(postings);
    std::fprintf(stderr, "[codec] %-13s postings=%llu bytes=%llu bits/posting=%.2f median=%.3fs ns/posting=%.3f "
                         "Mpostings/s=%.0f (checksum %llu)\n",
                 v.name, (unsigned long long)postings, (unsigned long long)bytes, bits, med, ns_per,
                 double(postings) / med / 1e6, (unsigned long long)checksum);
    js << (first ? "" : ",\n") << "    {\"codec\": \"" << v.name << "\", \"postings\": " << postings
       << ", \"postings_in_full_blocks\": " << full_postings << ", \"bytes\": " << bytes
       << ", \"bits_per_posting\": " << bits << ", \"median_seconds\": " << med << ", \"min_seconds\": " << secs.front()
       << ", \"max_seconds\": " << secs.back() << ", \"ns_per_posting\": " << ns_per
       << ", \"million_postings_per_second\": " << double(postings) / med / 1e6 << "}";
    first = false;
  }
  js << "\n  ]\n}\n";
  std::ofstream(out) << js.str();
  bench::write_meta(out, argc, argv, "bench/bench_lex_codec.cc", "warm (untimed decode pass first)");
  return 0;
}
