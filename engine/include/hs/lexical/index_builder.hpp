#pragma once
// Index construction with bounded memory: in-memory inversion into (term, doc, tf) runs,
// each run counting-sorted by term ID and spilled to a temporary file, then one k-way
// merge in lexicographic term order that computes block bounds and writes every codec.
//
// Peak memory is roughly mem_budget_mb (run buffer) + vocabulary (~40 B/term) + 9 B/doc.

#include <cstdint>
#include <string>
#include <vector>

#include "hs/lexical/codec.hpp"

namespace hs::lexical {

struct BuildOptions {
  std::string input_tsv;   // "pid \t text" lines
  std::string out_dir;
  std::string tmp_dir;     // default: <out_dir>/tmp (deleted afterwards)
  std::vector<Codec> codecs{Codec::VByte, Codec::BP128};
  int threads = 4;         // analysis workers + the inverting thread
  uint64_t mem_budget_mb = 1536;
  int zstd_level = 9;
  bool docstore = true;
  uint64_t skip_docs = 0;  // skip this many leading lines (shard slices)
  uint64_t max_docs = 0;   // 0 = all lines (after skip_docs)
  // Sharding with global BM25 statistics: when set, idf's N, avgdl and every term's df come
  // from this (full-collection) index instead of the documents being built, so a shard scores
  // exactly like the single index. Stored in meta.txt (global_*) and global_df.u32.
  std::string global_stats_dir;
  std::string shard_label;  // e.g. "2/4", recorded in meta.txt
  double min_free_gb = 3.0;  // refuse to start (or continue) below this much free disk
  bool verbose = true;
};

struct BuildReport {
  uint64_t num_docs = 0, docs_with_terms = 0, sum_dl = 0, num_terms = 0, num_postings = 0, num_blocks = 0;
  uint32_t num_runs = 0;
  double seconds_invert = 0, seconds_merge = 0, seconds_total = 0;
  uint64_t bytes_postings[kNumCodecs] = {0, 0};
  uint64_t bytes_lexicon = 0, bytes_blocks = 0, bytes_docstore = 0, bytes_docmeta = 0, bytes_terms = 0;
  uint64_t peak_tmp_bytes = 0;
};

BuildReport build_index(const BuildOptions& opt);

// Number of lines in a file (for computing shard slices).
uint64_t count_lines(const std::string& path);

// Free bytes on the filesystem holding `path`.
uint64_t free_disk_bytes(const std::string& path);

}  // namespace hs::lexical
