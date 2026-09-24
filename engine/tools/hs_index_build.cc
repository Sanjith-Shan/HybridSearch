// hs_index_build: collection.tsv -> lexical index directory.
//
//   hs_index_build --input data/raw/msmarco/collection.tsv --out data/indexes/lexical/full
//                  [--codecs vbyte,bp128] [--threads 4] [--mem-mb 1536] [--zstd-level 9]
//                  [--no-docstore] [--max-docs N] [--min-free-gb 3] [--tmp DIR] [--report FILE.json]
//                  [--shard i/N --global-stats FULL_INDEX_DIR]
//
// --shard i/N builds the i-th of N contiguous line slices (the input is sorted by passage ID,
// so each shard is a contiguous ID range). With --global-stats, BM25's N, avgdl and every
// term's df come from FULL_INDEX_DIR, so a shard's scores equal the single index's bit for bit.

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#include "hs/lexical/index_builder.hpp"

using namespace hs::lexical;

static void usage() {
  std::fprintf(stderr,
               "usage: hs_index_build --input TSV --out DIR [--codecs vbyte,bp128] [--threads N] [--mem-mb MB]\n"
               "                      [--zstd-level L] [--no-docstore] [--max-docs N] [--min-free-gb G] [--tmp DIR]\n"
               "                      [--report FILE.json] [--shard i/N --global-stats FULL_INDEX_DIR]\n");
}

int main(int argc, char** argv) {
  BuildOptions opt;
  std::string report, shard;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--input") opt.input_tsv = val();
    else if (a == "--out") opt.out_dir = val();
    else if (a == "--tmp") opt.tmp_dir = val();
    else if (a == "--threads") opt.threads = std::stoi(val());
    else if (a == "--mem-mb") opt.mem_budget_mb = std::stoull(val());
    else if (a == "--zstd-level") opt.zstd_level = std::stoi(val());
    else if (a == "--no-docstore") opt.docstore = false;
    else if (a == "--max-docs") opt.max_docs = std::stoull(val());
    else if (a == "--min-free-gb") opt.min_free_gb = std::stod(val());
    else if (a == "--report") report = val();
    else if (a == "--quiet") opt.verbose = false;
    else if (a == "--shard") shard = val();
    else if (a == "--global-stats") opt.global_stats_dir = val();
    else if (a == "--codecs") {
      opt.codecs.clear();
      std::istringstream ss(val());
      std::string c;
      while (std::getline(ss, c, ',')) {
        Codec cc;
        if (!parse_codec(c, &cc)) {
          std::fprintf(stderr, "unknown codec %s\n", c.c_str());
          return 2;
        }
        opt.codecs.push_back(cc);
      }
    } else {
      usage();
      return 2;
    }
  }
  if (opt.input_tsv.empty() || opt.out_dir.empty()) {
    usage();
    return 2;
  }
  try {
    if (!shard.empty()) {
      size_t slash = shard.find('/');
      if (slash == std::string::npos) throw std::invalid_argument("--shard expects i/N");
      uint64_t i = std::stoull(shard.substr(0, slash)), n = std::stoull(shard.substr(slash + 1));
      if (n == 0 || i >= n) throw std::invalid_argument("--shard: need 0 <= i < N");
      uint64_t lines = count_lines(opt.input_tsv);
      opt.skip_docs = lines * i / n;
      opt.max_docs = lines * (i + 1) / n - opt.skip_docs;
      opt.shard_label = shard;
      if (opt.global_stats_dir.empty())
        std::fprintf(stderr, "[build] warning: --shard without --global-stats scores with shard-local statistics\n");
      std::fprintf(stderr, "[build] shard %s: lines [%llu, %llu)\n", shard.c_str(), (unsigned long long)opt.skip_docs,
                   (unsigned long long)(opt.skip_docs + opt.max_docs));
    }
    BuildReport r = build_index(opt);
    std::fprintf(stderr,
                 "[build] done: %llu docs (%llu with terms), %llu terms, %llu postings, %llu blocks, %u runs, "
                 "invert %.1fs merge %.1fs total %.1fs\n",
                 (unsigned long long)r.num_docs, (unsigned long long)r.docs_with_terms, (unsigned long long)r.num_terms,
                 (unsigned long long)r.num_postings, (unsigned long long)r.num_blocks, r.num_runs, r.seconds_invert,
                 r.seconds_merge, r.seconds_total);
    if (!report.empty()) {
      std::FILE* f = std::fopen(report.c_str(), "w");
      if (!f) throw std::runtime_error("cannot write " + report);
      std::fprintf(f,
                   "{\n  \"input\": \"%s\",\n  \"out\": \"%s\",\n  \"threads\": %d,\n  \"mem_budget_mb\": %llu,\n"
                   "  \"num_docs\": %llu,\n  \"docs_with_terms\": %llu,\n  \"sum_dl\": %llu,\n  \"num_terms\": %llu,\n"
                   "  \"num_postings\": %llu,\n  \"num_blocks\": %llu,\n  \"num_runs\": %u,\n"
                   "  \"bytes_postings_vbyte\": %llu,\n  \"bytes_postings_bp128\": %llu,\n  \"bytes_lexicon\": %llu,\n"
                   "  \"bytes_blocks\": %llu,\n  \"bytes_terms\": %llu,\n  \"bytes_docmeta\": %llu,\n"
                   "  \"bytes_docstore\": %llu,\n  \"bytes_tmp_runs\": %llu,\n"
                   "  \"seconds_invert\": %.2f,\n  \"seconds_merge\": %.2f,\n  \"seconds_total\": %.2f\n}\n",
                   opt.input_tsv.c_str(), opt.out_dir.c_str(), opt.threads, (unsigned long long)opt.mem_budget_mb,
                   (unsigned long long)r.num_docs, (unsigned long long)r.docs_with_terms, (unsigned long long)r.sum_dl,
                   (unsigned long long)r.num_terms, (unsigned long long)r.num_postings,
                   (unsigned long long)r.num_blocks, r.num_runs, (unsigned long long)r.bytes_postings[0],
                   (unsigned long long)r.bytes_postings[1], (unsigned long long)r.bytes_lexicon,
                   (unsigned long long)r.bytes_blocks, (unsigned long long)r.bytes_terms,
                   (unsigned long long)r.bytes_docmeta, (unsigned long long)r.bytes_docstore,
                   (unsigned long long)r.peak_tmp_bytes, r.seconds_invert, r.seconds_merge, r.seconds_total);
      std::fclose(f);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hs_index_build: %s\n", e.what());
    return 1;
  }
  return 0;
}
