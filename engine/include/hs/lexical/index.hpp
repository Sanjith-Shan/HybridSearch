#pragma once
// The lexical index: what the shard server links against (docs/ARCHITECTURE.md,
// "Engine library API"). One header pulls in the analyzer, doc store and snippets too.
//
// On-disk layout of an index directory (all little-endian):
//   meta.txt           key=value: counts, stats, codecs, build parameters
//   docids.u64bin      ordinal -> global MS MARCO passage ID (hs/common/fbin.hpp format)
//   doclen.u32         exact analyzed length per ordinal (number of indexed tokens)
//   norms.u8           Lucene SmallFloat.intToByte4(doclen) per ordinal
//   terms.bin          term strings, sorted bytewise, concatenated
//   lexicon.bin        per term, varint: string length, df, cf, per-codec byte length,
//                      term-level score bounds (see BoundInfo)
//   blocks.bin         per block of every term with df > 128, varint: last doc delta,
//                      per-codec byte offset, block-level BoundInfo
//   postings.vbyte     postings, VByte codec     (either or both, see codec.hpp)
//   postings.bp128     postings, BP128 codec
//   docstore.bin/.idx  zstd-compressed passage text (docstore.hpp)

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "hs/common/deadline.hpp"
#include "hs/common/topk.hpp"
#include "hs/lexical/analyzer.hpp"
#include "hs/lexical/bm25.hpp"
#include "hs/lexical/codec.hpp"
#include "hs/lexical/docstore.hpp"
#include "hs/lexical/snippet.hpp"

namespace hs::lexical {

enum class Algorithm : uint8_t {
  Exhaustive = 0,  // term-at-a-time over every posting, accumulator array
  DAAT = 1,        // document-at-a-time, every posting scored, no pruning
  MaxScore = 2,    // Turtle & Flood 1995 (essential / non-essential lists)
  WAND = 3,        // Broder et al. 2003
  BMW = 4,         // block-max WAND, Ding & Suel 2011
};
constexpr int kNumAlgorithms = 5;
const char* algorithm_name(Algorithm a);
bool parse_algorithm(const std::string& s, Algorithm* out);

// Score upper-bound information for a posting range (a block, or a whole term). For each
// model, the posting whose contribution is largest under the index's build parameters
// (k1=0.9, b=0.4): because the score is monotone in tf * inv_norm for any positive query
// weight, that posting's score is the exact maximum for any boost. For other (k1, b) the
// safe bound (max_tf, min_dl) is used instead.
struct BoundInfo {
  uint32_t max_tf = 0, min_dl = 0;
  uint32_t best_tf[2] = {0, 0};  // indexed by Model
  uint32_t best_dl[2] = {0, 0};
};

struct SearchOptions {
  uint32_t k = 10;
  Algorithm algorithm = Algorithm::BMW;
  Model model = Model::Lucene;
  Codec codec = Codec::BP128;  // falls back to whichever codec the index has
  float k1 = kDefaultK1;
  float b = kDefaultB;
  bool explain = false;
};

struct TermContribution {
  std::string term;
  float score = 0;
  uint32_t tf = 0;
  uint32_t df = 0;
};

struct LexicalResult {
  std::vector<ScoredDoc> hits;  // doc = ordinal; sorted by ranks_before
  std::vector<std::vector<TermContribution>> explain;  // per hit, when requested
  bool partial = false;          // deadline cut the search short
  uint64_t docs_scored = 0;      // documents whose full score was computed
  uint64_t postings_decoded = 0; // postings decoded from blocks
  uint64_t blocks_decoded = 0;
};

struct IndexStats {
  uint64_t num_docs = 0;        // ordinals, including documents with no indexed terms
  uint64_t docs_with_terms = 0; // Lucene's docCount for the field: N in idf and avgdl
  uint64_t sum_dl = 0;
  uint64_t num_terms = 0;
  uint64_t num_postings = 0;
  uint64_t num_blocks = 0;      // blocks of multi-block terms (stored in blocks.bin)
};

class LexicalIndex {
 public:
  static std::unique_ptr<LexicalIndex> open(const std::string& dir);
  ~LexicalIndex();

  uint64_t num_docs() const { return stats_.num_docs; }
  uint64_t global_id(uint32_t ordinal) const { return docids_[ordinal]; }
  // Ordinal of a global ID, or -1 if this index does not hold it.
  int64_t ordinal_of(uint64_t global_id) const;
  const IndexStats& stats() const { return stats_; }
  float avgdl() const { return avgdl_; }
  bool has_codec(Codec c) const { return postings_[int(c)].data != nullptr; }
  uint64_t postings_bytes(Codec c) const { return postings_[int(c)].size; }
  const std::string& dir() const { return dir_; }
  uint32_t doc_length(uint32_t ordinal) const { return doclen_[ordinal]; }
  uint8_t norm(uint32_t ordinal) const { return norms_[ordinal]; }

  const Analyzer& analyzer() const { return analyzer_; }

  // Terms are analyzed terms, duplicates allowed (a repeated term is one clause with
  // boost = count, as Anserini's BagOfWordsQueryGenerator builds it).
  LexicalResult search(const std::vector<std::string>& terms, const SearchOptions& opt, Deadline& deadline) const;
  LexicalResult search_text(std::string_view query, const SearchOptions& opt, Deadline& deadline) const;

  // Term lookup; returns -1 if absent.
  int64_t term_id(std::string_view term) const;
  std::string_view term_string(uint32_t id) const;
  uint32_t df(uint32_t id) const;

  // Decode every posting of a term (tests, benchmarks).
  void postings(uint32_t id, Codec c, std::vector<uint32_t>& docs, std::vector<uint32_t>& tfs) const;

  // Internals shared with the query processors.
  struct TermInfo {
    uint64_t str_off = 0;
    uint32_t str_len = 0;
    uint32_t df = 0;
    uint64_t cf = 0;
    uint64_t post_off[kNumCodecs] = {0, 0};
    uint32_t first_block = 0;  // into blocks_ (only if df > kBlockSize)
    BoundInfo bound;
  };
  struct BlockInfo {
    uint32_t last_doc = 0;
    uint32_t off[kNumCodecs] = {0, 0};  // byte offset from the term's first posting
    BoundInfo bound;
  };
  const TermInfo& term(uint32_t id) const { return terms_[id]; }
  const BlockInfo* blocks(uint32_t id) const { return blocks_.data() + terms_[id].first_block; }
  const uint8_t* postings_base(Codec c) const { return postings_[int(c)].data; }
  const uint32_t* doclens() const { return doclen_.data(); }
  const uint8_t* norms() const { return norms_.data(); }
  float build_k1() const { return build_k1_; }
  float build_b() const { return build_b_; }

 private:
  LexicalIndex() = default;
  struct Mapped {
    const uint8_t* data = nullptr;
    uint64_t size = 0;
    void* base = nullptr;
    size_t map_len = 0;
  };

  std::string dir_;
  IndexStats stats_;
  float avgdl_ = 1;
  float build_k1_ = kDefaultK1, build_b_ = kDefaultB;
  std::vector<uint64_t> docids_;
  bool docids_sorted_ = true;
  std::vector<uint32_t> doclen_;
  std::vector<uint8_t> norms_;
  std::string term_strings_;
  std::vector<TermInfo> terms_;
  std::vector<BlockInfo> blocks_;
  Mapped postings_[kNumCodecs];
  Analyzer analyzer_;
};

}  // namespace hs::lexical
