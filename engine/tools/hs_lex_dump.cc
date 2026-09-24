// hs_lex_dump: print index statistics and the term dictionary.
//   hs_lex_dump --index DIR [--terms out.tsv]    ("term\tdf\tcf", bytewise term order)
// The format matches tools/lucene_ref indexstats, so vocabularies can be diffed directly.

#include <cstdio>
#include <string>

#include "hs/lexical/index.hpp"

using namespace hs::lexical;

int main(int argc, char** argv) {
  std::string dir, terms;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--index" && i + 1 < argc) dir = argv[++i];
    else if (a == "--terms" && i + 1 < argc) terms = argv[++i];
    else {
      std::fprintf(stderr, "usage: hs_lex_dump --index DIR [--terms out.tsv]\n");
      return 2;
    }
  }
  auto ix = LexicalIndex::open(dir);
  const auto& s = ix->stats();
  std::printf("num_docs\t%llu\ndocCount\t%llu\nsumTotalTermFreq\t%llu\nsumDocFreq\t%llu\nnumTerms\t%llu\n"
              "num_blocks\t%llu\navgdl\t%.9g\n",
              (unsigned long long)s.num_docs, (unsigned long long)s.docs_with_terms, (unsigned long long)s.sum_dl,
              (unsigned long long)s.num_postings, (unsigned long long)s.num_terms, (unsigned long long)s.num_blocks,
              double(ix->avgdl()));
  for (int c = 0; c < kNumCodecs; ++c)
    if (ix->has_codec(Codec(c)))
      std::printf("postings_bytes_%s\t%llu\n", codec_name(Codec(c)), (unsigned long long)ix->postings_bytes(Codec(c)));
  if (!terms.empty()) {
    std::FILE* f = std::fopen(terms.c_str(), "w");
    if (!f) return 1;
    for (uint32_t t = 0; t < s.num_terms; ++t) {
      auto str = ix->term_string(t);
      std::fprintf(f, "%.*s\t%u\t%llu\n", int(str.size()), str.data(), ix->term(t).df,
                   (unsigned long long)ix->term(t).cf);
    }
    std::fclose(f);
  }
  return 0;
}
