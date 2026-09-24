# M1 correctness gate: BM25 parity with Anserini

**Result: gate passed.** HybridSearch's Lucene-compatible BM25 on the full 8,841,823-passage
collection gets **MS MARCO dev MRR@10 = 0.1840** (0.183983 from the run file), against Anserini's
**0.1840** (0.183988). The gap is 0.000005; the target was ±0.002. For DL19 and DL20 we get nDCG@10
**0.5058 / 0.4796**, against Anserini's 0.5058 / 0.4796. All numbers come from our harness
(`python -m hybridsearch.eval`) scoring both run files the same way.

The reference is Anserini 2.3.0 (Lucene 10.5.0) via Pyserini, BM25 k1=0.9 b=0.4, prebuilt index
`lucene-inverted.msmarco-v1-passage.20221004.252b5e`, run file `data/runs/reference/anserini-bm25-default.*.trec`.
Our runs: `hs_lex_search --algo exhaustive --model {lucene,textbook} --k 1000`, on
`data/indexes/lexical/full`, stored in `data/runs/lexical/`.

## Headline table

| run | dev MRR@10 | dev R@1000 | DL19 nDCG@10 | DL19 R(rel≥2)@1000 | DL20 nDCG@10 | DL20 R(rel≥2)@1000 |
|---|---|---|---|---|---|---|
| Anserini reference run | 0.183988 | 0.8526 | 0.505831 | 0.750145 | 0.479637 | 0.786322 |
| **ours, Lucene BM25** (SmallFloat norms, no (k1+1)) | **0.183983** | **0.8526** | **0.505764** | 0.750012 | **0.479637** | 0.786322 |
| ours, textbook BM25 (exact lengths, with (k1+1)) | 0.184510 | 0.8526 | 0.505593 | 0.751170 | 0.478716 | 0.789099 |
| Δ Lucene model − Anserini | −0.000005 | 0.0000 | −0.000067 | −0.000133 | 0 | 0 |
| Δ textbook − Anserini | +0.000522 | 0.0000 | −0.000238 | +0.001025 | −0.000921 | +0.002777 |

Evaluated in true rank order (the `--raw-out` runs), the numbers are: Lucene model dev MRR@10 0.183981,
textbook 0.184593. The run files differ slightly because Anserini's tie-score formatting, which we copy,
is not order-preserving. See the 2026-09-23 entry in `docs/BUG_LOG.md`.

**Which BM25 matches Anserini?** The Lucene model, exactly. Textbook BM25 changes the top-10 of **3,253
of 6,980 dev queries (46.6%)**. Its MRR@10 moves by +0.0005, inside the ±0.002 band but not identical.
Every one of those 3,253 disagreements goes away under the Lucene model, so the only cause is Lucene's
one-byte length quantisation. The (k1+1) factor is a per-query constant and cannot change a ranking.

## Disagreements, bucketed per query (top-10 vs Anserini's top-10)

| run | queries | identical | tie order¹ | tie at the cut-off² | length quantisation³ | analyzer (vocab/df)⁴ | analyzer (doc length)⁴ | unexplained | mean top-10 overlap |
|---|---|---|---|---|---|---|---|---|---|
| dev, Lucene model | 6,980 | 6,799 | 146 | 35 | 0 | 0 | 0 | **0** | 9.995 / 10 |
| dev, textbook | 6,980 | 3,727 | 0 | 0 | 3,253 | 0 | 0 | 0 | 9.869 |
| DL19, Lucene model | 43 | 41 | 2 | 0 | 0 | 0 | 0 | 0 | 10.000 |
| DL19, textbook | 43 | 24 | 0 | 0 | 19 | 0 | 0 | 0 | 9.860 |
| DL20, Lucene model | 54 | 54 | 0 | 0 | 0 | 0 | 0 | 0 | 10.000 |
| DL20, textbook | 54 | 31 | 0 | 0 | 23 | 0 | 0 | 0 | 9.907 |

1. Same ten documents, different order, and every reordered document has an equal score (to Anserini's
   printed 1e-4). Lucene/Anserini break exact ties by docid **string** (`BREAK_SCORE_TIES_BY_DOCID`);
   our contract breaks them by numeric ID. This is also why DL19 nDCG@10 differs by 0.000067: graded
   gains make nDCG sensitive to the order of tied documents.
2. The ten documents differ only by documents tied with the 10th score. Again a tie-break choice, not a
   score difference. This caps top-10 overlap at 9 for 35 dev queries.
3. Textbook run only: the Lucene-model run agrees with Anserini on these queries.
4. Tested, not assumed. Every query term's df is compared between the two indexes, and so is every
   top-10 document's norm byte. Neither ever differs (next section), so both buckets are empty.

Tool: `py/hybridsearch/lexical/parity.py`. Per-query examples are in `parity_{dev,dl19,dl20}.json`.

## Why the Lucene model agrees exactly: index-level and score-level evidence

| check | result |
|---|---|
| Analyzer vs Lucene 10.5.0 (Anserini's DefaultEnglishAnalyzer chain) on 10,000 MS MARCO passages (every 884th line; 2,371 contain non-ASCII) | **0 mismatched passages** out of 396,841 tokens |
| Analyzer sweep: every code point U+0000–U+10FFFF (except surrogates, TAB, LF, CR) in 13 contexts | 14,456,793 lines, **0 mismatches** |
| Term dictionary vs Anserini's prebuilt index (read with Lucene 10.5.0) | 2,660,824 terms on both sides; the `term, df, cf` dumps are **byte-identical** |
| Collection statistics | docCount 8,841,823; sumTotalTermFreq 352,316,036; sumDocFreq (postings) 266,247,718, **all equal** |
| Norm bytes per passage (our SmallFloat.intToByte4 vs Lucene's stored norms) | 8,841,823 compared, **0 differ** |
| Our Lucene-model scores vs Lucene's own BM25 scores (ctest `Bm25.LuceneModelMatchesLuceneBitForBit`, test corpus, 300 queries, k=20) | **bit-identical floats** at every rank |
| Per-(query, doc) scores vs Anserini's dev run (`score_agreement_dev.json`) | 6,969,251 pairs in both runs; 99.911% equal after rounding to 1e-4. Of the 6,197 that differ, 6,153 are pairs where Anserini's printed score carries its −1e-6-per-tie offset. The other 44 differ by at most 4e-4. They are consistent with that offset reaching a multiple of 1e-4 (runs of 100 or more tied documents), but that was not verified pair by pair |
| Pairs present in only one top-1000 | 5,347 per side, the tie group cut at rank 1000 |

Files: `index_parity.json` (+ `.meta.json`), `score_agreement_dev.json`, `eval_{lucene,textbook}.{dev,dl19,dl20}.json`,
`parity_{dev,dl19,dl20}.json`.

## How exact score equality is achieved (what the interview answer is)

- **The analyzer** reproduces Lucene's StandardTokenizer grammar (UAX#29 rules from `StandardTokenizerImpl.jflex`,
  compiled to a DFA from UCD 12.1 tables, with JFlex longest-match semantics and the 255-UTF-16-unit token buffer
  that splits long runs) → English possessive → Java `Character.toLowerCase` → 33 stop words → Lucene's Porter
  stemmer, run over UTF-16 code units the way Java does.
- **Scoring** follows `BM25Similarity`: float `idf = (float) ln(1 + (N − df + 0.5)/(df + 0.5))`,
  `avgdl = (float)(sumTTF/N)`, a 256-entry float cache `1/(k1((1−b) + b·len/avgdl))` over
  `SmallFloat.byte4ToInt`, and `w − w/(1 + tf·inv)`. Per-term floats are summed in a double and cast to
  float, as BooleanScorer and MaxScoreBulkScorer do. FMA contraction is off (`#pragma clang fp contract(off)`),
  because clang contracts `1 + tf·inv` into an FMA by default on ARM, which rounds once instead of twice (a precaution: I did not measure how often it changes a score).
- **Query construction** follows Anserini's `BagOfWordsQueryGenerator`: one clause per distinct term, boosted
  by its count.
