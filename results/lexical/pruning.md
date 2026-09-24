# M2: codecs × query-processing algorithms

**Hardware label: dev-signal-only.** Apple M3 Pro (6P+6E), 18 GB, macOS 26 (Darwin 25.5.0), Apple
clang 17, `-O3 -mcpu=native`. Threads are not pinned (macOS can't pin them). The index is warm in the page
cache. Every timing below was taken while other sessions loaded the machine (load average 20–60, see
each `.meta.json`), so the latency columns are noisy. Back-to-back runs of one command differed by up to
3.4× at p50. The counter columns (docs scored, postings decoded) and every size column are deterministic.
Quiet-machine re-runs are queued in `TIMING_QUEUE.md`. Resume-grade numbers come only from a pinned
Linux box.

**Correctness first.** Every algorithm × codec returns the *identical* top-k (ordinals and float score
bits, in `ranks_before` order) as exhaustive TAAT:
- ctest `DifferentialGate.EveryAlgorithmAndCodecEqualsExhaustive`: test corpus, 300 queries × 2 models ×
  4 (k1, b) settings × 6 values of k × 5 algorithms × 2 codecs, more than 100,000 searches.
- ctest `RandomCorpora.AllAlgorithmsEqualBruteForce`: 24 random corpora, checked against a brute-force
  scorer that never touches the index.
- ctest `RealData.DevQueriesEveryAlgorithmAndCodecEqualsExhaustive`: 1M index, all 6,980 dev queries at
  k ∈ {10, 1000}, 0 mismatches (ctest defaults to the first 500 queries).
- On the **full 8.8M index**, the k=1000 runs of all 6,980 dev queries for every pruning algorithm ×
  codec (MaxScore, WAND, BMW × VByte, BP128) are byte-identical to the exhaustive run, including float
  bits (`full_gate.txt`).

A mutation test (bounds scaled by 0.97, which makes them unsafe) fails both gates immediately.

## Index size (full collection, 8,841,823 passages, 266,247,718 postings)

| component | bytes | per posting / note |
|---|---|---|
| postings, VByte | 628,986,128 | **18.90 bits/posting** (docid gap + tf) |
| postings, BP128 (SIMD-BP128 vertical layout, VByte tail blocks) | 451,022,817 | **13.55 bits/posting** |
| lexicon + block-max metadata + term strings | 79,975,480 | 2,660,824 terms, 2,026,755 blocks of 128 |
| docids + exact doc lengths + SmallFloat norms | 114,943,707 | 13 B/doc |
| doc store (zstd-9, 16 KB blocks) | 1,146,366,170 | 3.06 GB of text → 2.67× |
| **total, one codec (BP128) + doc store** | ≈1.79 GB | (both codecs: 2.3 GB on disk) |

Build: 869 s wall, peak RSS 477 MB (1.5 GB run budget, 4 spilled runs, 630 MB temp, deleted),
3 threads (`results/lexical/build_full.json`). 1M subset: VByte 19.40, BP128 13.69 bits/posting.

## Decode speed (1M index, every posting, `codec_1m.json`, load ≈49–59)

| codec | bits/posting | ns/posting | M postings/s |
|---|---|---|---|
| VByte | 19.40 | 12.5 | 80 |
| BP128, NEON kernel | 13.69 | 3.6 | 276 |
| BP128, scalar fallback (same layout) | 13.69 | 14.6 | 69 |

Full index (`codec_full.json`, 2026-09-24, **load average 117–125**, not usable): VByte 18.90 bits/posting,
18.7 ns/posting; BP128 13.55 bits/posting, NEON 9.3 ns, scalar 7.1 ns. The NEON-slower-than-scalar
inversion contradicts the 1M run and is a load artefact (the 1 GB of postings also competed for page
cache under swap pressure). The bits/posting figures are exact; the ns figures await a quiet re-run.

## Full index: 1,000 dev queries, k = 10 (`bench_full_k10.json`, load ≈20–33 at end)

| algorithm | codec | docs scored / q | postings decoded / q | blocks decoded / q | p50 ms | p99 ms | mean ms |
|---|---|---|---|---|---|---|---|
| exhaustive TAAT | vbyte | 923,730 | 1,010,670 | 7,898 | 96.8 | 859.2 | 154.0 |
| exhaustive TAAT | bp128 | 923,730 | 1,010,670 | 7,898 | 61.0 | 465.8 | 86.1 |
| DAAT | vbyte | 923,730 | 1,010,670 | 7,898 | 52.4 | 554.8 | 89.9 |
| DAAT | bp128 | 923,730 | 1,010,670 | 7,898 | 39.6 | 509.7 | 69.6 |
| MaxScore | vbyte | 17,213 | 359,094 | 2,807 | 20.2 | 234.5 | 36.6 |
| MaxScore | bp128 | 17,213 | 359,094 | 2,807 | 12.2 | 195.8 | 24.3 |
| WAND | vbyte | 17,667 | 676,347 | 5,286 | 19.1 | 294.3 | 40.4 |
| WAND | bp128 | 17,667 | 676,347 | 5,286 | 35.6* | 736.6* | 88.3* |
| BMW | vbyte | 1,633 | 533,358 | 4,168 | 19.2 | 242.1 | 35.0 |
| BMW | bp128 | 1,633 | 533,358 | 4,168 | 17.1 | 164.2 | 29.7 |

\* This row cannot be right as a codec effect: it decodes the same postings as WAND/vbyte, and BP128
decodes 3.4× faster. It coincides with a burst of load from another session. Re-measure before quoting it.

## 1M subset: all 6,980 dev queries, k = 10 (`bench_1m_k10.json`, load ≈39–48)

| algorithm | codec | docs scored / q | postings decoded / q | p50 ms | p99 ms |
|---|---|---|---|---|---|
| exhaustive | vbyte / bp128 | 107,776 | 119,234 | 9.6 / 5.8 | 263.6 / 58.8 |
| DAAT | vbyte / bp128 | 107,776 | 119,234 | 3.0 / 3.1 | 121.0 / 46.9 |
| MaxScore | vbyte / bp128 | 3,406 | 70,141 | 1.0 / 0.9 | 14.1 / 22.1 |
| WAND | vbyte / bp128 | 3,119 | 101,518 | 1.7 / 1.7 | 115.8 / 27.9 |
| BMW | vbyte / bp128 | 941 | 93,057 | 1.3 / 1.5 | 27.3 / 22.0 |

## Findings (from the deterministic counters)

- **BMW scores 566× fewer documents than exhaustive** on the full index (1,633 vs 923,730 per query at k=10).
  It scores 10.5× fewer than WAND, because block maxima reject most pivots that the term-level bounds
  admit. It still decodes about half the postings: every candidate block is decoded to find the next doc.
  BMW's win is in scoring, not in decoding.
- **MaxScore decodes the fewest postings** (359K/q, 36% of exhaustive), because non-essential lists are only
  probed with `next_geq`. On this collection MaxScore and BMW are close in latency, which matches what the
  PISA and Lucene results report for MS MARCO-style short queries.
- **BP128 vs VByte**: 28% smaller postings and 3.4× faster decode. The gain shows most where decoding
  dominates (exhaustive, DAAT).
- **Against Lucene 10.5.0** (the same 1,000 queries, `IndexSearcher.search(q, 10)` single-threaded, Anserini's
  prebuilt index): Lucene p50 **22.9 ms**, p99 273.7 ms (`vs_lucene_lucene.json`). It was run between our two
  runs of MaxScore/BMW on bp128, which measured p50 4.5/13.1 ms and 15.5/14.3 ms. That spread makes any
  ratio meaningless on this machine. The honest statement: at k=10 we are in the same range as Lucene, not
  measurably faster or slower. A pinned re-run is queued. Lucene's own top-k path uses block-max
  MaxScore/WAND-style pruning, so this is pruning compared against pruning.
