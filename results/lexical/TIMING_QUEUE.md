# Lexical timing runs waiting for a quiet machine

Every latency, throughput or decode-speed number in `results/lexical/` so far was measured while
other sessions were loading the laptop (load averages 20–60, recorded in each `.meta.json`). Treat
those numbers as a rough development signal: back-to-back runs of the same command differed by up to
3.4× at p50 (`vs_lucene_ours_a.json` vs `vs_lucene_ours_b.json`). Re-run the commands below on a quiet
machine. They are single-threaded and each writes a `.meta.json` with the load average. On macOS the
results are still `dev-signal-only`; resume numbers come from a pinned Linux box.

```bash
# 1. Codec decode speed (a 2026-09-24 attempt ran at load ~120 and is not usable) + bits/posting, full index (≈3 min)
engine/build-lex/bench_lex_codec --index data/indexes/lexical/full --out results/lexical/codec_full.json --reps 5

# 2. Per-algorithm x codec latency, full index, 1,000 dev queries, k=10 (≈15 min)
engine/build-lex/hs_lex_bench --index data/indexes/lexical/full --queries data/raw/msmarco/queries.dev.small.tsv \
  --limit 1000 --k 10 --warmup 2 --out results/lexical/bench_full_k10.json --label full-dev1000-k10

# 3. Same at k=1000 (the candidate depth fusion and reranking use)
engine/build-lex/hs_lex_bench --index data/indexes/lexical/full --queries data/raw/msmarco/queries.dev.small.tsv \
  --limit 1000 --k 1000 --warmup 1 --out results/lexical/bench_full_k1000.json --label full-dev1000-k1000

# 4. 1M subset, all 6,980 dev queries, k=10
engine/build-lex/hs_lex_bench --index data/indexes/lexical/1m --queries data/raw/msmarco/queries.dev.small.tsv \
  --k 10 --warmup 2 --out results/lexical/bench_1m_k10.json --label 1m-dev-k10
engine/build-lex/bench_lex_codec --index data/indexes/lexical/1m --out results/lexical/codec_1m.json --reps 5

# 5. Lucene 10.5.0 on the same queries, back to back with ours (needs the Anserini prebuilt index, which
#    was deleted to save disk; re-fetch lucene-inverted.msmarco-v1-passage.20221004.252b5e first)
J=.venv/lib/python3.12/site-packages/pyserini/resources/jars/anserini-2.3.0-fatjar.jar
C=data/cache/lucene/classes-anserini
/opt/homebrew/opt/openjdk@21/bin/javac -nowarn -cp $J -d $C tools/lucene_ref/LuceneRef.java
/opt/homebrew/opt/openjdk@21/bin/java -Xmx4g -cp $J:$C LuceneRef bench <prebuilt index dir> \
  data/raw/msmarco/queries.dev.small.tsv 10 1000 3 > results/lexical/vs_lucene_lucene.json
engine/build-lex/hs_lex_bench --index data/indexes/lexical/full --queries data/raw/msmarco/queries.dev.small.tsv \
  --limit 1000 --algos maxscore,bmw --codecs bp128 --warmup 3 --out results/lexical/vs_lucene_ours.json
```

After re-running, regenerate the latency columns of `pruning.md`.
