# Sharded lexical indexes with global BM25 statistics

**Problem.** BM25 depends on three collection-wide quantities: N (docs with at least one term),
avgdl, and each query term's df. If a shard builds its index from its own slice alone, it scores
with shard-local N, avgdl and df. The per-shard scores then drift from the single-index scores, and
a merge of the per-shard top-k no longer equals the single index's top-k, so it cannot match
Anserini either. This is the classic distributed-IDF problem.

**What we do: fix the statistics at build time.** `hs_index_build --shard i/N --global-stats FULL_INDEX_DIR`
builds lines `[floor(i*n/N), floor((i+1)*n/N))` of the ID-sorted collection, so each shard is a
contiguous passage-ID range. It reads N, sum_dl and every term's df from an existing full index and
stores them in the shard: `meta.txt` gets `global_docs_with_terms` and `global_sum_dl`, and
`global_df.u32` holds one df per shard term. At query time the shard's idf uses the global N and df.
Its avgdl, its Lucene norm cache and its textbook length normalisation use the global avgdl. At build
time, the per-block and per-term score-bound postings (the argmax of tf·inv_norm) are chosen under the
global avgdl, so WAND/BMW/MaxScore bounds are exact and safe for the scores the shard actually
computes. Local df is kept separately because it is the number of postings stored.

**The tradeoff.** Elasticsearch's `dfs_query_then_fetch` solves the same problem at query time. It
spends an extra round trip to every shard to collect term statistics, then scores with the summed
values. That costs a fan-out's worth of latency on every query, but it stays exact under continuous
indexing. Our build-time approach adds nothing at query time and needs no extra RPC in the broker.
The cost is that the statistics are frozen: adding or deleting documents in one shard makes every
shard's statistics stale until a rebuild. For a static collection like MS MARCO that cost is zero.
For a live index you would either tolerate slowly drifting statistics (Elasticsearch's default
`query_then_fetch` does exactly that) or refresh the global statistics periodically. The single
index's statistics also have to exist before the shards are built (a full pass or a stats-only pass).

## Verification (all correctness, no timings)

| check | result |
|---|---|
| ctest `Sharding.GlobalStatsMergedTopKEqualsSingleIndex`: test corpus (2,500 docs), N ∈ {2,3,4}, 300 queries × 5 algorithms × {Lucene, textbook} × k ∈ {1,10,100} | merged top-k **identical** (global IDs + float bits) in every one of 27,000 comparisons |
| ctest `Sharding.ShardLocalStatsDoNotMatch` (control, N=3, shard-local statistics) | merged top-10 differs from single index on most queries (asserted > 50% of non-empty queries), so the gate can fail |
| ctest `Sharding.RealData1mFourShards`: `data/indexes/lexical/1m-4shards` vs `data/indexes/lexical/1m`, first 500 dev queries, k=100, {exhaustive, MaxScore, WAND, BMW} | **0 mismatches in 2,000 comparisons** |
| shard docids concatenated == `data/subset/1m/docids.u64bin` | true. Shard ID ranges: [0, 2227464], [2227472, 4449140], [4449144, 6672832], [6672849, 8841814] |

Build: `scripts/build_serving_indexes.sh` (bp128 postings + doc store, 250,000 docs and ~7.4–7.7M postings
per shard, 57–59 MB per shard on disk). Per-shard build reports: `results/lexical/build_1m_shard{0..3}.json`.
