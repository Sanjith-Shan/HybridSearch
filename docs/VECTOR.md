# Vector index (M3): Vamana in memory, DiskANN-style on SSD

Owner: the vector module (`engine/include/hs/vector`, `engine/src/vector`,
`engine/tests/vector`, `engine/tools/hs_vec_*`, `engine/bench/bench_vec_*`,
`py/hybridsearch/bench/`, `scripts/vector/`, `scripts/full_scale/`).

The embeddings are BAAI/bge-base-en-v1.5 (768-d, CLS, L2-normalised). On the laptop they
were encoded by the data module over the 1M-passage subset; at full scale they are
Waterloo's precomputed vectors. The project is the index, not the encoder.

## What exists

| Piece | File | Notes |
|---|---|---|
| SIMD distances | `hs/vector/distance.hpp` | NEON (aarch64), AVX2+FMA (x86-64), scalar fallback; tested against scalar at 13 dims |
| Greedy search | `hs/vector/graph_search.hpp` | Algorithm 1 of the paper; open-addressing visited set; neighbour-vector prefetch |
| RobustPrune, Vamana build | `hs/vector/vamana.hpp`, `src/vector/vamana.cc` | random R-regular init, medoid start, two passes (alpha=1 then alpha), back-edge slack 1.3 |
| Brute force + ground truth | `hs/vector/bruteforce.hpp` | blocked exact inner-product top-k; the reference for every recall number |
| k-means, PQ | `hs/vector/kmeans.hpp`, `hs/vector/pq.hpp` | seeded Lloyd; PQ on centred data, M subspaces x 256 centroids (M=96 -> 96 B/vector at 768-d) |
| SSD index | `hs/vector/disk_index.hpp`, `src/vector/disk_{build,index}.cc` | 4 KB node blocks, PQ codes in RAM, beam search with W parallel reads, full-precision rerank, BFS node cache, deadline |
| Memory-limited build | `build_disk_index(... partitions / build_ram_gb ...)` | k-means into P clusters, each point in its 2 nearest, Vamana per cluster, spill, streaming merge (union, re-prune) |
| Tools | `engine/tools/hs_vec_{groundtruth,build,search,disk_build,disk_search,synth}.cc` | each prints / appends JSON lines |
| Benches | `engine/bench/bench_vec_{distance,coldread}.cc` | kernel microbench; cold-read honesty probe |
| Python | `py/hybridsearch/bench/` | `ann_compare` (hnswlib, FAISS HNSW / IVF-PQ / IVF-PQ+refine / flat), `mrr_points`, `plot_vector`, `meta`, `gt_to_trec`, `gt_from_pids`, `parquet_stream` |

## Design decisions worth defending

**Deterministic parallel build.** The paper's parallel build (and Microsoft's code) lets
threads insert concurrently under per-node locks, so the graph depends on scheduling.
Here each pass walks a seeded permutation in batches (prefix doubling up to 2% of n, as
in ParlayANN, Manohar et al. PPoPP 2024), and each batch has three barrier-separated
phases: (1) search + RobustPrune for every batch point against a frozen graph; (2) each
point installs its own out-list; (3) back-edges grouped by destination with a stable sort,
one task per destination. No phase reads what another task of the same phase writes, so
the graph is a pure function of (data, parameters, seed). `Vamana.RebuildIsBitIdentical
AcrossRunsAndThreadCounts` builds with 1, 3 and 8 threads (and twice with 8) and asserts
equal graph hashes; `DiskIndex.BuildIsDeterministic` asserts byte-identical `disk.index`,
`pq_codes.bin` and `pq_pivots.bin` for 2 vs 6 threads with 3 partitions. k-means and the
medoid use a fixed chunking of floating-point sums, independent of thread count. The cost
is staleness: points in the same batch do not see each other's edges during phase 1.

**alpha.** Applied to squared L2, as Microsoft's implementation does (alpha = 1.2 on
squared distance is sqrt(1.2) on Euclidean distance in the paper's notation). At alpha=1
RobustPrune is the HNSW/RNG heuristic: a candidate is dropped if any kept neighbour is at
least as close to it as p is, which leaves a sparse graph with short edges and long
routes. alpha > 1 only drops a candidate when a kept neighbour is *much* closer to it, so
longer edges survive and greedy search makes bigger jumps (fewer hops to the target).

**Back-edge slack.** Without it, every back-edge into a full node triggers a
RobustPrune; with lists allowed to reach 1.3R before pruning, those prunes are amortised.
A final pass prunes lists still above R.

**Disk layout.** `[float vec[768]][uint32 degree][uint32 nbrs[R]]` = 3,332 B at R=64, one
node per 4 KB block (R <= 255 fits one block). One read returns both the adjacency and
the full-precision vector, so the final rerank of expanded nodes costs no extra I/O.

**Beam search.** The candidate list is ordered by PQ (ADC) distance, computed from codes
in RAM. Each round reads the W closest unexpanded nodes concurrently (a shared pread
pool; the query thread does one read itself). The exact inner product of each expanded
node is kept, and the answer is the top-k of the expanded set by exact score. The
`hs::Deadline` is checked after every round; at least one round always completes, and the
result is flagged `partial`.

**Cold vs warm.** Cold = `F_NOCACHE` on macOS / `O_DIRECT` on Linux. On macOS,
`F_NOCACHE` does not bypass pages that are already cached (see `docs/BUG_LOG.md`,
2026-09-23), so cold mode first evicts the file with `msync(MS_INVALIDATE)`, and the
build writes the index with `F_NOCACHE`. Every row reports `us_per_io_round` as a check.

## Commands

```bash
scripts/build_engine.sh engine/build-vec        # or: cmake --build engine/build-vec -j4
ctest --test-dir engine/build-vec -R 'Vamana|Disk|PQ|KMeans|Distance|Containers|BruteForce|RobustPrune'
.venv/bin/python -m pytest -q py/tests/test_bench_vector.py

# build-parameter sweep on a real-embedding subset (train_tune queries)
scripts/vector/sweep_build_params.sh <base.fbin> 100000 <workdir>
# the 1M experiments, step by step (see the header of the script)
scripts/vector/run_1m.sh gt vamana vamana_dev disk disk_sweep disk_dev compare mrr plots
DISKV=disk_part scripts/vector/run_1m.sh disk_part disk_sweep disk_dev
```

## Full scale: DiskANN over 8.8M (rented Linux box)

Not run on the laptop (27 GB of fp32 vectors plus a 36 GB index do not fit). One script:
`scripts/full_scale/build_diskann_8m.sh [WORK=/mnt/nvme/hs]`.

**Box.** >= 64 GB RAM, >= 500 GB local NVMe, >= 16 cores, Ubuntu 22.04/24.04. Azure
L-series (e.g. L16s_v3: 16 vCPU, 128 GB, 1.9 TB NVMe) or the Tickerplant EPYC box.

**Vectors, streamed.** Waterloo's archive
`https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-bge-base-en-v1.5.parquet.tar`
(26 GB, MD5 `a55b3cb338ec4a1b1c36825bf0854648`, 8,841,823 passages; URL and checksum from
Anserini's reproduction page
`docs/reproduce/from-document-collection/msmarco-v1-passage.bge-base-en-v1.5.parquet.flat.onnx.md`)
is piped through `curl | tee >(md5sum) | python -m hybridsearch.bench.parquet_stream`,
which reads each parquet member from the tar stream in memory and appends it to
`passages.fbin` + `docids.u64bin`. The tar is never stored; the script checks the MD5
after the stream ends and aborts on mismatch. The parquet column names are not given on
that page: the converter auto-detects `docid|id|_id` and `vector|embedding|emb` and stops
(printing the schema) if it cannot, and it rejects non-unit vectors. **The converter has
not been run against the real archive** (it was never downloaded here).

**Steps and expected resources** (estimates, not measurements):

| Step | Output | Disk | Peak RAM | Notes |
|---|---|---|---|---|
| stream + convert | `passages.fbin` | 27.2 GB | ~1-2 GB (one parquet shard) | network-bound |
| query encode + exact GT (train_tune 2,000, dev 6,980) | `gt.*.bin` | < 10 MB | ~1 GB + mmap | brute force, 8.8M x 9K dot products |
| PQ (M=96, 500K-vector sample) | `pq_codes.bin` | 849 MB | ~2.5 GB | 96 B/vector in RAM at search time |
| partitioned Vamana, `--ram-gb 24` | spill files | ~4.6 GB (temporary) | ~24 GB + page cache | ~14 overlapping clusters of ~1.3M points |
| merge + write | `disk.index` | 8,841,823 x 4 KB = 36.2 GB | small | streams partitions in id order |
| sweeps | `results/vector/full/*.jsonl` | - | PQ codes 849 MB + cache | pinned with `taskset -c 0-7`, `drop_caches` before cold runs |

Only the full-scale numbers from that box, pinned and labelled, go on a resume; every
laptop number in `results/vector/` is `dev-signal-only`.

## Results on the laptop (dev-signal-only)

All timings: M3 Pro laptop, macOS, threads unpinned, machine shared with other projects
(load average 25-140 during these runs; each sidecar records it). Treat QPS and latency as
development signals; hops, distance computations, SSD reads, recall and MRR do not depend
on load. Recall is always against brute force (`hs_vec_groundtruth`, which matches the data
module's FAISS IndexFlatIP run exactly on dev: overlap@10 1.0000, `gt_crosscheck_dev_1m.json`).
Recall/QPS sweeps use the 2,000 `train_tune` queries (no tuning on dev); dev is used only for
MRR at the operating points.

**Build parameters (100K real vectors, `build_sweep_100000.*`).** Cost at matched recall@10
(`build_sweep_100000.matched_recall.json`), hops / distance computations per query:

| config | @0.95 | @0.98 | @0.99 |
|---|---|---|---|
| R=64, L=100, alpha=1.2 (default) | 28.6 / 1,074 | 62.1 / 1,945 | 116 / 3,159 |
| R=32 | 62.3 / 1,098 | 161 / 2,354 | not reached by L=240 |
| R=16 | 198 / 1,547 | not reached | not reached |
| R=96 | 22.1 / 1,086 | 46.4 / 1,884 | 79 / 2,836 |
| L_build=50 | 36.9 / 1,084 | 90.8 / 2,194 | 172 / 3,664 |
| L_build=200 | 26.8 / 1,080 | 58.4 / 1,938 | 107 / 3,082 |
| alpha=1.0 | 47.5 / 954 | 111 / 1,892 | 189 / 2,943 |
| alpha=1.4 | 33.5 / 1,177 | 81.8 / 2,305 | 159 / 3,854 |
| one pass at alpha=1.2 | 29.1 / 1,260 | 62.4 / 2,327 | 119 / 3,922 |

Reading: alpha=1 builds a sparse graph (average degree 22.8 of 64) that needs ~1.7x the
hops of alpha=1.2 at equal recall, while each hop is cheaper (fewer neighbours), so it wins
on distance computations in RAM but loses badly on disk, where a hop is a 4 KB read. That
is exactly why DiskANN uses alpha > 1. The second pass (alpha=1 then 1.2) saves ~15-20%
of distance computations over a single alpha=1.2 pass at the same hops.

**SSD index, one 250K shard (`shard0_250k.sweep.jsonl`, cold = F_NOCACHE + evicted).**
recall@10 0.90 at L=20 (W=4: 34.8 reads, 10 I/O rounds per query), 0.965 at L=50 (63 reads,
17 rounds), 0.984 at L=100 (112 reads, 29 rounds), 0.993 at L=200 (211 reads). Beam width
trades reads for rounds: at L=100, W=1 needs 105 rounds of 1 read, W=8 needs 17 rounds
and 122 reads. A 10,000-node BFS cache (33 MB) removes 10-12 reads/query at identical recall.
Sequential preads (no pool) at W=4 cost ~2.5 ms per round against ~0.35-0.6 ms with the
pool. RAM per shard: PQ codes 24.0 MB + pivots 0.8 MB (+ docids 2 MB); `disk.index` 1.02 GB.

**Serving shards, 1M (`shards_mod_1m.*`, `mrr_dev_1m_mod_shards.json`).** 4 modulo shards
(`docid % 4`), each built with 5 overlapping partitions under a 0.4 GB budget (6-20 min
per shard, 1.1-1.7 GB peak RSS). Exact check: merged exact top-10 == global exact top-10
for 2,000/2,000 queries (ranked lists identical), no row missing or duplicated; the same
holds for the range shards. Merged approximate recall@10 (train_tune, W=4): 0.930 / 0.957 /
0.978 / 0.987 / 0.992 at L = 10 / 20 / 50 / 100 / 200, with 103 / 138 / 251 / 447 / 842 SSD
reads per query summed over the 4 shards. RAM for the whole 1M: 96 MB PQ codes + 3.2 MB pivots.

**Dev MRR@10 cost (6,980 dev queries, the 1M subset).** Exact flat search: 0.6538. Merged
4-shard DiskANN: L=20 0.6347 (-0.0191), L=50 0.6449 (-0.0089), L=100 0.6487 (-0.0052),
L=200 0.6511 (-0.0027); all p=0.0001 (paired randomization). The 1M subset holds every
judged passage plus a random sample, so its absolute MRR is far above the 8.8M figure
(0.3583); only the delta transfers.

**Against hnswlib and FAISS (100K real vectors, same queries and ground truth, one search
thread, `compare_100k.*`; load average ~18-35).** Queries per CPU-second at matched recall@10:

| system | @0.90 | @0.95 | @0.98 | @0.99 | build (s, 7 threads) |
|---|---|---|---|---|---|
| this Vamana, R=64 L=100 alpha=1.2 | 5,191 | 3,136 | 1,639 | 984 | 59 |
| FAISS HNSW M=16 (efC=200) | 5,661 | 3,380 | 1,577 | 879 | 36 |
| FAISS HNSW M=32 | 4,237 | 2,771 | 1,460 | 911 | 56 |
| hnswlib M=16 | 1,446 | 872 | 461 | 280 | 62 |
| hnswlib M=32 | 1,136 | 750 | 488 | 334 | 88 |
| FAISS IVF-PQ (nlist 1024, PQ96) + exact refine x10 | 2,309 | 1,287 | - | - | 25 |
| FAISS IVF-PQ alone | tops out at recall 0.69 | | | | |
| FAISS IndexFlatIP (exact, batched GEMM) | 4,041 at recall 1.0 | | | | |

Reading: this Vamana is level with FAISS HNSW, about 8% behind at recall 0.90-0.95 and
about 8-12% ahead at 0.98-0.99. hnswlib is 3-4x slower than both. At equal M/ef it reaches the same
recall as FAISS HNSW, so the gap is per-distance cost: hnswlib's hand-written SIMD kernels are
x86-only (SSE/AVX), so on this arm64 laptop it most likely runs a scalar loop. That is an
inference, not profiled; expect the ranking to change on the x86 box. At 100K, batched exact
search (a BLAS matrix multiply over all queries) beats every graph index above recall ~0.93, because the
collection is too small for graph search to pay off against GEMM. That is why the 1M and 8.8M
numbers matter. PQ with 96 bytes caps recall at 0.69 without the full-precision rerank, which
is the reason DiskANN re-scores expanded nodes with their full vectors.

**Beam width on this laptop.** Wider beams cut I/O rounds as designed (L=100: 105 rounds at
W=1, 29 at W=4, 17 at W=8), but they did not raise cold single-query QPS here (L=100: 61 q/s
at W=1, 45 at W=4, 94 at W=8, with non-monotone curves). The runs overlapped other jobs
(load 12-60) and the cause is not isolated. This needs the pinned Linux NVMe box.

**Why modulo sharding, not ranges.** MS MARCO passage ids are grouped by source, and the
1M subset keeps every judged passage, so under range sharding 86.1% of the dev-relevant
passages sit in the top id quarter (shard 3). Under `docid % 4` each shard holds 24.3-25.3%
(computed from `data/subset/1m/qrels.dev.tsv` and `docids.u64bin`). Merged exact results are
identical either way (2,000/2,000 on both). Merged approximate recall is also close
(L=20 warm: range 0.960, modulo 0.957).

**Not done on the laptop.** The in-memory Vamana and hnswlib/FAISS comparison at 1M (done at
100K). io_uring (the Linux path uses a pread pool with O_DIRECT). The 8.8M build (script
ready, not run; the parquet converter has never seen the real archive).
