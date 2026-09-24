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
