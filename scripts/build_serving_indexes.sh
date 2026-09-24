#!/usr/bin/env bash
# Build the serving indexes for the sharded demo (1M subset, 4 shards).
# Shard i holds lines [floor(i*n/4), floor((i+1)*n/4)) of data/subset/1m/collection.tsv
# (n = 1,000,000; the file is sorted by global passage ID, same order as docids.u64bin).
# Sections are appended by each engine owner.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"
bin="${HS_ENGINE_BUILD:-engine/build-lex}"
coll=data/subset/1m/collection.tsv
nshards=4

# --- lexical -------------------------------------------------------------------------------
# Global BM25 statistics (N, avgdl, every term's df) come from the single 1M index, so each
# shard scores exactly like the single index and a ranks_before merge reproduces its top-k
# (verified by ctest Sharding.RealData1mFourShards). Postings in bp128 only, with doc store.
if [ ! -f data/indexes/lexical/1m/meta.txt ]; then
  "$bin/hs_index_build" --input "$coll" --out data/indexes/lexical/1m --threads 3 \
    --report results/lexical/build_1m.json
fi
for i in $(seq 0 $((nshards - 1))); do
  "$bin/hs_index_build" --input "$coll" --out "data/indexes/lexical/1m-4shards/shard$i" \
    --shard "$i/$nshards" --global-stats data/indexes/lexical/1m --codecs bp128 --threads 1 \
    --report "results/lexical/build_1m_shard$i.json"
done

# --- vector ----------------------------------------------------------------------------------
# One DiskANN-style SSD index per shard over rows [floor(i*n/4), floor((i+1)*n/4)) of
# data/embeddings/1m/passages.fbin (row order = docids.u64bin), each with its own
# docids.u64bin of global passage ids. Built with the paper's memory-limited procedure:
# k-means into overlapping partitions (each point in its 2 nearest), Vamana per partition
# (R=64, L=100, alpha=1.2), merge + re-prune; PQ M=96 (96 B/vector in RAM); one 4 KB block
# per node. Deterministic under the seed. One shard at a time, 4 threads, <= ~1.5 GB RSS.
# Verify with engine/build-vec/hs_vec_shard_check (exact merged top-10 == global top-10).
vbin="${HS_VECTOR_BUILD:-engine/build-vec}"
emb=data/embeddings/1m
[ -f "$emb/READY" ] || { echo "vector: $emb/READY missing (embeddings not finished)" >&2; exit 1; }
nvec=$(od -An -t u4 -N 4 "$emb/passages.fbin" | tr -d ' ')
for i in $(seq 0 $((nshards - 1))); do
  b=$(( i * nvec / nshards )); e=$(( (i + 1) * nvec / nshards ))
  out="data/indexes/vector/1m-4shards/shard$i"
  [ -f "$out/disk.index" ] && [ -f "$out/build.json" ] && continue
  HS_THREADS=4 "$vbin/hs_vec_disk_build" --base "$emb/passages.fbin" --row-begin "$b" --max-n $(( e - b )) \
    --docids data/subset/1m/docids.u64bin --out "$out" --R 64 --L 100 --alpha 1.2 \
    --pq-M 96 --pq-sample 100000 --ram-gb 0.4 --threads 4 --min-free-gb 8
  mkdir -p results/vector && cp "$out/build.json" "results/vector/serving_shard$i.build.json"
done
