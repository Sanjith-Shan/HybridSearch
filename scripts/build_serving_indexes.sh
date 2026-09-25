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
# One DiskANN-style SSD index per shard, each with its own docids.u64bin of global passage ids.
# Partitioning = the lexical rule: shard i holds the rows of data/embeddings/1m/passages.fbin
# (row order = data/subset/1m/docids.u64bin) whose global docid % 4 == i
# -> data/indexes/vector/1m-4shards-mod/shard{0..3}.  Range shards (rows [floor(i*n/4),
# floor((i+1)*n/4))) are kept only as the comparison: VEC_RANGE=1 -> data/indexes/vector/1m-4shards.
# Build = the paper's memory-limited procedure: k-means into overlapping partitions (each point
# in its 2 nearest; --ram-gb 0.4 -> 5 partitions per 250K shard), Vamana per partition
# (R=64, L=100, alpha=1.2), merge + re-prune; PQ M=96 (96 B/vector in RAM); one 4 KB block per
# node. Deterministic under the seed. ~1.1-1.7 GB peak RSS per shard.
# Verify: engine/build-vec/hs_vec_shard_check (exact merged top-10 == global exact top-10).
vbin="${HS_VECTOR_BUILD:-engine/build-vec}"
emb=data/embeddings/1m
vt="${VEC_THREADS:-4}"
[ -f "$emb/READY" ] || { echo "vector: $emb/READY missing (embeddings not finished)" >&2; exit 1; }
for i in $(seq 0 $((nshards - 1))); do
  out="data/indexes/vector/1m-4shards-mod/shard$i"
  [ -f "$out/disk.index" ] && [ -f "$out/build.json" ] && continue
  HS_THREADS="$vt" "$vbin/hs_vec_disk_build" --base "$emb/passages.fbin" --docids data/subset/1m/docids.u64bin \
    --docid-mod "$nshards" --docid-rem "$i" --out "$out" --R 64 --L 100 --alpha 1.2 \
    --pq-M 96 --pq-sample 100000 --ram-gb 0.4 --threads "$vt" --min-free-gb 6
  mkdir -p results/vector && cp "$out/build.json" "results/vector/serving_mod_shard$i.build.json"
done
if [ "${VEC_RANGE:-0}" = 1 ]; then
  nvec=$(od -An -t u4 -N 4 "$emb/passages.fbin" | tr -d ' ')
  for i in $(seq 0 $((nshards - 1))); do
    b=$(( i * nvec / nshards )); e=$(( (i + 1) * nvec / nshards ))
    out="data/indexes/vector/1m-4shards/shard$i"
    [ -f "$out/disk.index" ] && [ -f "$out/build.json" ] && continue
    HS_THREADS="$vt" "$vbin/hs_vec_disk_build" --base "$emb/passages.fbin" --row-begin "$b" --max-n $(( e - b )) \
      --docids data/subset/1m/docids.u64bin --out "$out" --R 64 --L 100 --alpha 1.2 \
      --pq-M 96 --pq-sample 100000 --ram-gb 0.4 --threads "$vt" --min-free-gb 6
    mkdir -p results/vector && cp "$out/build.json" "results/vector/serving_shard$i.build.json"
  done
fi

# Mod (hash) partitioning: shard i holds the passages whose global id % 4 == i. MS MARCO ids are
# clustered by source split (86% of dev-relevant passages fall in the top id quarter), so range
# shards are badly skewed; mod shards spread relevant passages and query load evenly. Same global
# BM25 statistics, so merged top-k == single-index top-k (ctest Sharding.RealData1mFourShardsMod).
for i in $(seq 0 $((nshards - 1))); do
  "$bin/hs_index_build" --input "$coll" --out "data/indexes/lexical/1m-4shards-mod/shard$i" \
    --shard "$i/$nshards" --partition mod --global-stats data/indexes/lexical/1m --codecs bp128 --threads 2 \
    --report "results/lexical/build_1m_mod_shard$i.json"
done
