#!/usr/bin/env bash
# Full-scale (8,841,823 passages) DiskANN-style build + evaluation on a rented Linux box.
# NOT run on the laptop. See "Full scale: DiskANN over 8.8M" in docs/VECTOR.md.
#
# Box: >= 64 GB RAM, >= 500 GB local NVMe, >= 16 cores, Ubuntu 22.04/24.04, pinnable cores.
# Usage: scripts/full_scale/build_diskann_8m.sh [WORK=/mnt/nvme/hs]
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$here"
WORK="${1:-/mnt/nvme/hs}"
URL=https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-bge-base-en-v1.5.parquet.tar
MD5=a55b3cb338ec4a1b1c36825bf0854648
THREADS="${THREADS:-$(nproc)}"
BIN=engine/build-full
EMB="$WORK/embeddings/full"; IDX="$WORK/indexes/vector/full"; RES=results/vector/full
mkdir -p "$EMB" "$IDX" "$RES"

# 0. Toolchain + engine (Release, -march=native, AVX2 path).
sudo apt-get install -y build-essential cmake python3-venv >/dev/null
scripts/build_engine.sh "$BIN" -DHS_BUILD_SERVER=OFF
ctest --test-dir "$BIN" -R 'Vamana|Disk|PQ|KMeans|Distance|BruteForce|RobustPrune|Containers'

# 1. Stream the 26 GB tar straight into passages.fbin (27 GB) + docids; never store the tar.
if [ ! -f "$EMB/passages.fbin" ]; then
  curl -sSfL "$URL" | tee >(md5sum | cut -d' ' -f1 > "$EMB/tar.md5") \
    | .venv/bin/python -m hybridsearch.bench.parquet_stream "$EMB"
  sleep 2; got=$(cat "$EMB/tar.md5")
  [ "$got" = "$MD5" ] || { echo "MD5 mismatch: $got != $MD5" >&2; exit 1; }
fi

# 2. Query embeddings (BGE query instruction, CLS, L2-norm) via the data module, then
#    exact ground truth by brute force (8.8M x 2000 train_tune queries; ~ minutes at 16+ cores).
.venv/bin/python -m hybridsearch.dense.encode queries --threads "$THREADS"
Q=data/embeddings/1m   # query files are corpus-independent
"$BIN/hs_vec_groundtruth" --base "$EMB/passages.fbin" --queries $Q/queries.train_tune.fbin --k 100 \
  --out "$EMB/gt.train_tune.bin" --threads "$THREADS"
"$BIN/hs_vec_groundtruth" --base "$EMB/passages.fbin" --queries $Q/queries.dev.fbin --k 100 \
  --out "$EMB/gt.dev.bin" --threads "$THREADS"

# 3. Memory-limited build: 96-byte PQ, partitions sized to a 24 GB budget (~ 14 overlapping
#    clusters at 8.8M), R=64 (one 4 KB block per node), L=100, alpha=1.2.
#    Expected: disk.index 8,841,823 x 4 KB = 36.2 GB; PQ codes 849 MB; spill ~ 4.6 GB temporary.
/usr/bin/time -v "$BIN/hs_vec_disk_build" --base "$EMB/passages.fbin" --docids "$EMB/docids.u64bin" \
  --out "$IDX" --R 64 --L 100 --alpha 1.2 --pq-M 96 --pq-sample 500000 --ram-gb 24 \
  --threads "$THREADS" 2> "$RES/build.time.txt"
cp "$IDX/build.json" "$RES/build.json"

# 4. Sweeps, pinned to cores 0-7 (taskset), cold (O_DIRECT) then warm, W in {2,4,8}, cache 0 / 100K nodes.
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
for cache in 0 100000; do
  for mode in cold warm; do
    drop_caches
    taskset -c 0-7 "$BIN/hs_vec_disk_search" --index "$IDX" --queries $Q/queries.train_tune.fbin \
      --gt "$EMB/gt.train_tune.bin" --L 10,20,30,50,75,100,150,200 --W 2,4,8 --cache $cache \
      --mode $mode --io-threads 8 --label diskann-8.8m --out "$RES/disk_sweep.jsonl"
  done
done

# 5. MRR@10 on dev at each operating point vs exact search (published BGE flat: 0.3583).
for L in 20 50 100 200; do
  taskset -c 0-7 "$BIN/hs_vec_disk_search" --index "$IDX" --queries $Q/queries.dev.fbin --gt "$EMB/gt.dev.bin" \
    --L $L --W 4 --mode cold --label diskann-8.8m-dev --trec-prefix "$RES/dev" --qids $Q/queries.dev.qids.txt \
    --out "$RES/disk_dev_points.jsonl"
done
.venv/bin/python -m hybridsearch.bench.gt_to_trec "$EMB/gt.dev.bin" $Q/queries.dev.qids.txt "$EMB/docids.u64bin" \
  "$RES/exact.dev.trec" exact-flat-8.8m --k 10
.venv/bin/python -m hybridsearch.bench.mrr_points --runs "$RES"/dev.W4.L*.trec \
  --qrels data/raw/msmarco/qrels.dev.small.tsv --reference "$RES/exact.dev.trec" \
  --points "$RES/disk_sweep.jsonl" "$RES/disk_dev_points.jsonl" --out "$RES/mrr_dev_8.8m.json"
for f in "$RES"/*.jsonl "$RES"/*.json; do
  .venv/bin/python -m hybridsearch.bench.meta "$f" --build-dir "$BIN" --threads 8 --cache "cold+warm" \
    --command "scripts/full_scale/build_diskann_8m.sh $WORK" --set pinned=true "taskset=0-7" "timing_label=Linux box, pinned (taskset 0-7), resume-grade"
done
