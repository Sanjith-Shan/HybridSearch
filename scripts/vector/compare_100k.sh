#!/usr/bin/env bash
# Vamana (R=64, L=100, alpha=1.2) vs hnswlib / FAISS on the same 100K real-embedding subset,
# same train_tune queries and brute-force ground truth. Single search thread everywhere.
# Usage: scripts/vector/compare_100k.sh BASE.fbin WORKDIR   (WORKDIR holds gt.train_tune.100000.bin from the sweep)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$here"
base="$1"; work="$2"; n=100000; B=engine/build-vec; export HS_THREADS="${HS_THREADS:-4}" PYTHONPATH=py
TT=data/embeddings/1m/queries.train_tune.fbin; gt="$work/gt.train_tune.$n.bin"; R=results/vector
$B/hs_vec_build --base "$base" --max-n $n --out "$work/g.bin" --R 64 --L 100 --alpha 1.2 --stats $R/compare_100k.vamana.build.jsonl
rm -f $R/compare_100k.vamana.jsonl $R/compare_100k.others.jsonl
$B/hs_vec_search --base "$base" --max-n $n --graph "$work/g.bin" --queries $TT --gt "$gt" \
  --L 10,15,20,30,40,60,80,120,160,240,320 --label vamana-R64 --out $R/compare_100k.vamana.jsonl
rm -f "$work/g.bin"
.venv/bin/python -m hybridsearch.bench.ann_compare --base "$base" --max-n $n --queries $TT --gt "$gt" \
  --out $R/compare_100k.others.jsonl --build-threads $HS_THREADS --nlist 1024
for f in $R/compare_100k.vamana.build.jsonl $R/compare_100k.vamana.jsonl $R/compare_100k.others.jsonl; do
  .venv/bin/python -m hybridsearch.bench.meta $f --command "scripts/vector/compare_100k.sh $*" --cache warm --threads $HS_THREADS \
    --set "queries=train_tune (2000)" "search_threads=1" "base=first $n rows of the 1M BGE subset" >/dev/null
done
.venv/bin/python -m hybridsearch.bench.plot_vector compare $R/compare_100k.png $R/compare_100k.vamana.jsonl $R/compare_100k.others.jsonl
