#!/usr/bin/env bash
# Vamana build-parameter sweep (R, L_build, alpha, one- vs two-pass) on a real-embedding
# subset, with an L_search sweep per build: recall@10 (vs brute force) against QPS.
# Queries: the train_tune set (no tuning on dev).
#
# Usage: scripts/vector/sweep_build_params.sh BASE.fbin N WORKDIR [OUT_PREFIX]
#   BASE.fbin  vectors (first N rows used)     WORKDIR  scratch for graphs + GT (deleted rows kept small)
# Env: HS_THREADS (build threads, default 4), BUILD (engine build dir, default engine/build-vec)
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$here"
base="$1"; n="$2"; work="$3"; prefix="${4:-results/vector/build_sweep_${n}}"
bin="${BUILD:-engine/build-vec}"
export HS_THREADS="${HS_THREADS:-4}"
queries=data/embeddings/1m/queries.train_tune.fbin
mkdir -p "$work" "$(dirname "$prefix")"
gt="$work/gt.train_tune.${n}.bin"
[ -f "$gt" ] || "$bin/hs_vec_groundtruth" --base "$base" --max-base "$n" --queries "$queries" --k 100 --out "$gt"
: > "$prefix.build.jsonl"; : > "$prefix.search.jsonl"
run() {  # label R L alpha [extra]
  local label="$1" R="$2" L="$3" a="$4"; shift 4
  local g="$work/graph.$label.bin"
  "$bin/hs_vec_build" --base "$base" --max-n "$n" --out "$g" --R "$R" --L "$L" --alpha "$a" "$@" \
     --stats "$prefix.build.jsonl"
  "$bin/hs_vec_search" --base "$base" --max-n "$n" --graph "$g" --queries "$queries" --gt "$gt" \
     --L 10,15,20,30,40,60,80,120,160,240 --label "$label" --out "$prefix.search.jsonl"
  rm -f "$g"
}
run R64-L100-a1.2 64 100 1.2
run R32-L100-a1.2 32 100 1.2
run R16-L100-a1.2 16 100 1.2
run R96-L100-a1.2 96 100 1.2
run R64-L50-a1.2  64 50  1.2
run R64-L200-a1.2 64 200 1.2
run R64-L100-a1.0 64 100 1.0
run R64-L100-a1.4 64 100 1.4
run R64-L100-a1.2-1pass 64 100 1.2 --single-pass
cmd="scripts/vector/sweep_build_params.sh $*"
for f in "$prefix.build.jsonl" "$prefix.search.jsonl"; do
  .venv/bin/python -m hybridsearch.bench.meta "$f" --command "$cmd" --cache warm --threads "$HS_THREADS" \
    --set "queries=train_tune (2000)" "base=first $n rows of $base" "query_threads=1"
done
