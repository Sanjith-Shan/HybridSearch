#!/usr/bin/env bash
# M5 ablations on the laptop GPU (Apple MPS): one small real training run per negative strategy.
# Each run takes data/locks/mps (mkdir mutex shared with the dense encoder) while it trains.
#   scripts/rerank_train_mps.sh [random bm25 dense mixed]   (default: all four, in that order)
#   STEPS=2000 scripts/rerank_train_mps.sh bm25
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-2000}
runs="$*"; [ -z "$runs" ] && runs="random bm25 dense mixed"
avail_gb=$(df -g data | awk 'NR==2{print $4}')
[ "$avail_gb" -lt 4 ] && { echo "refusing: ${avail_gb} GB free"; exit 1; }
mkdir -p data/rerank/logs
for r in $runs; do
  out=data/rerank/runs/mps_$r
  resume=""; [ -f "$out/last/trainer_state.pt" ] && resume="--resume"
  .venv/bin/python -m hybridsearch.rerank.train --config py/hybridsearch/rerank/configs/mps_$r.yaml \
    --set max_steps=$STEPS "label=M3 Pro MPS, $STEPS steps, ${r} negatives" --wait-lock $resume --drop-optimizer \
    2>&1 | grep -v -i -E "warn|Loading weights" | tee -a data/rerank/logs/train_mps_$r.log
  mkdir -p results/rerank/train
  cp $out/train_summary.json results/rerank/train/mps_$r.json
  cp $out/train_summary.json.meta.json results/rerank/train/mps_$r.json.meta.json
done
