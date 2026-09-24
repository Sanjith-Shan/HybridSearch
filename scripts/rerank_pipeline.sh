#!/usr/bin/env bash
# M5 end-to-end on the laptop, waiting for its inputs: the 1M passage vectors (dense agent),
# the train50k BM25 run (data agent) and a free MPS lock. Every GPU step takes the lock itself.
#   nohup scripts/rerank_pipeline.sh > data/rerank/logs/pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }
until [ -f data/embeddings/1m/passages.fbin ] && [ ! -d data/locks/mps ]; do sleep 120; done
log "passages.fbin ready, lock free"
[ -f data/rerank/positives.train50k.fbin ] || $PY -m hybridsearch.rerank.mine positives --backend mps 2>&1 | grep -v -i warn
# The MPS ablations need only part of train50k: mine from the full run if it exists, else from
# the finished parts (>= MIN_PARTS of the data agent's 10k-query shards). Queries without a BM25
# run are skipped, so all strategies see the same queries. Re-mine with the full run for A100.
P=data/runs/reference/train50k.parts
MIN_PARTS=${MIN_PARTS:-2}
runs() { if [ -f data/runs/reference/bm25.train50k.trec ]; then echo data/runs/reference/bm25.train50k.trec;
         else ls $P/run.?.trec 2>/dev/null | paste -sd, -; fi; }
until [ -f data/runs/reference/bm25.train50k.trec ] || [ "$(ls $P/run.?.trec 2>/dev/null | wc -l)" -ge "$MIN_PARTS" ]; do sleep 120; done
log "bm25 runs: $(runs)"
[ -f data/rerank/mined/train50k.jsonl ] || $PY -m hybridsearch.rerank.mine mine --bm25-run "$(runs)" 2>&1 | grep -v -i warn
mkdir -p results/rerank && cp data/rerank/mined/train50k.stats.json results/rerank/mining_stats.json
$PY -c "from hybridsearch.rerank.meta import write_meta; write_meta('results/rerank/mining_stats.json', what='hard-negative mining filter counts, train50k')"
log "mined"
# throughput probe (not a result): 60 steps, no validation
[ -f data/rerank/runs/probe/train_summary.json ] || $PY -m hybridsearch.rerank.train --config py/hybridsearch/rerank/configs/mps_bm25.yaml \
  --set max_steps=60 val_n=0 eval_every=0 log_every=20 out_dir=data/rerank/runs/probe label=probe --wait-lock --drop-optimizer 2>&1 | grep -E "pairs_per_s|data:|Error|Traceback"
log "probe done"
[ -z "${STEPS:-}" ] && { log "set STEPS=<n> (chosen from the probe rate) and rerun to train"; exit 0; }
STEPS=$STEPS scripts/rerank_train_mps.sh ${RUNS:-bm25 dense random mixed}
log "training done"
MODELS="public mps_bm25 mps_dense mps_random mps_mixed" scripts/rerank_eval.sh quality
log "quality eval done"
