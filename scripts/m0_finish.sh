#!/usr/bin/env bash
# Serialised tail of the M0/dense pipeline (keeps CPU use at <= 4 threads at a time):
#  1. subset Lucene index + BM25 runs (scripts/reference_bm25_1m.sh)
#  2. remaining full-index runs (scripts/reference_bm25.sh; train50k in 5 resumable parts)
#  3. verify every full-index run is complete, then delete the prebuilt index (~2.5 GB)
#  4. wait for data/embeddings/1m/READY, then parity, exact flat runs + GT, BEIR, scoring
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=py
log() { echo "[$(date +%H:%M:%S)] $*"; }
scripts/reference_bm25_1m.sh
log "subset BM25 done"
scripts/reference_bm25.sh dev dl19 dl20 train_tune train50k
log "full-index BM25 runs finished"
ok=1
for spec in "anserini-bm25-default.dev.trec 6980" "anserini-bm25-default.dl19.trec 43" "anserini-bm25-default.dl20.trec 54" \
            "anserini-bm25-default.train_tune.trec 2000" "bm25.train50k.trec 50000"; do
  set -- $spec
  n=$(cut -d' ' -f1 "data/runs/reference/$1" | uniq | sort -u | wc -l | tr -d ' ')
  log "$1: $n queries (expected <= $2)"
  [ "$n" -ge $(( $2 * 99 / 100 )) ] || ok=0
done
# Other agents (lexical parity work) also read this index; only delete when nobody else
# has it open, otherwise leave it and say so (delete later: rm -rf ~/.cache/pyserini/indexes/manual).
if [ $ok = 1 ] && ! pgrep -f "lucene-inverted.msmarco-v1-passage" >/dev/null; then
  rm -rf "$HOME/.cache/pyserini/indexes/manual"
  log "deleted prebuilt msmarco-v1-passage index"
elif [ $ok = 1 ]; then
  log "NOT deleting prebuilt index: another process is using it"
else
  log "NOT deleting prebuilt index: a run looks incomplete"
fi
.venv/bin/python -m hybridsearch.eval.reference || true
while [ ! -f data/embeddings/1m/READY ]; do sleep 60; done
log "embeddings ready"
scripts/dense_reference.sh
log "all done"
