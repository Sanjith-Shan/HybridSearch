#!/usr/bin/env bash
# M5 evaluation. Every rerank goes through hybridsearch.eval (the project's trec_eval-parity harness).
#   scripts/rerank_eval.sh quality            # public + ablations: BM25 (Anserini, full 8.8M) top-100, dl19 dl20 dev
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh subset1m  # chosen + public on the engine's own BM25 over the 1M subset
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh export    # ONNX fp32 + int8 -> data/models/reranker
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh int8      # ORT CPU fp32 vs int8 quality: dl19, dl20, first 1000 dev queries
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh cascade   # depth-200 scores (torch MPS; ORT int8 on dl19/dl20) + bench + curve
# CHOSEN is picked by train_tune validation RR@10 (results/rerank/train/*.json), never by dev.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=data/runs/reference
MODELS=${MODELS:-"public mps_bm25 mps_dense mps_random"}
model_path() { [ "$1" = public ] && echo cross-encoder/ms-marco-MiniLM-L6-v2 || echo data/rerank/runs/$1/best; }
rr() {  # name split candidates depth model backend label [extra args...]
  local name=$1 split=$2 cand=$3 depth=$4 model=$5 backend=$6 label=$7; shift 7
  local out=results/rerank/eval/$name.$split.json
  [ -s "$out" ] && { echo "exists: $out"; return; }
  $PY -m hybridsearch.rerank.rerank_run --name "$name" --split "$split" --candidates "$cand" --depth "$depth" \
     --model "$model" --backend "$backend" --candidate-label "$label" "$@" 2>&1 \
     | grep -v -i -E "warn|Loading weights" | grep -E '"(RR@10|nDCG@10)"|Error|Traceback' | head -4
}
FULL="Anserini BM25 default (k1=0.9,b=0.4), full 8.8M"
case "${1:-quality}" in
quality)
  for m in $MODELS; do
    mp=$(model_path $m)
    [ "$m" != public ] && [ ! -d "$mp" ] && { echo "skip $m (no checkpoint)"; continue; }
    for s in dl19 dl20 dev; do rr "$m.bm25full" $s $R/anserini-bm25-default.$s.trec 100 "$mp" torch "$FULL"; done
  done ;;
subset1m)
  for m in public $CHOSEN; do
    for s in dl19 dl20 dev; do
      sp=$( [ $s = dev ] && echo dev1m || echo ${s}_1m )
      rr "$m.hsbm25-1m" $sp data/runs/m4/hs-bm25-1m.$s.trec 100 "$(model_path $m)" torch "HybridSearch engine BM25 over the 1M subset"
    done
  done ;;
export)
  $PY -m hybridsearch.rerank.export --model data/rerank/runs/$CHOSEN/best --out data/models/reranker --int8 --tag reranker ;;
int8)
  for v in fp32 int8; do
    f=data/models/reranker/$( [ $v = fp32 ] && echo model.onnx || echo model.int8.onnx )
    for s in dl19 dl20; do rr "$CHOSEN-ort-$v.bm25full" $s $R/anserini-bm25-default.$s.trec 100 $f ort "$FULL" --batch 64; done
    rr "$CHOSEN-ort-$v.bm25full-q1000" dev $R/anserini-bm25-default.dev.trec 100 $f ort "$FULL" --batch 64 --max-queries 1000
  done ;;
cascade)
  for s in dl19 dl20 dev; do rr "$CHOSEN-d200.bm25full" $s $R/anserini-bm25-default.$s.trec 200 "$(model_path $CHOSEN)" torch "$FULL"; done
  for s in dl19 dl20; do rr "$CHOSEN-int8-d200.bm25full" $s $R/anserini-bm25-default.$s.trec 200 data/models/reranker/model.int8.onnx ort "$FULL" --batch 64; done
  $PY -m hybridsearch.rerank.bench --model-dir data/models/reranker --name reranker
  $PY -m hybridsearch.rerank.cascade --name $CHOSEN-d200.bm25full --int8-name $CHOSEN-int8-d200.bm25full --bench reranker ;;
esac
$PY -m hybridsearch.rerank.report > /dev/null
