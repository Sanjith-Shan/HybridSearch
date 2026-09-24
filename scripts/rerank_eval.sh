#!/usr/bin/env bash
# M5 evaluation: rerank first-stage top-100 with every model through the project harness.
#   scripts/rerank_eval.sh quality          # all models x {full-collection BM25, 1M BM25, 1M hybrid if present}
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh export   # ONNX fp32+int8 of the chosen model -> data/models/reranker
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh int8     # ORT CPU fp32 vs int8 quality (dl19, dl20[, dev])
#   CHOSEN=mps_bm25 scripts/rerank_eval.sh cascade  # depth-200 scores + bench + cascade curve
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=data/runs/reference
MODELS=${MODELS:-"public mps_random mps_bm25 mps_dense mps_mixed"}
model_path() { [ "$1" = public ] && echo cross-encoder/ms-marco-MiniLM-L6-v2 || echo data/rerank/runs/$1/best; }
rr() {  # name split candidates depth model [backend] [label]
  local out=results/rerank/eval/$1.$2.json
  [ -s "$out" ] && { echo "exists: $out"; return; }
  $PY -m hybridsearch.rerank.rerank_run --name "$1" --split "$2" --candidates "$3" --depth "$4" \
     --model "$5" --backend "${6:-torch}" --candidate-label "${7:-}" ${EXTRA:-} 2>&1 \
     | grep -v -i -E "warn|Loading weights" | grep -E '^\{|"(RR@10|nDCG@10)"|pairs/s' | tail -4
}
case "${1:-quality}" in
quality)
  for m in $MODELS; do
    mp=$(model_path $m)
    [ "$m" != public ] && [ ! -d "$mp" ] && { echo "skip $m (no checkpoint)"; continue; }
    for s in dl19 dl20 dev; do
      rr "$m.bm25full" $s $R/anserini-bm25-default.$s.trec 100 "$mp" torch "Anserini BM25 default (k1=0.9,b=0.4), full 8.8M"
    done
    for s in dev dl19 dl20; do
      f=$R/anserini-bm25-default-1m.$s.trec
      [ -f "$f" ] && rr "$m.bm25-1m" ${s/dev/dev1m}$( [ $s != dev ] && echo _1m ) $f 100 "$mp" torch "Anserini BM25 default over the 1M subset"
      h=$(ls data/runs/hybrid/*1m*.$s.trec 2>/dev/null | head -1 || true)
      [ -n "$h" ] && rr "$m.hybrid-1m" ${s/dev/dev1m}$( [ $s != dev ] && echo _1m ) "$h" 100 "$mp" torch "hybrid 1M: $h"
    done
  done ;;
export)
  $PY -m hybridsearch.rerank.export --model data/rerank/runs/$CHOSEN/best --out data/models/reranker --int8 --tag reranker ;;
int8)
  for v in model.onnx model.int8.onnx; do
    n=$CHOSEN-ort-$( [ $v = model.onnx ] && echo fp32 || echo int8 )
    for s in ${INT8_SPLITS:-dl19 dl20}; do
      EXTRA="--threads 4 --batch 64" rr "$n.bm25full" $s $R/anserini-bm25-default.$s.trec 100 data/models/reranker/$v ort "Anserini BM25 default, full 8.8M"
    done
  done ;;
cascade)
  for s in dl19 dl20 dev; do
    rr "$CHOSEN-d200.bm25full" $s $R/anserini-bm25-default.$s.trec 200 data/rerank/runs/$CHOSEN/best torch "Anserini BM25 default, full 8.8M"
  done
  for s in ${INT8_SPLITS:-dl19 dl20}; do
    EXTRA="--threads 4 --batch 64" rr "$CHOSEN-int8-d200.bm25full" $s $R/anserini-bm25-default.$s.trec 200 data/models/reranker/model.int8.onnx ort "Anserini BM25 default, full 8.8M"
  done
  $PY -m hybridsearch.rerank.bench --model-dir data/models/reranker --name reranker
  $PY -m hybridsearch.rerank.cascade --name $CHOSEN-d200.bm25full --int8-name $CHOSEN-int8-d200.bm25full --bench reranker ;;
esac
