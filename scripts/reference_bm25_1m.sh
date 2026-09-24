#!/usr/bin/env bash
# Subset-level lexical reference: index the 1M subset with Pyserini/Anserini
# (DefaultLuceneDocumentGenerator, default Lucene EnglishAnalyzer — the same settings as
# the msmarco-v1-passage regression, minus -storeRaw) and run BM25 default (k1=0.9, b=0.4)
# top-1000 for dev, dl19, dl20, train_tune.
# Index: data/indexes/anserini/1m   Runs: data/runs/reference/anserini-bm25-default-1m.{split}.trec
set -euo pipefail
cd "$(dirname "$0")/.."
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}
THREADS=${THREADS:-4}
IDX=data/indexes/anserini/1m
avail_gb=$(df -g data | awk 'NR==2{print $4}')
if [ "$avail_gb" -lt 4 ]; then echo "refusing: ${avail_gb} GB free (need ~1 GB + 3 GB headroom)"; exit 1; fi
if [ ! -f "$IDX/READY" ]; then
  J=data/indexes/anserini/1m-jsonl; mkdir -p "$J"
  .venv/bin/python - <<'PY'
import json
with open("data/subset/1m/collection.tsv", encoding="utf-8") as fin, open("data/indexes/anserini/1m-jsonl/docs.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        pid, text = line.rstrip("\n").split("\t", 1)
        fout.write(json.dumps({"id": pid, "contents": text}) + "\n")
PY
  rm -rf "$IDX"
  .venv/bin/python -m pyserini.index.lucene --collection JsonCollection --input "$J" --index "$IDX" \
    --generator DefaultLuceneDocumentGenerator --threads "$THREADS" --storePositions --storeDocvectors 2>&1 | grep -E "Total|indexed|ERROR|Exception" | tail -5
  rm -rf "$J"
  date > "$IDX/READY"
fi
spec() {
  case "$1" in
    dev) echo data/raw/msmarco/queries.dev.small.tsv ;;
    dl19) echo data/raw/trec-dl/dl19.queries.tsv ;;
    dl20) echo data/raw/trec-dl/dl20.queries.tsv ;;
    train_tune) echo data/subset/train_tune/queries.tsv ;;
  esac
}
splits="$*"; [ -z "$splits" ] && splits="dev dl19 dl20 train_tune"
for s in $splits; do
  out=data/runs/reference/anserini-bm25-default-1m.$s.trec
  [ -s "$out" ] && { echo "exists: $out"; continue; }
  .venv/bin/python -m pyserini.search.lucene --index "$IDX" --topics "$(spec $s)" --output "$out.tmp" \
    --output-format trec --bm25 --k1 0.9 --b 0.4 --hits 1000 --threads "$THREADS" --batch-size 128 2>&1 | grep -vE "it/s\]|Warning" | tail -2
  mv "$out.tmp" "$out"
done
