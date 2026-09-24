#!/usr/bin/env bash
# M0 reference runs: Pyserini/Anserini BM25 default (k1=0.9, b=0.4) on the prebuilt
# msmarco-v1-passage Lucene index (MD5 678876e8c99a89933d553609a0fd8793), top-1000.
#   scripts/reference_bm25.sh [split ...]   splits: dev dl19 dl20 train_tune train50k
set -euo pipefail
cd "$(dirname "$0")/.."
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home}
INDEX=${INDEX:-$HOME/.cache/pyserini/indexes/manual/lucene-inverted.msmarco-v1-passage.20221004.252b5e}
THREADS=${THREADS:-4}
OUT=data/runs/reference; mkdir -p "$OUT"
spec() {  # split -> "topics_file run_name"
  case "$1" in
    dev) echo "data/raw/msmarco/queries.dev.small.tsv anserini-bm25-default.dev.trec" ;;
    dl19) echo "data/raw/trec-dl/dl19.queries.tsv anserini-bm25-default.dl19.trec" ;;
    dl20) echo "data/raw/trec-dl/dl20.queries.tsv anserini-bm25-default.dl20.trec" ;;
    train_tune) echo "data/subset/train_tune/queries.tsv anserini-bm25-default.train_tune.trec" ;;
    train50k) echo "data/subset/train50k/queries.tsv bm25.train50k.trec" ;;
    *) echo "unknown split $1" >&2; exit 1 ;;
  esac
}
splits="$*"; [ -z "$splits" ] && splits="dev dl19 dl20 train_tune train50k"
for s in $splits; do
  set -- $(spec "$s"); topics=$1; out="$OUT/$2"
  [ -s "$out" ] && { echo "exists: $out"; continue; }
  if [ "$s" = train50k ]; then
    # 5 resumable parts of 10,000 queries, concatenated at the end
    mkdir -p "$OUT/train50k.parts"
    [ -f "$OUT/train50k.parts/q.a.tsv" ] || { split -l 10000 -a 1 "$topics" "$OUT/train50k.parts/q."
      for f in "$OUT"/train50k.parts/q.?; do mv "$f" "$f.tsv"; done; }   # pyserini needs .tsv
    for q in "$OUT"/train50k.parts/q.?.tsv; do
      p=${q%.tsv}; part="$OUT/train50k.parts/run.${p##*.}.trec"
      [ -s "$part" ] && continue
      .venv/bin/python -m pyserini.search.lucene --index "$INDEX" --topics "$q" --output "$part.tmp" \
        --output-format trec --bm25 --k1 0.9 --b 0.4 --hits 1000 --threads "$THREADS" --batch-size 128 2>&1 | grep -vE "it/s\]|Warning" | tail -1
      mv "$part.tmp" "$part"; echo "train50k part ${p##*.} done $(date +%H:%M:%S)"
    done
    cat "$OUT"/train50k.parts/run.?.trec > "$out.tmp" && mv "$out.tmp" "$out" && rm -rf "$OUT/train50k.parts"
    echo "train50k -> $out"; continue
  fi
  start=$(date +%s)
  .venv/bin/python -m pyserini.search.lucene --index "$INDEX" --topics "$topics" \
    --output "$out.tmp" --output-format trec --bm25 --k1 0.9 --b 0.4 --hits 1000 \
    --threads "$THREADS" --batch-size 128 2>&1 | grep -vE "it/s\]|Warning" | tail -3
  mv "$out.tmp" "$out"
  echo "$s -> $out ($(( $(date +%s) - start ))s)"
done
