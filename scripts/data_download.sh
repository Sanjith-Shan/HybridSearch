#!/usr/bin/env bash
# M0: download MS MARCO passage v1 + TREC DL 2019/2020 passage queries & qrels into
# data/downloads, then verify MD5s and lay out data/raw via hybridsearch.data.prepare.
set -euo pipefail
cd "$(dirname "$0")/.."
DL=data/downloads; mkdir -p "$DL"
BASE=https://msmarco.z22.web.core.windows.net/msmarcoranking
avail_gb=$(df -g data | awk 'NR==2{print $4}')
if [ "$avail_gb" -lt 7 ]; then echo "refusing: ${avail_gb} GB free (need ~4 GB + 3 GB headroom)"; exit 1; fi

# The 1.06 GB tarball, fetched as 8 parallel byte ranges (single stream ran at ~1 MB/s).
if [ ! -f "$DL/collectionandqueries.tar.gz" ] && [ ! -f "$DL/collectionandqueries.verified" ]; then
  N=$(curl -sIL "$BASE/collectionandqueries.tar.gz" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
  P=8; S=$(( N / P + 1 ))
  for i in $(seq 0 $((P-1))); do
    s=$((i*S)); e=$((s+S-1)); [ $e -ge $N ] && e=$((N-1))
    curl -sSL --retry 10 -r $s-$e -o "$DL/cq.part$i" "$BASE/collectionandqueries.tar.gz" &
  done
  wait
  cat "$DL"/cq.part{0..7} > "$DL/collectionandqueries.tar.gz" && rm "$DL"/cq.part*
fi
for f in msmarco-test2019-queries.tsv.gz msmarco-test2020-queries.tsv.gz; do
  [ -f "$DL/$f" ] || curl -sSL --retry 5 -o "$DL/$f" "$BASE/$f"
done
for y in 2019 2020; do
  [ -f "$DL/${y}qrels-pass.txt" ] || curl -sSL --retry 5 -o "$DL/${y}qrels-pass.txt" "https://trec.nist.gov/data/deep/${y}qrels-pass.txt"
done
.venv/bin/python -m hybridsearch.data.prepare
