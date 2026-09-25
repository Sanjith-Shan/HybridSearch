#!/usr/bin/env bash
# M4 hybrid retrieval study, end to end, one command:
#   1. BM25 runs on the 1M MS MARCO subset with HybridSearch's own engine (dev, dl19, dl20, train_tune)
#   2. BEIR scifact/nfcorpus/fiqa: build lexical indexes with hs_index_build, BM25 runs with hs_lex_search
#   3. fusion unit tests
#   4. tune RRF / weighted fusion on train_tune ONLY, evaluate on dev/DL19/DL20/BEIR, significance
#      (paired randomization + Holm, bootstrap CIs), and fuse with the approximate (DiskANN-style)
#      dense run at its default operating point (L=100, W=4) if that run file exists.
# Outputs: data/runs/m4/*.trec (gitignored), results/m4/fusion.{md,json} + .meta.json
# Needs: engine/build-lex (scripts/build_engine.sh), .venv, dense reference runs (scripts/dense_reference.sh).
# Env: THREADS (default 7), N_PERM (default 20000), APPROX_DEV (override the DiskANN dev run path).
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS=${THREADS:-7}
N_PERM=${N_PERM:-20000}
PY=.venv/bin/python
B=engine/build-lex
avail_gb=$(df -g data | awk 'NR==2{print $4}')
if [ "$avail_gb" -lt 8 ]; then echo "refusing: ${avail_gb} GB free (the fused runs take ~2 GB; keep >= 6 GB headroom)"; exit 1; fi
mkdir -p data/runs/m4 results/m4

echo "=== 1. BM25 on the 1M subset (HybridSearch engine) $(date +%T)"
for s in dev:data/raw/msmarco/queries.dev.small.tsv dl19:data/raw/trec-dl/dl19.queries.tsv \
         dl20:data/raw/trec-dl/dl20.queries.tsv train_tune:data/subset/train_tune/queries.tsv; do
  name=${s%%:*}; q=${s#*:}
  $B/hs_lex_search --index data/indexes/lexical/1m --queries "$q" --out "data/runs/m4/hs-bm25-1m.$name.trec" \
    --algo exhaustive --model lucene --k 1000 --threads "$THREADS" --tag hs-bm25-1m
done

echo "=== 2. BEIR BM25 (HybridSearch engine) $(date +%T)"
$PY scripts/m4_beir_bm25.py --threads "$THREADS" --force

echo "=== 3. tests $(date +%T)"
$PY -m pytest -q py/tests/test_fusion.py py/tests/test_fusion_fast.py

echo "=== 4. fusion study $(date +%T)"
approx=()
if [ -n "${APPROX_DEV:-}" ]; then
  approx=(--approx-dense "dev=$APPROX_DEV")
else
  # serving topology first: 4 modulo shards merged (scripts/vector), then single-index variants
  for f in results/vector/runs/shards_mod_1m.dev.W4.L100.trec results/vector/runs/disk_1m.dev.W4.L100.trec \
           results/vector/runs/diskpart_1m.dev.W4.L100.trec; do
    if [ -s "$f" ]; then approx=(--approx-dense "dev=$f"); break; fi
  done
fi
[ ${#approx[@]} -gt 0 ] && echo "approximate dense run: ${approx[1]}" || echo "no DiskANN dev run found; exact dense only"
PYTHONPATH=py $PY scripts/m4_fusion.py --n-perm "$N_PERM" ${approx[@]+"${approx[@]}"} ${approx[@]+--approx-label DiskANN-L100-W4}
echo "=== done $(date +%T): results/m4/fusion.md"
