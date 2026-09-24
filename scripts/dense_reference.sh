#!/usr/bin/env bash
# After data/embeddings/1m/READY: parity check vs sentence-transformers, exact FAISS
# flat runs + dev ground truth, BEIR embeddings + flat runs, then score everything.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=py
test -f data/embeddings/1m/READY || { echo "embeddings not ready"; exit 1; }
.venv/bin/python -m hybridsearch.dense.parity --n 200
.venv/bin/python -m hybridsearch.dense.flat --threads 8
.venv/bin/python -m hybridsearch.dense.beir --threads 8
.venv/bin/python -m hybridsearch.eval.reference
