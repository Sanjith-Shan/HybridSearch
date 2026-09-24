#!/usr/bin/env bash
# BGE-base-en-v1.5 embeddings for the 1M subset + query sets, on MPS, in the background.
# Resumable (re-run the same command after a crash). Holds data/locks/mps while running.
# Writes data/embeddings/1m/READY when passages.fbin is complete.
#   progress: tail -f data/embeddings/1m/encode.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/embeddings/1m
nohup env PYTHONPATH=py .venv/bin/python -m hybridsearch.dense.encode "${1:-all}" >> data/embeddings/1m/encode.log 2>&1 &
echo "started pid $!; log data/embeddings/1m/encode.log"
