#!/usr/bin/env bash
# M0: deterministic 1M subset (seed 20260923) + train_tune (2,000) + train50k queries.
# Writes data/subset/1m/READY when done.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=py .venv/bin/python -m hybridsearch.data.subset
