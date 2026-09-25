#!/usr/bin/env bash
# M9 (the feedback loop): every result under results/m9/, in order.
# All clicks are from SIMULATED users (click models; relevance from TREC DL graded qrels).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/m9_make_golden.py                        # tests/golden/interleaving.json
$PY -m pytest -q py/tests/test_clicks_models.py py/tests/test_clicks_ab.py \
    py/tests/test_clicks_analyze_events.py py/tests/test_interleaving.py py/tests/test_ltr.py
$PY -u scripts/m9_make_runs.py                       # BGE rerank / RRF runs over BM25 top-100 -> data/runs/m9/
$PY -u scripts/m9_sensitivity.py --pool 2e6 --trials 1000 "$@"                # sensitivity.{json,md,png}, sensitivity_n80.png, aa.json (A/A table is in sensitivity.md)
$PY -u scripts/m9_ltr.py                             # ltr.{json,md}, ltr_learning_curve.png
$PY -u scripts/m9_ltr.py --no-dense --tag _bm25only  # ablation
$PY -u scripts/m9_ltr.py --synthetic                 # ltr_synthetic.* (controlled stand-in)
$PY -u scripts/m9_simulate_events.py                 # simulated event log -> events report
