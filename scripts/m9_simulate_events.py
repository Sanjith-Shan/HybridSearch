#!/usr/bin/env python
"""Simulated experiment traffic -> /api/events JSONL -> the same analyser that runs on real logs.

    .venv/bin/python scripts/m9_simulate_events.py [--sessions 4000]

Writes the log to data/events_sim/simulated.jsonl (NOT data/events/, so it can never be
mistaken for broker traffic; every event also carries "simulated": <model>) and the
report to results/m9/events_report_simulated.{json,md}.
Real traffic: .venv/bin/python -m hybridsearch.clicks.analyze_events  (reads data/events/).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")  # shared laptop: <= 4 threads

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from hybridsearch.clicks import analyze_events as A  # noqa: E402
from hybridsearch.clicks import rankers as R  # noqa: E402
from hybridsearch.clicks.events import write_jsonl  # noqa: E402
from hybridsearch.clicks.models import make_model  # noqa: E402
from hybridsearch.clicks.simulate import simulate_experiment  # noqa: E402
from hybridsearch.data.io import DATA  # noqa: E402
from hybridsearch.rerank.meta import write_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--model", default="dbn-navigational")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    qrels, _ = R.load_dl_qrels()
    runs = R.discover_real_runs()
    bm25 = next((n for n in runs if "bm25" in n), None)
    runs = {k: v for k, v in runs.items() if k == bm25}
    runs["synthetic-oracle-sigma1.0"] = R.synthetic_noisy_oracle(qrels, 1.0, seed=1000)
    runs["synthetic-oracle-sigma1.25"] = R.synthetic_noisy_oracle(qrels, 1.25, seed=1000)
    table = R.build_table(runs, qrels, k=10)
    queries = R.load_dl_queries()
    ctrl = bm25 or "synthetic-oracle-sigma1.25"
    exps = [
        {"id": "sim-ab", "kind": "ab", "allocation": 0.5, "salt": "sim-ab-1", "control": ctrl,
         "treatment": "synthetic-oracle-sigma1.0"},
        {"id": "sim-aa", "kind": "aa", "allocation": 0.5, "salt": "sim-aa-1", "control": ctrl, "treatment": ctrl},
        {"id": "sim-interleave", "kind": "interleave", "allocation": 0.5, "salt": "sim-il-1",
         "control": ctrl, "treatment": "synthetic-oracle-sigma1.0"},
    ]
    kind, profile = args.model.split("-")
    model = make_model(kind, profile)
    events = []
    for e in exps:
        events += simulate_experiment(table, queries, e, model, args.sessions, rng)
    log = DATA / "events_sim/simulated.jsonl"
    write_jsonl(log, events)
    rep = A.analyse(events, {e["id"]: e for e in exps}, seed=args.seed)
    rep["source"] = str(log)
    rep["experiments_config"] = exps
    out = ROOT / "results/m9/events_report_simulated"
    Path(str(out) + ".json").write_text(json.dumps(rep, indent=2) + "\n")
    md = A.to_markdown(rep)
    Path(str(out) + ".md").write_text(md + "\n")
    meta = {"label": f"SIMULATED users (click model {model.name}, relevance from TREC DL 2019+2020 graded qrels)",
            "seed": args.seed, "sessions_per_experiment": args.sessions, "events": len(events),
            "experiments": exps}
    write_meta(str(out) + ".json", **meta)
    write_meta(str(out) + ".md", **meta)
    print(md)


if __name__ == "__main__":
    main()
