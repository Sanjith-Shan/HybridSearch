"""Interpolate cost (hops, distance computations, CPU us) at fixed recall@10 targets for
every series in a sweep jsonl: a load-independent comparison of build parameters.

    python -m hybridsearch.bench.matched_recall SWEEP.search.jsonl OUT.json [--targets 0.95 0.97 0.98 0.99]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

COST = ["hops_per_query", "dist_per_query", "cpu_us_per_query", "ssd_reads_per_query"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep"); ap.add_argument("out")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.97, 0.98, 0.99])
    a = ap.parse_args(argv)
    by = defaultdict(list)
    for line in open(a.sweep):
        r = json.loads(line)
        by[r["system"]].append(r)
    out = {"source": a.sweep, "method": "linear interpolation of cost vs recall@10 between measured L_search points",
           "targets": {}}
    for t in a.targets:
        row = {}
        for name, pts in by.items():
            pts = sorted(pts, key=lambda r: r["recall"])
            rec = [p["recall"] for p in pts]
            if rec[-1] < t:
                row[name] = None
                continue
            row[name] = {c: round(float(np.interp(t, rec, [p[c] for p in pts])), 1)
                         for c in COST if all(c in p for p in pts)}
        out["targets"][str(t)] = row
    json.dump(out, open(a.out, "w"), indent=2)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
