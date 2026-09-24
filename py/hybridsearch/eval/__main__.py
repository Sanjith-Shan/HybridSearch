"""CLI: python -m hybridsearch.eval run.trec qrels [--metrics RR@10 nDCG@10 ...]
                                   [--per-query out.tsv] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys

from ..data.io import read_qrels, read_run
from .metrics import DEFAULT_METRICS, evaluate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hybridsearch.eval")
    ap.add_argument("run")
    ap.add_argument("qrels")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                    help="RR@k, nDCG@k, R@k; add (rel=N) for a relevance threshold, e.g. 'R(rel=2)@1000'")
    ap.add_argument("--per-query", help="write qid<TAB>metric<TAB>value rows here")
    ap.add_argument("--json", help="write aggregate metrics as JSON here")
    args = ap.parse_args(argv)

    run = read_run(args.run)
    qrels = read_qrels(args.qrels)
    agg, per_q = evaluate(run, qrels, args.metrics)
    nq = len(qrels)
    missing = sum(1 for q in qrels if q not in run)
    print(f"# run={args.run} qrels={args.qrels} queries_in_qrels={nq} missing_from_run={missing}")
    for name, v in agg.items():
        print(f"{name}\t{v:.4f}\t(n={len(per_q[name])})")
    if args.per_query:
        with open(args.per_query, "w") as f:
            f.write("qid\tmetric\tvalue\n")
            for name, vals in per_q.items():
                f.writelines(f"{qid}\t{name}\t{vals[qid]:.6f}\n" for qid in sorted(vals, key=lambda x: (len(x), x)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"run": args.run, "qrels": args.qrels, "n_queries": nq,
                       "missing_from_run": missing,
                       "metrics": {k: round(v, 6) for k, v in agg.items()},
                       "n_per_metric": {k: len(v) for k, v in per_q.items()}}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
