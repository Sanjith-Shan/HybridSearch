"""MRR@10 at each ANN operating point against the exact-search reference on the
same subset: how much retrieval quality the approximation costs.

    python -m hybridsearch.bench.mrr_points --runs RUN.trec [RUN.trec ...] \
        --qrels data/subset/1m/qrels.dev.tsv \
        --reference data/runs/reference/dense-flat-1m.dev.trec \
        --points results/vector/x.jsonl --out results/vector/mrr_dev_1m.json

For every run: RR@10 (MRR@10) and R@10 against qrels, the delta to the reference
with a paired randomization test (two-sided, 10k sign flips) and a paired
bootstrap 95% CI, and "overlap@10": the mean fraction of the reference's top 10
the run returns (= recall@10 on dev queries against exact search). If --points is
given, the matching operating-point row (by run tag) is attached (recall on the
tuning queries, QPS, SSD reads...).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.io import read_qrels, read_run
from ..eval.metrics import evaluate, rank_run
from ..stats.significance import paired_bootstrap_ci, paired_randomization_test

METRICS = ["RR@10", "R@10", "nDCG@10"]


def overlap_at_10(run, ref) -> float:
    tot, n = 0.0, 0
    for qid, scores in ref.items():
        top_ref = set(rank_run(scores)[:10])
        top_run = set(rank_run(run.get(qid, {}))[:10])
        tot += len(top_ref & top_run) / max(1, len(top_ref))
        n += 1
    return tot / max(1, n)


def tag_of(path: str) -> str:
    with open(path) as f:
        return f.readline().split()[5]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hybridsearch.bench.mrr_points")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--points", nargs="*", default=[], help="jsonl files with operating-point rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    qrels = read_qrels(a.qrels)
    ref_run = read_run(a.reference)
    ref_agg, ref_pq = evaluate(ref_run, qrels, METRICS)
    points = {}
    for p in a.points:
        for line in Path(p).read_text().splitlines():
            row = json.loads(line)
            if "L" in row:
                key = f"{row['system']}-W{row['W']}-L{row['L']}" if "W" in row else f"{row['system']}-L{row['L']}"
                points[key] = row
    out = {"qrels": a.qrels, "reference": a.reference, "n_queries": len(qrels),
           "reference_metrics": {k: round(v, 5) for k, v in ref_agg.items()}, "runs": []}
    print(f"reference {a.reference}: " + ", ".join(f"{k}={v:.4f}" for k, v in ref_agg.items()))
    for path in a.runs:
        run = read_run(path)
        agg, pq = evaluate(run, qrels, METRICS)
        qids = sorted(qrels)
        x = [pq["RR@10"][q] for q in qids]
        y = [ref_pq["RR@10"][q] for q in qids]
        p = paired_randomization_test(x, y)
        _, lo, hi = paired_bootstrap_ci(x, y)
        lo, hi = round(lo, 5), round(hi, 5)
        tag = tag_of(path)
        row = {"run": path, "tag": tag, "metrics": {k: round(v, 5) for k, v in agg.items()},
               "delta_mrr10_vs_exact": round(agg["RR@10"] - ref_agg["RR@10"], 5),
               "delta_ci95": [lo, hi], "p_value_randomization": p,
               "overlap10_with_exact": round(overlap_at_10(run, ref_run), 5),
               "operating_point": points.get(tag)}
        out["runs"].append(row)
        print(f"{tag}: MRR@10={agg['RR@10']:.4f} (Δ {row['delta_mrr10_vs_exact']:+.4f}, p={p:.3g}) "
              f"overlap@10={row['overlap10_with_exact']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
