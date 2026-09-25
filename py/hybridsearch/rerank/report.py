"""Collect results/rerank/eval/*.json into one table (results/rerank/summary.md + .json).

    python -m hybridsearch.rerank.report
"""
from __future__ import annotations

import json

from hybridsearch.data.io import REPO_ROOT
from hybridsearch.rerank.meta import write_meta

RESULTS = REPO_ROOT / "results/rerank"


def main() -> None:
    rows = []
    for p in sorted((RESULTS / "eval").glob("*.json")):
        if p.name.endswith(".meta.json"):
            continue
        d = json.loads(p.read_text())
        rows.append({
            "file": f"results/rerank/eval/{p.name}", "name": d["name"], "split": d["split"],
            "candidates": d.get("candidate_label") or d["candidates"], "depth": d["depth"],
            "queries": d["n_queries_qrels"], "backend": f"{d['backend']}/{d['device']}",
            "metrics": d["metrics"], "first_stage": d["first_stage_metrics_at_depth"],
        })
    (RESULTS / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| model.candidates | split | queries | depth | backend | reranked | first stage | file |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["split"], r["name"])):
        fmt = lambda m: ", ".join(f"{k} {v:.4f}" for k, v in m.items())  # noqa: E731
        lines.append(f"| {r['name']} | {r['split']} | {r['queries']} | {r['depth']} | {r['backend']} | "
                     f"{fmt(r['metrics'])} | {fmt(r['first_stage'])} | {r['file']} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    write_meta(RESULTS / "summary.md", what="aggregation of results/rerank/eval/*.json (no new measurement)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
