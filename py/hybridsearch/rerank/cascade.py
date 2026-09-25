"""Cascade curve: rerank depth k -> quality gained vs CPU latency added.

Quality at depth k: the first stage's top-k is reordered by the cross-encoder, the rest of
the first-stage list stays below in its original order. Computed from the cached scores
of one depth-``max(k)`` rerank pass (``data/rerank/scores/<name>.<split>.npz``), so every
k uses exactly the same model scores. Latency at k: ORT CPU ``session.run`` p50 for one
request of k pairs, from ``results/rerank/bench_<bench>.json`` (macOS: dev-signal-only).

    python -m hybridsearch.rerank.cascade --name ours.bm25full --bench ours \
        --splits dev dl19 dl20 --int8-name ours-int8.bm25full
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hybridsearch.data.io import REPO_ROOT, read_qrels
from hybridsearch.eval.metrics import evaluate
from hybridsearch.rerank.meta import write_meta
from hybridsearch.rerank.rerank_run import SCORES, SPLITS, metrics_for

RESULTS = REPO_ROOT / "results/rerank"
DEPTHS = [10, 20, 50, 100, 200]


def cascade_quality(npz: Path, qrels, depths, metrics) -> dict[int, dict]:
    z = np.load(npz)
    qids, pids, sc, fs = z["qid"], z["pid"], z["score"], z["first_stage"]
    per_q: dict[str, list[tuple[str, float, float]]] = {}
    for q, p, s, f in zip(qids, pids, sc, fs):
        per_q.setdefault(str(q), []).append((str(p), float(s), float(f)))  # first-stage order
    avail = min(len(v) for v in per_q.values())
    out = {0: evaluate({q: {p: f for p, _, f in v} for q, v in per_q.items()}, qrels, metrics)[0]}
    for k in depths:
        if k > max(len(v) for v in per_q.values()):
            continue
        run = {}
        for q, v in per_q.items():
            head = sorted(v[:k], key=lambda x: -x[1])
            n = len(v)
            # reranked head gets ranks 1..k, the tail keeps first-stage order below it
            run[q] = {p: float(n - i) for i, (p, _, _) in enumerate(head + v[k:])}
        out[k] = evaluate(run, qrels, metrics)[0]
    out["min_candidates_per_query"] = avail
    return out


def latency(bench: Path, variant: str, k: int, threads: int = 7) -> float | None:
    if not bench.exists():
        return None
    rows = json.loads(bench.read_text())["results"].get(f"{variant}|threads={threads}", [])
    for r in rows:
        if r["batch"] == k:
            return r["p50_ms"]
    return None


def plot(table: list[dict], splits: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fp32": "#2a78d6", "int8": "#eb6834"}
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.6), squeeze=False)
    for ax, split in zip(axes[0], splits):
        metric = "RR@10" if not split.startswith("dl") else "nDCG@10"
        for var in ("fp32", "int8"):
            pts = [r for r in table if r["split"] == split and r["variant"] == var
                   and r["latency_p50_ms"] is not None and r["k"] > 0]
            if not pts:
                continue
            x = [r["latency_p50_ms"] for r in pts]
            y = [r[metric] for r in pts]
            ax.plot(x, y, "-o", color=colors[var], lw=2, ms=6, label=f"{var} ORT CPU")
            for r, xi, yi in zip(pts, x, y):
                if var == "fp32":
                    ax.annotate(f"k={r['k']}", (xi, yi), textcoords="offset points", xytext=(4, -12),
                                fontsize=8, color="#52514e")
        base = [r for r in table if r["split"] == split and r["k"] == 0]
        if base:
            ax.axhline(base[0][metric], color="#8a8985", lw=1, ls="--")
            ax.annotate("first stage only", (0, base[0][metric]), textcoords="offset points",
                        xytext=(2, 3), fontsize=8, color="#52514e", xycoords=("axes fraction", "data"))
        ax.set_xscale("log")
        ax.set_xlabel("rerank latency added, p50 ms (log)", fontsize=9)
        ax.set_ylabel(f"{split} {metric}", fontsize=9)
        ax.set_title(split, fontsize=10, loc="left")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Rerank depth k: quality vs CPU latency (M3 Pro, macOS, dev-signal-only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="score cache name of the fp32 model")
    ap.add_argument("--int8-name", default=None, help="score cache name of the int8 model")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "dl19", "dl20"])
    ap.add_argument("--threads", type=int, default=7)
    args = ap.parse_args()
    bench = RESULTS / f"bench_{args.bench}.json"
    table = []
    for split in args.splits:
        qrels = read_qrels(REPO_ROOT / SPLITS[split][1])
        ms = metrics_for(split)
        for var, name, onnx in (("fp32", args.name, "model.onnx"), ("int8", args.int8_name, "model.int8.onnx")):
            if not name or not (SCORES / f"{name}.{split}.npz").exists():
                continue
            q = cascade_quality(SCORES / f"{name}.{split}.npz", qrels, DEPTHS, ms)
            for k in [0, *DEPTHS]:
                if k not in q:
                    continue
                row = {"split": split, "variant": var, "k": k,
                       "latency_p50_ms": 0.0 if k == 0 else latency(bench, onnx, k, args.threads)}
                row.update({m: round(v, 4) for m, v in q[k].items()})
                table.append(row)
    out = RESULTS / "cascade.json"
    out.write_text(json.dumps(table, indent=2) + "\n")
    lines = ["| split | variant | k | CPU p50 ms added | metrics |", "|---|---|---|---|---|"]
    for r in table:
        mets = ", ".join(f"{k} {v:.4f}" for k, v in r.items() if "@" in k)
        lat = "-" if r["latency_p50_ms"] is None else f"{r['latency_p50_ms']:.1f}"
        lines.append(f"| {r['split']} | {r['variant']} | {r['k'] or 'none (first stage)'} | {lat} | {mets} |")
    (RESULTS / "cascade.md").write_text("\n".join(lines) + "\n")
    plot(table, args.splits, RESULTS / "cascade.png")
    for p in ("cascade.json", "cascade.md", "cascade.png"):
        write_meta(RESULTS / p, fp32_scores=args.name, int8_scores=args.int8_name, bench=str(bench),
                   threads=args.threads, timing_label="dev-signal-only (macOS)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
