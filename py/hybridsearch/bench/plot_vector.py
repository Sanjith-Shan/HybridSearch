"""Plots for results/vector/: recall@10 vs QPS curves and friends (matplotlib PNG).

    python -m hybridsearch.bench.plot_vector build-sweep results/vector/build_sweep_100000.search.jsonl
    python -m hybridsearch.bench.plot_vector compare OUT.png FILE.jsonl [FILE.jsonl ...]
    python -m hybridsearch.bench.plot_vector disk OUT_PREFIX disk.jsonl [mrr.json]

Colors: fixed categorical order (never cycled), plus a distinct marker per series
so identity never rests on color alone.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
YMETRIC = os.environ.get("HS_PLOT_Y", "cpu_qps")
YLABEL = {"cpu_qps": "queries / CPU-second, 1 thread (log)", "qps": "queries / s wall, 1 thread (log)",
          "dist_per_query": "distance computations / query (log)",
          "hops_per_query": "graph hops (nodes expanded) / query (log)"}.get(YMETRIC, YMETRIC)
LABEL = "dev-signal-only: macOS laptop, unpinned, shared/loaded machine"


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def curves(ax, series: dict, x="recall", y=None):
    y = y or YMETRIC
    for i, (name, pts) in enumerate(series.items()):
        pts = sorted([p for p in pts if p.get(y) is not None], key=lambda r: r[x])
        ax.plot([p[x] for p in pts], [p[y] for p in pts], color=PALETTE[i % 8], marker=MARKERS[i % 8],
                markersize=5, linewidth=2, label=name)
    if len(series) > 1:
        ax.legend(fontsize=8, frameon=False, labelcolor=INK)


def save(fig, out):
    fig.text(0.01, 0.005, LABEL, fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


def build_sweep(path):
    data = rows(path)
    by = defaultdict(list)
    for r in data:
        by[r["system"]].append(r)
    groups = {
        "Max degree R (L_build=100, α=1.2)": ["R16-L100-a1.2", "R32-L100-a1.2", "R64-L100-a1.2", "R96-L100-a1.2"],
        "Build list size L_build (R=64, α=1.2)": ["R64-L50-a1.2", "R64-L100-a1.2", "R64-L200-a1.2"],
        "α and passes (R=64, L_build=100)": ["R64-L100-a1.0", "R64-L100-a1.2", "R64-L100-a1.4",
                                            "R64-L100-a1.2-1pass"],
    }
    n = data[0]["n"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, (title, names) in zip(axes, groups.items()):
        curves(ax, {k: by[k] for k in names if k in by})
        ax.set_yscale("log")
        style(ax, title, "recall@10 (vs brute force)", YLABEL)
    fig.suptitle(f"Vamana build parameters, {n:,} BGE-base passages, train_tune queries: {YLABEL}", color=INK,
                 fontsize=12, x=0.01, ha="left")
    save(fig, str(Path(path).with_suffix("")).replace(".search", "") + f".{YMETRIC}.png")


def series_key(r):
    s = r["system"]
    if s.startswith("hnswlib") or s.startswith("faiss-hnsw"):
        return f"{s} M={r['M']}"
    if s.startswith("faiss-ivfpq"):
        return f"{s} (nlist={r['nlist']}, PQ{r['pq_M']})" + (f" k×{r['refine']}" if r.get("refine") else "")
    if "W" in r:
        return f"{s} W={r['W']} {r['mode']}" + (f" cache={r['cache_nodes']}" if r.get("cache_nodes") else "")
    return s


def compare(out, *paths):
    by = defaultdict(list)
    for p in paths:
        for r in rows(p):
            if r["system"] == "faiss-flat":
                continue
            by[series_key(r)].append(r)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    curves(ax, dict(list(by.items())[:8]))
    ax.set_yscale("log")
    n = next(iter(by.values()))[0]["n"]
    style(ax, f"recall@10 vs QPS, {n:,} vectors, single query thread", "recall@10 (vs brute force)",
          YLABEL)
    flat = [r for p in paths for r in rows(p) if r["system"] == "faiss-flat"]
    if flat:
        ax.axhline(flat[0][YMETRIC], color=INK2, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[0], flat[0][YMETRIC] * 1.1, f"exact (faiss IndexFlatIP) {flat[0][YMETRIC]:.0f}",
                fontsize=8, color=INK2)
    save(fig, out)


def disk(prefix, path, mrr_path=None):
    data = rows(path)
    by = defaultdict(list)
    for r in data:
        by[series_key(r)].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    curves(axes[0], by, y="qps")  # wall clock: CPU time would hide the SSD waits
    axes[0].set_yscale("log")
    style(axes[0], "recall@10 vs QPS (wall clock, includes SSD waits)", "recall@10", "queries / s wall, 1 thread (log)")
    curves(axes[1], by, x="ssd_reads_per_query", y="recall")
    style(axes[1], "recall@10 vs 4 KB SSD reads per query", "SSD reads / query", "recall@10")
    save(fig, prefix + ".png")
    if mrr_path:
        m = json.loads(Path(mrr_path).read_text())
        pts = [r for r in m["runs"] if r.get("operating_point")]
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
        grp = defaultdict(list)
        for r in pts:
            op = r["operating_point"]
            grp[series_key(op)].append({"recall": op["recall"], "mrr": r["metrics"]["RR@10"]})
        curves(ax, grp, x="recall", y="mrr")
        ref = m["reference_metrics"]["RR@10"]
        ax.axhline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[0], ref + 0.0005, f"exact search MRR@10 {ref:.4f}", fontsize=8, color=INK2)
        style(ax, "MS MARCO dev MRR@10 at each operating point", "recall@10 vs exact search (dev queries)", "MRR@10 (dev)")
        save(fig, prefix + ".mrr.png")


def shards(prefix, check_path, mrr_path=None):
    """Merged 4-shard results from hs_vec_shard_check (+ mrr_points json)."""
    data = [r for r in rows(check_path) if "L" in r]
    by = defaultdict(list)
    for r in data:
        by[f"{r['shards']} shards, W={r['W']} {r['mode']}"].append(r)
    fig, axes = plt.subplots(1, 2 if mrr_path else 1, figsize=(13 if mrr_path else 7, 4.8), facecolor=SURFACE)
    axes = axes if mrr_path else [axes]
    curves(axes[0], by, x="ssd_reads_per_query_all_shards", y="recall")
    style(axes[0], "Merged recall@10 vs 4 KB reads per query (sum over shards)", "SSD reads / query, all shards",
          "recall@10 vs global brute force")
    if mrr_path:
        m = json.loads(Path(mrr_path).read_text())
        pts = sorted([r for r in m["runs"] if r.get("operating_point")], key=lambda r: r["operating_point"]["L"])
        xs = [r["operating_point"]["recall"] for r in pts]
        ys = [r["metrics"]["RR@10"] for r in pts]
        ax = axes[1]
        ax.plot(xs, ys, color=PALETTE[0], marker="o", markersize=5, linewidth=2)
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(f"L={r['operating_point']['L']}", (x, y), textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color=INK2)
        ref = m["reference_metrics"]["RR@10"]
        ax.axhline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.text(min(xs), ref + 0.0004, f"exact search {ref:.4f}", fontsize=8, color=INK2)
        style(ax, "MS MARCO dev MRR@10 vs merged recall@10", "recall@10 vs exact (dev queries)", "MRR@10 (dev)")
    save(fig, prefix + ".png")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"build-sweep": build_sweep, "compare": compare, "disk": disk, "shards": shards}[cmd](*args)
