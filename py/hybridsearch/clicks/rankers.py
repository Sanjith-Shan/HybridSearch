"""Rankers and graded judgments for the M9 simulations.

Sources, in order of preference:

* TREC DL 2019 / 2020 graded qrels: ``data/raw/trec-dl/dl{19,20}.qrels`` (falls back to
  the raw NIST downloads ``data/downloads/20{19,20}qrels-pass.txt``).
* Real run files: ``data/runs/reference/*.dl19.trec`` and ``*.dl20.trec`` (Anserini BM25,
  BGE flat dense, ...). A ranker is used only if it has both a DL19 and a DL20 run.
* Fusion runs built here: reciprocal rank fusion (Cormack et al. 2009, k=60) of any two.
* Synthetic "noisy-oracle" rankers: score = DL grade + sigma * N(0,1) over the judged
  passages plus random unjudged fillers. Their quality ordering is known by
  construction. They are labelled ``synthetic-*`` everywhere.

nDCG@10 follows trec_eval (gain = grade, log2 discount, ideal from all judged grades,
unjudged = 0).
"""
from __future__ import annotations

import glob
import gzip
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hybridsearch.data.io import DATA, read_qrels, read_run

QRELS_PATHS = {
    "dl19": [DATA / "raw/trec-dl/dl19.qrels", DATA / "downloads/2019qrels-pass.txt"],
    "dl20": [DATA / "raw/trec-dl/dl20.qrels", DATA / "downloads/2020qrels-pass.txt"],
}
QUERY_PATHS = {
    "dl19": [DATA / "raw/trec-dl/dl19.queries.tsv", DATA / "downloads/msmarco-test2019-queries.tsv.gz"],
    "dl20": [DATA / "raw/trec-dl/dl20.queries.tsv", DATA / "downloads/msmarco-test2020-queries.tsv.gz"],
}
RUN_DIR = DATA / "runs/reference"


def _first_existing(paths):
    for p in paths:
        if Path(p).exists() and Path(p).stat().st_size > 0:
            return Path(p)
    return None


def load_dl_qrels() -> tuple[dict[str, dict[str, int]], dict]:
    """Merged DL19+DL20 qrels (topic IDs are disjoint). Returns (qrels, provenance)."""
    merged, prov = {}, {}
    for name, paths in QRELS_PATHS.items():
        p = _first_existing(paths)
        if p is None:
            raise FileNotFoundError(f"no qrels for {name}: tried {paths}")
        q = read_qrels(p)
        overlap = set(q) & set(merged)
        if overlap:
            raise ValueError(f"topic overlap between qrels: {sorted(overlap)[:5]}")
        merged.update(q)
        prov[name] = {"path": str(p), "topics": len(q), "judgments": sum(len(v) for v in q.values())}
    return merged, prov


def load_dl_queries() -> dict[str, str]:
    out = {}
    for paths in QUERY_PATHS.values():
        p = _first_existing(paths)
        if p is None:
            continue
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    qid, text = line.rstrip("\n").split("\t", 1)
                    out[qid] = text
    return out


# ------------------------------------------------------------------ run handling
def ranked(run_q: dict[str, float]) -> list[str]:
    """Score desc, ties by docid ascending (numeric when possible)."""
    def key(item):
        d, s = item
        return (-s, int(d) if d.isdigit() else d)
    return [d for d, _ in sorted(run_q.items(), key=key)]


M9_RUN_DIR = DATA / "runs/m9"
# other agents' run directories (HybridSearch's own BM25 engine, reranker runs) when present
EXTRA_RUN_DIRS = (DATA / "runs/lexical", DATA / "runs/rerank", DATA / "runs/dense", DATA / "runs/hybrid")


def discover_real_runs(run_dirs=(RUN_DIR, M9_RUN_DIR, *EXTRA_RUN_DIRS)) -> dict[str, dict[str, dict[str, float]]]:
    """{ranker: {qid: {docid: score}}} for every ranker with both DL19 and DL20 runs."""
    found: dict[str, dict[str, Path]] = {}
    paths = [p for d in ([run_dirs] if isinstance(run_dirs, Path) else run_dirs) for p in glob.glob(str(Path(d) / "*.trec"))]
    for p in paths:
        m = re.match(r"(.+)\.(dl19|dl20)\.trec$", os.path.basename(p))
        if m and os.path.getsize(p) > 0:
            found.setdefault(m.group(1), {})[m.group(2)] = Path(p)
    out = {}
    for name, parts in sorted(found.items()):
        if set(parts) >= {"dl19", "dl20"}:
            run = {}
            for part in ("dl19", "dl20"):
                run.update(read_run(parts[part]))
            out[name] = run
    return out


def rrf(runs: list[dict[str, dict[str, float]]], k: int = 60, depth: int = 1000) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    qids = set().union(*[set(r) for r in runs])
    for q in qids:
        acc: dict[str, float] = {}
        for r in runs:
            for rank, d in enumerate(ranked(r.get(q, {}))[:depth], start=1):
                acc[d] = acc.get(d, 0.0) + 1.0 / (k + rank)
        out[q] = acc
    return out


def synthetic_noisy_oracle(qrels, sigma: float, seed: int, n_fillers: int = 200,
                           depth: int = 100) -> dict[str, dict[str, float]]:
    """score = grade + sigma*N(0,1); fillers are unjudged (grade 0 by convention)."""
    rng = np.random.default_rng(seed)
    out = {}
    for q in sorted(qrels, key=lambda x: int(x)):
        docs = list(qrels[q]) + [f"syn{q}_{i}" for i in range(n_fillers)]
        g = np.array([qrels[q].get(d, 0) for d in docs], float)
        s = g + sigma * rng.standard_normal(len(g))
        top = np.argsort(-s, kind="stable")[:depth]
        out[q] = {docs[i]: float(s[i]) for i in top}
    return out


# ------------------------------------------------------------------ metrics
def ndcg_at_k(ranking: list[str], judged: dict[str, int], k: int = 10) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    gains = np.array([judged.get(d, 0) for d in ranking[:k]], float)
    dcg = float((gains * disc[: len(gains)]).sum())
    ideal = np.sort(np.array([g for g in judged.values() if g > 0], float))[::-1][:k]
    idcg = float((ideal * disc[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RankerTable:
    """Top-K lists of several rankers over a common query set, with grades."""

    qids: list[str]
    lists: dict[str, list[list[str]]]      # ranker -> per-query top-K docids
    qrels: dict[str, dict[str, int]]
    k: int

    def grades(self, ranker: str) -> np.ndarray:
        out = np.full((len(self.qids), self.k), -1, np.int64)
        for i, (q, lst) in enumerate(zip(self.qids, self.lists[ranker])):
            for r, d in enumerate(lst[: self.k]):
                out[i, r] = self.qrels[q].get(d, 0)
        return out

    def ndcg(self, ranker: str, k: int = 10) -> np.ndarray:
        return np.array([ndcg_at_k(lst, self.qrels[q], k) for q, lst in zip(self.qids, self.lists[ranker])])


def build_table(runs: dict[str, dict[str, dict[str, float]]], qrels, k: int = 10) -> RankerTable:
    qids = sorted(set(qrels).intersection(*[set(r) for r in runs.values()]), key=lambda x: int(x))
    # topics with no relevant judgments cannot distinguish rankers offline; keep them
    # anyway (they are part of the traffic) but they have nDCG 0 for everyone.
    lists = {name: [ranked(r[q])[:k] for q in qids] for name, r in runs.items()}
    return RankerTable(qids, lists, qrels, k)
