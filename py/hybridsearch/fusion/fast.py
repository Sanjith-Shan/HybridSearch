"""Vectorised fusion for parameter sweeps. Gives bit-identical scores and the same order as
``core.rrf`` / ``core.weighted`` (tested in py/tests/test_fusion_fast.py, and the M4 driver
re-checks every reported configuration against ``core``).

Per query, ``prepare`` builds the candidate union once (sorted by ``docid_key`` so a stable
sort on the negated score breaks ties by lower doc id), with each retriever's rank and its
normalised scores; normalisation reuses ``core.normalise`` so the floats are the same."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import NORMALISATIONS, Ranking, docid_key, normalise, truncate


@dataclass
class QueryCands:
    docs: list[str]
    lex_rank: np.ndarray      # 1-based, inf if absent
    dense_rank: np.ndarray
    norm: dict[str, tuple[np.ndarray, np.ndarray]]  # method -> (lex, dense), floor-filled


def prepare(lex: Ranking, dense: Ranking, depth: int | None = None) -> QueryCands:
    lex, dense = truncate(lex, depth), truncate(dense, depth)
    docs = sorted({d for d, _ in lex} | {d for d, _ in dense}, key=docid_key)
    pos = {d: i for i, d in enumerate(docs)}
    n = len(docs)
    ranks = []
    for r in (lex, dense):
        a = np.full(n, np.inf)
        for i, (d, _) in enumerate(r, start=1):
            a[pos[d]] = i
        ranks.append(a)
    norm = {}
    for m in NORMALISATIONS:
        pair = []
        for r in (lex, dense):
            ns = normalise([s for _, s in r], m)
            a = np.full(n, min(ns) if ns else 0.0)
            for (d, _), v in zip(r, ns):
                a[pos[d]] = v
            pair.append(a)
        norm[m] = (pair[0], pair[1])
    return QueryCands(docs, ranks[0], ranks[1], norm)


def _order(qc: QueryCands, score: np.ndarray, out_k: int | None) -> Ranking:
    idx = np.argsort(-score, kind="stable")
    if out_k is not None:
        idx = idx[:out_k]
    return [(qc.docs[i], float(score[i])) for i in idx]


def rrf_scores(qc: QueryCands, k: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        a = np.where(np.isfinite(qc.lex_rank), 1.0 / (k + qc.lex_rank), 0.0)
        b = np.where(np.isfinite(qc.dense_rank), 1.0 / (k + qc.dense_rank), 0.0)
    return a + b


def weighted_scores(qc: QueryCands, alpha: float, norm: str) -> np.ndarray:
    nl, nd = qc.norm[norm]
    return (1.0 - alpha) * nl + alpha * nd


def rrf_fast(qc: QueryCands, k: float, out_k: int | None = None) -> Ranking:
    return _order(qc, rrf_scores(qc, k), out_k)


def weighted_fast(qc: QueryCands, alpha: float, norm: str, out_k: int | None = None) -> Ranking:
    return _order(qc, weighted_scores(qc, alpha, norm), out_k)
