"""Rank fusion of a lexical and a dense ranking (M4). Pure functions, no I/O.

A *ranking* is a list of ``(docid, score)`` pairs in rank order (best first). Doc ids are
strings (MS MARCO ids are decimal strings; BEIR ids are arbitrary strings).

Methods
-------
* Reciprocal rank fusion (Cormack, Clarke & Buettcher 2009):
  ``rrf(d) = sum_i 1 / (k + rank_i(d))`` with ranks from 1; a document absent from ranking i
  contributes nothing for i.
* Weighted combination of normalised scores (a CombSUM with weights):
  ``fused(d) = alpha * norm_dense(d) + (1 - alpha) * norm_lex(d)``, alpha is the weight on dense.
  Normalisation is per query, over the (depth-truncated) candidate list of that retriever:
    - ``minmax``: (s - min) / (max - min); a list whose scores are all equal maps to 1.0.
    - ``zscore``: (s - mean) / std (population std); a list with std 0 maps to 0.0.
  A document the retriever did not return gets that retriever's *lowest* normalised score
  in the list (0 for min-max, the minimum z for z-score): it is treated as if it sat at the
  bottom of the list, never above anything the retriever actually returned. An empty list
  contributes 0 to every document.
* Candidate depth: every input ranking is truncated to its top ``depth`` before fusion
  (normalisation statistics are computed over the truncated list). ``depth=None`` keeps all.

Ties in the fused score are broken by lower doc id (numeric order when both ids are
decimal, else string order), matching the repo convention "ties by lower global id".
"""
from __future__ import annotations

import math
from collections.abc import Sequence

Ranking = list[tuple[str, float]]

NORMALISATIONS = ("minmax", "zscore")


def docid_key(d: str):
    """Sort key giving numeric order for decimal ids and string order otherwise
    (decimal ids sort before non-decimal ones)."""
    return (0, int(d), "") if d.isdigit() else (1, 0, d)


def truncate(r: Ranking, depth: int | None) -> Ranking:
    if depth is None:
        return list(r)
    if depth < 1:
        raise ValueError("depth must be >= 1 or None")
    return list(r[:depth])


def _check_unique(r: Ranking) -> None:
    seen = set()
    for d, _ in r:
        if d in seen:
            raise ValueError(f"duplicate docid {d!r} in a ranking")
        seen.add(d)


def sort_fused(scores: dict[str, float], out_k: int | None = None) -> Ranking:
    """Fused score descending, ties by lower doc id."""
    out = sorted(scores.items(), key=lambda x: (-x[1], docid_key(x[0])))
    return out if out_k is None else out[:out_k]


def rrf(rankings: Sequence[Ranking], k: float = 60.0, depth: int | None = None,
        out_k: int | None = None) -> Ranking:
    """Reciprocal rank fusion of any number of rankings."""
    if k < 0:
        raise ValueError("RRF k must be >= 0")
    scores: dict[str, float] = {}
    for r in rankings:
        r = truncate(r, depth)
        _check_unique(r)
        for rank, (d, _) in enumerate(r, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sort_fused(scores, out_k)


def normalise(scores: Sequence[float], method: str) -> list[float]:
    """Normalise one retriever's scores for one query."""
    s = [float(x) for x in scores]
    if not s:
        return []
    if method == "minmax":
        lo, hi = min(s), max(s)
        if hi - lo == 0:
            return [1.0] * len(s)
        return [(x - lo) / (hi - lo) for x in s]
    if method == "zscore":
        n = len(s)
        mu = sum(s) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in s) / n)
        if sd == 0:
            return [0.0] * n
        return [(x - mu) / sd for x in s]
    raise ValueError(f"unknown normalisation {method!r}; use one of {NORMALISATIONS}")


def weighted(lex: Ranking, dense: Ranking, alpha: float, norm: str = "minmax",
             depth: int | None = None, out_k: int | None = None) -> Ranking:
    """alpha * norm(dense) + (1 - alpha) * norm(lex); see the module docstring for missing docs."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    parts = []
    for r, w in ((lex, 1.0 - alpha), (dense, alpha)):
        r = truncate(r, depth)
        _check_unique(r)
        ns = normalise([s for _, s in r], norm)
        parts.append((dict(zip((d for d, _ in r), ns)), (min(ns) if ns else 0.0), w))
    docs = set().union(*(p[0].keys() for p in parts))
    scores = {d: sum(w * m.get(d, floor) for m, floor, w in parts) for d in docs}
    return sort_fused(scores, out_k)


def run_file_scores(r: Ranking, digits: int = 15) -> list[float]:
    """Scores to write in a TREC run file so that trec_eval-style re-sorting (score desc,
    ties by docid *descending string*) reproduces the ranking exactly: every score is forced
    strictly below the previous one, as printed with ``%.{digits}g``. Scores move by at most
    a few ulps-at-``digits`` per tie; the rank order is what counts."""
    out: list[float] = []
    prev_printed = math.inf
    for _, s in r:
        v = float(f"{s:.{digits}g}")
        if v >= prev_printed:
            step = max(abs(prev_printed), 1e-300) * 10.0 ** (1 - digits)
            v = float(f"{prev_printed - step:.{digits}g}")
            while v >= prev_printed:  # guard rounding
                step *= 2
                v = float(f"{prev_printed - step:.{digits}g}")
        out.append(v)
        prev_printed = v
    return out
