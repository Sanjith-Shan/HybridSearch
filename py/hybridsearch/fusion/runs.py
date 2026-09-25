"""Run-file plumbing for fusion: read rankings in rank order, fuse per query, write runs whose
scores re-sort to exactly the fused order."""
from __future__ import annotations

import os
from collections.abc import Callable

from ..eval.metrics import rank_run
from .core import Ranking, run_file_scores


def read_run_ranked(path: str | os.PathLike) -> dict[str, Ranking]:
    """TREC run -> {qid: [(docid, score), ...]} in the file's rank order (rank column,
    then file order). Fusion consumes the retriever's own order, not a re-sort by score."""
    rows: dict[str, list[tuple[int, int, str, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            p = line.split()
            if not p:
                continue
            if len(p) < 6:
                raise ValueError(f"{path}: bad run line {line!r}")
            rows.setdefault(p[0], []).append((int(p[3]), i, p[2], float(p[4])))
    return {q: [(d, s) for _, _, d, s in sorted(v)] for q, v in rows.items()}


def fuse_runs(lex: dict[str, Ranking], dense: dict[str, Ranking],
              fn: Callable[[Ranking, Ranking], Ranking], qids=None) -> dict[str, Ranking]:
    """Apply ``fn(lex_ranking, dense_ranking)`` per query over ``qids`` (default: the union
    of both runs' queries). A query missing from one run fuses with an empty ranking."""
    qs = qids if qids is not None else sorted(set(lex) | set(dense))
    return {q: fn(lex.get(q, []), dense.get(q, [])) for q in qs}


def as_score_dict(run: dict[str, Ranking]) -> dict[str, dict[str, float]]:
    """{qid: ranking} -> {qid: {docid: file score}} using the order-preserving scores that
    ``write_run_exact`` writes, so in-memory evaluation equals evaluating the file."""
    out = {}
    for q, r in run.items():
        sc = run_file_scores(r)
        out[q] = dict(zip((d for d, _ in r), sc))
    return out


def write_run_exact(path: str | os.PathLike, run: dict[str, Ranking], tag: str) -> dict[str, dict[str, float]]:
    """Write a TREC run and return the score dict that was written. Verifies that the
    evaluator's re-sort (score desc, ties by docid desc) reproduces every query's order."""
    scores = as_score_dict(run)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q, r in run.items():
            sd = scores[q]
            if rank_run(sd) != [d for d, _ in r]:
                raise AssertionError(f"query {q}: written scores do not reproduce the fused order")
            f.writelines(f"{q} Q0 {d} {i} {sd[d]:.15g} {tag}\n" for i, (d, _) in enumerate(r, start=1))
    os.replace(tmp, path)
    return scores
