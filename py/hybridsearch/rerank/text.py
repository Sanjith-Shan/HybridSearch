"""Passage-text lookup and query/qrels/run loading for reranking."""
from __future__ import annotations

from pathlib import Path

from hybridsearch.data.io import DATA, read_qrels, read_queries, read_run  # noqa: F401

FULL_COLLECTION = DATA / "raw/msmarco/collection.tsv"
SUBSET_COLLECTION = DATA / "subset/1m/collection.tsv"


def load_passages(needed: set[int] | set[str] | None, path: Path = FULL_COLLECTION) -> dict[str, str]:
    """Stream ``pid \\t text`` and keep only ``needed`` pids (as strings). ``None`` keeps all."""
    want = None if needed is None else {str(x) for x in needed}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            tab = line.index("\t")
            pid = line[:tab]
            if want is None or pid in want:
                out[pid] = line[tab + 1:].rstrip("\n")
                if want is not None and len(out) == len(want):
                    break
    return out


def top_k(run: dict[str, dict[str, float]], k: int) -> dict[str, list[tuple[str, float]]]:
    """Per query, hits sorted by score desc then docid asc (the evaluator's order), cut to k."""
    out = {}
    for qid, hits in run.items():
        items = sorted(hits.items(), key=lambda x: (-x[1], int(x[0]) if x[0].isdigit() else x[0]))
        out[qid] = items[:k]
    return out


def iter_run_groups(path: Path, max_hits: int | None = None):
    """Stream a TREC run grouped by consecutive qid: yields (qid, {docid: score}).
    Constant memory per query; the run must have each query's lines contiguous (true for
    Pyserini/Anserini and the engine's writers). Raises if a qid reappears later."""
    seen: set[str] = set()
    cur, hits = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            qid, docid, score = parts[0], parts[2], float(parts[4])
            if qid != cur:
                if cur is not None:
                    yield cur, hits
                if qid in seen:
                    raise ValueError(f"{path}: qid {qid} is not contiguous")
                seen.add(qid)
                cur, hits = qid, {}
            if max_hits is None or len(hits) < max_hits:
                hits[docid] = score
    if cur is not None:
        yield cur, hits
