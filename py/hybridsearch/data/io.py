"""File formats shared across HybridSearch.

- ``.fbin``   : uint32 n, uint32 dim, then n*dim float32 row-major (little endian).
- ``.u64bin`` : uint64 n, then n uint64 values (little endian).
- qrels       : TREC qrels, ``qid iter docid grade`` (whitespace separated).
- queries     : ``qid \\t text``.
- run files   : TREC ``qid Q0 docid rank score tag``.

These match ``engine/include/hs/common/fbin.hpp``.
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"


# ---------------------------------------------------------------- binary formats
def write_u64bin(path: str | os.PathLike, ids) -> None:
    arr = np.ascontiguousarray(np.asarray(ids, dtype="<u8"))
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(np.uint64(arr.shape[0]).astype("<u8").tobytes())
        f.write(arr.tobytes())
    os.replace(tmp, path)


def read_u64bin(path: str | os.PathLike) -> np.ndarray:
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(8), dtype="<u8")[0])
        arr = np.frombuffer(f.read(8 * n), dtype="<u8")
    if arr.shape[0] != n:
        raise ValueError(f"{path}: header says {n} ids, file has {arr.shape[0]}")
    return arr.copy()


def write_fbin(path: str | os.PathLike, mat: np.ndarray) -> None:
    mat = np.ascontiguousarray(mat, dtype="<f4")
    if mat.ndim != 2:
        raise ValueError("fbin needs a 2-D matrix")
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(np.array(mat.shape, dtype="<u4").tobytes())
        f.write(mat.tobytes())
    os.replace(tmp, path)


def fbin_header(path: str | os.PathLike) -> tuple[int, int]:
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype="<u4")
    return int(n), int(d)


def read_fbin(path: str | os.PathLike, mmap: bool = True) -> np.ndarray:
    n, d = fbin_header(path)
    if mmap:
        return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(n, d))
    with open(path, "rb") as f:
        f.seek(8)
        return np.frombuffer(f.read(4 * n * d), dtype="<f4").reshape(n, d).copy()


# ---------------------------------------------------------------- text formats
def read_queries(path: str | os.PathLike) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            out[qid] = text
    return out


def write_queries(path: str | os.PathLike, queries) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(f"{qid}\t{text}\n" for qid, text in queries)


def read_qrels(path: str | os.PathLike) -> dict[str, dict[str, int]]:
    """TREC qrels -> {qid: {docid: grade}}. Accepts 4-column (qid iter docid grade)
    whitespace- or tab-separated files."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}: bad qrels line {line!r}")
            qid, _, docid, grade = parts
            qrels[qid][docid] = int(grade)
    return dict(qrels)


def write_qrels(path: str | os.PathLike, qrels: dict[str, dict[str, int]]) -> None:
    def key(x):
        return (int(x), x) if x.isdigit() else (0, x)

    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(qrels, key=key):
            f.writelines(f"{qid} 0 {docid} {qrels[qid][docid]}\n" for docid in sorted(qrels[qid], key=key))


def read_run(path: str | os.PathLike) -> dict[str, dict[str, float]]:
    """TREC run -> {qid: {docid: score}}. Ranks in the file are ignored; ordering is
    recomputed from scores by the evaluator (score desc, ties by docid)."""
    run: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 6:
                raise ValueError(f"{path}: bad run line {line!r}")
            qid, _, docid, _rank, score = parts[:5]
            run[qid][docid] = float(score)
    return dict(run)


def write_run(path: str | os.PathLike, results, tag: str) -> None:
    """``results``: iterable of (qid, [(docid, score), ...]) already in rank order."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for qid, hits in results:
            f.writelines(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n" for rank, (docid, score) in enumerate(hits, start=1))
    os.replace(tmp, path)


def iter_collection(path: str | os.PathLike):
    """Yield (pid:int, text:str) from a ``pid \\t text`` collection file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            yield int(pid), text
