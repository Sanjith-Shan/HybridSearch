"""Readers for the engine's vector files (.fbin, ground truth) as numpy arrays."""
from __future__ import annotations

import numpy as np


def read_fbin(path: str, max_rows: int | None = None, mmap: bool = True) -> np.ndarray:
    n, d = (int(x) for x in np.fromfile(path, dtype=np.uint32, count=2))
    if max_rows:
        n = min(n, max_rows)
    if mmap:
        return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))
    return np.fromfile(path, dtype=np.float32, offset=8, count=n * d).reshape(n, d)


def write_fbin(path: str, x: np.ndarray) -> None:
    x = np.ascontiguousarray(x, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.tofile(f)


def read_gt(path: str) -> tuple[np.ndarray, np.ndarray]:
    """hs_vec_groundtruth layout: uint32 nq, uint32 k, uint32 ids[nq*k], float scores[nq*k]."""
    nq, k = (int(x) for x in np.fromfile(path, dtype=np.uint32, count=2))
    ids = np.fromfile(path, dtype=np.uint32, offset=8, count=nq * k).reshape(nq, k)
    scores = np.fromfile(path, dtype=np.float32, offset=8 + 4 * nq * k, count=nq * k).reshape(nq, k)
    return ids, scores


def recall_at_k(approx: np.ndarray, gt_ids: np.ndarray, k: int = 10) -> float:
    hits = 0
    for a, g in zip(approx[:, :k], gt_ids[:, :k]):
        hits += len(set(a.tolist()) & set(g.tolist()))
    return hits / (k * len(approx))
