"""Dense (BGE-base-en-v1.5) similarity for LTR candidates only.

The LTR study needs a dense score for each BM25 top-20 candidate of ~2,100 queries
(~42k passages), not a dense index over the collection. When the dense run files are not
on disk, this module encodes exactly those passages and queries itself, with the
project's encoder convention (``hybridsearch.dense.encode.Encoder``: CLS pooling,
L2-normalised, BGE query instruction), on the GPU (MPS) only when the shared lock ``data/locks/mps`` is free, otherwise
on CPU, so it never contends with the collection encoder. Results record that the vectors were computed here and
on which hardware (honesty rule 2 of the spec).

Vectors are cached in ``data/cache/m9/`` (pid list + float32 matrix).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from hybridsearch.data.io import DATA

CACHE = DATA / "cache/m9"


def _load(name):
    ids_p, vec_p = CACHE / f"{name}.ids.json", CACHE / f"{name}.npy"
    if ids_p.exists() and vec_p.exists():
        ids = json.loads(ids_p.read_text())
        return dict(zip(ids, np.load(vec_p)))
    return {}


def _save(name, d):
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = list(d)
    np.save(CACHE / f"{name}.npy", np.stack([d[i] for i in ids]).astype(np.float32) if ids else np.zeros((0, 768), np.float32))
    (CACHE / f"{name}.ids.json").write_text(json.dumps(ids))


def encode_needed(queries: dict[str, str], passages: dict[str, str], threads: int = 4, log=print) -> dict:
    """Returns {"q": {qid: vec}, "p": {pid: vec}, "info": {...}}; encodes only what the
    cache lacks."""
    import torch

    from hybridsearch.dense.encode import Encoder

    qv, pv = _load("bge_queries"), _load("bge_passages")
    need_q = [q for q in queries if q not in qv]
    need_p = [p for p in passages if p not in pv]
    info = {"device": None, "threads": threads, "model": "BAAI/bge-base-en-v1.5",
            "encoded_queries_now": len(need_q), "encoded_passages_now": len(need_p)}
    if need_q or need_p:
        device, locked = "cpu", False
        if torch.backends.mps.is_available():
            # take the GPU only if nobody holds the shared MPS lock (data/locks/mps, the
            # protocol hybridsearch.dense.encode uses); never wait for it
            from hybridsearch.dense.encode import LOCK
            try:
                LOCK.parent.mkdir(parents=True, exist_ok=True)
                os.mkdir(LOCK)
                (LOCK / "owner").write_text(f"m9-ltr-dense-feature pid={os.getpid()}\n")
                device, locked = "mps", True
            except FileExistsError:
                pass
        info["device"] = device
        try:
            torch.set_num_threads(threads)
            enc = Encoder(device=device)
            t0 = time.time()
            if need_q:
                for q, v in zip(need_q, enc.encode_queries([queries[q] for q in need_q])):
                    qv[q] = v
                _save("bge_queries", qv)
            step = 500
            for i in range(0, len(need_p), step):
                chunk = need_p[i:i + step]
                for p, v in zip(chunk, enc.encode_passages([passages[p] for p in chunk])):
                    pv[p] = v
                _save("bge_passages", pv)
                log(f"  dense feature: {min(i + step, len(need_p)):,}/{len(need_p):,} passages "
                    f"({(min(i + step, len(need_p))) / (time.time() - t0):.0f} p/s, {device})")
            info["encode_seconds"] = round(time.time() - t0, 1)
        finally:
            if locked:
                from hybridsearch.dense.encode import release_lock
                release_lock()
    return {"q": qv, "p": pv, "info": info}


def passage_texts(pids: set[str]) -> dict[str, str]:
    out = {}
    with open(DATA / "raw/msmarco/collection.tsv", encoding="utf-8") as f:
        for line in f:
            pid, text = line.rstrip("\n").split("\t", 1)
            if pid in pids:
                out[pid] = text
    return out


def dense_run_for_candidates(cands: dict[str, list[str]], vecs: dict) -> dict[str, dict[str, float]]:
    """A pseudo-run {qid: {pid: inner product}} over each query's candidates only."""
    out = {}
    for q, ps in cands.items():
        if q not in vecs["q"]:
            continue
        qv = vecs["q"][q]
        out[q] = {p: float(vecs["p"][p] @ qv) for p in ps if p in vecs["p"]}
    return out
