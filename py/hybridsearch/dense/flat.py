"""Exact dense reference on the 1M subset: FAISS IndexFlatIP, depth 1000.

    python -m hybridsearch.dense.flat [--splits dev dl19 dl20 train_tune] [--block 100000]

Writes data/runs/reference/dense-flat-1m.{split}.trec (tag dense-flat-1m) and the
exact ground truth for dev:
  data/embeddings/1m/gt.dev.top100.u64bin  : .u64bin (uint64 n, then n uint64), n = nq*100,
                                             row-major [query][rank]; values are GLOBAL
                                             MS MARCO pids; query order = queries.dev.qids.txt
  data/embeddings/1m/gt.dev.top100.scores.fbin : .fbin nq x 100 float32 inner products
Ranking order: inner product desc, ties by lower global id (repo convention).

To keep RAM low on the shared laptop, the corpus is searched in blocks of ``--block``
rows (one IndexFlatIP per block, exact) and per-query top-k lists are merged; this is
mathematically identical to one IndexFlatIP over all rows.
"""
from __future__ import annotations

import argparse
import sys
import time

import faiss
import numpy as np

from ..data.io import DATA, read_fbin, read_u64bin, write_fbin, write_run, write_u64bin
from ..eval.results import write_result
from .encode import OUT

RUNS = DATA / "runs" / "reference"


def exact_topk(queries: np.ndarray, corpus: np.ndarray, ids: np.ndarray, k: int, block: int):
    nq = queries.shape[0]
    best_s = np.full((nq, 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, 0), dtype=np.int64)
    q = np.ascontiguousarray(queries, dtype=np.float32)
    for s in range(0, corpus.shape[0], block):
        e = min(corpus.shape[0], s + block)
        index = faiss.IndexFlatIP(corpus.shape[1])
        index.add(np.ascontiguousarray(corpus[s:e], dtype=np.float32))
        kk = min(k, e - s)
        D, I = index.search(q, kk)
        gi = ids[I + s].astype(np.int64)
        cs = np.concatenate([best_s, D], axis=1)
        ci = np.concatenate([best_i, gi], axis=1)
        # sort by (-score, id) and keep k
        order = np.lexsort((ci, -cs), axis=1)[:, :k]
        best_s = np.take_along_axis(cs, order, axis=1)
        best_i = np.take_along_axis(ci, order, axis=1)
        del index
    return best_s, best_i


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "dl19", "dl20", "train_tune"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--block", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    faiss.omp_set_num_threads(args.threads)
    corpus = read_fbin(OUT / "passages.fbin")
    ids = read_u64bin(DATA / "subset" / "1m" / "docids.u64bin")
    assert corpus.shape[0] == ids.shape[0]
    RUNS.mkdir(parents=True, exist_ok=True)
    timings = {}
    for split in args.splits:
        qmat = read_fbin(OUT / f"queries.{split}.fbin", mmap=False)
        qids = (OUT / f"queries.{split}.qids.txt").read_text().split()
        t = time.time()
        S, I = exact_topk(qmat, corpus, ids, args.k, args.block)
        timings[split] = {"queries": len(qids), "seconds": round(time.time() - t, 1)}
        write_run(RUNS / f"dense-flat-1m.{split}.trec",
                  ((qids[r], list(zip(I[r].tolist(), S[r].tolist()))) for r in range(len(qids))),
                  tag="dense-flat-1m")
        if split == "dev":
            write_u64bin(OUT / "gt.dev.top100.u64bin", I[:, :100].reshape(-1).astype(np.uint64))
            write_fbin(OUT / "gt.dev.top100.scores.fbin", S[:, :100])
        print(split, timings[split], flush=True)
    write_result("dense/flat_1m_timing.json", {"note": "exact IndexFlatIP, blocked; dev-signal-only timing",
                                               "block": args.block, "threads": args.threads, "k": args.k,
                                               "splits": timings},
                 command="python -m hybridsearch.dense.flat " + " ".join(argv or sys.argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
