"""Parity: our BGE pipeline (MPS, token-budgeted length-sorted batches) vs a plain
sentence-transformers encode (CPU) on a seeded sample of subset passages already
written to passages.fbin(.partial). Also checks query embeddings the same way.

    python -m hybridsearch.dense.parity [--n 200]
Writes results/dense/parity.json (+ .meta.json).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from ..data.io import DATA, read_fbin, read_queries, read_u64bin
from ..eval.results import write_result
from .encode import MODEL, OUT, QUERY_PREFIX, QUERY_SETS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MODEL, device="cpu")
    ids = read_u64bin(DATA / "subset" / "1m" / "docids.u64bin")
    path = OUT / "passages.fbin"
    done_rows = None
    if not path.exists():
        path = OUT / "passages.fbin.partial"
        chunks = sorted(int(x) for x in (OUT / "passages.progress").read_text().split())
        from .encode import CHUNK

        done_rows = np.concatenate([np.arange(c * CHUNK, min(len(ids), (c + 1) * CHUNK)) for c in chunks])
    mat = read_fbin(path)
    pool = done_rows if done_rows is not None else np.arange(len(ids))
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(pool, args.n, replace=False))
    want = set(rows.tolist())
    texts = {}
    with open(DATA / "subset" / "1m" / "collection.tsv", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                pid, t = line.rstrip("\n").split("\t", 1)
                assert int(pid) == int(ids[i])
                texts[i] = t
    ref = st.encode([texts[i] for i in rows], batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    ours = np.asarray(mat[rows])
    cos_p = (ref * ours).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(ours, axis=1)

    q = read_queries(QUERY_SETS["dev"])
    qids = (OUT / "queries.dev.qids.txt").read_text().split()
    qmat = read_fbin(OUT / "queries.dev.fbin")
    qrows = np.sort(rng.choice(len(qids), args.n, replace=False))
    st.max_seq_length = 64
    qref = st.encode([QUERY_PREFIX + q[qids[i]] for i in qrows], batch_size=32, normalize_embeddings=True)
    cos_q = (qref * np.asarray(qmat[qrows])).sum(1)

    res = {
        "model": MODEL,
        "reference": "sentence-transformers SentenceTransformer.encode(normalize_embeddings=True), CPU fp32",
        "ours": "hybridsearch.dense.encode (transformers AutoModel, CLS, L2, MPS fp32, length-sorted batches)",
        "n_passages": int(args.n), "seed": args.seed, "source_file": path.name,
        "passage_cos_min": float(cos_p.min()), "passage_cos_mean": float(cos_p.mean()),
        "n_queries": int(args.n), "query_set": "dev",
        "query_cos_min": float(cos_q.min()), "query_cos_mean": float(cos_q.mean()),
        "threshold": 0.9999,
        "pass": bool(cos_p.min() >= 0.9999 and cos_q.min() >= 0.9999),
    }
    write_result("dense/parity.json", res, command=" ".join(["python -m hybridsearch.dense.parity"] + (argv or sys.argv[1:])))
    print(json.dumps(res, indent=2))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
