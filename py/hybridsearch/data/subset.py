"""Deterministic 1M-passage subset, the train_tune query sample, and the train50k
hard-negative-mining query sample.

    python -m hybridsearch.data.subset

Algorithm (seed 20260923, numpy PCG64 via ``np.random.default_rng``):
  1. R = every pid with grade > 0 in qrels.dev.small, DL19 or DL20 passage qrels.
  2. C = all pids in collection.tsv not in R, ascending.
  3. S = rng.choice(C, 1_000_000 - |R|, replace=False)   (first draw from rng)
  4. subset = sorted(R ∪ S).
  5. train_tune: train queries (with qrels) whose *every* relevant pid is in the subset,
     ascending by qid; draw 2,000 with rng.choice(..., replace=False) (second draw), then
     sort by qid.
  6. train50k: from the remaining judged train queries (train_tune excluded), draw
     50,000 with a fresh rng seeded 20260923 + 1, sorted by qid. Not restricted to the
     subset: its BM25 runs are over the full 8.8M collection.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

from .io import DATA, read_qrels, read_queries, write_qrels, write_queries, write_u64bin
from .prepare import md5sum, require_free

SEED = 20260923
SUBSET_SIZE = 1_000_000
N_TRAIN_TUNE = 2_000
N_TRAIN50K = 50_000

MS = DATA / "raw" / "msmarco"
TDL = DATA / "raw" / "trec-dl"
OUT = DATA / "subset" / "1m"
TT = DATA / "subset" / "train_tune"
T50 = DATA / "subset" / "train50k"

QRELS = {
    "dev": MS / "qrels.dev.small.tsv",
    "dl19": TDL / "dl19.qrels",
    "dl20": TDL / "dl20.qrels",
}


def relevant_pids(qrels) -> set[int]:
    return {int(d) for docs in qrels.values() for d, g in docs.items() if g > 0}


def main() -> int:
    t0 = time.time()
    require_free(DATA, 0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "READY").unlink(missing_ok=True)

    qrels = {k: read_qrels(p) for k, p in QRELS.items()}
    R: set[int] = set()
    for q in qrels.values():
        R |= relevant_pids(q)
    print(f"relevant pids: {len(R)}")

    # All collection pids (verifies the file is sorted and unique on the way).
    with open(MS / "collection.tsv", "rb") as f:
        pids = np.fromiter((int(line.split(b"\t", 1)[0]) for line in f), dtype=np.int64)
    assert np.all(np.diff(pids) > 0), "collection.tsv not strictly ascending by pid"
    Rarr = np.array(sorted(R), dtype=np.int64)
    assert np.isin(Rarr, pids).all(), "a relevant pid is missing from the collection"
    comp = pids[~np.isin(pids, Rarr)]

    rng = np.random.default_rng(SEED)
    sample = rng.choice(comp, SUBSET_SIZE - len(Rarr), replace=False)
    subset = np.sort(np.concatenate([Rarr, sample]))
    assert subset.shape[0] == SUBSET_SIZE and np.unique(subset).shape[0] == SUBSET_SIZE
    in_subset = set(subset.tolist())

    # collection subset, streamed in file order (== ascending pid)
    n = 0
    with open(MS / "collection.tsv", encoding="utf-8") as fin, open(
        OUT / "collection.tsv.tmp", "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if int(line.split("\t", 1)[0]) in in_subset:
                fout.write(line)
                n += 1
    assert n == SUBSET_SIZE
    (OUT / "collection.tsv.tmp").rename(OUT / "collection.tsv")
    write_u64bin(OUT / "docids.u64bin", subset.astype(np.uint64))

    stats = {}
    for k, q in qrels.items():
        filt = {qid: {d: g for d, g in docs.items() if int(d) in in_subset} for qid, docs in q.items()}
        filt = {qid: docs for qid, docs in filt.items() if docs}
        write_qrels(OUT / f"qrels.{k}.tsv", filt)
        stats[k] = {
            "topics": len(q),
            "topics_in_subset_qrels": len(filt),
            "judgments_full": sum(len(v) for v in q.values()),
            "judgments_in_subset": sum(len(v) for v in filt.values()),
            "relevant_full": sum(1 for v in q.values() for g in v.values() if g > 0),
            "relevant_in_subset": sum(1 for v in filt.values() for g in v.values() if g > 0),
        }
        assert stats[k]["relevant_full"] == stats[k]["relevant_in_subset"]

    # train_tune
    train_q = read_queries(MS / "queries.train.tsv")
    train_qrels = read_qrels(MS / "qrels.train.tsv")
    judged = sorted((q for q in train_qrels if q in train_q), key=int)
    eligible = [
        q for q in judged
        if all(int(d) in in_subset for d, g in train_qrels[q].items() if g > 0)
        and any(g > 0 for g in train_qrels[q].values())
    ]
    pick = rng.choice(len(eligible), N_TRAIN_TUNE, replace=False)
    tt = sorted((eligible[i] for i in pick), key=int)
    TT.mkdir(parents=True, exist_ok=True)
    write_queries(TT / "queries.tsv", [(q, train_q[q]) for q in tt])
    write_qrels(TT / "qrels.tsv", {q: train_qrels[q] for q in tt})

    # train50k (full-collection hard-negative mining sample), disjoint from train_tune
    tt_set = set(tt)
    pool = [q for q in judged if q not in tt_set]
    rng2 = np.random.default_rng(SEED + 1)
    pick2 = rng2.choice(len(pool), N_TRAIN50K, replace=False)
    t50 = sorted((pool[i] for i in pick2), key=int)
    T50.mkdir(parents=True, exist_ok=True)
    write_queries(T50 / "queries.tsv", [(q, train_q[q]) for q in t50])
    write_qrels(T50 / "qrels.tsv", {q: train_qrels[q] for q in t50})

    manifest = {
        "seed": SEED,
        "subset_size": SUBSET_SIZE,
        "relevant_pids_forced": len(Rarr),
        "sampled_pids": len(sample),
        "min_pid": int(subset[0]),
        "max_pid": int(subset[-1]),
        "docids_sha256": hashlib.sha256(subset.astype("<u8").tobytes()).hexdigest(),
        "collection_md5": md5sum(OUT / "collection.tsv"),
        "qrels": stats,
        "train_judged_queries": len(judged),
        "train_tune": {"eligible": len(eligible), "picked": len(tt)},
        "train50k": {"pool": len(pool), "picked": len(t50), "seed": SEED + 1},
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "READY").write_text(json.dumps({"docids_sha256": manifest["docids_sha256"]}) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
