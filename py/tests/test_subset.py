"""Integrity checks on the built 1M subset (skipped when data/ is not present)."""
import json

import numpy as np
import pytest
from hybridsearch.data.io import DATA, read_qrels, read_queries, read_u64bin

SUB = DATA / "subset" / "1m"
pytestmark = pytest.mark.skipif(not (SUB / "READY").exists(), reason="subset not built")


def test_docids_sorted_unique_and_match_collection():
    ids = read_u64bin(SUB / "docids.u64bin")
    assert ids.shape[0] == 1_000_000
    assert np.all(np.diff(ids.astype(np.int64)) > 0)
    with open(SUB / "collection.tsv") as f:
        head = [int(line.split("\t", 1)[0]) for line, _ in zip(f, range(1000))]
    assert head == ids[:1000].tolist()


def test_all_relevant_docs_included():
    ids = set(read_u64bin(SUB / "docids.u64bin").tolist())
    for src in [DATA / "raw/msmarco/qrels.dev.small.tsv", DATA / "raw/trec-dl/dl19.qrels", DATA / "raw/trec-dl/dl20.qrels"]:
        for docs in read_qrels(src).values():
            assert all(int(d) in ids for d, g in docs.items() if g > 0)


def test_train_tune_and_train50k():
    ids = set(read_u64bin(SUB / "docids.u64bin").tolist())
    tt = read_queries(DATA / "subset/train_tune/queries.tsv")
    ttq = read_qrels(DATA / "subset/train_tune/qrels.tsv")
    assert len(tt) == 2000 and set(tt) == set(ttq)
    assert all(int(d) in ids for docs in ttq.values() for d, g in docs.items() if g > 0)
    t50 = read_queries(DATA / "subset/train50k/queries.tsv")
    assert len(t50) == 50_000 and not (set(t50) & set(tt))


def test_manifest_hash_matches():
    import hashlib

    m = json.loads((SUB / "manifest.json").read_text())
    ids = read_u64bin(SUB / "docids.u64bin")
    assert hashlib.sha256(ids.astype("<u8").tobytes()).hexdigest() == m["docids_sha256"]
    assert m["seed"] == 20260923
