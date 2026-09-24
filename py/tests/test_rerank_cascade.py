"""Cascade depth logic: only the top-k is reordered; the tail keeps first-stage order."""
from __future__ import annotations

import numpy as np

from hybridsearch.rerank.cascade import cascade_quality


def test_cascade_reorders_only_head(tmp_path):
    # one query, first stage order d0..d9; the relevant doc d7 sits at rank 8.
    pids = [f"d{i}" for i in range(10)]
    first = np.arange(10, 0, -1, dtype=np.float32)
    ce = np.zeros(10, np.float32)
    ce[7] = 5.0  # the cross-encoder loves d7
    np.savez(tmp_path / "s.npz", qid=np.array(["1"] * 10), pid=np.array(pids), score=ce, first_stage=first)
    qrels = {"1": {"d7": 1}}
    q = cascade_quality(tmp_path / "s.npz", qrels, [5, 8, 10], ["RR@10"])
    assert q[0]["RR@10"] == 1 / 8          # first stage only
    assert q[5]["RR@10"] == 1 / 8          # d7 outside the reranked head: unchanged
    assert q[8]["RR@10"] == 1.0            # inside the head: promoted to rank 1
    assert q[10]["RR@10"] == 1.0
    assert q["min_candidates_per_query"] == 10


def test_cascade_skips_depths_beyond_candidates(tmp_path):
    np.savez(tmp_path / "s.npz", qid=np.array(["1"] * 3), pid=np.array(["a", "b", "c"]),
             score=np.array([0, 1, 2], np.float32), first_stage=np.array([3, 2, 1], np.float32))
    q = cascade_quality(tmp_path / "s.npz", {"1": {"c": 1}}, [2, 3, 10], ["RR@10"])
    assert 10 not in q and q[3]["RR@10"] == 1.0 and q[2]["RR@10"] == 1 / 3
