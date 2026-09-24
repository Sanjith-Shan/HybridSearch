"""Hard-negative mining end to end on a synthetic 1M-subset stand-in (FAISS, tiny)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import hybridsearch.rerank.mine as M
from hybridsearch.data.io import write_fbin, write_u64bin


@pytest.fixture()
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_sub, d = 300, 16
    subset = np.sort(rng.choice(2000, n_sub, replace=False)).astype(np.uint64)
    (tmp_path / "subset/1m").mkdir(parents=True)
    write_u64bin(tmp_path / "subset/1m/docids.u64bin", subset)
    X = rng.normal(size=(n_sub, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    emb = tmp_path / "emb"
    emb.mkdir()
    write_fbin(emb / "passages.fbin", X)
    qids = [str(100 + i) for i in range(20)]
    Q = rng.normal(size=(20, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    write_fbin(emb / "queries.t.fbin", Q)
    (emb / "queries.t.qids.txt").write_text("\n".join(qids) + "\n")
    # positives: a pid outside the subset (5000+i) for each query, plus one judged pid inside it
    qrels, pos_pids = [], []
    for i, q in enumerate(qids):
        qrels.append(f"{q} 0 {5000 + i} 1\n")
        qrels.append(f"{q} 0 {int(subset[i])} 1\n")
        pos_pids += [5000 + i, int(subset[i])]
    (tmp_path / "qrels.tsv").write_text("".join(qrels))
    (tmp_path / "queries.tsv").write_text("".join(f"{q}\tquery {q}\n" for q in qids))
    P = np.concatenate([Q[i:i + 1] * 0.2 + X[i:i + 1] * 0 for i in range(20)] * 2)  # weak positives
    pos_sorted = sorted(pos_pids)
    rows = np.stack([P[pos_pids.index(p)] for p in pos_sorted]).astype(np.float32)
    (tmp_path / "rerank").mkdir()
    write_fbin(tmp_path / "rerank/positives.t.fbin", rows)
    (tmp_path / "rerank/positives.t.fbin.pids.txt").write_text("\n".join(map(str, pos_sorted)) + "\n")
    # BM25 run: first 18 queries only; hits = the judged subset pid + subset docs + outside docs
    lines = []
    for i, q in enumerate(qids[:18]):
        hits = [int(subset[i])] + [int(x) for x in subset[20:60]] + list(range(3000, 3020))
        for r, p in enumerate(hits):
            lines.append(f"{q} Q0 {p} {r + 1} {100 - r:.2f} bm25\n")
    (tmp_path / "bm25.trec").write_text("".join(lines))
    monkeypatch.setattr(M, "DATA", tmp_path)
    monkeypatch.setattr(M, "EMB", emb)
    monkeypatch.setattr(M, "MINED", tmp_path / "mined")
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(M.SPLITS, "t", (tmp_path / "queries.tsv", tmp_path / "qrels.tsv"))
    return tmp_path, subset


def test_mine_end_to_end(world):
    tmp, subset = world
    out = M.mine(M.MineConfig(split="t", bm25_run="bm25.trec", pool_depth=10, dense_depth=40))
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    stats = json.loads((tmp / "mined/t.stats.json").read_text())
    assert len(recs) == 18 and stats["queries"] == 18 and stats["skipped_no_bm25_run"] == 2
    qrels = {}
    for line in (tmp / "qrels.tsv").read_text().splitlines():
        q, _, p, _ = line.split()
        qrels.setdefault(q, set()).add(p)
    sub = {str(int(x)) for x in subset}
    for r in recs:
        for src in ("bm25", "dense", "random"):
            assert not set(r[src]) & qrels[r["qid"]], f"judged pid leaked into {src}"
            assert len(r[src]) == len(set(r[src])) <= 10
        assert set(r["dense"]) <= sub                       # dense pool comes from the subset
        assert len(r["random"]) == 10
    assert stats["leak_removed_bm25"] == 18                   # the judged subset pid at BM25 rank 1
    total = sum(stats[k] for k in stats if k.startswith(("fn_dropped", "verified_kept", "unverifiable")))
    assert total > 0


def test_mine_is_deterministic(world):
    tmp, _ = world
    cfg = M.MineConfig(split="t", bm25_run="bm25.trec", pool_depth=10, dense_depth=40)
    a = M.mine(cfg).read_text()
    b = M.mine(cfg).read_text()
    assert a == b


def test_exact_topk_matches_bruteforce():
    rng = np.random.default_rng(1)
    xb = rng.normal(size=(500, 8)).astype(np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)  # unit rows: a row is its own nearest
    q = rng.normal(size=(7, 8)).astype(np.float32)
    xb[10] = xb[20]  # a tie: lower row id must come first
    q[0] = xb[10]
    S, I = M.exact_topk(xb, q, 25, batch=3, chunk=64)  # forces running-top-k merges
    full = q @ xb.T
    for j in range(7):
        want = sorted(range(500), key=lambda r: (-full[j, r], r))[:25]
        assert list(I[j]) == want
        np.testing.assert_allclose(S[j], full[j, want], rtol=1e-6)
    assert list(I[0][:2]) == [10, 20]
