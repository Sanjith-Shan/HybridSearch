"""Negative sampler + mining rule properties (offline, synthetic data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _rerank_tiny import synthetic_corpus
from hybridsearch.rerank.mine import false_negative, random_pool
from hybridsearch.rerank.sampler import GroupSampler, needed_pids, norm_text, read_mined
from hybridsearch.rerank.text import iter_run_groups, load_passages


@pytest.fixture()
def corpus(tmp_path):
    info = synthetic_corpus(tmp_path)
    mined = read_mined(tmp_path / "mined.jsonl")
    passages = load_passages(None, tmp_path / "collection.tsv")
    judged: dict[str, set[str]] = {}
    for line in (tmp_path / "qrels.tsv").read_text().splitlines():
        q, _, p, _ = line.split()
        judged.setdefault(q, set()).add(p)
    return tmp_path, info, mined, passages, judged


def make(corpus, strategy, n_neg=7, seed=1, **kw):
    _, info, mined, passages, judged = corpus
    return GroupSampler(mined, info["queries"], passages, strategy, n_neg, seed, judged=judged, **kw)


@pytest.mark.parametrize("strategy", ["random", "bm25-hard", "dense-hard", "mixed"])
def test_no_positive_or_judged_leaks_and_no_duplicates(corpus, strategy):
    _, _, mined, passages, judged = corpus
    s = make(corpus, strategy)
    by_q = {m.qid: m for m in mined}
    for step in range(30):
        for qid, pos, negs in s.batch(step, 8):
            assert pos in by_q[qid].pos
            assert len(negs) == 7
            assert len(set(negs)) == len(negs), "repeated negative in a group"
            assert not set(negs) & judged[qid], "a judged pid leaked into negatives"
            assert pos not in negs
            ptext = norm_text(passages[pos])
            assert all(norm_text(passages[n]) != ptext for n in negs), "duplicate-text negative"


def test_duplicate_text_of_positive_is_skipped(corpus):
    _, _, mined, passages, _ = corpus
    assert passages["399"] == passages["3"]
    s = make(corpus, "bm25-hard", n_neg=20)
    m0 = next(m for m in s.mined if m.pos == ["3"])
    for seed in range(20):
        _, _, negs = s.group(m0, np.random.default_rng(seed))
        assert "399" not in negs


def test_strategies_draw_from_their_pool(corpus):
    _, _, mined, _, _ = corpus
    by_q = {m.qid: m for m in mined}
    for strategy, field in [("bm25-hard", "bm25"), ("dense-hard", "dense"), ("random", "random")]:
        s = make(corpus, strategy, n_neg=5)
        for qid, _, negs in s.batch(0, 16):
            pool = set(getattr(by_q[qid], field))
            # a short pool is topped up from random; otherwise every negative is from the pool
            extra = set(negs) - pool
            assert not extra or extra <= set(by_q[qid].random)


def test_mixed_ratio(corpus):
    _, _, mined, _, _ = corpus
    by_q = {m.qid: m for m in mined}
    s = make(corpus, "mixed", n_neg=6, mix_bm25=0.5)
    for qid, _, negs in s.batch(0, 16):
        m = by_q[qid]
        from_bm25 = [n for n in negs[:3]]
        assert all(n in m.bm25 for n in from_bm25)


def test_topup_when_pool_short(corpus):
    s = make(corpus, "dense-hard", n_neg=40)
    groups = s.batch(0, 4)
    assert all(len(n) == 40 for _, _, n in groups)
    assert s.topups > 0


def test_deterministic_and_resumable(corpus):
    a, b = make(corpus, "mixed"), make(corpus, "mixed")
    seq_a = [a.batch(s, 8) for s in range(12)]
    # a "resumed" sampler asked only for steps 6.. must match the uninterrupted one
    seq_b = [b.batch(s, 8) for s in range(6, 12)]
    assert seq_a[6:] == seq_b
    c = make(corpus, "mixed", seed=2)
    assert [c.batch(s, 8) for s in range(12)] != seq_a


def test_epochs_cover_every_query_once(corpus):
    s = make(corpus, "random", n_neg=1)
    n = len(s)
    seen = [g[0] for step in range(n // 4) for g in s.batch(step, 4)]
    assert sorted(seen) == sorted(m.qid for m in s.mined)


def test_needed_pids_covers_sampler(corpus):
    _, _, mined, _, _ = corpus
    for strategy in ["random", "bm25-hard", "dense-hard", "mixed"]:
        need = needed_pids(mined, strategy)
        s = make(corpus, strategy, n_neg=5)
        for _, pos, negs in s.batch(0, 16):
            assert pos in need and set(negs) <= need


def test_false_negative_rule():
    # above the positive by the margin in both -> drop
    assert false_negative(0.80, 0.70, 30.0, 25.0, 0.05, 0.10) is True
    # dense above, BM25 not (by relative margin) -> keep
    assert false_negative(0.80, 0.70, 26.0, 25.0, 0.05, 0.10) is False
    # BM25 above, dense within margin -> keep
    assert false_negative(0.72, 0.70, 30.0, 25.0, 0.05, 0.10) is False
    # a score missing from either retriever -> unverifiable, kept
    assert false_negative(None, 0.70, 30.0, 25.0, 0.05, 0.10) is None
    assert false_negative(0.9, 0.70, None, 25.0, 0.05, 0.10) is None


def test_random_pool_excludes_and_unique():
    rng = np.random.default_rng(0)
    ex = {str(i) for i in range(0, 8_841_823, 3)}
    pool = random_pool(rng, ex, 200)
    assert len(pool) == len(set(pool)) == 200
    assert not set(pool) & ex


def test_iter_run_groups_contiguity(tmp_path: Path):
    p = tmp_path / "r.trec"
    p.write_text("1 Q0 a 1 3 t\n1 Q0 b 2 2 t\n2 Q0 c 1 5 t\n")
    assert list(iter_run_groups(p)) == [("1", {"a": 3.0, "b": 2.0}), ("2", {"c": 5.0})]
    assert list(iter_run_groups(p, max_hits=1)) == [("1", {"a": 3.0}), ("2", {"c": 5.0})]
    p.write_text("1 Q0 a 1 3 t\n2 Q0 c 1 5 t\n1 Q0 b 2 2 t\n")
    with pytest.raises(ValueError):
        list(iter_run_groups(p))


def test_mined_file_roundtrip(tmp_path):
    rec = {"qid": "5", "pos": ["1"], "bm25": ["2"], "dense": [], "random": ["9"]}
    (tmp_path / "m.jsonl").write_text(json.dumps(rec) + "\n")
    [m] = read_mined(tmp_path / "m.jsonl")
    assert (m.qid, m.pos, m.bm25, m.dense, m.random) == ("5", ["1"], ["2"], [], ["9"])
