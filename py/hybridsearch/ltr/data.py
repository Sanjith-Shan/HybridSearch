"""Learning-to-rank datasets for the counterfactual LTR study.

A dataset is a set of queries, each with the logging (production) ranker's top-K
candidates in logging order, a feature vector per candidate, and the candidate's true
grade. Features (all computed from run files and the collection, nothing learned):

  0 bm25_z      BM25 score, z-scored within the query's BM25 run list
  1 dense_z     BGE inner product, z-scored within the query's dense run list
  2 bm25_rr     1 / BM25 rank (0 if not retrieved in the run depth)
  3 dense_rr    1 / dense rank (0 if not retrieved)
  4 log_len     log(1 + passage length in whitespace tokens)
  5 qlen_bm25   log(1 + query length) * bm25_z   (query length only matters through
  6 qlen_dense  log(1 + query length) * dense_z   interactions in a listwise model)

Candidates missing from one run get that run's z-score floor (min - 0.5) and rank 0.
Features are standardised with statistics from the *training* split only.

``synthetic`` builds a stand-in with the same shape, used by tests and as a fallback when
the real runs are not on disk; results from it are labelled synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybridsearch.clicks.rankers import ranked
from hybridsearch.ltr.model import ListData

FEATURES = ["bm25_z", "dense_z", "bm25_rr", "dense_rr", "log_len", "qlen_bm25", "qlen_dense"]


@dataclass
class LTRSet:
    name: str
    qids: list[str]
    data: ListData
    grades: np.ndarray          # (N,) candidate grades (DL scale 0..3)
    ideal: list[np.ndarray]     # per query: all judged grades (for nDCG ideal)
    K: int
    feature_names: list[str]

    def grades_log_order(self) -> np.ndarray:
        Q = self.data.n_queries
        out = np.full((Q, self.K), -1, np.int64)
        for q in range(Q):
            a, b = self.data.offsets[q], self.data.offsets[q + 1]
            out[q, : b - a] = self.grades[a:b]
        return out


def _zscores(run_q: dict[str, float]):
    docs = ranked(run_q)
    s = np.array([run_q[d] for d in docs], float)
    mu, sd = s.mean(), s.std() + 1e-9
    z = {d: (run_q[d] - mu) / sd for d in docs}
    rank = {d: i + 1 for i, d in enumerate(docs)}
    floor = (s.min() - mu) / sd - 0.5
    return z, rank, floor


def build_from_runs(name, qids, logging_run, bm25_run, dense_run, qrels, queries, doclen,
                    K: int = 20, label_map=None) -> LTRSet:
    """``label_map``: maps qrels labels to DL grades (e.g. MS MARCO train/dev label 1 → 2)."""
    feats, grades, offsets, ideal, kept = [], [], [0], [], []
    for q in qids:
        if q not in logging_run or q not in qrels:
            continue
        cands = ranked(logging_run[q])[:K]
        zb, rb, fb = _zscores(bm25_run.get(q, {"_": 0.0}))
        zd, rd, fd = _zscores(dense_run.get(q, {"_": 0.0}))
        ql = np.log1p(len(queries.get(q, "").split()))
        judged = qrels[q]
        lab = (lambda g: label_map.get(g, g)) if label_map else (lambda g: g)
        for d in cands:
            b = zb.get(d, fb)
            de = zd.get(d, fd)
            feats.append([b, de, 1.0 / rb[d] if d in rb else 0.0, 1.0 / rd[d] if d in rd else 0.0,
                          np.log1p(doclen.get(d, 0)), ql * b, ql * de])
            grades.append(lab(judged.get(d, 0)))
        offsets.append(offsets[-1] + len(cands))
        ideal.append(np.array([lab(g) for g in judged.values()], np.int64))
        kept.append(q)
    X = np.asarray(feats, float)
    return LTRSet(name, kept, ListData(X, np.asarray(offsets)), np.asarray(grades, np.int64), ideal, K,
                  list(FEATURES))


def standardise(train: LTRSet, *others: LTRSet) -> None:
    mu = train.data.X.mean(0)
    sd = train.data.X.std(0) + 1e-9
    for s in (train, *others):
        s.data.X = (s.data.X - mu) / sd


def synthetic(n_queries: int, K: int, rng: np.random.Generator, name="synthetic") -> LTRSet:
    """Grades ~ DL-like marginal; the 'dense' feature is more informative than 'bm25';
    the logging ranker orders by the bm25 feature. Same feature layout as real data."""
    pg = np.array([0.55, 0.2, 0.15, 0.10])
    N = n_queries * K
    g = rng.choice(4, size=N, p=pg)
    bm = 0.5 * g + rng.standard_normal(N)
    de = 0.9 * g + rng.standard_normal(N)
    ln = rng.normal(4.0, 0.4, N) - 0.05 * g
    ql = np.repeat(np.log1p(rng.integers(2, 12, n_queries)), K)
    X = np.stack([bm, de, np.zeros(N), np.zeros(N), ln, ql * bm, ql * de], 1)
    offsets = np.arange(0, N + 1, K)
    grades = np.empty(N, np.int64)
    ideal = []
    # order each list by the logging feature (bm25) and add reciprocal-rank features
    for q in range(n_queries):
        sl = slice(q * K, (q + 1) * K)
        o = np.argsort(-X[sl, 0], kind="stable")
        Xq = X[sl][o]
        rb = np.empty(K)
        rb[:] = 1.0 / np.arange(1, K + 1)
        rd = np.empty(K)
        rd[np.argsort(-Xq[:, 1], kind="stable")] = 1.0 / np.arange(1, K + 1)
        Xq[:, 2], Xq[:, 3] = rb, rd
        X[sl] = Xq
        grades[sl] = g[sl][o]
        ideal.append(grades[sl].copy())
    return LTRSet(name, [f"s{q}" for q in range(n_queries)], ListData(X, offsets), grades, ideal, K,
                  list(FEATURES))
