"""Linear listwise ranker trained from (simulated) logged clicks.

Data are per-query candidate lists: features ``X_q`` (n_q, F) and non-negative label
weights ``w_q`` (n_q,). Loss (listwise softmax cross-entropy, as in the IPS-weighted
softmax of Ai et al. 2018 / the counterfactual LTR survey of Oosterhuis & de Rijke):

    L(theta) = - sum_q sum_d w_qd * log softmax_q(X_q theta)_d  +  lam/2 * ||theta||^2

Label weights by estimator (clicks aggregated over logged sessions):

* naive  : w_qd = #clicks on d                      (click = relevance; biased by position)
* IPS    : w_qd = sum over clicks of 1 / eta_rank   (rank at which the click happened),
           optionally clipped at ``clip`` (Joachims et al. 2017; Strehl et al. 2010)
* oracle : w_qd = DL grade                          (true graded relevance, no clicks)
* skyline: w_qd = alpha(grade) * sessions_q         (what IPS converges to with infinite
           logs: click probability given examination)

The objective is convex in theta; it is minimised with L-BFGS. Label weights are
normalised by their global total so ``lam`` means the same thing at every log size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ListData:
    """Concatenated candidate lists. ``offsets`` has len(queries)+1 entries."""

    X: np.ndarray        # (N, F)
    offsets: np.ndarray  # (Q+1,)

    @property
    def n_queries(self) -> int:
        return len(self.offsets) - 1

    def seg(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_queries), np.diff(self.offsets))


def _loss_grad(theta, X, w, seg, starts, lam):
    s = X @ theta
    # per-query log-sum-exp
    smax = np.maximum.reduceat(s, starts)
    e = np.exp(s - smax[seg])
    z = np.add.reduceat(e, starts)
    lse = np.log(z) + smax
    wq = np.add.reduceat(w, starts)
    loss = -(w @ s) + wq @ lse + 0.5 * lam * theta @ theta
    p = e / z[seg]
    grad = -(X.T @ w) + X.T @ (p * wq[seg]) + lam * theta
    return loss, grad


def fit(data: ListData, w: np.ndarray, lam: float = 1e-3, theta0=None) -> np.ndarray:
    w = np.asarray(w, float)
    tot = w.sum()
    if tot <= 0:
        return np.zeros(data.X.shape[1])
    w = w / tot
    seg = data.seg()
    starts = data.offsets[:-1]
    # drop empty lists (reduceat needs non-empty segments)
    keep = np.diff(data.offsets) > 0
    if not keep.all():
        raise ValueError("empty candidate list")
    th0 = np.zeros(data.X.shape[1]) if theta0 is None else theta0
    res = minimize(_loss_grad, th0, args=(data.X, w, seg, starts, lam), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "gtol": 1e-9})
    return res.x


def rank_lists(data: ListData, theta: np.ndarray) -> list[np.ndarray]:
    """Per query: candidate indices (local) in descending score order, stable."""
    s = data.X @ theta
    out = []
    for q in range(data.n_queries):
        a, b = data.offsets[q], data.offsets[q + 1]
        out.append(np.argsort(-s[a:b], kind="stable"))
    return out


def ndcg10(order: np.ndarray, grades: np.ndarray, ideal_grades: np.ndarray, k: int = 10) -> float:
    """trec_eval-style nDCG@k of candidates ``order`` (local indices), ideal from all
    judged grades of the query (not just the candidates)."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    g = grades[order[:k]].astype(float)
    dcg = float(g @ disc[: len(g)])
    ideal = np.sort(ideal_grades[ideal_grades > 0].astype(float))[::-1][:k]
    idcg = float(ideal @ disc[: len(ideal)])
    return dcg / idcg if idcg > 0 else 0.0
