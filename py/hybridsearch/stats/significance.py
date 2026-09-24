"""Paired randomization (sign-flip permutation) test, paired bootstrap CI, and
Holm-Bonferroni step-down correction.

All functions are deterministic given ``seed`` (numpy PCG64).

Randomization test (Smucker, Allan & Carterette 2007): under H0 the two systems are
exchangeable per query, so each per-query difference d_i = a_i - b_i keeps or flips
its sign with probability 1/2. The test statistic is the mean difference.
p = (1 + #{perm : |mean(perm)| >= |mean(d)|}) / (1 + n_perm)  (two-sided; the +1 makes
the estimate a valid p-value, never exactly 0). A tolerance absorbs float noise so
identical systems give p = 1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_EPS = 1e-12


def align(a: dict[str, float], b: dict[str, float], strict: bool = True) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align two per-query metric dicts on qid. With strict=True, the qid sets must match."""
    if strict and set(a) != set(b):
        only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
        raise ValueError(f"qid sets differ: {only_a} only in a, {only_b} only in b")
    qids = sorted(set(a) & set(b), key=lambda x: (len(x), x))
    return qids, np.array([a[q] for q in qids], dtype=np.float64), np.array([b[q] for q in qids], dtype=np.float64)


def paired_randomization_test(
    a, b, n_perm: int = 10_000, seed: int = 20260923, alternative: str = "two-sided", batch: int = 1000
) -> float:
    """p-value for H0: mean(a - b) == 0 under random sign flips of the paired differences.

    alternative: "two-sided", "greater" (a > b), or "less" (a < b).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("need a non-empty 1-D array of paired values")
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    rng = np.random.default_rng(seed)
    n = d.size
    obs = d.mean()
    count = 0
    done = 0
    while done < n_perm:
        m = min(batch, n_perm - done)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8) * 2 - 1
        means = (signs * d).mean(axis=1)
        if alternative == "two-sided":
            count += int(np.sum(np.abs(means) >= abs(obs) - _EPS))
        elif alternative == "greater":
            count += int(np.sum(means >= obs - _EPS))
        elif alternative == "less":
            count += int(np.sum(means <= obs + _EPS))
        else:
            raise ValueError(f"unknown alternative {alternative!r}")
        done += m
    return (count + 1) / (n_perm + 1)


def paired_bootstrap_ci(
    a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 20260923, batch: int = 1000
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean(a - b), resampling queries with replacement.
    Returns (mean_diff, lo, hi)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0:
        raise ValueError("empty")
    rng = np.random.default_rng(seed)
    n = d.size
    stats = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        stats[done: done + m] = d[idx].mean(axis=1)
        done += m
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def holm_bonferroni(pvals, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm's step-down procedure. Returns (adjusted_p, reject) in the input order.

    adjusted p_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j)) over ascending p; reject
    iff adjusted p <= alpha (equivalent to the sequential rule)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha


@dataclass
class Comparison:
    name: str
    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    delta: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_holm: float | None = None
    significant: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compare(
    family: dict[str, tuple[dict[str, float], dict[str, float]]],
    metric: str = "",
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260923,
) -> list[Comparison]:
    """Run the randomization test + bootstrap CI for each (a, b) pair in ``family``
    (name -> (per_query_a, per_query_b)), then Holm-correct across the family."""
    out = []
    for name, (pa, pb) in family.items():
        qids, a, b = align(pa, pb)
        p = paired_randomization_test(a, b, n_perm=n_perm, seed=seed)
        delta, lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
        out.append(Comparison(name, metric, len(qids), float(a.mean()), float(b.mean()), delta, lo, hi, p))
    adj, rej = holm_bonferroni([c.p_value for c in out], alpha)
    for c, pa_, r in zip(out, adj, rej):
        c.p_holm, c.significant = float(pa_), bool(r)
    return out
