"""A/B and interleaving statistics.

Metric definitions (per result-page impression, averaged over impressions):

* ``ctr``          clicks per impression (total clicks / impressions; can exceed 1).
* ``clicks_at_1``  fraction of impressions with a click on rank 1.
* ``abandonment``  fraction of impressions with no click at all.
* ``rr_first``     reciprocal rank of the first click, 0 when there is none
                   (its mean is the "MRR of first click").

``higher_is_better`` says which direction means "the ranker is better" for each metric.

Tests: two-sample Welch t-test for A/B metrics (normal reference distribution; every
sample size used here is large), exact two-sided sign test on wins vs losses for
interleaving (ties dropped), percentile bootstrap CIs, chi-square sample-ratio-mismatch.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

METRICS = ("ctr", "clicks_at_1", "abandonment", "rr_first")
HIGHER_IS_BETTER = {"ctr": True, "clicks_at_1": True, "abandonment": False, "rr_first": True}


def impression_metrics(clicks: np.ndarray) -> dict[str, np.ndarray]:
    clicks = np.asarray(clicks, bool)
    n, k = clicks.shape
    any_c = clicks.any(1)
    first = np.argmax(clicks, axis=1)
    return {
        "ctr": clicks.sum(1).astype(float),
        "clicks_at_1": clicks[:, 0].astype(float),
        "abandonment": (~any_c).astype(float),
        "rr_first": np.where(any_c, 1.0 / (first + 1), 0.0),
    }


# ----------------------------------------------------------------- tests (vectorised)
def welch_p(m1, v1, n1, m2, v2, n2):
    """Two-sided p-value for mean difference; arrays broadcast. v = sample variance."""
    m1, v1, n1, m2, v2, n2 = map(lambda x: np.asarray(x, float), (m1, v1, n1, m2, v2, n2))
    se = np.sqrt(v1 / n1 + v2 / n2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (m2 - m1) / se
    p = 2 * stats.norm.sf(np.abs(z))
    return np.where(se > 0, p, np.where(m1 == m2, 1.0, 0.0))


def sign_test_p(wins, losses):
    """Exact two-sided binomial sign test, H0: P(win | not tie) = 1/2. Vectorised."""
    wins = np.asarray(wins)
    losses = np.asarray(losses)
    n = wins + losses
    lo = np.minimum(wins, losses)
    p = np.minimum(1.0, 2 * stats.binom.cdf(lo, n, 0.5))
    return np.where(n > 0, p, 1.0)


def one_sample_p(mean, var, n):
    mean, var, n = (np.asarray(x, float) for x in (mean, var, n))
    se = np.sqrt(var / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = mean / se
    return np.where(se > 0, 2 * stats.norm.sf(np.abs(z)), np.where(mean == 0, 1.0, 0.0))


# ----------------------------------------------------------------- bootstrap
def _boot_means(x: np.ndarray, rng, n_boot: int) -> np.ndarray:
    """Bootstrap replicates of the mean. Per-impression metrics take few distinct
    values, so resampling n items with replacement is done exactly as a multinomial
    draw over the distinct values (same distribution, O(values) instead of O(n))."""
    vals, cnt = np.unique(x, return_counts=True)
    if len(vals) <= 256:
        draws = rng.multinomial(len(x), cnt / cnt.sum(), size=n_boot)
        return draws @ vals / len(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(1)


def bootstrap_mean_ci(x, rng, n_boot=2000, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    means = _boot_means(x, rng, n_boot)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist())


def bootstrap_diff_ci(x_a, x_b, rng, n_boot=2000, alpha=0.05):
    """CI for mean(x_b) - mean(x_a), resampling each arm independently."""
    x_a, x_b = np.asarray(x_a, float), np.asarray(x_b, float)
    if len(x_a) == 0 or len(x_b) == 0:
        return (float("nan"), float("nan"))
    ma = _boot_means(x_a, rng, n_boot)
    mb = _boot_means(x_b, rng, n_boot)
    return tuple(np.quantile(mb - ma, [alpha / 2, 1 - alpha / 2]).tolist())


def ab_report(clicks_a: np.ndarray, clicks_b: np.ndarray, rng, n_boot=2000) -> dict:
    """Per-metric control (a) / treatment (b) means, delta, bootstrap CI, Welch p."""
    ma, mb = impression_metrics(clicks_a), impression_metrics(clicks_b)
    out = {"n_a": int(len(clicks_a)), "n_b": int(len(clicks_b)), "metrics": {}}
    for m in METRICS:
        xa, xb = ma[m], mb[m]
        p = float(welch_p(xa.mean(), xa.var(ddof=1), len(xa), xb.mean(), xb.var(ddof=1), len(xb))) \
            if len(xa) > 1 and len(xb) > 1 else float("nan")
        out["metrics"][m] = {
            "a": float(xa.mean()) if len(xa) else float("nan"),
            "b": float(xb.mean()) if len(xb) else float("nan"),
            "delta": float(xb.mean() - xa.mean()) if len(xa) and len(xb) else float("nan"),
            "delta_ci95": bootstrap_diff_ci(xa, xb, rng, n_boot),
            "p_welch": p,
            "higher_is_better": HIGHER_IS_BETTER[m],
        }
    return out


def interleave_report(outcomes: np.ndarray, rng, n_boot=2000) -> dict:
    """``outcomes``: +1 A wins, -1 B wins, 0 tie (per impression).
    Delta_AB = P(A wins) + P(tie)/2 - 1/2 (Chapelle et al. 2012); > 0 prefers A."""
    o = np.asarray(outcomes)
    w, l_, t = int((o > 0).sum()), int((o < 0).sum()), int((o == 0).sum())
    n = len(o)
    delta = (w + 0.5 * t) / n - 0.5 if n else float("nan")
    ci = bootstrap_mean_ci(np.clip(o, -1, 1) / 2.0, rng, n_boot) if n else (float("nan"),) * 2
    return {"impressions": n, "wins_a": w, "wins_b": l_, "ties": t,
            "delta_ab": delta, "delta_ci95": ci, "p_sign": float(sign_test_p(w, l_))}


# ----------------------------------------------------------------- SRM
def srm_check(counts, expected_fractions, threshold: float = 0.0005) -> dict:
    """Sample-ratio-mismatch: chi-square goodness of fit of observed arm counts to the
    configured allocation. Flag at p < ``threshold`` (a strict threshold is conventional
    because SRM is checked on every experiment, every day; Fabijan et al. KDD 2019)."""
    counts = np.asarray(counts, float)
    exp = np.asarray(expected_fractions, float)
    exp = exp / exp.sum() * counts.sum()
    chi2, p = stats.chisquare(counts, exp)
    return {"counts": counts.astype(int).tolist(), "expected": exp.tolist(),
            "chi2": float(chi2), "p": float(p), "srm": bool(p < threshold), "threshold": threshold}
