"""Sensitivity of online evaluation methods under simulated users.

Question: how many query impressions does each online method need to pick the ranker
that is better by offline nDCG@10 (DL19+DL20 graded qrels), at two-sided p < 0.05 with
80% power?

Method (all simulated users; relevance from TREC DL graded judgments):

1. **Impressions.** Each impression is a query drawn uniformly from the DL19+DL20 topics
   that both rankers answered; the user sees the top 10 and clicks under a click model.
   Total impressions ``N`` is the budget for every method: interleaving shows ``N``
   interleaved pages, A/B shows ``N/2`` pages per arm.
2. **Per-impression outcome distribution.** Every per-impression quantity used by a test
   is discrete (TDI/BI outcome in {-1,0,+1}; CTR in 0..10 clicks; clicks@1 and
   abandonment in {0,1}; first-click rank in 0..10; PI's marginalised outcome takes
   finitely many values). We simulate a large pool of ``M`` impressions per arm/method
   with the click model and tabulate the empirical distribution of each quantity.
3. **Trials.** Impressions are i.i.d., so a trial of size ``N`` is exactly a multinomial
   draw of ``N`` outcomes from that distribution (= resampling ``N`` impressions from
   the pool). Each trial runs the test; power(N) = fraction of trials that are
   significant **in the offline-correct direction**. ``T`` trials per grid point.
4. **N80** = smallest N where the isotonic (monotone) fit of power(N) reaches 0.8,
   log-linearly interpolated between grid points. The grid resolution is reported.
5. Because (3) resamples a finite pool, results whose N80 exceeds ``M / 10`` are marked
   ``pool_limited`` (the pool's own sampling error in the effect is then not negligible).
   ``validate_direct`` re-checks selected N80 values with fresh click simulations per
   trial (no pool).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

from hybridsearch.clicks import ab as abstats
from hybridsearch.clicks.interleave import (
    balanced_outcome_batch,
    precompute_balanced,
    precompute_probabilistic,
    precompute_team_draft,
    probabilistic_marginal_outcome,
    team_draft_outcome_batch,
)
from hybridsearch.clicks.models import ClickModel
from hybridsearch.clicks.rankers import RankerTable

AB_METRICS = abstats.METRICS
INTERLEAVE_METHODS = ("tdi", "bi", "pi")


# ------------------------------------------------------------------ pair data
@dataclass
class Entries:
    """Flattened interleavings over all queries: grades (E,K), method aux, prob (E,)."""

    grades: np.ndarray
    teams: np.ndarray | None = None
    aux_a: np.ndarray | None = None
    aux_b: np.ndarray | None = None
    prob: np.ndarray | None = None


@dataclass
class Pair:
    a: str
    b: str
    grades_a: np.ndarray
    grades_b: np.ndarray
    ndcg_a: np.ndarray
    ndcg_b: np.ndarray
    entries: dict[str, Entries] = field(default_factory=dict)

    @property
    def offline_delta(self) -> float:
        """mean nDCG@10(A) - mean nDCG@10(B)."""
        return float(self.ndcg_a.mean() - self.ndcg_b.mean())


def build_pair(table: RankerTable, a: str, b: str, rng: np.random.Generator,
               pi_samples: int = 256, methods=INTERLEAVE_METHODS) -> Pair:
    k = table.k
    pair = Pair(a, b, table.grades(a), table.grades(b), table.ndcg(a), table.ndcg(b))
    nq = len(table.qids)
    acc = {m: {"grades": [], "teams": [], "aux_a": [], "aux_b": [], "prob": []} for m in methods}
    for i, q in enumerate(table.qids):
        la, lb = table.lists[a][i], table.lists[b][i]
        docs = list(dict.fromkeys(la + lb))
        idx = {d: j for j, d in enumerate(docs)}
        g = np.array([table.qrels[q].get(d, 0) for d in docs] + [-1], np.int64)  # last = empty
        ia, ib = [idx[d] for d in la], [idx[d] for d in lb]
        for m in methods:
            if m == "tdi":
                pool = precompute_team_draft(ia, ib, k)
            elif m == "bi":
                pool = precompute_balanced(ia, ib, k)
            else:
                pool = precompute_probabilistic(ia, ib, k, rng, m=pi_samples)
            acc[m]["grades"].append(g[pool.docs])  # -1 index hits the "empty" sentinel
            acc[m]["prob"].append(pool.prob / nq)
            for key, val in (("teams", pool.teams), ("aux_a", pool.aux_a), ("aux_b", pool.aux_b)):
                if val is not None:
                    acc[m][key].append(val)
    for m in methods:
        d = acc[m]
        pair.entries[m] = Entries(
            grades=np.concatenate(d["grades"]),
            teams=np.concatenate(d["teams"]) if d["teams"] else None,
            aux_a=np.concatenate(d["aux_a"]) if d["aux_a"] else None,
            aux_b=np.concatenate(d["aux_b"]) if d["aux_b"] else None,
            prob=np.concatenate(d["prob"]),
        )
    return pair


# ------------------------------------------------------------------ outcome computation
def interleave_outcomes(method: str, ent: Entries, idx: np.ndarray, clicks: np.ndarray,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-impression outcome, > 0 means A preferred."""
    if method == "tdi":
        return team_draft_outcome_batch(ent.teams[idx], clicks).astype(float)
    if method == "bi":
        return balanced_outcome_batch(ent.aux_a[idx], ent.aux_b[idx], clicks).astype(float)
    if method == "pi":
        # Stochastic rounding to a 0.01 grid (round up with probability equal to the
        # remainder): leaves every outcome's expectation unchanged, adds variance
        # <= 0.0025/4, and caps the support at 201 values so multinomial trials stay cheap.
        o = probabilistic_marginal_outcome(ent.aux_a[idx], clicks) * 100.0
        lo = np.floor(o)
        return (lo + (rng.random(len(o)) < (o - lo))) / 100.0
    raise ValueError(method)


def ab_codes(clicks: np.ndarray) -> dict[str, np.ndarray]:
    """Integer codes for each A/B metric (value = VALUES[metric][code])."""
    any_c = clicks.any(1)
    first = np.argmax(clicks, axis=1)
    return {
        "ctr": clicks.sum(1),
        "clicks_at_1": clicks[:, 0].astype(np.int64),
        "abandonment": (~any_c).astype(np.int64),
        "rr_first": np.where(any_c, first + 1, 0),
    }


def ab_values(k: int) -> dict[str, np.ndarray]:
    return {
        "ctr": np.arange(k + 1, dtype=float),
        "clicks_at_1": np.array([0.0, 1.0]),
        "abandonment": np.array([0.0, 1.0]),
        "rr_first": np.concatenate([[0.0], 1.0 / np.arange(1, k + 1)]),
    }


@dataclass
class Dist:
    """Discrete distribution: values and probabilities, plus the pool size."""

    values: np.ndarray
    probs: np.ndarray
    m: int

    @property
    def mean(self) -> float:
        return float(self.values @ self.probs)

    @property
    def var(self) -> float:
        return float((self.values ** 2) @ self.probs - self.mean ** 2)


def pool_ab(grades: np.ndarray, model: ClickModel, m: int, rng, chunk: int = 250_000) -> dict[str, Dist]:
    k = grades.shape[1]
    vals = ab_values(k)
    counts = {mt: np.zeros(len(vals[mt]), np.int64) for mt in AB_METRICS}
    done = 0
    while done < m:
        n = min(chunk, m - done)
        q = rng.integers(0, len(grades), size=n)
        c = model.simulate(grades[q], rng).clicks
        for mt, code in ab_codes(c).items():
            counts[mt] += np.bincount(code, minlength=len(vals[mt]))
        done += n
    return {mt: Dist(vals[mt], counts[mt] / m, m) for mt in AB_METRICS}


def pool_interleave(method: str, ent: Entries, model: ClickModel, m: int, rng,
                    chunk: int = 250_000) -> Dist:
    acc: dict[float, int] = {}
    cdf = np.cumsum(ent.prob)
    cdf /= cdf[-1]
    done = 0
    while done < m:
        n = min(chunk, m - done)
        idx = np.minimum(np.searchsorted(cdf, rng.random(n), side="right"), len(cdf) - 1)
        c = model.simulate(ent.grades[idx], rng).clicks
        o = interleave_outcomes(method, ent, idx, c, rng)
        v, cnt = np.unique(o, return_counts=True)
        for vv, cc in zip(v.tolist(), cnt.tolist()):
            acc[vv] = acc.get(vv, 0) + cc
        done += n
    v = np.array(sorted(acc))
    p = np.array([acc[x] for x in v], float) / m
    return Dist(v, p, m)


# ------------------------------------------------------------------ power
def n_grid(lo: float = 1, hi: float = 8, per_decade: int = 20) -> np.ndarray:
    return np.unique(np.round(np.logspace(lo, hi, int((hi - lo) * per_decade) + 1)).astype(np.int64))


def power_interleave(method: str, dist: Dist, ns: np.ndarray, trials: int, rng,
                     a_is_better: bool, alpha: float = 0.05):
    """Returns (power_correct, rate_wrong_direction) arrays over ``ns``."""
    pc, pw = np.zeros(len(ns)), np.zeros(len(ns))
    sgn = 1.0 if a_is_better else -1.0
    for i, n in enumerate(ns):
        cnt = rng.multinomial(int(n), dist.probs, size=trials)
        if method in ("tdi", "bi"):
            wins = cnt[:, dist.values > 0].sum(1)
            losses = cnt[:, dist.values < 0].sum(1)
            p = abstats.sign_test_p(wins, losses)
            direction = np.sign(wins - losses) * sgn
        else:
            mean = cnt @ dist.values / n
            var = np.maximum((cnt @ dist.values ** 2 - n * mean ** 2) / max(n - 1, 1), 0)
            p = abstats.one_sample_p(mean, var, n)
            direction = np.sign(mean) * sgn
        sig = p < alpha
        pc[i] = np.mean(sig & (direction > 0))
        pw[i] = np.mean(sig & (direction < 0))
    return pc, pw


def power_ab(metric: str, da: Dist, db: Dist, ns: np.ndarray, trials: int, rng,
             a_is_better: bool, alpha: float = 0.05):
    pc, pw = np.zeros(len(ns)), np.zeros(len(ns))
    good = 1.0 if abstats.HIGHER_IS_BETTER[metric] else -1.0
    sgn = 1.0 if a_is_better else -1.0
    for i, n in enumerate(ns):
        na = max(int(n) // 2, 2)
        nb = max(int(n) - na, 2)
        ca = rng.multinomial(na, da.probs, size=trials)
        cb = rng.multinomial(nb, db.probs, size=trials)
        ma, mb = ca @ da.values / na, cb @ db.values / nb
        va = np.maximum((ca @ da.values ** 2 - na * ma ** 2) / (na - 1), 0)
        vb = np.maximum((cb @ db.values ** 2 - nb * mb ** 2) / (nb - 1), 0)
        p = abstats.welch_p(mb, vb, nb, ma, va, na)  # sign convention irrelevant (two-sided)
        direction = np.sign(ma - mb) * good * sgn     # > 0: metric favours offline-better
        sig = p < alpha
        pc[i] = np.mean(sig & (direction > 0))
        pw[i] = np.mean(sig & (direction < 0))
    return pc, pw


def n_at_power(ns: np.ndarray, power: np.ndarray, target: float = 0.8):
    """Smallest N where the isotonic fit of power(log N) reaches ``target``, log-linearly
    interpolated. None if never reached on the grid."""
    x = np.log10(ns.astype(float))
    fit = IsotonicRegression(increasing=True).fit_transform(x, power)
    hit = np.nonzero(fit >= target)[0]
    if len(hit) == 0:
        return None
    j = hit[0]
    if j == 0:
        return float(ns[0])
    x0, x1, y0, y1 = x[j - 1], x[j], fit[j - 1], fit[j]
    xt = x1 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float(10 ** xt)


def analytic_n80_ab(da: Dist, db: Dist, alpha=0.05, power=0.8):
    """Normal-approximation cross-check: total N (both arms, N/2 each)."""
    from scipy.stats import norm
    d = da.mean - db.mean
    if d == 0:
        return None
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(2 * z ** 2 * (da.var + db.var) / d ** 2)


def analytic_n80_one_sample(dist: Dist, alpha=0.05, power=0.8):
    from scipy.stats import norm
    if dist.mean == 0:
        return None
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z ** 2 * dist.var / dist.mean ** 2)


# ------------------------------------------------------------------ direct simulation
def direct_interleave_pvalues(method: str, ent: Entries, model: ClickModel, n: int, trials: int,
                              rng, chunk_rows: int = 500_000):
    """Fresh click simulation for every trial (no pool). Returns (p_values, direction)
    where direction > 0 means A preferred."""
    cdf = np.cumsum(ent.prob)
    cdf /= cdf[-1]
    per = max(1, chunk_rows // n)
    ps, dirs = [], []
    t = 0
    while t < trials:
        tt = min(per, trials - t)
        idx = np.minimum(np.searchsorted(cdf, rng.random(tt * n), side="right"), len(cdf) - 1)
        c = model.simulate(ent.grades[idx], rng).clicks
        o = interleave_outcomes(method, ent, idx, c, rng).reshape(tt, n)
        if method in ("tdi", "bi"):
            w, l_ = (o > 0).sum(1), (o < 0).sum(1)
            ps.append(abstats.sign_test_p(w, l_))
            dirs.append(np.sign(w - l_))
        else:
            ps.append(abstats.one_sample_p(o.mean(1), o.var(1, ddof=1), n))
            dirs.append(np.sign(o.mean(1)))
        t += tt
    return np.concatenate(ps), np.concatenate(dirs)


def direct_ab_pvalues(grades_a, grades_b, model: ClickModel, n: int, trials: int, rng,
                      chunk_rows: int = 500_000):
    """Fresh simulation per trial; N/2 impressions per arm. Returns {metric: (p, dir)}
    where dir = sign(mean_A - mean_B)."""
    na, nb = n // 2, n - n // 2
    per = max(1, chunk_rows // n)
    k = grades_a.shape[1]
    vals = ab_values(k)
    out = {m: ([], []) for m in AB_METRICS}
    t = 0
    while t < trials:
        tt = min(per, trials - t)
        qa = rng.integers(0, len(grades_a), size=tt * na)
        qb = rng.integers(0, len(grades_b), size=tt * nb)
        cda = ab_codes(model.simulate(grades_a[qa], rng).clicks)
        cdb = ab_codes(model.simulate(grades_b[qb], rng).clicks)
        for m in AB_METRICS:
            xa = vals[m][cda[m]].reshape(tt, na)
            xb = vals[m][cdb[m]].reshape(tt, nb)
            p = abstats.welch_p(xa.mean(1), xa.var(1, ddof=1), na, xb.mean(1), xb.var(1, ddof=1), nb)
            out[m][0].append(p)
            out[m][1].append(np.sign(xa.mean(1) - xb.mean(1)))
        t += tt
    return {m: (np.concatenate(v[0]), np.concatenate(v[1])) for m, v in out.items()}
