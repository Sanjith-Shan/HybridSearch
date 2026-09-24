"""Simulated click logs from a production (logging) ranker, with optional randomised
interventions, and position-bias estimation from those interventions.

All clicks are from simulated users (PBM, relevance from TREC DL / MS MARCO judgments).

Logged data are aggregated as ``clicks_at[i, r]``: clicks on candidate ``i`` (global index
into the concatenated candidate lists) while displayed at rank ``r``. That is sufficient
for every estimator here, including IPS with any propensity estimate.

Interventions (Joachims, Swaminathan & Schnabel, WSDM 2017, Section 5):

* ``randtop``: with probability ``eps`` a session shows a uniformly random permutation of
  the top ``n`` results. In those sessions relevance is independent of position, so
  CTR@r / CTR@1 = eta_r / eta_1.
* ``swap`` (RandPair-style swap of ranks 1 and k): with probability ``eps`` a session is an
  intervention for a uniformly random k in 2..K; with probability 1/2 the results at
  ranks 1 and k are swapped. For the originally-top result,
  CTR(shown at k | swapped) / CTR(shown at 1 | not swapped) = eta_k / eta_1.
  This touches only two positions per intervened session, so it costs much less user
  experience than ``randtop``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hybridsearch.clicks.models import PBM


@dataclass
class Logs:
    clicks_at: np.ndarray            # (N_candidates, K) clicks by displayed rank
    sessions_per_query: np.ndarray   # (Q,)
    n_sessions: int
    # intervention statistics
    randtop_clicks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    randtop_sessions: int = 0
    randtop_n: int = 0
    swap_click_at_k: np.ndarray = field(default_factory=lambda: np.zeros(0))   # [k] clicks on orig-top at k
    swap_n_swapped: np.ndarray = field(default_factory=lambda: np.zeros(0))
    swap_click_at_1: np.ndarray = field(default_factory=lambda: np.zeros(0))   # [k] clicks on orig-top at 1
    swap_n_control: np.ndarray = field(default_factory=lambda: np.zeros(0))


def simulate_logs(grades_log_order: np.ndarray, offsets: np.ndarray, model: PBM, n_sessions: int,
                  rng: np.random.Generator, intervention: str | None = None, eps: float = 0.0,
                  randtop_n: int = 10, chunk: int = 200_000) -> Logs:
    """``grades_log_order`` (Q, K): grade of the candidate the logging ranker puts at each
    rank (-1 = no candidate). Candidate i of query q (local index = its logging rank) is
    global index ``offsets[q] + i``."""
    Q, K = grades_log_order.shape
    n_cand = offsets[-1]
    clicks_at = np.zeros((n_cand, K), np.int64)
    spq = np.zeros(Q, np.int64)
    rt_clicks = np.zeros(K, np.int64)
    rt_sessions = 0
    sw_k = np.zeros(K, np.int64)
    sw_ns = np.zeros(K, np.int64)
    sw_1 = np.zeros(K, np.int64)
    sw_nc = np.zeros(K, np.int64)
    n_real = np.diff(offsets)
    done = 0
    while done < n_sessions:
        n = min(chunk, n_sessions - done)
        q = rng.integers(0, Q, size=n)
        np.add.at(spq, q, 1)
        perm = np.tile(np.arange(K), (n, 1))  # perm[s, r] = local candidate shown at rank r
        is_rt = np.zeros(n, bool)
        sw_kk = np.full(n, -1)
        sw_do = np.zeros(n, bool)
        if intervention == "randtop" and eps > 0:
            # only lists with at least ``randtop_n`` candidates, so every shuffled rank
            # holds a real candidate and CTR ratios are not diluted by empty slots
            is_rt = (rng.random(n) < eps) & (n_real[q] >= randtop_n)
            m = int(is_rt.sum())
            if m:
                keys = rng.random((m, K))
                keys[:, randtop_n:] = np.arange(randtop_n, K) + 10.0  # tail keeps its order
                perm[is_rt] = np.argsort(keys, axis=1)
        elif intervention == "swap" and eps > 0:
            inter = rng.random(n) < eps
            idx = np.nonzero(inter)[0]
            if len(idx):
                kmax = np.minimum(n_real[q[idx]], K)
                ok = kmax >= 2
                idx, kmax = idx[ok], kmax[ok]
                kk = 1 + (rng.random(len(idx)) * (kmax - 1)).astype(np.int64)  # 1..kmax-1 (0-based)
                do = rng.random(len(idx)) < 0.5
                sw_kk[idx] = kk
                sw_do[idx] = do
                rows = idx[do]
                perm[rows, 0] = kk[do]
                perm[rows, kk[do]] = 0
        g = np.take_along_axis(grades_log_order[q], perm, axis=1)
        c = model.simulate(g, rng).clicks
        # accumulate clicks by (global candidate, displayed rank)
        s_idx, r_idx = np.nonzero(c)
        gidx = offsets[q[s_idx]] + perm[s_idx, r_idx]
        np.add.at(clicks_at, (gidx, r_idx), 1)
        if is_rt.any():
            rt_clicks += c[is_rt].sum(0)
            rt_sessions += int(is_rt.sum())
        if (sw_kk >= 0).any():
            sel = sw_kk >= 0
            kk = sw_kk[sel]
            do = sw_do[sel]
            cc = c[sel]
            rows = np.arange(len(kk))
            # swapped: orig-top shown at kk; control: orig-top shown at rank 1 (index 0)
            np.add.at(sw_k, kk[do], cc[rows[do], kk[do]])
            np.add.at(sw_ns, kk[do], 1)
            np.add.at(sw_1, kk[~do], cc[rows[~do], 0])
            np.add.at(sw_nc, kk[~do], 1)
        done += n
    return Logs(clicks_at, spq, n_sessions, rt_clicks, rt_sessions, randtop_n, sw_k, sw_ns, sw_1, sw_nc)


def estimate_propensity(logs: Logs, method: str) -> np.ndarray:
    """Relative position bias eta_r / eta_1 (eta_1 := 1). NaN where there is no data."""
    K = logs.clicks_at.shape[1]
    if method == "randtop":
        c = logs.randtop_clicks.astype(float)
        out = np.full(K, np.nan)
        if c[0] > 0:
            n = logs.randtop_n
            out[:n] = c[:n] / c[0]  # ranks beyond the shuffled window are not identified
        return out
    if method == "swap":
        with np.errstate(divide="ignore", invalid="ignore"):
            ctr_k = logs.swap_click_at_k / logs.swap_n_swapped
            ctr_1 = logs.swap_click_at_1 / logs.swap_n_control
            est = ctr_k / ctr_1
        est = est.astype(float)
        est[0] = 1.0
        return est
    raise ValueError(method)


def fill_propensity(est: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Replace NaN/inf/non-positive entries (ranks with no intervention data) by the last
    finite value to their left, and floor at ``floor``; ranks beyond the randomised
    window are extrapolated as flat (a stated, conservative choice)."""
    out = np.array(est, float)
    last = 1.0
    for i in range(len(out)):
        if not np.isfinite(out[i]) or out[i] <= 0:
            out[i] = last
        last = out[i]
    return np.maximum(out, floor)


def propensity_support(logs: Logs, method: str) -> np.ndarray:
    """Weight behind each rank's ratio estimate: the smaller of the two click counts the
    ratio is made of (the log-ratio's variance is ~ 1/c1 + 1/c2). 0 = unidentified; a
    ratio built on zero clicks is treated as unknown, not as eta = 0."""
    K = logs.clicks_at.shape[1]
    if method == "randtop":
        w = np.zeros(K)
        n = logs.randtop_n
        w[:n] = np.minimum(logs.randtop_clicks[:n], logs.randtop_clicks[0])
        return w.astype(float)
    w = np.minimum(logs.swap_click_at_k, logs.swap_click_at_1).astype(float)
    w[0] = max(w.max(), 1.0)
    return w


def monotone_propensity(est: np.ndarray, support: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Weighted isotonic (non-increasing in rank) fit of the raw ratio estimates, weights =
    clicks behind each rank's ratio; unidentified ranks take the last fitted value.
    Raw per-rank ratios at deep ranks rest on few clicks and can be near 0, which turns
    into huge IPS weights; the examination hypothesis says eta is non-increasing in rank
    for a top-down scan, so we impose that."""
    from sklearn.isotonic import IsotonicRegression

    est = np.asarray(est, float)
    ok = np.isfinite(est) & (support > 0) & (est > 0)
    out = np.empty_like(est)
    if ok.sum() < 2:
        return fill_propensity(est, floor)
    r = np.arange(len(est))
    fit = IsotonicRegression(increasing=False).fit(r[ok], est[ok], sample_weight=support[ok])
    out[:] = fit.predict(r)
    last_ok = r[ok].max()
    out[r > last_ok] = out[last_ok]
    out = out / out[0] if out[0] > 0 else out
    return np.maximum(out, floor)


def ips_weights(clicks_at: np.ndarray, eta: np.ndarray | None, clip: float | None = None) -> np.ndarray:
    """Per-candidate label weight: sum_r clicks_at[:, r] * min(1/eta_r, clip).
    ``eta=None`` gives the naive (unweighted) click count."""
    if eta is None:
        return clicks_at.sum(1).astype(float)
    inv = 1.0 / np.asarray(eta, float)
    if clip is not None:
        inv = np.minimum(inv, clip)
    return clicks_at @ inv


# ------------------------------------------------------------------ intervention harvesting
def simulate_logs_multi(grades: np.ndarray, orders: np.ndarray, offsets: np.ndarray, model: PBM,
                        n_sessions: int, rng: np.random.Generator, chunk: int = 200_000):
    """Several production rankers ("loggers") share traffic: each session picks logger j
    uniformly and shows candidates in ``orders[j, q]`` (local candidate indices by rank,
    -1 padded). ``grades`` (Q, K) is indexed by local candidate index. No deliberate
    randomisation of any single ranking (Agarwal et al., WSDM 2019).
    Returns (clicks_at (N_cand, K), sessions (Q, L))."""
    L, Q, K = orders.shape
    clicks_at = np.zeros((offsets[-1], K), np.int64)
    sessions = np.zeros((Q, L), np.int64)
    done = 0
    while done < n_sessions:
        n = min(chunk, n_sessions - done)
        q = rng.integers(0, Q, size=n)
        j = rng.integers(0, L, size=n)
        np.add.at(sessions, (q, j), 1)
        perm = orders[j, q]                              # (n, K) local candidate at each rank
        g = np.where(perm >= 0, np.take_along_axis(grades[q], np.maximum(perm, 0), axis=1), -1)
        c = model.simulate(g, rng).clicks
        s_idx, r_idx = np.nonzero(c)
        np.add.at(clicks_at, (offsets[q[s_idx]] + perm[s_idx, r_idx], r_idx), 1)
        done += n
    return clicks_at, sessions


def harvest_propensity(clicks_at: np.ndarray, sessions: np.ndarray, orders: np.ndarray,
                       offsets: np.ndarray, floor: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Intervention harvesting (Agarwal, Zaitsev, Wang, Li, Najork & Joachims, WSDM 2019),
    count-aggregated form: for every candidate that two loggers show at different ranks
    (r, r'), relevance is the same, so summed over such candidates
    (C_r / N_r) / (C_r' / N_r') estimates eta_r / eta_r'. The pairwise log-ratios are then
    combined by weighted least squares in log space with eta_1 = 1.
    Returns (eta estimate, support = number of informative rank pairs per rank)."""
    L, Q, K = orders.shape
    C = np.zeros((K, K))   # C[r, r']: clicks at r on docs that the other logger shows at r'
    N = np.zeros((K, K))   # sessions behind C
    for q in range(Q):
        rank = np.full((L, K), -1)
        for j in range(L):
            o = orders[j, q]
            ok = o >= 0
            rank[j, o[ok]] = np.nonzero(ok)[0]
        for a in range(L):
            for b in range(L):
                if a == b:
                    continue
                m = (rank[a] >= 0) & (rank[b] >= 0) & (rank[a] != rank[b])
                if not m.any():
                    continue
                ra, rb = rank[a, m], rank[b, m]
                cl = clicks_at[offsets[q] + np.nonzero(m)[0], ra]
                np.add.at(C, (ra, rb), cl)
                np.add.at(N, (ra, rb), sessions[q, a])
    rows, rhs, w = [], [], []
    for r in range(K):
        for r2 in range(r + 1, K):
            c1, c2 = C[r, r2], C[r2, r]
            if c1 > 0 and c2 > 0 and N[r, r2] > 0 and N[r2, r] > 0:
                x = np.zeros(K)
                x[r], x[r2] = 1.0, -1.0
                rows.append(x)
                rhs.append(np.log(c1 / N[r, r2]) - np.log(c2 / N[r2, r]))
                w.append(1.0 / (1.0 / c1 + 1.0 / c2))           # inverse Poisson variance of the log-ratio
    support = np.zeros(K)
    if not rows:
        return np.full(K, np.nan), support
    A, y, W = np.array(rows), np.array(rhs), np.sqrt(np.array(w))
    for x in A:
        support[x != 0] += 1
    ident = support > 0
    ident[0] = True
    A2 = (A[:, ident])[:, 1:] * W[:, None]                     # drop eta_1 (fixed at log 1 = 0)
    sol, *_ = np.linalg.lstsq(A2, y * W, rcond=None)
    est = np.full(K, np.nan)
    est[ident] = np.exp(np.concatenate([[0.0], sol]))
    return np.maximum(est, floor), support
