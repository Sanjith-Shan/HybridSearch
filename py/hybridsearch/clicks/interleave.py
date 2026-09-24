"""Interleaved comparison of two rankers.

* **Team-draft interleaving** (Radlinski, Kurup & Joachims, CIKM 2008). The exact
  variant implemented here is pinned by ``tests/golden/interleaving.json``, which the C#
  broker's implementation must also pass:

  - ``I = []``, ``TA = TB = 0`` (team sizes).
  - While ``len(I) < k`` and at least one ranking still has a document not in ``I``:
    - If ``TA < TB`` A picks; if ``TB < TA`` B picks; if equal, **consume the next coin**:
      ``true`` means A picks.
    - The picking team appends its highest-ranked document not already in ``I`` and
      that position is credited to it. If the picking team has no document left, the
      other team picks instead (and owns the position). The coin, if one was consumed,
      is not re-flipped.
  - Credit: each click counts for the team owning that position; the team with more
    clicks wins the impression; equal counts (including zero clicks) are a tie.

* **Balanced interleaving** (Joachims 2002), with the standard credit rule.
* **Probabilistic interleaving** (Hofmann, Whiteson & de Rijke, CIKM 2011): softmax over
  ranks with ``tau = 3`` and the *marginalised* credit (expectation of the outcome over
  every team assignment that could have produced the observed list).

Scalar functions work on Python lists (used by the golden tests and by the event
analyser). The ``*_batch`` / ``precompute_*`` functions are the vectorised forms used in
the simulation study.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

TEAM_A, TEAM_B, NO_TEAM = 0, 1, -1


# ------------------------------------------------------------------ team-draft (scalar)
class CoinsExhausted(ValueError):
    pass


def team_draft(a: Sequence, b: Sequence, k: int, coins: Iterable[bool]):
    """Returns ``(interleaved, teams, coins_consumed)``; ``teams`` holds "A"/"B"."""
    coins = iter(coins)
    out: list = []
    teams: list[str] = []
    seen: set = set()
    ta = tb = 0
    ia = ib = 0  # cursors: everything before them is already in ``seen``
    used = 0

    def advance(lst, i):
        while i < len(lst) and lst[i] in seen:
            i += 1
        return i

    while len(out) < k:
        ia, ib = advance(a, ia), advance(b, ib)
        a_left, b_left = ia < len(a), ib < len(b)
        if not a_left and not b_left:
            break
        if ta < tb:
            pick_a = True
        elif tb < ta:
            pick_a = False
        else:
            try:
                pick_a = bool(next(coins))
            except StopIteration:
                raise CoinsExhausted("ran out of coins") from None
            used += 1
        if pick_a and not a_left:
            pick_a = False
        elif not pick_a and not b_left:
            pick_a = True
        if pick_a:
            doc = a[ia]
            ta += 1
            teams.append("A")
        else:
            doc = b[ib]
            tb += 1
            teams.append("B")
        out.append(doc)
        seen.add(doc)
    return out, teams, used


def team_draft_random(a, b, k, rng: np.random.Generator):
    out, teams, _ = team_draft(a, b, k, (rng.random() < 0.5 for _ in itertools.count()))
    return out, teams


def team_draft_credit(teams: Sequence[str], clicked_positions: Iterable[int]):
    """``clicked_positions`` are 0-based positions in the interleaved list.
    Returns ``(clicks_a, clicks_b, winner)`` with winner in {"A", "B", "tie"}."""
    ca = cb = 0
    for p in set(clicked_positions):
        if teams[p] == "A":
            ca += 1
        elif teams[p] == "B":
            cb += 1
    return ca, cb, ("A" if ca > cb else "B" if cb > ca else "tie")


def enumerate_team_draft(a, b, k):
    """Every outcome of team-draft with its probability. Coins are consumed only on
    team-size ties, so there are at most 2^ceil(k/2) outcomes.
    Returns list of (interleaved, teams, prob)."""
    results = []

    def rec(prefix: tuple[bool, ...]):
        try:
            out, teams, used = team_draft(a, b, k, prefix)
        except CoinsExhausted:
            rec(prefix + (True,))
            rec(prefix + (False,))
            return
        results.append((out, teams, 0.5 ** used))

    rec(())
    return results


# ------------------------------------------------------------------ balanced (scalar)
def balanced(a: Sequence, b: Sequence, k: int, a_first: bool):
    out: list = []
    seen: set = set()
    ka = kb = 0
    while len(out) < k and (ka < len(a) or kb < len(b)):
        take_a = (ka < kb or (ka == kb and a_first)) if (ka < len(a) and kb < len(b)) else ka < len(a)
        if take_a:
            d = a[ka]
            ka += 1
        else:
            d = b[kb]
            kb += 1
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def balanced_credit(a, b, interleaved, clicked_positions):
    clicked = sorted(set(clicked_positions))
    if not clicked:
        return 0, 0, "tie"
    ra = {d: i + 1 for i, d in enumerate(a)}
    rb = {d: i + 1 for i, d in enumerate(b)}
    inf = 10**9
    last = interleaved[clicked[-1]]
    kk = min(ra.get(last, inf), rb.get(last, inf))
    docs = [interleaved[p] for p in clicked]
    ha = sum(1 for d in docs if ra.get(d, inf) <= kk)
    hb = sum(1 for d in docs if rb.get(d, inf) <= kk)
    return ha, hb, ("A" if ha > hb else "B" if hb > ha else "tie")


# ------------------------------------------------------------------ probabilistic
def _softmax_rank_weights(n: int, tau: float) -> np.ndarray:
    return 1.0 / np.arange(1, n + 1, dtype=float) ** tau


def probabilistic_sample(a, b, k, rng, tau: float = 3.0):
    """Returns (interleaved, teams, p_a) where p_a[r] is the posterior probability that
    position r was contributed by A given the list (used for marginalised credit)."""
    wa = {d: w for d, w in zip(a, _softmax_rank_weights(len(a), tau))}
    wb = {d: w for d, w in zip(b, _softmax_rank_weights(len(b), tau))}
    rem_a, rem_b = dict(wa), dict(wb)
    out, teams, pa = [], [], []
    while len(out) < k and (rem_a or rem_b):
        sa, sb = sum(rem_a.values()), sum(rem_b.values())
        pick_a = rng.random() < 0.5
        if pick_a and not rem_a:
            pick_a = False
        if not pick_a and not rem_b:
            pick_a = True
        src = rem_a if pick_a else rem_b
        docs = list(src)
        w = np.array([src[d] for d in docs])
        d = docs[int(rng.choice(len(docs), p=w / w.sum()))]
        la = rem_a.get(d, 0.0) / sa if sa > 0 else 0.0
        lb = rem_b.get(d, 0.0) / sb if sb > 0 else 0.0
        pa.append(la / (la + lb))
        out.append(d)
        teams.append("A" if pick_a else "B")
        rem_a.pop(d, None)
        rem_b.pop(d, None)
    return out, teams, np.array(pa)


def probabilistic_marginal_outcome(p_a: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    """Vectorised marginalised PI credit. ``p_a``, ``clicks``: (n, K).
    Returns P(A wins) - P(B wins) per impression, in [-1, 1]."""
    n, k = clicks.shape
    dist = np.zeros((n, 2 * k + 1))
    dist[:, k] = 1.0
    pa = np.where(clicks, p_a, 0.0)
    pb = np.where(clicks, 1.0 - p_a, 0.0)
    stay = 1.0 - pa - pb
    for r in range(k):
        new = dist * stay[:, r:r + 1]
        new[:, 1:] += dist[:, :-1] * pa[:, r:r + 1]
        new[:, :-1] += dist[:, 1:] * pb[:, r:r + 1]
        dist = new
    return dist[:, k + 1:].sum(1) - dist[:, :k].sum(1)


# ------------------------------------------------------------------ vectorised helpers
def team_draft_outcome_batch(teams: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    """teams (n,K) in {0,1,-1}; returns sign(clicks_A - clicks_B) in {+1 (A), 0, -1 (B)}."""
    ca = (clicks & (teams == TEAM_A)).sum(1)
    cb = (clicks & (teams == TEAM_B)).sum(1)
    return np.sign(ca - cb).astype(np.int8)


def balanced_outcome_batch(rank_a: np.ndarray, rank_b: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    """rank_a/rank_b (n,K): 1-based rank of the doc at each interleaved position in A / B
    (large sentinel if absent)."""
    n, k = clicks.shape
    any_click = clicks.any(1)
    last = k - 1 - np.argmax(clicks[:, ::-1], axis=1)
    rows = np.arange(n)
    kk = np.minimum(rank_a[rows, last], rank_b[rows, last])[:, None]
    ha = (clicks & (rank_a <= kk)).sum(1)
    hb = (clicks & (rank_b <= kk)).sum(1)
    return np.where(any_click, np.sign(ha - hb), 0).astype(np.int8)


@dataclass
class InterleavePool:
    """Precomputed interleavings for one query: ``docs`` (m, K) indices into the query's
    doc table (-1 empty), ``aux`` per method, ``prob`` (m,) sampling probabilities."""

    docs: np.ndarray
    teams: np.ndarray | None
    aux_a: np.ndarray | None
    aux_b: np.ndarray | None
    prob: np.ndarray


def _pad(lst, k, fill=-1):
    arr = np.full(k, fill, dtype=np.int64)
    arr[: len(lst)] = lst[:k]
    return arr


def precompute_team_draft(a, b, k) -> InterleavePool:
    outs = enumerate_team_draft(list(a), list(b), k)
    docs = np.stack([_pad(o, k) for o, _, _ in outs])
    teams = np.stack([_pad([TEAM_A if t == "A" else TEAM_B for t in t_], k) for _, t_, _ in outs])
    prob = np.array([p for _, _, p in outs])
    return InterleavePool(docs, teams, None, None, prob / prob.sum())


def precompute_balanced(a, b, k) -> InterleavePool:
    big = 10**6
    ra = {d: i + 1 for i, d in enumerate(a)}
    rb = {d: i + 1 for i, d in enumerate(b)}
    lists = [balanced(list(a), list(b), k, af) for af in (True, False)]
    docs = np.stack([_pad(l_, k) for l_ in lists])
    rank_a = np.stack([_pad([ra.get(d, big) for d in l_], k, big) for l_ in lists])
    rank_b = np.stack([_pad([rb.get(d, big) for d in l_], k, big) for l_ in lists])
    return InterleavePool(docs, None, rank_a, rank_b, np.array([0.5, 0.5]))


def precompute_probabilistic(a, b, k, rng, m: int = 512, tau: float = 3.0) -> InterleavePool:
    docs, pas = [], []
    for _ in range(m):
        out, _, pa = probabilistic_sample(list(a), list(b), k, rng, tau)
        docs.append(_pad(out, k))
        p = np.zeros(k)
        p[: len(pa)] = pa
        pas.append(p)
    return InterleavePool(np.stack(docs), None, np.stack(pas), None, np.full(m, 1.0 / m))
