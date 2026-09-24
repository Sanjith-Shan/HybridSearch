"""Interleaving: golden fixture, structural properties, credit assignment."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from hybridsearch.clicks.interleave import (
    balanced,
    balanced_credit,
    balanced_outcome_batch,
    enumerate_team_draft,
    precompute_balanced,
    probabilistic_marginal_outcome,
    probabilistic_sample,
    team_draft,
    team_draft_credit,
    team_draft_outcome_batch,
    team_draft_random,
)

GOLDEN = Path(__file__).resolve().parents[2] / "tests/golden/interleaving.json"


def _golden():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("case", _golden()["cases"], ids=lambda c: c["name"])
def test_golden_team_draft(case):
    out, teams, used = team_draft(case["rankingA"], case["rankingB"], case["k"], case["coins"])
    exp = case["expected"]
    assert out == exp["interleaved"]
    assert teams == exp["teams"]
    assert used == exp["coinsConsumed"]
    for t in case["clickTests"]:
        pos = [out.index(d) for d in t["clickedDocIds"]]
        assert team_draft_credit(teams, pos) == (t["clicksA"], t["clicksB"], t["winner"])


@pytest.mark.parametrize("dist", _golden()["distributions"], ids=lambda d: d["name"])
def test_golden_distributions(dist):
    got = {}
    for out, teams, p in enumerate_team_draft(dist["rankingA"], dist["rankingB"], dist["k"]):
        key = (tuple(out), tuple(teams))
        got[key] = got.get(key, 0) + p
    want = {(tuple(o["interleaved"]), tuple(o["teams"])): o["probability"] for o in dist["outcomes"]}
    assert got == pytest.approx(want)


def test_hand_example_from_paper_shape():
    # identical lists: team labels must still be balanced and random
    assert team_draft([1, 2, 3, 4], [1, 2, 3, 4], 4, [True, False]) == ([1, 2, 3, 4], ["A", "B", "B", "A"], 2)


def _random_lists(rng, universe=30):
    a = list(rng.permutation(universe)[: rng.integers(0, 15)])
    b = list(rng.permutation(universe)[: rng.integers(0, 15)])
    return [int(x) for x in a], [int(x) for x in b]


def test_team_draft_properties_random():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a, b = _random_lists(rng)
        k = int(rng.integers(1, 12))
        out, teams = team_draft_random(a, b, k, rng)
        # no duplicates, only docs from A ∪ B, full length when possible
        assert len(out) == len(set(out))
        assert set(out) <= set(a) | set(b)
        assert len(out) == min(k, len(set(a) | set(b)))
        # order preservation: each team's picks appear in that ranker's order
        for team, lst in (("A", a), ("B", b)):
            picked = [d for d, t in zip(out, teams) if t == team]
            ranks = [lst.index(d) for d in picked]
            assert ranks == sorted(ranks)
        # team balance: sizes differ by at most one while both lists still have docs
        na, nb = teams.count("A"), teams.count("B")
        if not (set(a) <= set(out) or set(b) <= set(out)):
            assert abs(na - nb) <= 1
        # each team's pick is its highest-ranked doc not yet shown
        shown = set()
        for d, t in zip(out, teams):
            lst = a if t == "A" else b
            rem = [x for x in lst if x not in shown]
            assert rem and rem[0] == d
            shown.add(d)


def test_enumeration_probabilities_and_rng_agree():
    a, b = [1, 2, 3, 4, 5], [3, 1, 6, 7, 2]
    outs = enumerate_team_draft(a, b, 5)
    assert sum(p for *_, p in outs) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    n = 40_000
    counts = {}
    for _ in range(n):
        o, t = team_draft_random(a, b, 5, rng)
        counts[(tuple(o), tuple(t))] = counts.get((tuple(o), tuple(t)), 0) + 1
    for o, t, p in outs:
        f = counts.get((tuple(o), tuple(t)), 0) / n
        assert abs(f - p) < 5 * np.sqrt(p * (1 - p) / n)


def test_team_draft_is_fair_under_random_clicks():
    """With identical rankers and clicks independent of documents, A and B win equally
    often (exactly, by enumeration over coins and click patterns)."""
    a = b = [1, 2, 3, 4]
    pa = pb = 0.0
    for out, teams, p in enumerate_team_draft(a, b, 4):
        for mask in itertools.product([0, 1], repeat=4):
            pm = p * 0.3 ** sum(mask) * 0.7 ** (4 - sum(mask))
            _, _, w = team_draft_credit(teams, [i for i, m in enumerate(mask) if m])
            pa += pm * (w == "A")
            pb += pm * (w == "B")
    assert pa == pytest.approx(pb)


def test_vectorised_credit_matches_scalar():
    rng = np.random.default_rng(2)
    teams = rng.integers(0, 2, size=(500, 10))
    clicks = rng.random((500, 10)) < 0.2
    vec = team_draft_outcome_batch(teams, clicks)
    for i in range(500):
        t = ["A" if x == 0 else "B" for x in teams[i]]
        _, _, w = team_draft_credit(t, np.nonzero(clicks[i])[0])
        assert vec[i] == {"A": 1, "B": -1, "tie": 0}[w]


def test_balanced_interleaving_and_credit():
    a, b = [1, 2, 3, 4], [5, 1, 6, 2]
    # Joachims (2002): a duplicate still uses up that ranker's turn
    assert balanced(a, b, 4, True) == [1, 5, 2, 3]
    assert balanced(a, b, 4, False) == [5, 1, 2, 6]
    inter = balanced(a, b, 4, True)
    # click on doc 5 (B's top): k = 1 -> A top-1 {1} has no click, B top-1 {5} has one
    assert balanced_credit(a, b, inter, [1]) == (0, 1, "B")
    # click on doc 1 (A rank 1, B rank 2): k = 1 -> A wins
    assert balanced_credit(a, b, inter, [0]) == (1, 0, "A")
    # clicks on 1 and 5: k = 1 -> one each -> tie
    assert balanced_credit(a, b, inter, [0, 1]) == (1, 1, "tie")
    # vectorised version agrees on random click patterns
    pool = precompute_balanced(a, b, 4)
    rng = np.random.default_rng(3)
    for _ in range(200):
        i = int(rng.integers(0, 2))
        c = rng.random(4) < 0.4
        vec = balanced_outcome_batch(pool.aux_a[[i]], pool.aux_b[[i]], c[None, :])[0]
        lst = [int(x) for x in pool.docs[i]]
        _, _, w = balanced_credit(a, b, lst, np.nonzero(c)[0])
        assert vec == {"A": 1, "B": -1, "tie": 0}[w]


def test_probabilistic_marginal_credit_matches_bruteforce():
    rng = np.random.default_rng(4)
    p_a = rng.random((200, 6))
    clicks = rng.random((200, 6)) < 0.4
    got = probabilistic_marginal_outcome(p_a, clicks)
    for i in range(200):
        idx = np.nonzero(clicks[i])[0]
        exp = 0.0
        for assign in itertools.product([0, 1], repeat=len(idx)):
            pr = np.prod([p_a[i, j] if s else 1 - p_a[i, j] for j, s in zip(idx, assign)])
            na = sum(assign)
            exp += pr * np.sign(na - (len(idx) - na))
        assert got[i] == pytest.approx(exp)


def test_probabilistic_sample_valid():
    rng = np.random.default_rng(5)
    out, teams, pa = probabilistic_sample([1, 2, 3], [3, 4, 5], 5, rng)
    assert len(out) == len(set(out)) == 5 and set(out) == {1, 2, 3, 4, 5}
    assert np.all((pa >= 0) & (pa <= 1))
    # a doc only A has must be credited to A with certainty
    for d, p in zip(out, pa):
        if d in (1, 2):
            assert p == 1.0
        if d in (4, 5):
            assert p == 0.0
