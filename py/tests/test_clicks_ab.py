"""A/B statistics, A/A calibration, SRM, assignment, power engine."""
import numpy as np
import pytest
from scipy import stats

from hybridsearch.clicks import ab
from hybridsearch.clicks import sensitivity as S
from hybridsearch.clicks.events import assign, bucket
from hybridsearch.clicks.models import make_model


def test_sign_test_matches_scipy():
    for w, l_ in [(0, 0), (5, 5), (10, 2), (2, 10), (60, 40), (500, 440), (0, 7)]:
        want = stats.binomtest(w, w + l_, 0.5).pvalue if w + l_ else 1.0
        assert ab.sign_test_p(w, l_) == pytest.approx(want, rel=1e-9)


def test_welch_matches_scipy_large_n():
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 5000), rng.normal(0.05, 2, 7000)
    got = ab.welch_p(x.mean(), x.var(ddof=1), len(x), y.mean(), y.var(ddof=1), len(y))
    want = stats.ttest_ind(x, y, equal_var=False).pvalue
    assert got == pytest.approx(want, rel=1e-2)  # normal vs t reference: <1% apart at n=5000


def test_impression_metrics():
    c = np.array([[0, 1, 0], [0, 0, 0], [1, 1, 0]], bool)
    m = ab.impression_metrics(c)
    assert m["ctr"].tolist() == [1, 0, 2]
    assert m["clicks_at_1"].tolist() == [0, 0, 1]
    assert m["abandonment"].tolist() == [0, 1, 0]
    assert m["rr_first"].tolist() == [0.5, 0, 1]


def _aa_pvalues(kind, trials=400, n=400, seed=0):
    rng = np.random.default_rng(seed)
    model = make_model(kind, "informational")
    grades = rng.integers(0, 4, size=(50, 10))
    out = S.direct_ab_pvalues(grades, grades, model, n, trials, rng)
    return out


def test_aa_pvalues_uniform_for_ab_metrics():
    """A/A null calibration: same ranker in both arms -> Welch p-values ~ U(0,1)."""
    out = _aa_pvalues("dbn")
    for m in ("ctr", "rr_first"):
        p = out[m][0]
        assert stats.kstest(p, "uniform").pvalue > 0.001, m
        assert abs(np.mean(p < 0.05) - 0.05) < 0.035, m


def test_aa_interleaving_false_positive_rate():
    """Interleaving a ranker with itself: the sign test must not reject more than alpha
    (it is conservative because the binomial is discrete)."""
    from hybridsearch.clicks.rankers import RankerTable
    rng = np.random.default_rng(1)
    qrels = {str(q): {str(d): int(rng.integers(0, 4)) for d in range(30)} for q in range(20)}
    lists = [[str(d) for d in rng.permutation(30)[:10]] for _ in range(20)]
    table = RankerTable([str(q) for q in range(20)], {"x": lists, "y": lists}, qrels, 10)
    pair = S.build_pair(table, "x", "y", rng, methods=("tdi",))
    p, _ = S.direct_interleave_pvalues("tdi", pair.entries["tdi"], make_model("pbm", "navigational"), 500, 600, rng)
    fpr = np.mean(p < 0.05)
    assert fpr <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 600)
    assert fpr > 0.01


def test_srm_detection():
    assert not ab.srm_check([5000, 5040], [1, 1])["srm"]
    r = ab.srm_check([5000, 5400], [1, 1])
    assert r["srm"] and r["p"] < 1e-4
    # unequal configured allocation is respected
    assert not ab.srm_check([2000, 8000], [0.2, 0.8])["srm"]


def test_hash_assignment_is_deterministic_balanced_and_passes_srm():
    ids = [f"{i:016x}" for i in range(40_000)]
    arms = [assign("salt-1", s, 0.5) for s in ids]
    assert arms == [assign("salt-1", s, 0.5) for s in ids]  # sticky
    enrolled = [a for a in arms if a]
    assert abs(len(enrolled) / len(ids) - 0.5) < 0.01
    c = [enrolled.count("control"), enrolled.count("treatment")]
    assert not ab.srm_check(c, [1, 1])["srm"]
    # enrolment and arm are independent hashes: arm split is 50/50 inside the enrolled set
    assert abs(c[0] / sum(c) - 0.5) < 0.015
    assert 0 <= bucket("s", "x") < 10_000


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(200):
        x = rng.random(300) < 0.3
        lo, hi = ab.bootstrap_mean_ci(x.astype(float), rng, 500)
        hits += lo <= 0.3 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def test_n_at_power_interpolates_isotonic():
    ns = np.array([10, 100, 1000, 10000])
    assert S.n_at_power(ns, np.array([0.1, 0.5, 0.9, 1.0])) == pytest.approx(10 ** 2.75)
    assert S.n_at_power(ns, np.array([0.1, 0.2, 0.3, 0.4])) is None


def test_pool_power_matches_direct_simulation():
    """The multinomial-resampling power engine agrees with brute-force simulation."""
    from hybridsearch.clicks.rankers import RankerTable
    rng = np.random.default_rng(4)
    qrels = {str(q): {str(d): int(rng.integers(0, 4)) for d in range(30)} for q in range(30)}
    good = [sorted(map(str, range(30)), key=lambda d: -qrels[str(q)][d] + rng.normal(0, 1.5))[:10] for q in range(30)]
    bad = [[str(d) for d in rng.permutation(30)[:10]] for _ in range(30)]
    table = RankerTable([str(q) for q in range(30)], {"good": good, "bad": bad}, qrels, 10)
    pair = S.build_pair(table, "good", "bad", rng, methods=("tdi",))
    model = make_model("dbn", "navigational")
    dist = S.pool_interleave("tdi", pair.entries["tdi"], model, 400_000, rng)
    n = 60
    pc, _ = S.power_interleave("tdi", dist, np.array([n]), 4000, rng, True)
    p, d = S.direct_interleave_pvalues("tdi", pair.entries["tdi"], model, n, 1500, rng)
    direct = np.mean((p < 0.05) & (d > 0))
    assert abs(pc[0] - direct) < 4 * np.sqrt(direct * (1 - direct) / 1500) + 0.01
