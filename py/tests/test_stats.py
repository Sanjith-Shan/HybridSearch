import numpy as np
import pytest
from hybridsearch.stats import (
    align,
    compare,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_randomization_test,
)


def test_identical_runs_p_is_one():
    rng = np.random.default_rng(0)
    a = rng.random(500)
    assert paired_randomization_test(a, a.copy(), n_perm=10_000) == 1.0
    d, lo, hi = paired_bootstrap_ci(a, a.copy())
    assert d == 0 and lo == 0 and hi == 0


def test_shifted_distribution_small_p():
    rng = np.random.default_rng(1)
    b = rng.random(300)
    a = b + 0.05 + rng.normal(0, 0.05, 300)
    p = paired_randomization_test(a, b, n_perm=10_000, seed=3)
    assert p == pytest.approx(1 / 10_001)  # no permutation reaches the observed mean
    assert paired_randomization_test(a, b, alternative="greater") < 0.001
    assert paired_randomization_test(a, b, alternative="less") > 0.99
    d, lo, hi = paired_bootstrap_ci(a, b)
    assert 0 < lo < d < hi


def test_null_p_values_roughly_uniform():
    rng = np.random.default_rng(2)
    ps = [paired_randomization_test(rng.normal(size=50), rng.normal(size=50), n_perm=2000, seed=i) for i in range(200)]
    frac = np.mean(np.array(ps) < 0.05)
    assert 0.01 < frac < 0.10


def test_matches_exact_enumeration_small_n():
    # n=10 -> 1024 sign patterns; exact two-sided p is computable.
    d = np.array([0.3, 0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.25, 0.05, 0.12])
    import itertools

    obs = abs(d.mean())
    exact = np.mean([abs((np.array(s) * d).mean()) >= obs - 1e-12 for s in itertools.product([-1, 1], repeat=10)])
    p = paired_randomization_test(d, np.zeros_like(d), n_perm=50_000, seed=5)
    assert p == pytest.approx(exact, abs=0.005)


def test_deterministic_seed():
    rng = np.random.default_rng(4)
    a, b = rng.random(100), rng.random(100)
    assert paired_randomization_test(a, b, seed=9) == paired_randomization_test(a, b, seed=9)


def test_holm_hand_computed():
    # p = [0.01, 0.04, 0.03, 0.005], m=4; sorted: 0.005, 0.01, 0.03, 0.04
    # raw adj: 4*0.005=0.02, 3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.04 -> monotone: .02 .03 .06 .06
    adj, rej = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]


def test_holm_caps_at_one_and_empty():
    adj, rej = holm_bonferroni([0.5, 0.6])
    assert adj.tolist() == [1.0, 1.0] and not rej.any()
    adj, rej = holm_bonferroni([])
    assert adj.size == 0


def test_holm_matches_statsmodels_if_available():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(3).random(12) ** 3
    rej, adj, _, _ = sm.multipletests(p, alpha=0.05, method="holm")
    ours, our_rej = holm_bonferroni(p)
    assert ours == pytest.approx(adj) and (our_rej == rej).all()


def test_compare_family_and_align():
    rng = np.random.default_rng(6)
    base = {str(i): float(v) for i, v in enumerate(rng.random(200))}
    better = {q: v + 0.1 for q, v in base.items()}
    same = dict(base)
    res = compare({"better": (better, base), "same": (same, base)}, metric="RR@10", n_perm=2000, n_boot=2000)
    assert res[0].significant and not res[1].significant
    assert res[1].p_value == 1.0 and res[0].delta == pytest.approx(0.1)
    with pytest.raises(ValueError):
        align({"1": 0.0}, {"2": 0.0})
