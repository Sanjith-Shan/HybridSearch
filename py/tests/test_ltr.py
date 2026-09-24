"""Counterfactual LTR: IPS unbiasedness, propensity estimation, training."""
import numpy as np
import pytest

from hybridsearch.clicks.models import PBM, make_model, position_bias
from hybridsearch.ltr import data as D
from hybridsearch.ltr import logsim as L
from hybridsearch.ltr import model as M


def _toy_model():
    # alpha by DL grade: grade g -> alpha[g]
    alpha = np.array([0.1, 0.3, 0.6, 0.9])
    return PBM("toy", alpha, np.zeros(4), eta=position_bias(3, 1.0))


def test_ips_is_unbiased_on_toy_problem():
    """One query, three docs shown in a fixed order with PBM eta = (1, 1/2, 1/3).
    E[clicks_d / eta_rank(d)] per session = alpha_d exactly (analytic truth)."""
    model = _toy_model()
    grades = np.array([[1, 3, 2]])          # alpha = 0.3, 0.9, 0.6 at ranks 1,2,3
    alpha = model.alpha[grades[0]]
    n = 400_000
    rng = np.random.default_rng(0)
    logs = L.simulate_logs(grades, np.array([0, 3]), model, n, rng)
    ips = L.ips_weights(logs.clicks_at, model.eta) / n
    naive = L.ips_weights(logs.clicks_at, None) / n
    se = np.sqrt(alpha * model.eta * (1 - alpha * model.eta) / n) / model.eta
    assert np.all(np.abs(ips - alpha) < 5 * se)
    # the naive estimate converges to eta * alpha, not alpha, and here inverts the true
    # order of docs 0 (alpha 0.3, rank 1) and 2 (alpha 0.6, rank 3)
    assert np.all(np.abs(naive - model.eta * alpha) < 5 * se)
    assert naive[0] > naive[2] and alpha[0] < alpha[2]
    assert ips[0] < ips[2]
    # IPS estimate of a new ranking's utility sum_d alpha_d * lambda(rank_new(d)) is unbiased
    lam = 1.0 / np.log2(np.arange(2, 5))
    new_rank = np.array([2, 0, 1])          # new ranking puts doc 1 first
    truth = float(alpha @ lam[new_rank])
    assert float(ips @ lam[new_rank]) == pytest.approx(truth, abs=5 * float(se @ lam[new_rank]))


def test_clipping_bounds_weights_and_adds_bias():
    model = _toy_model()
    rng = np.random.default_rng(1)
    logs = L.simulate_logs(np.array([[3, 3, 3]]), np.array([0, 3]), model, 100_000, rng)
    w_full = L.ips_weights(logs.clicks_at, model.eta)
    w_clip = L.ips_weights(logs.clicks_at, model.eta, clip=2.0)
    assert w_clip[2] < w_full[2]            # rank 3 weight 3 clipped to 2
    assert w_clip[0] == w_full[0]
    assert np.all(w_clip <= logs.clicks_at.sum(1) * 2.0 + 1e-9)


@pytest.mark.parametrize("method", ["randtop", "swap"])
def test_propensity_estimation_recovers_eta(method):
    rng = np.random.default_rng(2)
    ds = D.synthetic(300, 10, rng)
    model = make_model("pbm", "navigational", k=10)
    logs = L.simulate_logs(ds.grades_log_order(), ds.data.offsets, model, 600_000, rng, method, eps=0.5,
                           randtop_n=10)
    est = L.estimate_propensity(logs, method)
    assert est[0] == 1.0
    assert np.allclose(est[:6], model.eta[:6], rtol=0.12, atol=0.02), est


def test_fill_propensity():
    out = L.fill_propensity(np.array([1.0, 0.5, np.nan, -1.0, 0.2]))
    assert out.tolist() == [1.0, 0.5, 0.5, 0.5, 0.2]


def test_loss_gradient_finite_difference():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 4))
    data = M.ListData(X, np.array([0, 5, 12]))
    w = rng.random(12)
    th = rng.normal(size=4)
    seg, starts = data.seg(), data.offsets[:-1]
    f, g = M._loss_grad(th, X, w, seg, starts, 0.1)
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1e-6
        fp, _ = M._loss_grad(th + e, X, w, seg, starts, 0.1)
        fm, _ = M._loss_grad(th - e, X, w, seg, starts, 0.1)
        assert (fp - fm) / 2e-6 == pytest.approx(g[i], rel=1e-5, abs=1e-7)


def test_ips_beats_naive_on_synthetic():
    rng = np.random.default_rng(4)
    tr, te = D.synthetic(600, 20, rng), D.synthetic(300, 20, rng)
    D.standardise(tr, te)
    model = make_model("pbm", "navigational", k=20)
    logs = L.simulate_logs(tr.grades_log_order(), tr.data.offsets, model, 200_000, rng)

    def score(theta):
        return np.mean([M.ndcg10(o, te.grades[te.data.offsets[q]:te.data.offsets[q + 1]], te.ideal[q])
                        for q, o in enumerate(M.rank_lists(te.data, theta))])

    naive = score(M.fit(tr.data, L.ips_weights(logs.clicks_at, None)))
    ips = score(M.fit(tr.data, L.ips_weights(logs.clicks_at, model.eta)))
    oracle = score(M.fit(tr.data, tr.grades.astype(float)))
    logging = score(np.eye(7)[0])
    assert ips > naive + 0.02
    assert oracle > logging + 0.05
    assert ips > oracle - 0.03


def test_ndcg10_matches_rankers_module():
    from hybridsearch.clicks.rankers import ndcg_at_k
    grades = np.array([0, 3, 1, 2, 0])
    order = np.array([1, 3, 2, 0, 4])
    judged = {str(i): int(g) for i, g in enumerate(grades)}
    assert M.ndcg10(order, grades, grades) == pytest.approx(ndcg_at_k([str(i) for i in order], judged))


def test_monotone_propensity_repairs_noisy_deep_ranks():
    est = np.array([1.0, 0.52, 0.30, 0.004, 0.27, np.nan, np.nan])
    sup = np.array([100, 100, 100, 3, 100, 0, 0], float)
    out = L.monotone_propensity(est, sup)
    assert out[0] == 1.0
    assert np.all(np.diff(out) <= 1e-12)          # non-increasing
    assert out[3] > 0.2                            # the 3-session outlier no longer dominates
    assert out[5] == out[4] == out[6]              # unidentified ranks carried flat


def test_intervention_harvesting_recovers_eta():
    """Two loggers that disagree on order; no deliberate randomisation."""
    rng = np.random.default_rng(7)
    ds = D.synthetic(400, 10, rng)
    model = make_model("pbm", "navigational", k=10)
    Q, K = 400, 10
    grades = ds.grades_log_order()
    orders = np.stack([np.tile(np.arange(K), (Q, 1)),
                       np.stack([rng.permutation(K) for _ in range(Q)])])
    clicks_at, sessions = L.simulate_logs_multi(grades, orders, ds.data.offsets, model, 400_000, rng)
    est, support = L.harvest_propensity(clicks_at, sessions, orders, ds.data.offsets)
    assert est[0] == 1.0 and (support[:6] > 0).all()
    assert np.allclose(est[:6], model.eta[:6], rtol=0.15, atol=0.02), est
