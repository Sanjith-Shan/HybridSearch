"""Click-model properties (simulated users)."""
import numpy as np
import pytest

from hybridsearch.clicks.events import click_matrix, impression_events, pages_from_events, SIM_EPOCH
from hybridsearch.clicks.models import DBN, PBM, PROFILES, dl_profile_arrays, make_model, position_bias


def _se(p, n):
    return np.sqrt(p * (1 - p) / n)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_pbm_ctr_is_eta_times_alpha(profile):
    rng = np.random.default_rng(1)
    m = make_model("pbm", profile)
    n = 200_000
    for g in range(4):
        grades = np.full((n, 10), g)
        ctr = m.simulate(grades, rng).clicks.mean(0)
        want = m.eta * m.alpha[g]
        assert np.all(np.abs(ctr - want) <= 5 * _se(want, n) + 1e-4), (g, ctr, want)


def test_pbm_eta_shape():
    assert np.allclose(position_bias(4, 1.0), [1, 1 / 2, 1 / 3, 1 / 4])
    assert np.allclose(position_bias(3, 2.0), [1, 1 / 4, 1 / 9])


def test_cascade_stops_after_satisfied_click():
    rng = np.random.default_rng(2)
    m = make_model("cascade", "navigational")
    grades = rng.integers(0, 4, size=(50_000, 10))
    r = m.simulate(grades, rng)
    first_sat = np.where(r.satisfied.any(1), np.argmax(r.satisfied, 1), 10)
    after = np.arange(10)[None, :] > first_sat[:, None]
    assert not (r.clicks & after).any()
    assert not (r.examined & after).any()
    # with no satisfied click, cascade (gamma=1) examines everything
    none = ~r.satisfied.any(1)
    assert r.examined[none].all()


def test_pure_cascade_at_most_one_click():
    rng = np.random.default_rng(3)
    alpha, _ = dl_profile_arrays("informational")
    m = DBN("pure-cascade", alpha, np.ones(4), gamma=1.0)
    r = m.simulate(rng.integers(0, 4, size=(20_000, 10)), rng)
    assert r.clicks.sum(1).max() == 1


def test_dbn_examination_decays_with_gamma():
    rng = np.random.default_rng(4)
    alpha = np.zeros(4)  # never click: examination only decays through gamma
    m = DBN("dbn", alpha, np.zeros(4), gamma=0.8)
    n = 200_000
    ex = m.simulate(np.zeros((n, 10), int), rng).examined.mean(0)
    want = 0.8 ** np.arange(10)
    assert np.all(np.abs(ex - want) < 5 * _se(want, n) + 1e-4)


def test_empty_slots_never_clicked_and_grade_mapping_monotone():
    rng = np.random.default_rng(5)
    for kind in ("pbm", "cascade", "dbn"):
        m = make_model(kind, "informational")
        g = np.full((1000, 10), -1)
        g[:, :3] = 3
        c = m.simulate(g, rng).clicks
        assert not c[:, 3:].any()
    for p in PROFILES:
        a, s = dl_profile_arrays(p)
        assert np.all(np.diff(a) >= 0) and np.all(np.diff(s) >= 0)


def test_seeded_reproducibility():
    g = np.random.default_rng(0).integers(0, 4, size=(1000, 10))
    for kind in ("pbm", "cascade", "dbn"):
        m = make_model(kind, "navigational")
        a = m.simulate(g, np.random.default_rng(42)).clicks
        b = m.simulate(g, np.random.default_rng(42)).clicks
        assert (a == b).all()


def test_events_roundtrip_schema():
    rng = np.random.default_rng(6)
    clicks = np.array([True, False, True] + [False] * 7)
    ev = impression_events(session_id="s1", request_id="r1", query="q", doc_ids=list(range(100, 110)),
                           clicks=clicks, dwell_ms=np.full(10, 5000), experiment_id="e", variant="interleaved",
                           teams=list("ABABABABAB"), t0=SIM_EPOCH, simulated="pbm-perfect")
    allowed = {"type", "sessionId", "requestId", "query", "docId", "rank", "dwellMs", "experimentId",
               "variant", "team", "clientTs", "simulated"}
    for e in ev:
        assert set(e) <= allowed
        assert e["clientTs"].endswith("Z")
    types = [e["type"] for e in ev]
    assert types.count("impression") == 10 and types.count("click") == 2 and types.count("dwell") == 2
    pages, dq = pages_from_events(ev)
    assert len(pages) == 1 and not dq
    assert (click_matrix(pages)[0] == clicks).all()
    assert pages[0].teams[1] == "A" and pages[0].teams[2] == "B"
    # abandoned page
    ev2 = impression_events(session_id="s1", request_id="r2", query="q", doc_ids=[1, 2], clicks=[False, False],
                            dwell_ms=[0, 0], t0=SIM_EPOCH)
    assert ev2[-1]["type"] == "abandon"
