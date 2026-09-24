"""The event-log analyser on simulated logs (same code path as real broker logs)."""
import json

import numpy as np

from hybridsearch.clicks import analyze_events as A
from hybridsearch.clicks.events import read_events, write_jsonl
from hybridsearch.clicks.models import make_model
from hybridsearch.clicks.rankers import RankerTable
from hybridsearch.clicks.simulate import simulate_experiment


def _table(rng):
    qrels = {str(q): {str(1000 * q + d): int(rng.integers(0, 4)) for d in range(30)} for q in range(40)}
    good = [sorted(qrels[str(q)], key=lambda d: -qrels[str(q)][d])[:10] for q in range(40)]
    bad = [list(qrels[str(q)])[::-1][:10] for q in range(40)]
    return RankerTable([str(q) for q in range(40)], {"good": good, "bad": bad}, qrels, 10)


def test_ab_report_on_simulated_log(tmp_path):
    rng = np.random.default_rng(0)
    table = _table(rng)
    exp = {"id": "ab1", "kind": "ab", "allocation": 0.8, "salt": "s", "control": "bad", "treatment": "good"}
    ev = simulate_experiment(table, {}, exp, make_model("dbn", "navigational"), 3000, rng)
    write_jsonl(tmp_path / "e.jsonl", ev)
    rep = A.analyse(read_events(tmp_path), {"ab1": {"kind": "ab"}}, seed=1, n_boot=300)
    assert rep["label"].startswith("SIMULATED")
    e = rep["experiments"]["ab1"]
    assert not e["srm"]["srm"]
    assert rep["not_enrolled"]["pages"] > 0
    m = e["ab"]["metrics"]
    assert m["rr_first"]["delta"] > 0 and m["rr_first"]["p_welch"] < 0.05
    assert m["abandonment"]["delta"] < 0
    lo, hi = m["ctr"]["delta_ci95"]
    assert lo <= m["ctr"]["delta"] <= hi
    # markdown renders and JSON serialises
    assert "SRM check" in A.to_markdown(rep)
    json.dumps(rep)


def test_srm_flagged_when_treatment_events_are_lost(tmp_path):
    rng = np.random.default_rng(1)
    table = _table(rng)
    exp = {"id": "ab2", "kind": "ab", "allocation": 1.0, "salt": "t", "control": "bad", "treatment": "good"}
    ev = simulate_experiment(table, {}, exp, make_model("pbm", "navigational"), 6000, rng)
    # a logging bug drops 15% of treatment sessions
    lost = {e["sessionId"] for e in ev if e.get("variant") == "treatment" and int(e["sessionId"], 16) % 100 < 15}
    ev = [e for e in ev if e["sessionId"] not in lost]
    rep = A.analyse(ev, {}, n_boot=100)
    assert rep["experiments"]["ab2"]["srm"]["srm"]


def test_interleaving_report_on_simulated_log():
    rng = np.random.default_rng(2)
    table = _table(rng)
    exp = {"id": "il", "kind": "interleave", "allocation": 1.0, "salt": "u", "control": "good", "treatment": "bad"}
    ev = simulate_experiment(table, {}, exp, make_model("cascade", "navigational"), 1500, rng)
    rep = A.analyse(ev, {}, n_boot=300)
    e = rep["experiments"]["il"]
    assert e["kind"] == "interleave"
    r = e["interleaving"]
    assert r["wins_a"] > r["wins_b"] and r["p_sign"] < 1e-6 and r["delta_ab"] > 0
    assert e["team_balance_rank1"]["p"] > 0.001


def test_cli_missing_log_returns_nonzero(tmp_path):
    assert A.main(["--events", str(tmp_path / "nope"), "--out", str(tmp_path / "r")]) == 1
