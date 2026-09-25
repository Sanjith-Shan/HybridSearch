"""Hand-computed cases for py/hybridsearch/fusion."""
import math

import pytest
from hybridsearch.eval.metrics import rank_run
from hybridsearch.fusion import (
    docid_key,
    fuse_runs,
    normalise,
    read_run_ranked,
    rrf,
    run_file_scores,
    truncate,
    weighted,
    write_run_exact,
)

LEX = [("a", 10.0), ("b", 6.0), ("c", 2.0)]
DENSE = [("b", 0.9), ("d", 0.5)]


def ids(r):
    return [d for d, _ in r]


# ------------------------------------------------------------------ RRF
def test_rrf_hand_computed_k60():
    r = rrf([LEX, DENSE], k=60)
    s = dict(r)
    assert s["a"] == pytest.approx(1 / 61)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["c"] == pytest.approx(1 / 63)
    assert s["d"] == pytest.approx(1 / 62)
    assert ids(r) == ["b", "a", "d", "c"]


def test_rrf_small_k_changes_order():
    # k=0: a = 1, b = 1/2 + 1 = 1.5, c = 1/3, d = 1/2
    r = rrf([LEX, DENSE], k=0)
    assert [round(s, 6) for _, s in r] == [1.5, 1.0, 0.5, round(1 / 3, 6)]
    assert ids(r) == ["b", "a", "d", "c"]


def test_rrf_ties_by_lower_numeric_docid():
    r = rrf([[("10", 5.0)], [("9", 0.1)]], k=60)
    assert ids(r) == ["9", "10"]  # numeric, not string, order
    assert r[0][1] == r[1][1] == pytest.approx(1 / 61)


def test_rrf_depth_truncates_each_input_before_fusion():
    # depth 1: only a (from LEX) and b (from DENSE) survive, both 1/61 -> tie -> "a" first
    r = rrf([LEX, DENSE], k=60, depth=1)
    assert ids(r) == ["a", "b"]
    assert r[0][1] == r[1][1]


def test_rrf_out_k_and_empty_inputs():
    assert rrf([[], []]) == []
    assert ids(rrf([LEX, []], out_k=2)) == ["a", "b"]  # single ranking: order kept


def test_rrf_rejects_duplicates_and_negative_k():
    with pytest.raises(ValueError):
        rrf([[("a", 1.0), ("a", 0.5)]])
    with pytest.raises(ValueError):
        rrf([LEX], k=-1)


# ------------------------------------------------------------------ normalisation
def test_minmax():
    assert normalise([10, 6, 2], "minmax") == [1.0, 0.5, 0.0]
    assert normalise([3, 3], "minmax") == [1.0, 1.0]
    assert normalise([], "minmax") == []


def test_zscore():
    z = normalise([1, 2, 3], "zscore")
    sd = math.sqrt(2 / 3)
    assert z == pytest.approx([-1 / sd, 0.0, 1 / sd])
    assert normalise([4, 4, 4], "zscore") == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        normalise([1], "l2")


# ------------------------------------------------------------------ weighted
def test_weighted_minmax_alpha_025():
    # lex norm: a 1, b .5, c 0; dense norm: b 1, d 0; missing -> 0
    r = weighted(LEX, DENSE, alpha=0.25, norm="minmax")
    s = dict(r)
    assert s == pytest.approx({"a": 0.75, "b": 0.625, "c": 0.0, "d": 0.0})
    assert ids(r) == ["a", "b", "c", "d"]  # c, d tied at 0 -> lower id first


def test_weighted_minmax_alpha_08():
    r = weighted(LEX, DENSE, alpha=0.8, norm="minmax")
    assert dict(r) == pytest.approx({"a": 0.2, "b": 0.9, "c": 0.0, "d": 0.0})
    assert ids(r) == ["b", "a", "c", "d"]


def test_weighted_alpha_extremes_reproduce_single_retriever():
    assert ids(weighted(LEX, DENSE, alpha=0.0))[:3] == ["a", "b", "c"]
    # alpha=1: b 1; d is dense's bottom (0) and ties with the lex-only docs a, c (missing -> 0)
    assert ids(weighted(LEX, DENSE, alpha=1.0)) == ["b", "a", "c", "d"]


def test_weighted_zscore_missing_gets_list_minimum():
    lex = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    dense = [("a", 5.0), ("d", 1.0)]
    z = 1 / math.sqrt(2 / 3)  # 1.2247
    r = weighted(lex, dense, alpha=0.5, norm="zscore")
    s = dict(r)
    # a: .5*z + .5*1 ; b: .5*0 + .5*(-1) (missing in dense -> min z = -1)
    # c: .5*(-z) + .5*(-1) ; d: .5*(-z) (missing in lex -> min z = -z) + .5*(-1)
    assert s["a"] == pytest.approx(0.5 * z + 0.5)
    assert s["b"] == pytest.approx(-0.5)
    assert s["c"] == pytest.approx(-0.5 * z - 0.5)
    assert s["d"] == pytest.approx(-0.5 * z - 0.5)
    assert ids(r) == ["a", "b", "c", "d"]


def test_weighted_depth_changes_normalisation_stats():
    # depth 2: lex [a 10, b 6] -> a 1, b 0 ; dense [b .9, d .5] -> b 1, d 0 ; alpha .5
    r = weighted(LEX, DENSE, alpha=0.5, norm="minmax", depth=2)
    assert dict(r) == pytest.approx({"a": 0.5, "b": 0.5, "d": 0.0})
    assert ids(r) == ["a", "b", "d"]


def test_weighted_empty_side_is_single_retriever():
    r = weighted([], DENSE, alpha=0.3, norm="zscore")
    assert ids(r) == ["b", "d"]
    with pytest.raises(ValueError):
        weighted(LEX, DENSE, alpha=1.5)


# ------------------------------------------------------------------ run plumbing
def test_docid_key_orders_numeric_then_strings():
    assert sorted(["10", "9", "PLAIN-2", "PLAIN-10"], key=docid_key) == ["9", "10", "PLAIN-10", "PLAIN-2"]


def test_truncate():
    assert truncate(LEX, 2) == LEX[:2]
    assert truncate(LEX, None) == LEX
    with pytest.raises(ValueError):
        truncate(LEX, 0)


def test_run_file_scores_break_ties_in_our_order():
    r = [("1", 0.5), ("2", 0.5), ("3", 0.5), ("4", 0.4)]
    sc = run_file_scores(r)
    assert all(x > y for x, y in zip(sc, sc[1:]))
    assert rank_run(dict(zip(ids(r), sc))) == ["1", "2", "3", "4"]
    assert sc[0] == 0.5 and sc[3] == pytest.approx(0.4)
    neg = run_file_scores([("1", -1.0), ("2", -1.0)])
    assert neg[0] > neg[1]


def test_write_read_roundtrip_and_fuse_runs(tmp_path):
    lex = {"q1": LEX, "q2": [("x", 1.0)]}
    dense = {"q1": DENSE, "q3": [("y", 0.2)]}
    fused = fuse_runs(lex, dense, lambda a, b: rrf([a, b], k=60))
    assert set(fused) == {"q1", "q2", "q3"}
    assert ids(fused["q2"]) == ["x"] and ids(fused["q3"]) == ["y"]
    p = tmp_path / "f.trec"
    written = write_run_exact(p, fused, "t")
    back = read_run_ranked(p)
    assert {q: ids(r) for q, r in back.items()} == {q: ids(r) for q, r in fused.items()}
    assert {q: dict(r) for q, r in back.items()} == written


def test_read_run_ranked_uses_rank_column(tmp_path):
    p = tmp_path / "r.trec"
    p.write_text("q Q0 b 2 5.0 t\nq Q0 a 1 5.0 t\n")
    assert ids(read_run_ranked(p)["q"]) == ["a", "b"]
