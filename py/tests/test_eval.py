"""Harness parity: our metrics vs trec_eval (pytrec_eval) and ir_measures, to 1e-4
aggregate and 1e-9 per query, on synthetic runs with ties, graded qrels, missing
docs, and (when present) the real reference run files."""
from __future__ import annotations

import random
from pathlib import Path

import ir_measures
import pytest
import pytrec_eval
from hybridsearch.data.io import read_qrels, read_run, write_qrels, write_run
from hybridsearch.eval import evaluate, parse_metric, rank_run
from hybridsearch.eval.__main__ import main as cli_main

REPO = Path(__file__).resolve().parents[2]

PYTREC = {"RR@10": "recip_rank", "nDCG@10": "ndcg_cut_10", "R@100": "recall_100", "R@1000": "recall_1000"}


def synth(seed: int, nq=60, ndocs=400, depth=150, graded=True, ties=True):
    rng = random.Random(seed)
    qrels, run = {}, {}
    for q in range(nq):
        qid = str(1000 + q)
        judged = rng.sample(range(ndocs), rng.randint(1, 40))
        qrels[qid] = {str(d): (rng.randint(0, 3) if graded else 1) for d in judged}
        if not any(v > 0 for v in qrels[qid].values()):
            qrels[qid][str(judged[0])] = 1
        docs = rng.sample(range(ndocs), depth)
        # coarse scores create many ties
        run[qid] = {str(d): (round(rng.random() * 5, 1) if ties else rng.random()) for d in docs}
    return qrels, run


def pytrec_scores(qrels, run, cut10_run=None):
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_100", "recall_1000"})
    res = ev.evaluate(run)
    # MRR@10: trec_eval recip_rank on the top-10-truncated run (trec_eval ordering)
    trunc = {q: {d: run[q][d] for d in rank_run(run[q])[:10]} for q in run}
    ev2 = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    rr = ev2.evaluate(trunc)
    for q in res:
        res[q]["recip_rank"] = rr[q]["recip_rank"]
    return res


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ties", [True, False])
def test_per_query_matches_pytrec_eval(seed, ties):
    qrels, run = synth(seed, ties=ties)
    agg, per_q = evaluate(run, qrels)
    ref = pytrec_scores(qrels, run)
    for ours, theirs in PYTREC.items():
        for qid in run:
            assert per_q[ours][qid] == pytest.approx(ref[qid][theirs], abs=1e-9), (ours, qid)
        mean_ref = sum(ref[q][theirs] for q in run) / len(run)
        assert round(agg[ours], 4) == round(mean_ref, 4)


@pytest.mark.parametrize("seed", [7, 8])
def test_aggregate_matches_ir_measures(seed):
    qrels, run = synth(seed, ties=False)
    measures = [ir_measures.parse_measure(m) for m in ["RR@10", "nDCG@10", "R@100", "R@1000", "R(rel=2)@100"]]
    ref = ir_measures.calc_aggregate(measures, qrels, run)
    agg, _ = evaluate(run, qrels, ["RR@10", "nDCG@10", "R@100", "R@1000", "R(rel=2)@100"])
    for m in measures:
        # queries with no rel>=2 doc score 0 and count, in both ir_measures and ours
        assert agg[str(m)] == pytest.approx(ref[m], abs=5e-5), str(m)


def test_missing_queries_count_as_zero():
    qrels = {"1": {"a": 1}, "2": {"b": 1}}
    run = {"1": {"a": 3.0, "c": 1.0}}
    agg, per_q = evaluate(run, qrels, ["RR@10", "nDCG@10"])
    assert agg["RR@10"] == 0.5 and per_q["RR@10"] == {"1": 1.0, "2": 0.0}


def test_mrr_cutoff_and_ties():
    qrels = {"q": {"d11": 1}}
    run = {"q": {f"d{i:02d}": 1.0 for i in range(20)}}  # all tied
    # trec_eval order among ties: docid descending -> d19, d18, ..., d11 is rank 9
    assert rank_run(run["q"])[:3] == ["d19", "d18", "d17"]
    agg, _ = evaluate(run, qrels, ["RR@10"])
    assert agg["RR@10"] == pytest.approx(1 / 9)
    run2 = {"q": {f"d{i:02d}": 1.0 for i in range(12)}}
    run2["q"]["d11"] = 0.5  # pushed to rank 12 -> outside the cutoff
    assert evaluate(run2, qrels, ["RR@10"])[0]["RR@10"] == 0.0


def test_parse_metric():
    assert parse_metric("MRR@10").name == "RR@10"
    assert parse_metric("ndcg@10").name == "nDCG@10"
    assert parse_metric("R(rel=2)@1000").rel == 2
    with pytest.raises(ValueError):
        parse_metric("MAP")


def test_io_roundtrip_and_cli(tmp_path):
    qrels, run = synth(11)
    qp, rp = tmp_path / "q.txt", tmp_path / "r.trec"
    write_qrels(qp, qrels)
    write_run(rp, [(q, sorted(run[q].items(), key=lambda x: -x[1])) for q in run], tag="t")
    assert read_qrels(qp) == qrels
    back = read_run(rp)
    assert set(back) == set(run) and all(set(back[q]) == set(run[q]) for q in run)
    out = tmp_path / "pq.tsv"
    assert cli_main([str(rp), str(qp), "--per-query", str(out), "--json", str(tmp_path / "a.json")]) == 0
    rows = out.read_text().splitlines()
    assert rows[0] == "qid\tmetric\tvalue" and len(rows) == 1 + 4 * len(qrels)


REF_RUNS = [
    ("data/runs/reference/anserini-bm25-default.dev.trec", "data/raw/msmarco/qrels.dev.small.tsv"),
    ("data/runs/reference/anserini-bm25-default.dl19.trec", "data/raw/trec-dl/dl19.qrels"),
    ("data/runs/reference/anserini-bm25-default.dl20.trec", "data/raw/trec-dl/dl20.qrels"),
]


@pytest.mark.parametrize("run_path,qrels_path", REF_RUNS)
def test_reference_runs_match_trec_eval(run_path, qrels_path):
    rp, qp = REPO / run_path, REPO / qrels_path
    if not rp.exists() or not qp.exists():
        pytest.skip("reference run not present")
    run, qrels = read_run(rp), read_qrels(qp)
    metrics = ["RR@10", "nDCG@10", "R@100", "R@1000"]
    agg, per_q = evaluate(run, qrels, metrics)
    ref = pytrec_scores(qrels, run)
    for ours, theirs in PYTREC.items():
        # every qrels query is in these runs, so trec_eval -c and plain agree
        assert len(per_q[ours]) == len(qrels)
        mean_ref = sum(ref[q][theirs] for q in qrels) / len(qrels)
        assert round(agg[ours], 4) == round(mean_ref, 4), ours
