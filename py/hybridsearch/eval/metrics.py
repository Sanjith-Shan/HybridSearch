"""Metric definitions, chosen to match trec_eval conventions exactly.

Conventions (all match trec_eval 9.x as exposed by pytrec_eval / ir_measures):
  * Ranking: trec_eval ignores the rank column and sorts by score descending, breaking
    ties by docid in *descending string* order. We do the same so tie-heavy runs score
    identically. (HybridSearch's own engines break ties by lower global id, which only
    matters for tied scores.)
  * Queries: a query is evaluated if it has qrels. Queries in qrels but missing from
    the run contribute 0 (trec_eval ``-c`` behaviour, which is what MS MARCO dev MRR
    uses: divide by 6,980). Queries in the run but not in the qrels are ignored.
  * RR@10 (MRR@10): reciprocal rank of the first doc with grade >= rel_level within the
    top 10, else 0. The MS MARCO dev convention (msmarco_eval.py) is the same.
  * nDCG@k: trec_eval ``ndcg_cut``: gain = grade (linear, not 2^g-1), discount
    log2(rank+1), ideal from all judged grades of the query (grade > 0).
  * R@k: |relevant (grade >= rel_level) in top k| / |relevant|. A query that has qrels
    but no doc at rel_level scores 0 and still counts in the mean (trec_eval and
    ir_measures do the same; verified in py/tests/test_eval.py).

``rel_level`` defaults to 1. For TREC DL passage, R@k and MRR conventionally use
rel >= 2 (Anserini's ``-l 2``); pass ``rel=2`` (metric names like ``R(rel=2)@1000``).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

DEFAULT_METRICS = ["RR@10", "nDCG@10", "R@100", "R@1000"]


@dataclass(frozen=True)
class Metric:
    name: str  # canonical string, e.g. "RR@10", "R(rel=2)@1000"
    kind: str  # "RR" | "nDCG" | "R"
    k: int
    rel: int = 1


_METRIC_RE = re.compile(r"^(RR|MRR|nDCG|NDCG|ndcg|R|Recall|recall)(?:\(rel=(\d+)\))?@(\d+)$")


def parse_metric(s: str) -> Metric:
    m = _METRIC_RE.match(s.strip())
    if not m:
        raise ValueError(f"unknown metric {s!r}; use RR@k, nDCG@k, R@k, optionally R(rel=2)@k")
    kind = {"rr": "RR", "mrr": "RR", "ndcg": "nDCG", "r": "R", "recall": "R"}[m.group(1).lower()]
    rel = int(m.group(2)) if m.group(2) else 1
    k = int(m.group(3))
    name = f"{kind}{f'(rel={rel})' if rel != 1 else ''}@{k}"
    return Metric(name, kind, k, rel)


def rank_run(scores: dict[str, float]) -> list[str]:
    """trec_eval ordering: score desc, then docid desc (string compare)."""
    return [d for d, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)]


def mrr_at_k(ranked: list[str], qrel: dict[str, int], k: int = 10, rel: int = 1) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrel.get(d, 0) >= rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        g = qrel.get(d, 0)
        if g > 0:
            dcg += g / math.log2(i + 2)
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int, rel: int = 1) -> float:
    nrel = sum(1 for g in qrel.values() if g >= rel)
    if nrel == 0:
        return 0.0
    hit = sum(1 for d in ranked[:k] if qrel.get(d, 0) >= rel)
    return hit / nrel


def evaluate(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metrics: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Returns (aggregate, per_query). per_query[metric][qid] = value.

    Every query in ``qrels`` appears in per_query[metric] (missing from run -> 0).
    """
    ms = [parse_metric(m) for m in (metrics or DEFAULT_METRICS)]
    per_query: dict[str, dict[str, float]] = {m.name: {} for m in ms}
    ranked_cache: dict[str, list[str]] = {}
    for qid, qrel in qrels.items():
        if qid not in ranked_cache:
            ranked_cache[qid] = rank_run(run.get(qid, {}))
        ranked = ranked_cache[qid]
        for m in ms:
            if m.kind == "RR":
                v = mrr_at_k(ranked, qrel, m.k, m.rel)
            elif m.kind == "nDCG":
                v = ndcg_at_k(ranked, qrel, m.k)
            else:
                v = recall_at_k(ranked, qrel, m.k, m.rel)
            per_query[m.name][qid] = v
    agg = {
        name: (sum(vals.values()) / len(vals) if vals else 0.0) for name, vals in per_query.items()
    }
    return agg, per_query
