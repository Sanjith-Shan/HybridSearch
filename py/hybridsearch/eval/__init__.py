"""HybridSearch evaluation harness: MRR@10, nDCG@k, R@k over TREC run files.

Implemented from scratch; parity with trec_eval (via pytrec_eval / ir_measures) is
checked in py/tests/test_eval.py.
"""
from .metrics import (  # noqa: F401
    DEFAULT_METRICS,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    parse_metric,
    rank_run,
    recall_at_k,
)
