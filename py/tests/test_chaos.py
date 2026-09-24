from hybridsearch.serving.chaos import Sample, mrr_at_10, summarize


def test_mrr_counts_first_relevant_in_top10_and_errors_as_zero():
    qrels = {"1": {10: 1}, "2": {20: 1}, "3": {30: 1}}
    samples = [
        Sample("fault", "1", 5.0, 200, doc_ids=[7, 10, 11]),        # rr = 1/2
        Sample("fault", "2", 5.0, 200, doc_ids=list(range(100, 111)) + [20]),  # rank 12 -> 0
        Sample("fault", "3", 5.0, 599),                               # error -> 0
        Sample("fault", "9", 5.0, 200, doc_ids=[1]),                  # unjudged, ignored
    ]
    assert mrr_at_10(samples, qrels) == (0.5 + 0 + 0) / 3


def test_summarize_rates():
    qrels = {"1": {1: 1}}
    samples = [
        Sample("fault", "1", 10.0, 200, doc_ids=[1], degradation_level=1, failed_shards=1),
        Sample("fault", "1", 30.0, 200, doc_ids=[2]),
    ]
    s = summarize(samples, qrels)
    assert s["requests"] == 2
    assert s["degraded_rate"] == 0.5
    assert s["failed_shard_rate"] == 0.5
    assert s["mrr_at_10"] == 0.5
