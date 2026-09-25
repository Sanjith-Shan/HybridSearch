"""The vectorised sweep path must equal the pure reference implementation exactly."""
import numpy as np
import pytest
from hybridsearch.fusion import rrf, weighted
from hybridsearch.fusion.fast import prepare, rrf_fast, weighted_fast


def _random_pair(rng, n_docs=60, n_lex=25, n_dense=30, ties=False):
    pool = [str(x) for x in rng.choice(10_000, n_docs, replace=False)]
    lex_ids = list(rng.choice(pool, n_lex, replace=False))
    dense_ids = list(rng.choice(pool, n_dense, replace=False))
    ls = np.sort(rng.random(n_lex) * 20)[::-1]
    ds = np.sort(rng.random(n_dense))[::-1]
    if ties:
        ls = np.round(ls)
        ds = np.round(ds, 1)
    return list(zip(lex_ids, map(float, ls))), list(zip(dense_ids, map(float, ds)))


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("depth", [None, 10])
def test_fast_equals_core(seed, depth):
    rng = np.random.default_rng(seed)
    lex, dense = _random_pair(rng, ties=seed % 2 == 0)
    qc = prepare(lex, dense, depth)
    for k in (0, 10, 60):
        assert rrf_fast(qc, k) == rrf([lex, dense], k=k, depth=depth)
    for norm in ("minmax", "zscore"):
        for alpha in (0.0, 0.3, 0.5, 0.9, 1.0):
            assert weighted_fast(qc, alpha, norm) == weighted(lex, dense, alpha, norm, depth=depth)


def test_fast_empty_side():
    lex = [("5", 3.0), ("2", 1.0)]
    qc = prepare(lex, [], None)
    assert rrf_fast(qc, 60) == rrf([lex, []], k=60)
    assert weighted_fast(qc, 0.4, "zscore") == weighted(lex, [], 0.4, "zscore")
