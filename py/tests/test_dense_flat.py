import faiss
import numpy as np
from hybridsearch.data.io import read_fbin, read_u64bin, write_fbin, write_u64bin
from hybridsearch.dense.flat import exact_topk


def test_blocked_matches_single_index():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = rng.normal(size=(20, 32)).astype(np.float32)
    ids = np.sort(rng.choice(10**6, 5000, replace=False)).astype(np.uint64)
    S, I = exact_topk(Q, X, ids, k=50, block=777)
    idx = faiss.IndexFlatIP(32)
    idx.add(X)
    D, J = idx.search(Q, 50)
    assert np.allclose(S, D, atol=1e-6)
    assert (I == ids[J].astype(np.int64)).all()


def test_ties_broken_by_lower_id():
    X = np.ones((10, 4), dtype=np.float32) / 2
    ids = np.arange(100, 110, dtype=np.uint64)[::-1].copy()  # file order descending ids
    _, I = exact_topk(np.ones((1, 4), dtype=np.float32), X, ids, k=5, block=3)
    assert I[0].tolist() == [100, 101, 102, 103, 104]


def test_fbin_u64bin_roundtrip(tmp_path):
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_fbin(tmp_path / "a.fbin", m)
    raw = (tmp_path / "a.fbin").read_bytes()
    assert raw[:8] == np.array([3, 4], dtype="<u4").tobytes() and len(raw) == 8 + 48
    assert (read_fbin(tmp_path / "a.fbin") == m).all()
    write_u64bin(tmp_path / "b.u64bin", [5, 2**40])
    raw = (tmp_path / "b.u64bin").read_bytes()
    assert raw[:8] == np.uint64(2).tobytes() and len(raw) == 24
    assert read_u64bin(tmp_path / "b.u64bin").tolist() == [5, 2**40]
