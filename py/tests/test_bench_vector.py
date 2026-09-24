"""Tests for the vector bench helpers (tiny synthetic data; no big files)."""
import json

import numpy as np

from hybridsearch.bench import gt_from_pids, gt_to_trec
from hybridsearch.bench.io import read_fbin, read_gt, recall_at_k, write_fbin
from hybridsearch.data.io import write_u64bin


def _write_gt(path, ids, scores):
    with open(path, "wb") as f:
        np.array(ids.shape, dtype=np.uint32).tofile(f)
        ids.astype(np.uint32).tofile(f)
        scores.astype(np.float32).tofile(f)


def test_fbin_roundtrip(tmp_path):
    x = np.random.default_rng(0).standard_normal((7, 5)).astype(np.float32)
    write_fbin(tmp_path / "x.fbin", x)
    assert np.array_equal(read_fbin(str(tmp_path / "x.fbin")), x)
    assert read_fbin(str(tmp_path / "x.fbin"), max_rows=3).shape == (3, 5)


def test_recall_at_k():
    gt = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(np.array([[3, 2, 1], [4, 9, 9]]), gt, k=3) == 4 / 6


def test_gt_to_trec_maps_ordinals_to_pids(tmp_path):
    ids = np.array([[2, 0], [1, 2]])
    sc = np.array([[0.9, 0.5], [0.8, 0.1]])
    _write_gt(tmp_path / "gt.bin", ids, sc)
    (tmp_path / "q.txt").write_text("q1\nq2\n")
    write_u64bin(tmp_path / "d.u64bin", np.array([100, 200, 300], dtype=np.uint64))
    gt_to_trec.main([str(tmp_path / "gt.bin"), str(tmp_path / "q.txt"), str(tmp_path / "d.u64bin"),
                     str(tmp_path / "r.trec"), "exact", "--k", "2"])
    lines = (tmp_path / "r.trec").read_text().split("\n")
    assert lines[0] == "q1 Q0 300 1 0.900000 exact"
    assert lines[3].startswith("q2 Q0 300 2")


def test_gt_from_pids_roundtrip_and_check(tmp_path, capsys):
    docids = np.array([50, 10, 30, 20], dtype=np.uint64)  # ordinal -> pid, unsorted on purpose
    pids = np.array([[30, 10], [20, 50]], dtype=np.uint64)
    scores = np.array([[0.9, 0.2], [0.7, 0.6]], dtype=np.float32)
    write_u64bin(tmp_path / "p.u64bin", pids.ravel())
    write_fbin(tmp_path / "s.fbin", scores)
    write_u64bin(tmp_path / "d.u64bin", docids)
    _write_gt(tmp_path / "ours.bin", np.array([[2, 1], [3, 0]]), scores)
    gt_from_pids.main([str(tmp_path / "p.u64bin"), str(tmp_path / "s.fbin"), str(tmp_path / "d.u64bin"),
                       str(tmp_path / "out.bin"), "--check", str(tmp_path / "ours.bin")])
    ids, sc = read_gt(str(tmp_path / "out.bin"))
    assert ids.tolist() == [[2, 1], [3, 0]]
    rep = json.loads(capsys.readouterr().out)
    assert rep["top10_set_identical_frac"] == 1.0
