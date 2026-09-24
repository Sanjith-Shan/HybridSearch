"""Convert the data module's exact dev ground truth (global pids, .u64bin + scores .fbin,
from hybridsearch.dense.flat) into the engine's ground-truth format (ordinals), and
optionally cross-check it against an hs_vec_groundtruth file on the same queries.

    python -m hybridsearch.bench.gt_from_pids PIDS.u64bin SCORES.fbin DOCIDS.u64bin OUT.bin \
        [--check OURS.bin]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..data.io import read_u64bin
from .io import read_fbin, read_gt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hybridsearch.bench.gt_from_pids")
    ap.add_argument("pids"); ap.add_argument("scores"); ap.add_argument("docids"); ap.add_argument("out")
    ap.add_argument("--check")
    a = ap.parse_args(argv)
    scores = np.asarray(read_fbin(a.scores, mmap=False))
    nq, k = scores.shape
    pids = np.asarray(read_u64bin(a.pids)).reshape(nq, k)
    docids = np.asarray(read_u64bin(a.docids))
    order = np.argsort(docids)
    pos = np.searchsorted(docids[order], pids)
    ords = order[pos].astype(np.uint32)
    assert np.all(docids[ords] == pids), "pid not in docids"
    with open(a.out, "wb") as f:
        np.array([nq, k], dtype=np.uint32).tofile(f)
        ords.tofile(f)
        scores.astype(np.float32).tofile(f)
    report = {"nq": nq, "k": k}
    if a.check:
        ours, oscore = read_gt(a.check)
        m = min(len(ours), nq)
        kk = min(10, ours.shape[1])
        same_set = np.mean([set(ours[i, :kk]) == set(ords[i, :kk]) for i in range(m)])
        max_score_diff = float(np.max(np.abs(oscore[:m, :kk] - scores[:m, :kk])))
        report.update({"checked_queries": m, "top10_set_identical_frac": float(same_set),
                       "max_abs_score_diff_top10": max_score_diff})
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
