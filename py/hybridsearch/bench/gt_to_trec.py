"""Turn an hs_vec_groundtruth file (ordinals) into a TREC run over global passage ids:
the exact-search reference for MRR comparisons.

    python -m hybridsearch.bench.gt_to_trec GT.bin QIDS.txt DOCIDS.u64bin OUT.trec TAG [--k 10]
"""
from __future__ import annotations

import argparse

import numpy as np

from ..data.io import read_u64bin
from .io import read_gt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hybridsearch.bench.gt_to_trec")
    ap.add_argument("gt"); ap.add_argument("qids"); ap.add_argument("docids"); ap.add_argument("out"); ap.add_argument("tag")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(argv)
    ids, scores = read_gt(a.gt)
    qids = [l.strip() for l in open(a.qids) if l.strip()]
    docids = np.asarray(read_u64bin(a.docids))
    with open(a.out, "w") as f:
        for qi, qid in enumerate(qids[: len(ids)]):
            for r in range(min(a.k, ids.shape[1])):
                f.write(f"{qid} Q0 {int(docids[ids[qi, r]])} {r + 1} {scores[qi, r]:.6f} {a.tag}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
