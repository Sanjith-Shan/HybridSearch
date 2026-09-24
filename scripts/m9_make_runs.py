#!/usr/bin/env python
"""Extra real rankers for the M9 sensitivity study, built from Anserini's BM25 DL19/DL20
runs and BGE-base-en-v1.5 similarities computed here (CPU) for BM25's top-100:

* ``bm25-bge-rerank100``: BM25 top-100 reordered by BGE inner product (a dense reranker
  over BM25 candidates; *not* dense retrieval over the collection).
* ``rrf-bm25-bge100``: reciprocal rank fusion (k=60) of BM25 and that reordering.

They are real rankers judged with real graded qrels; the dense vectors were encoded by
the M9 code on CPU with the project's encoder convention (spec honesty rule 2).
Written to data/runs/m9/{name}.{dl19,dl20}.trec.

    .venv/bin/python scripts/m9_make_runs.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")  # shared laptop: <= 4 threads

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from hybridsearch.clicks.rankers import RUN_DIR, load_dl_queries, ranked, rrf  # noqa: E402
from hybridsearch.data.io import DATA, read_run, write_run  # noqa: E402
from hybridsearch.ltr import dense_feature as DF  # noqa: E402
from hybridsearch.rerank.meta import write_meta  # noqa: E402

OUT = DATA / "runs/m9"
DEPTH = 100
BM25 = "anserini-bm25-default"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = load_dl_queries()
    info = {}
    for split in ("dl19", "dl20"):
        bm = read_run(RUN_DIR / f"{BM25}.{split}.trec")  # full-collection Anserini run, explicitly
        cands = {q: ranked(r)[:DEPTH] for q, r in bm.items()}
        pids = set().union(*cands.values())
        vecs = DF.encode_needed({q: queries[q] for q in cands}, DF.passage_texts(pids),
                                log=lambda m: print(m, flush=True))
        info[split] = vecs["info"]
        dense = DF.dense_run_for_candidates(cands, vecs)
        rer = {q: sorted(d.items(), key=lambda x: (-x[1], int(x[0]))) for q, d in dense.items()}
        write_run(OUT / f"bm25-bge-rerank100.{split}.trec", sorted(rer.items(), key=lambda x: int(x[0])),
                  "bm25-bge-rerank100")
        bm100 = {q: {d: bm[q][d] for d in c} for q, c in cands.items()}
        fused = rrf([bm100, dense], k=60)
        fr = {q: sorted(d.items(), key=lambda x: (-x[1], int(x[0]))) for q, d in fused.items()}
        write_run(OUT / f"rrf-bm25-bge100.{split}.trec", sorted(fr.items(), key=lambda x: int(x[0])),
                  "rrf-bm25-bge100")
    for name in ("bm25-bge-rerank100", "rrf-bm25-bge100"):
        for split in ("dl19", "dl20"):
            write_meta(OUT / f"{name}.{split}.trec", depth=DEPTH, dense_encode=info[split],
                       note="BGE-base-en-v1.5 encoded by M9 on CPU for BM25 top-100 candidates only")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
