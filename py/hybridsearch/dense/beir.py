"""BGE-base-en-v1.5 embeddings + exact dense runs for the BEIR subset
(SciFact, NFCorpus, FiQA), test split.

    python -m hybridsearch.dense.beir [--datasets scifact nfcorpus fiqa]

Inputs  data/raw/beir/{ds}/corpus.jsonl, queries.jsonl, qrels/test.tsv (BEIR format).
Outputs data/embeddings/beir/{ds}/corpus.fbin + corpus.docids.txt (row i = line i),
        queries.test.fbin + queries.test.qids.txt,
        data/runs/reference/dense-flat.beir-{ds}.test.trec (depth 1000),
        data/raw/beir/{ds}/qrels/test.trec (TREC-format copy of qrels/test.tsv).
Passage text = f"{title} {text}".strip() (the BEIR / sentence-transformers convention);
queries get the BGE retrieval instruction prefix. Holds data/locks/mps while encoding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import faiss
import torch

from ..data.io import DATA, write_fbin, write_run
from ..eval.results import write_result
from .encode import Encoder, acquire_lock, release_lock

BEIR = DATA / "raw" / "beir"
EMB = DATA / "embeddings" / "beir"
RUNS = DATA / "runs" / "reference"


def load(ds: str):
    with open(BEIR / ds / "corpus.jsonl", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    queries = {}
    with open(BEIR / ds / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            queries[str(o["_id"])] = o["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(BEIR / ds / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            q, d, s = line.rstrip("\n").split("\t")
            qrels.setdefault(q, {})[d] = int(s)
    return corpus, queries, qrels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    ap.add_argument("--k", type=int, default=1000)
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    stats = {}
    acquire_lock()
    try:
        enc = Encoder()
        for ds in args.datasets:
            corpus, queries, qrels = load(ds)
            out = EMB / ds
            out.mkdir(parents=True, exist_ok=True)
            docids = [str(c["_id"]) for c in corpus]
            texts = [f"{c.get('title', '') or ''} {c.get('text', '') or ''}".strip() for c in corpus]
            t = time.time()
            C = enc.encode_passages(texts)
            tc = time.time() - t
            write_fbin(out / "corpus.fbin", C)
            (out / "corpus.docids.txt").write_text("".join(f"{d}\n" for d in docids))
            qids = [q for q in qrels if q in queries]
            t = time.time()
            Q = enc.encode_queries([queries[q] for q in qids])
            tq = time.time() - t
            write_fbin(out / "queries.test.fbin", Q)
            (out / "queries.test.qids.txt").write_text("".join(f"{q}\n" for q in qids))
            with open(BEIR / ds / "qrels" / "test.trec", "w") as f:
                for q in qrels:
                    f.writelines(f"{q} 0 {d} {s}\n" for d, s in qrels[q].items())
            index = faiss.IndexFlatIP(C.shape[1])
            index.add(C)
            k = min(args.k, len(docids))
            D, I = index.search(Q, k)
            write_run(RUNS / f"dense-flat.beir-{ds}.test.trec",
                      ((qids[r], [(docids[j], float(s)) for j, s in zip(I[r], D[r])]) for r in range(len(qids))),
                      tag="dense-flat")
            stats[ds] = {"corpus": len(docids), "test_queries": len(qids), "missing_query_text": len(qrels) - len(qids),
                         "corpus_encode_seconds": round(tc, 1), "passages_per_sec": round(len(docids) / tc, 1),
                         "query_encode_seconds": round(tq, 2)}
            print(ds, stats[ds], flush=True)
    finally:
        release_lock()
    write_result("dense/beir_encode.json", {"model": "BAAI/bge-base-en-v1.5", "device": enc.device, "datasets": stats},
                 command="python -m hybridsearch.dense.beir " + " ".join(argv or sys.argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
