"""ONNX Runtime CPU latency / throughput of a reranker, fp32 vs int8.

A "request" is what the broker sends: one query with its top-``b`` candidates in one batch
(padded to the longest pair), so the batch sizes double as rerank depths. Candidates are
real: BM25 (Anserini, full collection) top hits for DL19 + dev queries, texts from
collection.tsv, so sequence lengths are realistic.

Timed: ``session.run`` only (the broker tokenises in C#); tokenisation time in Python is
reported separately for reference. Warm-up runs are discarded. **macOS: dev-signal-only**
(threads cannot be pinned, E/P cores are mixed); Linux-box numbers must be re-measured.

    python -m hybridsearch.rerank.bench --model-dir data/models/reranker --name ours
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hybridsearch.data.io import REPO_ROOT, read_queries
from hybridsearch.rerank.export import INPUTS, OUTPUT, ort_session
from hybridsearch.rerank.model import EVAL_MAX_LEN, encode_pairs
from hybridsearch.rerank.text import iter_run_groups, load_passages
from hybridsearch.rerank.meta import write_meta

RESULTS = REPO_ROOT / "results/rerank"
DEFAULT_BATCHES = [1, 10, 16, 20, 50, 64, 100, 200]


def bench_requests(n_queries: int, depth: int, seed: int = 0):
    """``n_queries`` real (query, top-``depth`` BM25 candidates) requests: DL19 then dev."""
    srcs = [("data/raw/trec-dl/dl19.queries.tsv", "data/runs/reference/anserini-bm25-default.dl19.trec"),
            ("data/raw/msmarco/queries.dev.small.tsv", "data/runs/reference/anserini-bm25-default.dev.trec")]
    reqs = []
    for qpath, rpath in srcs:
        if len(reqs) >= n_queries or not (REPO_ROOT / rpath).exists():
            continue
        queries = read_queries(REPO_ROOT / qpath)
        for qid, hits in iter_run_groups(REPO_ROOT / rpath, max_hits=depth):
            if qid in queries and len(hits) >= depth:
                items = sorted(hits.items(), key=lambda x: (x[1], x[0]), reverse=True)
                reqs.append((queries[qid], [p for p, _ in items]))
            if len(reqs) >= n_queries:
                break
    text = load_passages({p for _, ps in reqs for p in ps})
    return [(q, [text[p] for p in ps]) for q, ps in reqs]


def time_model(onnx_path: Path, tok, reqs, batches, threads: int, iters: int, warmup: int = 3) -> list[dict]:
    sess = ort_session(onnx_path, threads)
    rows = []
    for b in batches:
        feeds, tok_s = [], []
        for q, ps in reqs[:iters]:
            t = time.perf_counter()
            enc = encode_pairs(tok, [q] * b, ps[:b], EVAL_MAX_LEN, return_tensors="np")
            tok_s.append(time.perf_counter() - t)
            feeds.append({n: enc[n].astype(np.int64) for n in INPUTS})
        for f in feeds[:warmup]:
            sess.run([OUTPUT], f)
        lat = []
        for f in feeds:
            t = time.perf_counter()
            sess.run([OUTPUT], f)
            lat.append(time.perf_counter() - t)
        lat_ms = np.array(lat) * 1e3
        rows.append({
            "batch": b, "requests": len(lat),
            "mean_ms": round(float(lat_ms.mean()), 3),
            "p50_ms": round(float(np.percentile(lat_ms, 50)), 3),
            "p90_ms": round(float(np.percentile(lat_ms, 90)), 3),
            "p99_ms": round(float(np.percentile(lat_ms, 99)), 3),
            "pairs_per_s": round(b * len(lat) / float(np.sum(lat)), 1),
            "mean_seq_len": round(float(np.mean([f["input_ids"].shape[1] for f in feeds])), 1),
            "py_tokenize_mean_ms": round(float(np.mean(tok_s)) * 1e3, 3),
        })
        print(json.dumps({"model": onnx_path.name, "threads": threads, **rows[-1]}), flush=True)
    return rows


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--iters", type=int, default=50, help="requests (distinct queries) per batch size")
    args = ap.parse_args()
    md = args.model_dir if args.model_dir.is_absolute() else REPO_ROOT / args.model_dir
    tok = AutoTokenizer.from_pretrained(str(md))
    reqs = bench_requests(args.iters + 3, max(args.batches))
    res = {"name": args.name, "model_dir": str(args.model_dir), "results": {}}
    for variant in ("model.onnx", "model.int8.onnx"):
        if not (md / variant).exists():
            continue
        for th in args.threads:
            res["results"][f"{variant}|threads={th}"] = time_model(md / variant, tok, reqs, args.batches, th, args.iters)
    out = RESULTS / f"bench_{args.name}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    write_meta(out, what="ORT CPU session.run latency per request (1 query + b candidates)",
               requests=args.iters, warmup=3, timing_label="dev-signal-only (macOS)",
               cache="warm (model in RAM, repeated runs)")


if __name__ == "__main__":
    main()
