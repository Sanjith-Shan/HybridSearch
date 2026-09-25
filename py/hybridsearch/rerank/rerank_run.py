"""Rerank a TREC run's top-``depth`` candidates with a cross-encoder and evaluate it.

Backends: ``torch`` (a HF checkpoint dir or hub id, on cpu/mps/cuda) or ``ort`` (an .onnx
file with the ``score`` interface, ONNX Runtime CPU). Evaluation goes through the project's
harness (``hybridsearch.eval.metrics.evaluate``, trec_eval parity verified by the data agent).

    python -m hybridsearch.rerank.rerank_run --model data/rerank/runs/mps_bm25/best \
        --split dev --candidates data/runs/reference/anserini-bm25-default.dev.trec --depth 100 \
        --name mps_bm25.bm25full

Writes ``results/rerank/runs/<name>.<split>.trec`` (gitignored-size run) and
``results/rerank/eval/<name>.<split>.json`` (+ .meta.json); raw scores are cached in
``data/rerank/scores/<name>.<split>.npz`` so the cascade curve reuses them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hybridsearch.data.io import DATA, REPO_ROOT, read_qrels, read_queries, write_run
from hybridsearch.rerank.meta import write_meta
from hybridsearch.rerank.model import EVAL_MAX_LEN, clip_query, pick_device
from hybridsearch.rerank.text import FULL_COLLECTION, iter_run_groups, load_passages

SPLITS = {
    "dev": ("data/raw/msmarco/queries.dev.small.tsv", "data/raw/msmarco/qrels.dev.small.tsv"),
    "dl19": ("data/raw/trec-dl/dl19.queries.tsv", "data/raw/trec-dl/dl19.qrels"),
    "dl20": ("data/raw/trec-dl/dl20.queries.tsv", "data/raw/trec-dl/dl20.qrels"),
    "train_tune": ("data/subset/train_tune/queries.tsv", "data/subset/train_tune/qrels.tsv"),
    # the 1M-subset qrels (identical judgments for relevant docs; see subset manifest)
    "dev1m": ("data/raw/msmarco/queries.dev.small.tsv", "data/subset/1m/qrels.dev.tsv"),
    "dl19_1m": ("data/raw/trec-dl/dl19.queries.tsv", "data/subset/1m/qrels.dl19.tsv"),
    "dl20_1m": ("data/raw/trec-dl/dl20.queries.tsv", "data/subset/1m/qrels.dl20.tsv"),
}
METRICS = {
    "dev": ["RR@10", "nDCG@10", "R@100"],
    "dl": ["nDCG@10", "RR(rel=2)@10", "R(rel=2)@100"],
}
RESULTS = REPO_ROOT / "results/rerank"
SCORES = DATA / "rerank/scores"


def metrics_for(split: str) -> list[str]:
    return METRICS["dl" if split.startswith("dl") else "dev"]


def load_candidates(run_path: Path, qids: set[str], depth: int) -> dict[str, list[tuple[str, float]]]:
    """Top-``depth`` per query in the evaluator's order (score desc, docid desc on ties)."""
    out = {}
    for qid, hits in iter_run_groups(run_path):
        if qid in qids:
            items = sorted(hits.items(), key=lambda x: (x[1], x[0]), reverse=True)[:depth]
            out[qid] = items
    return out


class Scorer:
    def __init__(self, model: str, backend: str, device: str = "auto", threads: int | None = None,
                 max_length: int = EVAL_MAX_LEN):
        from transformers import AutoTokenizer

        self.backend, self.max_length = backend, max_length
        if backend == "ort":
            from hybridsearch.rerank.export import ort_session

            self.sess = ort_session(Path(model), threads, spinning=False)
            self.tok = AutoTokenizer.from_pretrained(str(Path(model).parent))
            self.device = "cpu"
        else:
            from hybridsearch.rerank.model import load_cross_encoder

            self.device = pick_device(device)
            if self.device == "mps":
                torch_mod = __import__("torch")
                torch_mod.mps.set_per_process_memory_fraction(0.3)
            self.tok, self.model = load_cross_encoder(model, self.device)

    def score(self, Q: list[str], P: list[str], batch: int = 128, progress: bool = False) -> np.ndarray:
        """Scores in input order; internally length-sorted so padding is minimal."""
        import torch

        from hybridsearch.rerank.export import ort_score
        from hybridsearch.rerank.model import encode_pairs

        out = np.empty(len(Q), np.float32)
        order = np.argsort([len(q) + len(p) for q, p in zip(Q, P)], kind="stable")
        t = time.time()
        for n, s in enumerate(range(0, len(order), batch)):
            idx = order[s:s + batch]
            q, p = [Q[i] for i in idx], [P[i] for i in idx]
            if self.backend == "ort":
                out[idx] = ort_score(self.sess, self.tok, q, p, self.max_length, batch)
            else:
                with torch.no_grad():
                    enc = encode_pairs(self.tok, q, p, self.max_length,
                                       pad_multiple=32 if self.device == "mps" else None)
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    out[idx] = self.model(**enc).logits[:, 0].float().cpu().numpy()
            if self.backend == "torch" and self.device == "mps" and n % 100 == 0:
                torch.mps.empty_cache()
            if progress and n % 200 == 0:
                print(f"  {s}/{len(Q)} pairs, {s / max(time.time() - t, 1e-9):.0f} pairs/s", flush=True)
        return out


def rerank(name: str, split: str, candidates: Path, depth: int, model: str, backend: str,
           device: str = "auto", threads: int | None = None, collection: Path = FULL_COLLECTION,
           batch: int = 128, candidate_label: str = "", max_queries: int | None = None) -> dict:
    from hybridsearch.rerank import cap_threads

    cap_threads()
    qpath, qrels_path = SPLITS[split]
    queries = read_queries(REPO_ROOT / qpath)
    qrels = read_qrels(REPO_ROOT / qrels_path)
    qids = {q for q in queries if q in qrels}
    if max_queries:  # first N judged queries by qid; qrels restricted so the mean is over N
        qids = set(sorted(qids, key=int)[:max_queries])
        qrels = {q: qrels[q] for q in qids}
    cands = load_candidates(candidates, qids, depth)
    t0 = time.time()
    need = {p for hits in cands.values() for p, _ in hits}
    text = load_passages(need, collection)
    missing = need - set(text)
    if missing:
        raise KeyError(f"{len(missing)} candidate pids missing from {collection}")
    scorer = Scorer(model, backend, device, threads)
    keys = [(q, p) for q in sorted(cands, key=int) for p, _ in cands[q]]
    Q = [clip_query(scorer.tok, queries[q]) for q, _ in keys]
    P = [text[p] for _, p in keys]
    from hybridsearch.rerank.train import MPSLock

    with MPSLock(scorer.device, f"rerank-score {name}.{split}", wait=True):
        t1 = time.time()
        scores = scorer.score(Q, P, batch=batch, progress=True)
        score_seconds = time.time() - t1

    SCORES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORES / f"{name}.{split}.npz", qid=np.array([k[0] for k in keys]),
                        pid=np.array([k[1] for k in keys]), score=scores,
                        first_stage=np.array([s for q in sorted(cands, key=int) for _, s in cands[q]], np.float32))
    run: dict[str, dict[str, float]] = {}
    for (q, p), s in zip(keys, scores):
        run.setdefault(q, {})[p] = float(s)
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    run_path = RESULTS / "runs" / f"{name}.{split}.trec"
    write_run(run_path, [(q, sorted(h.items(), key=lambda x: (-x[1], x[0]))) for q, h in
                         sorted(run.items(), key=lambda x: int(x[0]))], name)

    from hybridsearch.eval.metrics import evaluate

    ms = metrics_for(split)
    agg, _ = evaluate(run, qrels, ms)
    first = {q: dict(h) for q, h in cands.items()}
    base, _ = evaluate(first, qrels, ms)
    res = {
        "name": name, "split": split, "model": model, "backend": backend, "device": scorer.device,
        "candidates": str(candidates.relative_to(REPO_ROOT)) if candidates.is_absolute() else str(candidates),
        "candidate_label": candidate_label, "depth": depth,
        "n_queries_qrels": len(qrels), "max_queries": max_queries, "n_queries_with_candidates": len(cands),
        "n_pairs": len(keys), "metrics": {k: round(v, 4) for k, v in agg.items()},
        "first_stage_metrics_at_depth": {k: round(v, 4) for k, v in base.items()},
        "score_seconds": round(score_seconds, 1), "pairs_per_s": round(len(keys) / score_seconds, 1),
        "load_seconds": round(t1 - t0, 1), "max_length": scorer.max_length,
    }
    (RESULTS / "eval").mkdir(parents=True, exist_ok=True)
    out = RESULTS / "eval" / f"{name}.{split}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    write_meta(out, pairs_scored=len(keys), note="pairs_per_s is throughput on this laptop, dev-signal-only")
    print(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["torch", "ort"], default="torch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--candidate-label", default="")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--collection", type=Path, default=FULL_COLLECTION)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="evaluate only the first N judged queries (by qid)")
    args = ap.parse_args()
    cand = args.candidates if args.candidates.is_absolute() else REPO_ROOT / args.candidates
    rerank(args.name, args.split, cand, args.depth, args.model, args.backend, args.device,
           args.threads, args.collection, args.batch, args.candidate_label, args.max_queries)


if __name__ == "__main__":
    main()
