"""Hard-negative mining from HybridSearch's own retrievers, with false-negative filtering.

Sources, per training query (``data/subset/train50k``):

- **bm25**: the BM25 run over the full 8.8M collection (``data/runs/reference/bm25.train50k.trec``
  from the data agent, Anserini default k1=0.9 b=0.4; later the engine's own run, same format).
- **dense**: exact inner-product search (BLAS matmul + top-k) of the BGE query vector over the
  1M-subset passage vectors (``data/embeddings/1m``). Exact, so no ANN error in the pool.
- **random**: pids drawn uniformly from the whole collection (seeded).

Filters, in order (every drop is counted in the output ``stats``):

1. **No positive leaks.** Any pid judged for the query in ``qrels.train`` (all grades > 0) is
   removed from every pool. This is an invariant, tested in ``py/tests/test_rerank_sampler.py``.
2. **Likely false negatives (both-retrievers rule).** MS MARCO train has ~1 judged passage
   per query, so many top-ranked "negatives" are unjudged relevant passages. A candidate ``n``
   is dropped when **both** retrievers put it above the query's best positive by a margin:

       dense:  s_d(n) - max_pos s_d(pos) >= dense_margin              (cosine; default 0.05)
       bm25:   s_b(n) - B            >= bm25_rel_margin * B           (relative; default 0.10)

   where ``s_d(pos)`` is computed exactly (the positives are encoded with the same BGE model,
   ``encode_positives``), and ``B`` is the best positive's BM25 score if a positive is in the
   BM25 top-1000, else the lowest score in that list (an upper bound on the positive's score,
   so the margin used is a lower bound and the rule drops *fewer* candidates, never more).
   A candidate only one retriever can score (a BM25 hit outside the 1M subset has no vector;
   a dense hit below the BM25 top-1000 has no BM25 score) cannot be confirmed by both and is
   **kept**; how many were unverifiable is reported. Margins are fixed a priori, not tuned.
3. Duplicate text of a positive is removed later, in the sampler, where texts are loaded.

Output: ``data/rerank/mined/<split>.jsonl``: one record per query
``{"qid", "pos": [pid...], "bm25": [pid...], "dense": [pid...], "random": [pid...]}``
(lists in retriever rank order, already filtered, cut to ``pool_depth``), plus
``<split>.stats.json``.

    python -m hybridsearch.rerank.mine positives   # encode train50k positives (ORT CPU)
    python -m hybridsearch.rerank.mine dense-runs  # exact FAISS top-k over 1M for every split
    python -m hybridsearch.rerank.mine mine        # write the jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hybridsearch.data.io import (DATA, REPO_ROOT, read_fbin, read_qrels, read_queries,
                                  read_u64bin, write_fbin, write_run)

N_COLLECTION = 8_841_823
EMB = DATA / "embeddings/1m"
MINED = DATA / "rerank/mined"
DENSE_RUNS = DATA / "runs/rerank"
SPLITS = {
    "train50k": (DATA / "subset/train50k/queries.tsv", DATA / "subset/train50k/qrels.tsv"),
    "train_tune": (DATA / "subset/train_tune/queries.tsv", DATA / "subset/train_tune/qrels.tsv"),
    "dev": (DATA / "raw/msmarco/queries.dev.small.tsv", DATA / "subset/1m/qrels.dev.tsv"),
    "dl19": (DATA / "raw/trec-dl/dl19.queries.tsv", DATA / "subset/1m/qrels.dl19.tsv"),
    "dl20": (DATA / "raw/trec-dl/dl20.queries.tsv", DATA / "subset/1m/qrels.dl20.tsv"),
}


@dataclass
class MineConfig:
    split: str = "train50k"
    bm25_run: str = "data/runs/reference/bm25.train50k.trec"   # or several files, comma-separated
    pool_depth: int = 50          # candidates kept per source after filtering
    dense_depth: int = 200        # exact dense top-k searched (before filtering)
    dense_margin: float = 0.05
    bm25_rel_margin: float = 0.10
    seed: int = 20260923


# ------------------------------------------------------------------ dense side
def load_query_vectors(split: str) -> tuple[list[str], np.ndarray]:
    qids = (EMB / f"queries.{split}.qids.txt").read_text().split()
    vecs = read_fbin(EMB / f"queries.{split}.fbin", mmap=False)
    if len(qids) != vecs.shape[0]:
        raise ValueError(f"{split}: {len(qids)} qids vs {vecs.shape[0]} vectors")
    return qids, vecs


def exact_topk(xb: np.ndarray, queries: np.ndarray, k: int, batch: int = 256):
    """Exact inner-product top-k by BLAS matmul + argpartition. Returns (scores, row ids),
    sorted by score desc, ties by lower row id. (FAISS is not used in this module: its
    OpenMP runtime aborts alongside torch's, see docs/BUG_LOG.md.)"""
    k = min(k, xb.shape[0])
    S = np.empty((queries.shape[0], k), np.float32)
    I = np.empty((queries.shape[0], k), np.int64)
    for i in range(0, queries.shape[0], batch):
        sc = np.asarray(queries[i:i + batch], np.float32) @ xb.T
        part = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        for j in range(sc.shape[0]):
            cand = part[j]
            order = np.lexsort((cand, -sc[j, cand]))
            I[i + j] = cand[order]
            S[i + j] = sc[j, I[i + j]]
    return S, I


def load_passage_matrix() -> tuple[np.ndarray, np.ndarray]:
    passages_path = EMB / "passages.fbin"
    if not passages_path.exists():
        raise FileNotFoundError(f"{passages_path} not ready (dense encoding still running?)")
    docids = read_u64bin(DATA / "subset/1m/docids.u64bin")
    xb = read_fbin(passages_path, mmap=False)
    if xb.shape[0] != docids.shape[0]:
        raise ValueError("passages.fbin rows != docids")
    return xb, docids


def exact_dense_search(queries: np.ndarray, k: int):
    """Exact IP top-k over the 1M passages. Returns (scores[nq,k], pids[nq,k] global ids)."""
    xb, docids = load_passage_matrix()
    S, I = exact_topk(xb, queries, k)
    return S, docids[I].astype(np.int64)


def dense_runs(splits: list[str], k: int) -> None:
    """Write exact dense top-k runs over the 1M subset for each split."""
    DENSE_RUNS.mkdir(parents=True, exist_ok=True)
    xb, docids = load_passage_matrix()
    for split in splits:
        t = time.time()
        qids, q = load_query_vectors(split)
        S, I = exact_topk(xb, q, k)
        results = [(qids[j], [(str(int(docids[r])), float(v)) for r, v in zip(I[j], S[j])])
                   for j in range(len(qids))]
        out = DENSE_RUNS / f"dense1m-exact.{split}.trec"
        write_run(out, results, "hs-dense1m-exact")
        print(f"{split}: {len(qids)} queries -> {out} ({time.time() - t:.1f}s)")


# ------------------------------------------------------------------ positives
def encode_positives(split: str = "train50k", batch: int = 32, threads: int | None = None,
                     backend: str = "mps") -> Path:
    """Encode every judged passage of ``split`` with BGE-base (no prefix, max 512, CLS, L2).
    ``backend="mps"`` uses the dense agent's torch Encoder under the data/locks/mps mutex;
    ``"ort"`` uses the exported ONNX encoder on CPU (identical vectors to ~1e-6, far slower on
    a contended laptop). Writes data/rerank/positives.<split>.fbin + .pids.txt."""
    from hybridsearch.rerank.text import load_passages

    _, qrels_path = SPLITS[split]
    qrels = read_qrels(qrels_path)
    pids = sorted({int(d) for docs in qrels.values() for d, g in docs.items() if g > 0})
    text = load_passages(set(pids))
    texts = [text[str(p)] for p in pids]
    t = time.time()
    if backend == "ort":
        from transformers import AutoTokenizer

        from hybridsearch.dense.export_query_encoder import OUT_DIR, ort_encode, ort_session

        tok = AutoTokenizer.from_pretrained(str(OUT_DIR))
        sess = ort_session(OUT_DIR / "model.onnx", threads or 4)
        order = sorted(range(len(pids)), key=lambda i: len(texts[i]))
        vecs = np.empty((len(pids), 768), np.float32)
        for s in range(0, len(order), batch * 16):
            idx = order[s:s + batch * 16]
            vecs[idx] = ort_encode(sess, tok, [texts[i] for i in idx], 512, batch)
            print(f"  {s + len(idx)}/{len(pids)} {(s + len(idx)) / (time.time() - t):.0f} p/s", flush=True)
    else:
        from hybridsearch.dense.encode import Encoder
        from hybridsearch.rerank.train import MPSLock

        with MPSLock(backend, f"rerank-mine positives {split}", wait=True):
            t = time.time()
            vecs = Encoder(backend).encode_passages(texts)
    out = DATA / f"rerank/positives.{split}.fbin"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_fbin(out, vecs)
    Path(f"{out}.pids.txt").write_text("\n".join(map(str, pids)) + "\n")
    print(f"encoded {len(pids)} positives on {backend} in {time.time() - t:.0f}s -> {out}")
    return out


# ------------------------------------------------------------------ mining
def false_negative(s_d_neg: float | None, s_d_pos: float | None,
                   s_b_neg: float | None, s_b_pos_ub: float | None,
                   dense_margin: float, bm25_rel_margin: float) -> bool | None:
    """True = drop (both retrievers rank it above the best positive by the margin),
    False = keep, None = unverifiable (a score is missing) -> kept."""
    if s_d_neg is None or s_d_pos is None or s_b_neg is None or s_b_pos_ub is None:
        return None
    dense_above = (s_d_neg - s_d_pos) >= dense_margin
    bm25_above = (s_b_neg - s_b_pos_ub) >= bm25_rel_margin * abs(s_b_pos_ub)
    return bool(dense_above and bm25_above)


def random_pool(rng: np.random.Generator, exclude: set[str], n: int) -> list[str]:
    out: list[str] = []
    seen = set(exclude)
    while len(out) < n:
        p = str(int(rng.integers(0, N_COLLECTION)))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def dict_or_stream(run_paths: list[Path]):
    """Returns f(qids_in_order) yielding (qid, hits) for every qid: BM25 hits streamed from
    the run file(s) (constant memory), then an empty dict for queries no run contains."""
    from hybridsearch.rerank.text import iter_run_groups

    def gen(qids):
        want = set(qids)
        done = set()
        for run_path in run_paths:
            for qid, hits in iter_run_groups(run_path):
                if qid in want and qid not in done:
                    done.add(qid)
                    yield qid, hits
        for qid in qids:
            if qid not in done:
                yield qid, {}
    return gen


def mine(cfg: MineConfig) -> Path:
    from hybridsearch.rerank import cap_threads

    cap_threads()
    queries_path, qrels_path = SPLITS[cfg.split]
    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)
    bm25_groups = dict_or_stream([REPO_ROOT / p for p in cfg.bm25_run.split(",")])
    qids, qvecs = load_query_vectors(cfg.split)
    qrow = {q: i for i, q in enumerate(qids)}

    # passage vectors for the 1M subset (dense scores of BM25 hits), and for positives
    docids = read_u64bin(DATA / "subset/1m/docids.u64bin")
    ordinal = {int(p): i for i, p in enumerate(docids)}
    xb = read_fbin(EMB / "passages.fbin", mmap=True)
    pos_path = DATA / f"rerank/positives.{cfg.split}.fbin"
    pos_vecs = read_fbin(pos_path, mmap=False)
    pos_row = {p: i for i, p in enumerate(Path(f"{pos_path}.pids.txt").read_text().split())}

    t = time.time()
    dS, dP = exact_dense_search(qvecs, cfg.dense_depth)
    print(f"dense search {len(qids)} x 1M in {time.time() - t:.0f}s", flush=True)

    rng = np.random.default_rng(cfg.seed)
    stats = {k: 0 for k in ("queries", "no_bm25_hits", "pos_in_bm25_top1000", "leak_removed_bm25",
                            "leak_removed_dense", "fn_dropped_bm25", "fn_dropped_dense",
                            "unverifiable_bm25", "unverifiable_dense", "verified_kept_bm25",
                            "verified_kept_dense")}
    MINED.mkdir(parents=True, exist_ok=True)
    out = MINED / f"{cfg.split}.jsonl"
    tmp = Path(f"{out}.tmp")
    with open(tmp, "w") as f:
        for qid, hits in bm25_groups(sorted(queries, key=int)):
            if qid not in qrels or qid not in qrow:
                continue
            stats["queries"] += 1
            pos = sorted(d for d, g in qrels[qid].items() if g > 0)
            judged = set(qrels[qid])
            qv = qvecs[qrow[qid]]
            s_d_pos = max(float(pos_vecs[pos_row[p]] @ qv) for p in pos if p in pos_row) \
                if any(p in pos_row for p in pos) else None
            if not hits:
                # not in the (possibly partial) BM25 run: skipped, so every strategy trains
                # on exactly the same queries
                stats["no_bm25_hits"] += 1
                stats["queries"] -= 1
                continue
            pos_b = [hits[p] for p in pos if p in hits]
            if pos_b:
                stats["pos_in_bm25_top1000"] += 1
            s_b_pos_ub = max(pos_b) if pos_b else (min(hits.values()) if hits else None)

            def dense_score(pid: str) -> float | None:
                o = ordinal.get(int(pid))
                return None if o is None else float(np.asarray(xb[o]) @ qv)

            def keep(pid: str, s_d: float | None, s_b: float | None, src: str) -> bool:
                if pid in judged:
                    stats[f"leak_removed_{src}"] += 1
                    return False
                fn = false_negative(s_d, s_d_pos, s_b, s_b_pos_ub, cfg.dense_margin, cfg.bm25_rel_margin)
                if fn is None:
                    stats[f"unverifiable_{src}"] += 1
                    return True
                if fn:
                    stats[f"fn_dropped_{src}"] += 1
                    return False
                stats[f"verified_kept_{src}"] += 1
                return True

            bm25_pool = []
            for pid, s_b in sorted(hits.items(), key=lambda x: (-x[1], int(x[0]))):
                if len(bm25_pool) >= cfg.pool_depth:
                    break
                if keep(pid, dense_score(pid), s_b, "bm25"):
                    bm25_pool.append(pid)
            dense_pool = []
            r = qrow[qid]
            for pid, s_d in zip(dP[r], dS[r]):
                if len(dense_pool) >= cfg.pool_depth:
                    break
                pid = str(int(pid))
                if keep(pid, float(s_d), hits.get(pid), "dense"):
                    dense_pool.append(pid)
            rnd = random_pool(rng, judged, cfg.pool_depth)
            f.write(json.dumps({"qid": qid, "pos": pos, "bm25": bm25_pool, "dense": dense_pool,
                                "random": rnd}) + "\n")
    tmp.replace(out)
    stats["seconds"] = round(time.time() - t, 1)
    stats["config"] = asdict(cfg)
    Path(f"{MINED / cfg.split}.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["positives", "dense-runs", "mine"])
    ap.add_argument("--split", default="train50k")
    ap.add_argument("--splits", nargs="+", default=["dev", "dl19", "dl20", "train_tune"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--pool-depth", type=int, default=50)
    ap.add_argument("--bm25-run", default=MineConfig.bm25_run, help="comma-separated run files")
    ap.add_argument("--backend", default="mps", choices=["mps", "cuda", "cpu", "ort"])
    args = ap.parse_args()
    if args.cmd == "positives":
        encode_positives(args.split, threads=args.threads, backend=args.backend)
    elif args.cmd == "dense-runs":
        dense_runs(args.splits, args.k)
    else:
        mine(MineConfig(split=args.split, pool_depth=args.pool_depth, bm25_run=args.bm25_run))


if __name__ == "__main__":
    main()
