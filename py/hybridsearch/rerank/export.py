"""Export a cross-encoder to ONNX (fp32 + dynamic int8) and check parity.

Graph: ``input_ids, attention_mask, token_type_ids`` (int64 [batch, seq]) -> ``score``
(float32 [batch], the raw relevance logit; higher is more relevant).

    python -m hybridsearch.rerank.export --model cross-encoder/ms-marco-MiniLM-L6-v2 \
        --out data/models/reranker-public
    python -m hybridsearch.rerank.export --model runs/<ckpt> --out data/models/reranker --int8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hybridsearch.data.io import DATA, REPO_ROOT, read_qrels, read_queries
from hybridsearch.dense.export_query_encoder import export_onnx, save_tokenizer
from hybridsearch.rerank.meta import write_meta
from hybridsearch.rerank.model import (EVAL_MAX_LEN, CrossEncoderGraph, encode_pairs,
                                       load_cross_encoder)

INPUTS = ["input_ids", "attention_mask", "token_type_ids"]
OUTPUT = "score"


def export_cross_encoder(model_id: str, out_dir: Path, max_length: int = EVAL_MAX_LEN) -> Path:
    tok, clf = load_cross_encoder(model_id, "cpu")
    graph = CrossEncoderGraph(clf)
    enc = encode_pairs(tok, ["what is a cat", "q"], ["a cat is a small mammal.", "p p p"], max_length)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"
    export_onnx(graph, path, INPUTS, OUTPUT, tuple(enc[n] for n in INPUTS))
    save_tokenizer(tok, model_id, out_dir)
    cfg = {
        "source_model": model_id,
        "max_length": max_length,
        "pair_format": "[CLS] query [SEP] passage [SEP]; token_type 0 then 1; truncation only_second",
        "do_lower_case": True,
        "inputs": {n: "int64[batch,seq]" for n in INPUTS},
        "output": {OUTPUT: "float32[batch] raw logit, higher = more relevant"},
    }
    (out_dir / "reranker_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return path


def quantize_int8(fp32: Path, out: Path | None = None) -> Path:
    """Dynamic (weight-only static, activation dynamic) int8 quantisation of MatMul/Gemm."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = out or fp32.with_name("model.int8.onnx")
    pre = fp32.with_name("model.pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)  # symbolic inference fails on the traced BERT mask subgraph
    quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"])
    pre.unlink()
    return out


def ort_session(path: Path, threads: int | None = None, spinning: bool = True):
    """``spinning=False`` stops idle intra-op threads from busy-waiting. On a machine shared
    with other jobs, spinning threads fight for cores: measured 14.0 s vs 8.0 s per batch of 64
    at 4 threads under load ~50 on 12 cores. Latency benches keep the default (a dedicated
    serving box spins); batch scoring on the shared laptop turns it off."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    from hybridsearch.rerank import MAX_CPU_THREADS

    so.intra_op_num_threads = min(threads or MAX_CPU_THREADS, MAX_CPU_THREADS)
    so.inter_op_num_threads = 1
    if not spinning:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_score(sess, tok, queries, passages, max_length: int = EVAL_MAX_LEN, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(queries), batch):
        enc = encode_pairs(tok, list(queries[i:i + batch]), list(passages[i:i + batch]),
                           max_length, return_tensors="np")
        out.append(sess.run([OUTPUT], {n: enc[n].astype(np.int64) for n in INPUTS})[0])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@torch.no_grad()
def torch_score(model, tok, queries, passages, max_length: int = EVAL_MAX_LEN, batch: int = 64,
                device: str = "cpu") -> np.ndarray:
    out = []
    for i in range(0, len(queries), batch):
        enc = encode_pairs(tok, list(queries[i:i + batch]), list(passages[i:i + batch]), max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out.append(model(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def parity_pairs(n_queries: int = 100, per_query: int = 5, seed: int = 0) -> tuple[list, list, list]:
    """``n_queries`` dev queries, each with its first judged passage + (per_query-1) passages
    drawn uniformly from the collection (seeded). Returns (qids, queries, passages)."""
    from hybridsearch.rerank.text import FULL_COLLECTION, load_passages

    queries = read_queries(DATA / "raw/msmarco/queries.dev.small.tsv")
    qrels = read_qrels(DATA / "raw/msmarco/qrels.dev.small.tsv")
    qids = [q for q in queries if q in qrels][:n_queries]
    rng = random.Random(seed)
    pos = {q: sorted(qrels[q])[0] for q in qids}
    # uniform random pids that exist: MS MARCO pids are 0..8841822 contiguous
    rand = {q: [str(rng.randrange(8_841_823)) for _ in range(per_query - 1)] for q in qids}
    need = set(pos.values()) | {p for v in rand.values() for p in v}
    text = load_passages(need, FULL_COLLECTION)
    Q, P, I = [], [], []
    for q in qids:
        for pid in [pos[q], *rand[q]]:
            I.append(q)
            Q.append(queries[q])
            P.append(text[pid])
    return I, Q, P


def parity(model_id: str, onnx_path: Path, n_queries: int = 100, per_query: int = 5,
           tol: float = 1e-3) -> dict:
    tok, model = load_cross_encoder(model_id, "cpu")
    qids, Q, P = parity_pairs(n_queries, per_query)
    ref = torch_score(model, tok, Q, P)
    got = ort_score(ort_session(onnx_path), tok, Q, P)
    diff = np.abs(ref - got)
    # ranking agreement inside each query group
    same_order = 0
    groups = 0
    for s in range(0, len(Q), per_query):
        groups += 1
        same_order += int(np.array_equal(np.argsort(-ref[s:s + per_query], kind="stable"),
                                         np.argsort(-got[s:s + per_query], kind="stable")))
    return {
        "n_pairs": len(Q), "n_queries": groups,
        "pairs": f"dev.small first {n_queries} judged queries x (1 judged passage + {per_query - 1} random pids, seed 0)",
        "reference": f"transformers {model_id} torch CPU fp32 logits",
        "candidate": str(onnx_path.relative_to(REPO_ROOT)) if onnx_path.is_absolute() else str(onnx_path),
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "score_range": [float(ref.min()), float(ref.max())],
        "groups_same_order": same_order,
        "tolerance": tol, "pass": bool(diff.max() <= tol),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--int8", action="store_true", help="also write model.int8.onnx")
    ap.add_argument("--tag", default=None, help="results/rerank/export_<tag>_parity.json")
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()
    tag = args.tag or args.out.name
    if not args.parity_only:
        t = time.time()
        p = export_cross_encoder(args.model, args.out)
        print(f"exported {p} ({p.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.1f}s")
        if args.int8:
            q = quantize_int8(p)
            print(f"quantised {q} ({q.stat().st_size / 1e6:.1f} MB)")
    res = {"fp32": parity(args.model, args.out / "model.onnx")}
    if (args.out / "model.int8.onnx").exists():
        # int8 is lossy by design: record its deviation, no pass/fail at fp32 tolerance
        r8 = parity(args.model, args.out / "model.int8.onnx", tol=float("inf"))
        r8.pop("pass")
        r8.pop("tolerance")
        res["int8"] = r8
    print(json.dumps(res, indent=2))
    out = REPO_ROOT / f"results/rerank/export_{tag}_parity.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    write_meta(out, model=args.model, onnx_dir=str(args.out))
    if not res["fp32"]["pass"]:
        raise SystemExit("PARITY FAILED")


if __name__ == "__main__":
    main()
