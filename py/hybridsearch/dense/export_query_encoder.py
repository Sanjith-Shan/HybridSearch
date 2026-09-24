"""Export the BGE-base-en-v1.5 *query* encoder to ONNX for the C# broker.

The exported graph is the whole query path after tokenisation:

    input_ids, attention_mask (int64, [batch, seq])
      -> BERT -> last_hidden_state[:, 0]  (CLS pooling)
      -> L2 normalise
      -> "embedding" (float32, [batch, 768])

so the broker only tokenises (BERT WordPiece, lowercase, ``[CLS] prefix+query [SEP]``,
max 64 tokens) and runs one session. ``token_type_ids`` is fixed to zeros inside the
graph (single-segment input), so it is not an input.

Usage::

    python -m hybridsearch.dense.export_query_encoder            # export + parity (500 queries)
    python -m hybridsearch.dense.export_query_encoder --parity-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hybridsearch.data.io import DATA, REPO_ROOT, read_queries

MODEL_ID = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 64
OUT_DIR = DATA / "models" / "query-encoder"
OPSET = 17


class QueryEncoderGraph(torch.nn.Module):
    """BERT + CLS pooling + L2 normalisation, as one module."""

    def __init__(self, bert: torch.nn.Module):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
        ).last_hidden_state
        return F.normalize(out[:, 0], p=2.0, dim=-1)


def export_onnx(module: torch.nn.Module, path: Path, input_names: list[str], output_name: str,
                example: tuple[torch.Tensor, ...]) -> None:
    """TorchScript-based export with dynamic batch and sequence axes, single file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    dyn = {n: {0: "batch", 1: "seq"} for n in input_names}
    dyn[output_name] = {0: "batch"}
    tmp = path.with_suffix(".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(
            module, example, str(tmp),
            input_names=input_names, output_names=[output_name],
            dynamic_axes=dyn, opset_version=OPSET, do_constant_folding=True,
            dynamo=False,
        )
    import onnx

    m = onnx.load(str(tmp))  # pulls in any external data
    onnx.checker.check_model(m)
    onnx.save_model(m, str(path), save_as_external_data=False)
    tmp.unlink()
    for extra in path.parent.glob("*.tmp.onnx.data"):
        extra.unlink()


def load_bert(model_dir_or_id: str):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir_or_id)
    bert = AutoModel.from_pretrained(model_dir_or_id, add_pooling_layer=False)
    bert.eval()
    return tok, bert


def export_query_encoder(model_id: str = MODEL_ID, out_dir: Path = OUT_DIR,
                         max_len: int = MAX_LEN, prefix: str = QUERY_PREFIX) -> Path:
    tok, bert = load_bert(model_id)
    graph = QueryEncoderGraph(bert)
    enc = tok([prefix + "what is the capital of peru", prefix + "x"], padding=True,
              truncation=True, max_length=max_len, return_tensors="pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    export_onnx(graph, onnx_path, ["input_ids", "attention_mask"], "embedding",
                (enc["input_ids"], enc["attention_mask"]))
    save_tokenizer(tok, model_id, out_dir)
    cfg = {
        "model_id": model_id,
        "query_prefix": prefix,
        "max_length": max_len,
        "do_lower_case": True,
        "pooling": "cls",
        "normalize": "l2",
        "inputs": {"input_ids": "int64[batch,seq]", "attention_mask": "int64[batch,seq]"},
        "output": {"embedding": f"float32[batch,{bert.config.hidden_size}]"},
        "opset": OPSET,
    }
    (out_dir / "encoder_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return onnx_path


def save_tokenizer(tok, model_id: str, out_dir: Path) -> None:
    """tokenizer.json/config via save_pretrained, plus vocab.txt, which transformers 5
    no longer writes for fast tokenizers (the C# side needs it)."""
    tok.save_pretrained(str(out_dir))
    vocab = out_dir / "vocab.txt"
    if not vocab.exists():
        src = Path(model_id) / "vocab.txt"
        if not src.exists():
            from huggingface_hub import hf_hub_download

            src = Path(hf_hub_download(model_id, "vocab.txt"))
        vocab.write_bytes(src.read_bytes())


def ort_session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_encode(sess, tok, texts: list[str], max_len: int, batch: int = 64) -> np.ndarray:
    outs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len,
                  return_tensors="np")
        outs.append(sess.run(["embedding"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0])
    return np.concatenate(outs)


def parity_queries(n: int) -> tuple[list[str], str]:
    """First ``n`` queries (file order) from the first available query file."""
    candidates = [
        DATA / "raw/msmarco/queries.dev.small.tsv",
        DATA / "subset/train_tune/queries.tsv",
        DATA / "raw/msmarco/queries.train.tsv",
    ]
    for p in candidates:
        if p.exists():
            qs = list(read_queries(p).values())
            if len(qs) >= n:
                return qs[:n], str(p.relative_to(REPO_ROOT))
    raise FileNotFoundError(f"no query file with >= {n} queries among {candidates}")


def parity(model_id: str = MODEL_ID, out_dir: Path = OUT_DIR, n: int = 500,
           max_len: int = MAX_LEN, prefix: str = QUERY_PREFIX) -> dict:
    """ONNX Runtime CPU vs sentence-transformers (torch CPU), cosine per query."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    queries, source = parity_queries(n)
    st = SentenceTransformer(model_id, device="cpu")
    st.max_seq_length = max_len
    ref = st.encode([prefix + q for q in queries], batch_size=64, normalize_embeddings=True,
                    convert_to_numpy=True)
    tok = AutoTokenizer.from_pretrained(str(out_dir))
    sess = ort_session(out_dir / "model.onnx")
    got = ort_encode(sess, tok, [prefix + q for q in queries], max_len)
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    return {
        "n_queries": len(queries),
        "query_source": source,
        "reference": f"sentence-transformers {model_id} on torch CPU, max_seq_length={max_len}",
        "candidate": "onnxruntime CPUExecutionProvider, data/models/query-encoder/model.onnx",
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "max_abs_diff": float(np.abs(ref - got).max()),
        "norm_min": float(np.linalg.norm(got, axis=1).min()),
        "norm_max": float(np.linalg.norm(got, axis=1).max()),
        "threshold": 0.9999,
        "pass": bool(cos.min() >= 0.9999),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()
    if not args.parity_only:
        t = time.time()
        p = export_query_encoder(args.model, args.out)
        print(f"exported {p} ({p.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.1f}s")
    res = parity(args.model, args.out, args.n)
    print(json.dumps(res, indent=2))
    from hybridsearch.rerank.meta import write_meta

    out = REPO_ROOT / "results/rerank/query_encoder_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    write_meta(out, model=args.model, onnx=str(args.out / "model.onnx"))
    write_sanity(args.out)
    if not res["pass"]:
        raise SystemExit("PARITY FAILED")


def write_sanity(out_dir: Path = OUT_DIR, text: str = "what is the capital of peru") -> dict:
    """Record ids + first 8 output components for one query so C# can assert them."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(out_dir))
    sess = ort_session(out_dir / "model.onnx")
    ids = tok(QUERY_PREFIX + text)["input_ids"]
    vec = ort_encode(sess, tok, [QUERY_PREFIX + text], MAX_LEN)[0]
    sanity = {"query": text, "input_ids": ids, "embedding_first8": [round(float(x), 6) for x in vec[:8]]}
    cfg_path = out_dir / "encoder_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["sanity"] = sanity
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return sanity


if __name__ == "__main__":
    main()
