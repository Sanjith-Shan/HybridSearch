"""ONNX export parity on tiny random-init models (fast, offline)."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from _rerank_tiny import WORDS, tiny_bert, tiny_cross_encoder
from hybridsearch.dense.export_query_encoder import (QueryEncoderGraph, export_onnx, ort_encode,
                                                     save_tokenizer)
from hybridsearch.rerank.export import (INPUTS, OUTPUT, export_cross_encoder, ort_score,
                                        quantize_int8, torch_score)
from hybridsearch.data.io import REPO_ROOT
from hybridsearch.rerank.model import load_cross_encoder


def _texts(n, seed=0, lo=1, hi=40):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, rng.integers(lo, hi))) for _ in range(n)]


def test_query_encoder_export_parity(tmp_path):
    from transformers import AutoModel, AutoTokenizer

    src = tiny_bert(tmp_path / "src")
    tok = AutoTokenizer.from_pretrained(str(src))
    bert = AutoModel.from_pretrained(str(src), add_pooling_layer=False).eval()
    graph = QueryEncoderGraph(bert)
    ex = tok(["a b", "c"], padding=True, return_tensors="pt")
    out = tmp_path / "out"
    export_onnx(graph, out / "model.onnx", ["input_ids", "attention_mask"], "embedding",
                (ex["input_ids"], ex["attention_mask"]))
    save_tokenizer(tok, str(src), out)
    assert (out / "vocab.txt").exists()
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input_ids", "attention_mask"]
    assert all(i.type == "tensor(int64)" for i in sess.get_inputs())
    texts = _texts(50)
    for batch in (1, 7, 50):  # dynamic batch and sequence length
        got = ort_encode(sess, tok, texts, 64, batch)
        with torch.no_grad():
            enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            ref = graph(enc["input_ids"], enc["attention_mask"]).numpy()
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
        assert ((got * ref).sum(1)).min() >= 0.9999


def test_cross_encoder_export_parity_and_int8(tmp_path):
    src = tiny_cross_encoder(tmp_path / "src")
    out = tmp_path / "out"
    export_cross_encoder(str(src), out, max_length=64)
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == INPUTS
    assert [o.name for o in sess.get_outputs()] == [OUTPUT]
    tok, model = load_cross_encoder(str(src))
    Q, P = _texts(40, 1, 1, 8), _texts(40, 2, 5, 60)
    ref = torch_score(model, tok, Q, P, 64, batch=16)
    for batch in (1, 16, 40):
        got = ort_score(sess, tok, Q, P, 64, batch)
        assert got.shape == (40,)
        np.testing.assert_allclose(got, ref, atol=1e-4)
    q8 = quantize_int8(out / "model.onnx")
    assert q8.exists() and q8.stat().st_size < (out / "model.onnx").stat().st_size
    s8 = ort.InferenceSession(str(q8), providers=["CPUExecutionProvider"])
    got8 = ort_score(s8, tok, Q, P, 64, 16)
    assert got8.shape == (40,) and np.isfinite(got8).all()
    # structural check: weights really are int8 MatMuls (quality loss is measured on the real
    # model through the eval harness; a random-init tiny model has too little score variance)
    import onnx

    ops = {n.op_type for n in onnx.load(str(q8)).graph.node}
    assert {"DynamicQuantizeLinear", "MatMulInteger"} <= ops


def test_export_is_single_file(tmp_path):
    src = tiny_cross_encoder(tmp_path / "src")
    out = tmp_path / "out"
    export_cross_encoder(str(src), out, max_length=64)
    assert sorted(p.name for p in out.glob("*.onnx*")) == ["model.onnx"]


REAL_QE = REPO_ROOT / "data/models/query-encoder/model.onnx"


@pytest.mark.skipif(not REAL_QE.exists(), reason="real export not present")
def test_real_query_encoder_interface():
    sess = ort.InferenceSession(str(REAL_QE), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input_ids", "attention_mask"]
    assert [o.name for o in sess.get_outputs()] == ["embedding"]
    assert sess.get_outputs()[0].shape[1] == 768
