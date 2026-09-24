"""End-to-end tiny training on CPU: runs, validates, checkpoints, resumes exactly."""
from __future__ import annotations

import json

import torch
from safetensors.torch import load_file

from _rerank_tiny import synthetic_corpus, tiny_cross_encoder
from hybridsearch.rerank.train import TrainConfig, train


def _cfg(tmp_path, out, **kw):
    d = tmp_path / "data"
    base = dict(base_model=str(tmp_path / "base"), strategy="mixed", loss="listwise", n_neg=3,
                batch_groups=4, lr=1e-3, max_steps=6, max_len=64, device="cpu",
                mined=str(d / "mined.jsonl"), queries=str(d / "queries.tsv"), qrels=str(d / "qrels.tsv"),
                val_run=str(d / "val.trec"), val_queries=str(d / "queries.tsv"), val_qrels=str(d / "qrels.tsv"),
                val_n=10, val_depth=10, eval_every=3, save_every=3, log_every=1,
                out_dir=str(out), collection=str(d / "collection.tsv"), label="pytest tiny CPU")
    base.update(kw)
    return TrainConfig(**base)


def _weights(path):
    return load_file(str(path / "model.safetensors"))


def test_train_checkpoints_and_determinism(tmp_path):
    synthetic_corpus(tmp_path / "data")
    tiny_cross_encoder(tmp_path / "base")
    full = train(_cfg(tmp_path, tmp_path / "full"))
    assert full["steps"] == 6 and full["pairs_seen"] == 6 * 4 * 4
    assert (tmp_path / "full/best/model.safetensors").exists()
    assert (tmp_path / "full/best/best.json").exists()
    assert (tmp_path / "full/last/trainer_state.pt").exists()
    assert json.loads((tmp_path / "full/train_summary.json").read_text())["groups_seen"] == 24
    assert (tmp_path / "full/train_summary.json.meta.json").exists()
    # same seed + config -> bit-identical weights on CPU
    train(_cfg(tmp_path, tmp_path / "again"))
    a, b = _weights(tmp_path / "full/last"), _weights(tmp_path / "again/last")
    assert all(torch.equal(a[k], b[k]) for k in a)
    # a different seed -> different weights
    train(_cfg(tmp_path, tmp_path / "other", seed=7, eval_every=0, val_n=0))
    c = _weights(tmp_path / "other/last")
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_resume_matches_uninterrupted(tmp_path):
    synthetic_corpus(tmp_path / "data")
    tiny_cross_encoder(tmp_path / "base")
    ref = _cfg(tmp_path, tmp_path / "ref", eval_every=0, val_n=0)
    train(ref)
    # same config, but stop after step 3 by simulating a crash: run with a save at 3,
    # then truncate the run by resuming from last/ written at step 3
    cut = _cfg(tmp_path, tmp_path / "cut", eval_every=0, val_n=0)
    import hybridsearch.rerank.train as T

    orig_save = cut.save_every
    # first session: train 3 of 6 steps (raise after the step-3 save)
    class Stop(Exception):
        pass

    real_rename = T.Path.rename
    calls = {"n": 0}

    def rename(self, target):
        r = real_rename(self, target)
        if str(target).endswith("/last"):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Stop()
        return r

    T.Path.rename = rename
    try:
        try:
            train(cut)
        except Stop:
            pass
    finally:
        T.Path.rename = real_rename
    assert orig_save == 3
    st = torch.load(tmp_path / "cut/last/trainer_state.pt", weights_only=False)
    assert st["step"] == 3
    train(cut, resume=True)
    a, b = _weights(tmp_path / "ref/last"), _weights(tmp_path / "cut/last")
    assert all(torch.equal(a[k], b[k]) for k in a), "resumed run diverged from uninterrupted run"


def test_bce_and_all_strategies_run(tmp_path):
    synthetic_corpus(tmp_path / "data")
    tiny_cross_encoder(tmp_path / "base")
    for strat in ("random", "bm25-hard", "dense-hard"):
        s = train(_cfg(tmp_path, tmp_path / strat, strategy=strat, loss="bce", max_steps=2,
                       eval_every=0, val_n=0))
        assert s["steps"] == 2
