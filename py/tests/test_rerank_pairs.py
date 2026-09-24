"""Pair formatting, truncation, and group losses."""
from __future__ import annotations

import math

import pytest
import torch

from _rerank_tiny import tiny_tokenizer
from hybridsearch.rerank.model import clip_query, encode_pairs
from hybridsearch.rerank.train import group_loss


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    return tiny_tokenizer(tmp_path_factory.mktemp("tok"))


def test_pair_layout_and_segments(tok):
    enc = encode_pairs(tok, ["what is the capital"], ["lima is the capital of peru ."], 64)
    ids = enc["input_ids"][0].tolist()
    tt = enc["token_type_ids"][0].tolist()
    cls, sep = tok.cls_token_id, tok.sep_token_id
    assert ids[0] == cls and ids.count(sep) == 2 and ids[-1] == sep
    first_sep = ids.index(sep)
    assert tok.decode(ids[1:first_sep]) == "what is the capital"
    assert all(t == 0 for t in tt[: first_sep + 1])
    assert all(t == 1 for t in tt[first_sep + 1:])


def test_truncates_only_passage(tok):
    q = "what is the capital of peru"
    p = " ".join(["water"] * 500)
    enc = encode_pairs(tok, [q, q], [p, "lima"], 32)
    ids = enc["input_ids"]
    assert ids.shape[1] == 32
    row = ids[0].tolist()
    assert tok.decode(row[1:row.index(tok.sep_token_id)]) == q  # query intact
    # padding: the short pair is padded with 0 and masked out
    assert enc["attention_mask"][1].sum() < 32
    assert (ids[1][enc["attention_mask"][1] == 0] == tok.pad_token_id).all()


def test_clip_query(tok):
    long = " ".join(["city"] * 100)
    assert len(tok.tokenize(clip_query(tok, long, 10))) == 10
    assert clip_query(tok, "short query", 10) == "short query"


def test_listwise_loss_matches_cross_entropy():
    logits = torch.tensor([2.0, 0.5, -1.0, 0.0, 1.0, 3.0])  # 2 groups of 3, pos first
    got = group_loss(logits, 3, "listwise")
    lg = logits.view(2, 3)
    want = -(torch.log_softmax(lg, 1)[:, 0]).mean()
    assert torch.allclose(got, want)


def test_bce_loss_labels_positive_first():
    logits = torch.tensor([10.0, -10.0, -10.0, 10.0, -10.0, -10.0])
    assert group_loss(logits, 3, "bce") < 1e-3
    flipped = torch.tensor([-10.0, 10.0, 10.0, -10.0, 10.0, 10.0])
    assert group_loss(flipped, 3, "bce") > 5


def test_losses_prefer_correct_order():
    good = torch.tensor([3.0, 0.0, 0.0, 0.0])
    bad = torch.tensor([0.0, 3.0, 0.0, 0.0])
    for kind in ("bce", "listwise"):
        assert group_loss(good, 4, kind) < group_loss(bad, 4, kind)
    assert math.isclose(float(group_loss(torch.zeros(4), 4, "listwise")), math.log(4), rel_tol=1e-6)


def test_group_loss_rejects_ragged():
    with pytest.raises(RuntimeError):
        group_loss(torch.zeros(5), 3, "listwise")
