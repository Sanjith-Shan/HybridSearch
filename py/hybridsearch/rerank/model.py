"""Cross-encoder model pieces shared by training, scoring and export.

Pair format (BERT two-segment input), identical for training, torch scoring and ONNX:

    [CLS] query [SEP] passage [SEP]
    token_type_ids: 0 for [CLS] query [SEP], 1 for passage [SEP]

Truncation is ``only_second``: the query is never cut, the passage is cut to fit
``max_length``. Queries longer than ``max_query_len`` WordPiece tokens are cut first
(MS MARCO queries are short; this only guards against pathological input).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

TRAIN_MAX_LEN = 256
EVAL_MAX_LEN = 512
MAX_QUERY_LEN = 64


@dataclass
class PairBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor

    def to(self, device) -> "PairBatch":
        return PairBatch(self.input_ids.to(device), self.attention_mask.to(device),
                         self.token_type_ids.to(device))

    def as_dict(self) -> dict:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask,
                "token_type_ids": self.token_type_ids}


def clip_query(tok, query: str, max_query_len: int = MAX_QUERY_LEN) -> str:
    toks = tok.tokenize(query)
    if len(toks) <= max_query_len:
        return query
    return tok.convert_tokens_to_string(toks[:max_query_len])


def encode_pairs(tok, queries: list[str], passages: list[str], max_length: int,
                 return_tensors: str = "pt", pad_multiple: int | None = None):
    """Tokenise (query, passage) pairs. Returns the tokenizer's BatchEncoding with
    input_ids, attention_mask, token_type_ids, padded to the longest pair."""
    if len(queries) != len(passages):
        raise ValueError("queries and passages must have equal length")
    return tok(queries, passages, padding=True, truncation="only_second",
               max_length=max_length, return_tensors=return_tensors,
               return_token_type_ids=True, pad_to_multiple_of=pad_multiple)


def pair_batch(tok, queries, passages, max_length, pad_multiple: int | None = 32) -> PairBatch:
    """Padded to a multiple of 32 tokens: on MPS every distinct shape gets its own cached
    graph and buffers, so bucketing lengths bounds memory (docs/BUG_LOG.md 2026-09-24)."""
    enc = encode_pairs(tok, list(queries), list(passages), max_length, pad_multiple=pad_multiple)
    return PairBatch(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])


class CrossEncoderGraph(torch.nn.Module):
    """Sequence classifier with one logit, squeezed to ``score[batch]`` for export."""

    def __init__(self, clf: torch.nn.Module):
        super().__init__()
        self.clf = clf

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.clf(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits
        return logits[:, 0]


def load_cross_encoder(path_or_id: str, device: str = "cpu"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path_or_id)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_id, num_labels=1)
    model.to(device).eval()
    return tok, model


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
