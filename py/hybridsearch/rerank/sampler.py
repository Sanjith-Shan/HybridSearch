"""Training groups for the cross-encoder: (query, 1 positive, n negatives).

Negative strategies (``strategy`` in the training config):

- ``random``     : uniform from the collection (the mined ``random`` pool).
- ``bm25-hard``  : from the filtered BM25 top-``pool_depth`` pool.
- ``dense-hard`` : from the filtered dense (exact BGE over the 1M subset) pool.
- ``mixed``      : ``round(n * mix_bm25)`` from BM25, the rest from dense.

Sampling is uniform *within* the pool (not "always the top n"), which is the common
recipe and keeps the very top ranks, where unjudged relevant passages concentrate, from
dominating. If a hard pool runs short, the group is topped up from the random pool, so
every group has exactly ``n_neg`` negatives. Counts of top-ups are exposed.

Determinism: the group for (global step s, slot i) depends only on (seed, s, i) and the
mined file, via ``np.random.default_rng([seed, s])``, so a resumed run replays exactly the
batches it would have seen.

Invariants (tested): a negative is never a judged pid of its query, never has the same
normalised text as a positive, and never repeats inside a group.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STRATEGIES = ("random", "bm25-hard", "dense-hard", "mixed")
_WS = re.compile(r"\s+")


def norm_text(t: str) -> str:
    return _WS.sub(" ", t.strip().lower())


@dataclass
class MinedQuery:
    qid: str
    pos: list[str]
    bm25: list[str]
    dense: list[str]
    random: list[str]


def read_mined(path: Path, limit: int | None = None) -> list[MinedQuery]:
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append(MinedQuery(r["qid"], r["pos"], r.get("bm25", []), r.get("dense", []),
                                  r.get("random", [])))
            if limit and len(out) >= limit:
                break
    return out


def needed_pids(mined: list[MinedQuery], strategy: str) -> set[str]:
    need: set[str] = set()
    for m in mined:
        need.update(m.pos)
        need.update(m.random)
        if strategy in ("bm25-hard", "mixed"):
            need.update(m.bm25)
        if strategy in ("dense-hard", "mixed"):
            need.update(m.dense)
    return need


class GroupSampler:
    def __init__(self, mined: list[MinedQuery], queries: dict[str, str], passages: dict[str, str],
                 strategy: str, n_neg: int, seed: int, mix_bm25: float = 0.5,
                 judged: dict[str, set[str]] | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.queries, self.passages = queries, passages
        self.strategy, self.n_neg, self.seed, self.mix_bm25 = strategy, n_neg, seed, mix_bm25
        # keep only queries whose query text and at least one positive text are available
        self.mined = [m for m in mined if m.qid in queries and any(p in passages for p in m.pos)]
        self.judged = judged or {}
        self.topups = 0
        self._orders: dict[int, np.ndarray] = {}
        self._pos_text = {m.qid: {norm_text(passages[p]) for p in m.pos if p in passages}
                          for m in self.mined}

    def __len__(self) -> int:
        return len(self.mined)

    def epoch_order(self, epoch: int) -> np.ndarray:
        if epoch not in self._orders:
            if len(self._orders) > 2:
                self._orders.clear()
            self._orders[epoch] = np.random.default_rng([self.seed, 10_000 + epoch]).permutation(len(self.mined))
        return self._orders[epoch]

    def _ok(self, m: MinedQuery, pid: str, taken: set[str]) -> bool:
        if pid in taken or pid in m.pos or pid in self.judged.get(m.qid, ()):
            return False
        t = self.passages.get(pid)
        return t is not None and norm_text(t) not in self._pos_text[m.qid]

    def _draw(self, rng, m: MinedQuery, pool: list[str], n: int, taken: set[str]) -> list[str]:
        out = []
        for j in rng.permutation(len(pool)):
            if len(out) == n:
                break
            if self._ok(m, pool[j], taken):
                out.append(pool[j])
                taken.add(pool[j])
        return out

    def group(self, m: MinedQuery, rng: np.random.Generator) -> tuple[str, str, list[str]]:
        cands = [p for p in m.pos if p in self.passages]
        pos = cands[int(rng.integers(len(cands)))]
        taken: set[str] = set()
        if self.strategy == "random":
            negs = self._draw(rng, m, m.random, self.n_neg, taken)
        elif self.strategy == "bm25-hard":
            negs = self._draw(rng, m, m.bm25, self.n_neg, taken)
        elif self.strategy == "dense-hard":
            negs = self._draw(rng, m, m.dense, self.n_neg, taken)
        else:
            nb = int(round(self.n_neg * self.mix_bm25))
            negs = self._draw(rng, m, m.bm25, nb, taken)
            negs += self._draw(rng, m, m.dense, self.n_neg - len(negs), taken)
        if len(negs) < self.n_neg:
            self.topups += self.n_neg - len(negs)
            negs += self._draw(rng, m, m.random, self.n_neg - len(negs), taken)
        return m.qid, pos, negs

    def batch(self, step: int, batch_size: int) -> list[tuple[str, str, list[str]]]:
        """Groups for global ``step``: queries ``step*B .. step*B+B-1`` of the epoch order."""
        n = len(self.mined)
        rng = np.random.default_rng([self.seed, step])
        out = []
        for i in range(batch_size):
            k = step * batch_size + i
            epoch, off = divmod(k, n)
            m = self.mined[int(self.epoch_order(epoch)[off])] if n else None
            out.append(self.group(m, rng))
        return out
