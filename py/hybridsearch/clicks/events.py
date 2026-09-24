"""Event logs in the exact ``POST /api/events`` schema (docs/ARCHITECTURE.md).

The simulator writes the same JSON Lines the broker appends, so
``analyze_events.py`` runs unchanged on simulated logs and on real ones.

Event = {type, sessionId, requestId, query, docId?, rank?, dwellMs?, experimentId?,
         variant?, team?, clientTs}

Simulated logs mark themselves with ``"simulated": "<click model>"`` on every event
(an extra field; the broker's own events never carry it), so a simulated log can never
be mistaken for traffic.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xxhash

EVENT_TYPES = ("impression", "click", "dwell", "query", "abandon")


# ------------------------------------------------------------------ assignment
def bucket(salt: str, session_id: str, buckets: int = 10_000) -> int:
    """``xxhash64(salt + sessionId) mod 10000`` (ARCHITECTURE.md, Experiments)."""
    return xxhash.xxh64_intdigest((salt + session_id).encode("utf-8")) % buckets


def assign(salt: str, session_id: str, allocation: float) -> str | None:
    """None (not enrolled) | "control" | "treatment". The arm uses a second hash with a
    different key (``salt + "|arm|" + sessionId``) so enrolment and arm are independent."""
    if bucket(salt, session_id) >= allocation * 10_000:
        return None
    arm = xxhash.xxh64_intdigest((salt + "|arm|" + session_id).encode("utf-8")) % 2
    return "treatment" if arm else "control"


# ------------------------------------------------------------------ writing
def impression_events(*, session_id, request_id, query, doc_ids, clicks, dwell_ms,
                      experiment_id=None, variant=None, teams=None, t0: datetime,
                      simulated: str | None = None) -> list[dict]:
    """Events for one served result page. ``doc_ids`` in rank order; ``clicks`` bool per
    rank; ``teams`` "A"/"B" per rank for interleaved pages."""
    def base(kind, ts):
        e = {"type": kind, "sessionId": session_id, "requestId": request_id, "query": query,
             "clientTs": ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")}
        if experiment_id is not None:
            e["experimentId"] = experiment_id
            e["variant"] = variant
        if simulated:
            e["simulated"] = simulated
        return e

    ev = [base("query", t0)]
    t = t0 + timedelta(milliseconds=300)
    for r, d in enumerate(doc_ids):
        e = base("impression", t)
        e["docId"] = int(d)
        e["rank"] = r + 1
        if teams is not None:
            e["team"] = teams[r]
        ev.append(e)
    t += timedelta(seconds=2)
    any_click = False
    for r, d in enumerate(doc_ids):
        if not clicks[r]:
            continue
        any_click = True
        for kind in ("click", "dwell"):
            e = base(kind, t)
            e["docId"] = int(d)
            e["rank"] = r + 1
            if teams is not None:
                e["team"] = teams[r]
            if kind == "dwell":
                e["dwellMs"] = int(dwell_ms[r])
                t += timedelta(milliseconds=int(dwell_ms[r]))
            ev.append(e)
        t += timedelta(seconds=1)
    if not any_click:
        ev.append(base("abandon", t))
    return ev


def write_jsonl(path: str | os.PathLike, events) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------ reading
def read_events(paths) -> list[dict]:
    """Reads ``*.jsonl`` (and ``*.parquet`` if the broker switches format)."""
    if isinstance(paths, (str, os.PathLike)):
        p = str(paths)
        paths = sorted(glob.glob(os.path.join(p, "*.jsonl")) + glob.glob(os.path.join(p, "*.parquet"))) \
            if os.path.isdir(p) else [p]
    out: list[dict] = []
    for p in paths:
        if str(p).endswith(".parquet"):
            import pandas as pd
            out.extend(pd.read_parquet(p).to_dict("records"))
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # tolerate batch envelopes {"events": [...]}
                if isinstance(obj, dict) and "events" in obj and "type" not in obj:
                    out.extend(obj["events"])
                else:
                    out.append(obj)
    return out


@dataclass
class Page:
    """One served result page reconstructed from its events."""

    request_id: str
    session_id: str
    query: str
    experiment_id: str | None = None
    variant: str | None = None
    docs: dict[int, int] = field(default_factory=dict)          # rank -> docId
    teams: dict[int, str] = field(default_factory=dict)         # rank -> "A"/"B"
    clicked_ranks: set[int] = field(default_factory=set)
    dwell: dict[int, int] = field(default_factory=dict)
    abandoned: bool = False
    simulated: str | None = None

    @property
    def interleaved(self) -> bool:
        return bool(self.teams)


def pages_from_events(events: list[dict]) -> tuple[list[Page], dict]:
    """Group events by requestId. Returns (pages, data-quality counters)."""
    pages: dict[str, Page] = {}
    dq = defaultdict(int)
    for e in events:
        t = e.get("type")
        if t not in EVENT_TYPES:
            dq["unknown_type"] += 1
            continue
        rid = e.get("requestId")
        if not rid:
            dq["missing_request_id"] += 1
            continue
        pg = pages.get(rid)
        if pg is None:
            pg = pages[rid] = Page(rid, e.get("sessionId", ""), e.get("query", ""))
        if e.get("experimentId") and pg.experiment_id is None:
            pg.experiment_id, pg.variant = e["experimentId"], e.get("variant")
        if e.get("simulated"):
            pg.simulated = e["simulated"]
        rank = e.get("rank")
        if t in ("impression", "click", "dwell") and rank is None:
            dq[f"{t}_missing_rank"] += 1
            continue
        if t == "impression":
            pg.docs[int(rank)] = int(e["docId"])
            if e.get("team"):
                pg.teams[int(rank)] = e["team"]
        elif t == "click":
            pg.clicked_ranks.add(int(rank))
            if e.get("team"):
                pg.teams.setdefault(int(rank), e["team"])
        elif t == "dwell":
            pg.dwell[int(rank)] = int(e.get("dwellMs") or 0)
        elif t == "abandon":
            pg.abandoned = True
    for pg in pages.values():
        if any(r not in pg.docs for r in pg.clicked_ranks):
            dq["click_without_impression"] += 1
    return list(pages.values()), dict(dq)


def click_matrix(pages: list[Page], k: int = 10) -> np.ndarray:
    m = np.zeros((len(pages), k), bool)
    for i, pg in enumerate(pages):
        for r in pg.clicked_ranks:
            if 1 <= r <= k:
                m[i, r - 1] = True
    return m


# Simulated logs use a fixed epoch so they are byte-for-byte reproducible under a seed.
SIM_EPOCH = datetime(2026, 9, 23, tzinfo=timezone.utc)
