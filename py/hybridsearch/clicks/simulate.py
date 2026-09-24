"""Simulated experiment traffic, written as ``/api/events`` logs.

Sessions get random IDs, are assigned with the same hash rule as the broker
(``events.assign``), issue 1 + Geometric queries each (queries uniform over the DL
topics), and see either one arm's top-10 (A/B, A/A) or a team-draft interleaving of both
(interleave). Clicks come from a click model with relevance from TREC DL graded qrels.
Every event carries ``"simulated": <model name>``.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import xxhash

from hybridsearch.clicks.events import SIM_EPOCH, assign, impression_events
from hybridsearch.clicks.interleave import team_draft_random
from hybridsearch.clicks.models import ClickModel, sample_dwell_ms
from hybridsearch.clicks.rankers import RankerTable


def simulate_experiment(table: RankerTable, queries: dict[str, str], exp: dict, model: ClickModel,
                        n_sessions: int, rng: np.random.Generator, mean_queries: float = 2.0) -> list[dict]:
    """``exp``: {"id", "kind": ab|aa|interleave, "allocation", "salt",
    "control": ranker name, "treatment": ranker name}."""
    k = table.k
    ev: list[dict] = []
    t = SIM_EPOCH
    req = 0
    for s in range(n_sessions):
        sid = f"{rng.integers(0, 2**63):016x}"
        arm = assign(exp["salt"], sid, exp["allocation"])
        nq = int(rng.geometric(1.0 / mean_queries))  # >= 1, mean ``mean_queries``
        for _ in range(nq):
            qi = int(rng.integers(0, len(table.qids)))
            qid = table.qids[qi]
            req += 1
            rid = f"sim-{exp['id']}-{req:08d}"
            teams = None
            if arm is None:
                docs, variant, eid = table.lists[exp["control"]][qi], None, None
            elif exp["kind"] == "interleave":
                docs, teams = team_draft_random(table.lists[exp["control"]][qi],
                                                table.lists[exp["treatment"]][qi], k, rng)
                variant, eid = "interleaved", exp["id"]
            else:
                ranker = exp["control"] if (arm == "control" or exp["kind"] == "aa") else exp["treatment"]
                docs, variant, eid = table.lists[ranker][qi], arm, exp["id"]
            g = np.array([[table.qrels[qid].get(d, 0) for d in docs] + [-1] * (k - len(docs))])
            res = model.simulate(g, rng)
            dwell = sample_dwell_ms(res.satisfied[0], rng)
            # docIds must be integers in the schema; synthetic fillers get a stable hash
            ids = [int(d) if str(d).isdigit() else 10**12 + xxhash.xxh64_intdigest(str(d).encode()) % 10**9 for d in docs]
            ev.extend(impression_events(
                session_id=sid, request_id=rid, query=queries.get(qid, qid), doc_ids=ids,
                clicks=res.clicks[0][: len(docs)], dwell_ms=dwell, experiment_id=eid, variant=variant,
                teams=teams, t0=t, simulated=model.name))
            t += timedelta(seconds=30)
    return ev
