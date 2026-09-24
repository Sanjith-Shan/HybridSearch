"""Chaos experiments that measure retrieval quality under failure, not just uptime.

A run replays dev queries against the broker at a fixed arrival rate through
three phases (baseline, fault, recovery). A fault is injected and removed by
shell hooks, so the same runner drives Docker (`docker kill`), bare processes
(`kill -STOP`), or Toxiproxy (latency). For each phase it reports latency
percentiles from every sample, the fraction of degraded / partial responses,
and MRR@10 over the queries answered in that phase, so "one shard dead"
becomes a number in both milliseconds and ranking quality.

Open-loop: requests are sent on schedule whether or not earlier ones have
returned, so a slow shard cannot hide the requests queued behind it.

    python -m hybridsearch.serving.chaos --name kill-shard-1 \
        --queries data/raw/msmarco/queries.dev.small.tsv \
        --qrels data/raw/msmarco/qrels.dev.small.tsv \
        --rate 50 --phase-seconds 30 \
        --fault-start "docker kill hs-shard1" --fault-stop "docker start hs-shard1"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np


@dataclass
class Sample:
    phase: str
    qid: str
    latency_ms: float
    status: int
    doc_ids: list[int] = field(default_factory=list)
    degradation_level: int = 0
    partial_shards: int = 0
    failed_shards: int = 0
    hedged_shards: int = 0


def load_queries(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        if line:
            qid, text = line.split("\t", 1)
            out.append((qid, text))
    return out


def load_qrels(path: Path) -> dict[str, dict[int, int]]:
    qrels: dict[str, dict[int, int]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4:
            qid, _, did, rel = parts
            qrels[qid][int(did)] = int(rel)
    return qrels


def mrr_at_10(samples: list[Sample], qrels: dict[str, dict[int, int]]) -> float | None:
    """MS MARCO dev convention: reciprocal rank of the first relevant doc in the top 10,
    averaged over judged queries that were answered. Failed requests score 0, since a
    user who got an error got nothing relevant."""
    scores = []
    for s in samples:
        rels = qrels.get(s.qid)
        if not rels:
            continue
        rr = 0.0
        if s.status == 200:
            for rank, did in enumerate(s.doc_ids[:10], start=1):
                if rels.get(did, 0) > 0:
                    rr = 1.0 / rank
                    break
        scores.append(rr)
    return float(np.mean(scores)) if scores else None


def summarize(samples: list[Sample], qrels) -> dict:
    lat = np.array([s.latency_ms for s in samples]) if samples else np.array([0.0])
    ok = [s for s in samples if s.status == 200]
    n = max(len(samples), 1)
    return {
        "requests": len(samples),
        "error_rate": 1 - len(ok) / n,
        "p50_ms": float(np.percentile(lat, 50)),
        "p99_ms": float(np.percentile(lat, 99)),
        "p999_ms": float(np.percentile(lat, 99.9)),
        "degraded_rate": sum(s.degradation_level > 0 for s in ok) / n,
        "partial_rate": sum(s.partial_shards > 0 for s in ok) / n,
        "failed_shard_rate": sum(s.failed_shards > 0 for s in ok) / n,
        "hedged_rate": sum(s.hedged_shards > 0 for s in ok) / n,
        "mrr_at_10": mrr_at_10(samples, qrels),
    }


async def one_request(client: httpx.AsyncClient, base: str, params: dict, qid: str, text: str,
                      phase: str, timeout_s: float) -> Sample:
    t0 = time.perf_counter()
    try:
        r = await client.get(f"{base}/api/search", params={**params, "q": text}, timeout=timeout_s)
        latency = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return Sample(phase, qid, latency, r.status_code)
        body = r.json()
        deg = body.get("degradation") or {}
        return Sample(
            phase, qid, latency, 200,
            doc_ids=[int(x["docId"]) for x in body.get("results", [])],
            degradation_level=int(deg.get("level", 0)),
            partial_shards=len(deg.get("partialShards", [])),
            failed_shards=len(deg.get("failedShards", [])),
            hedged_shards=len(deg.get("hedgedShards", [])),
        )
    except httpx.HTTPError:
        return Sample(phase, qid, (time.perf_counter() - t0) * 1000, 599)


async def run_phase(client, base, params, queries, phase, rate, seconds, timeout_s, rng) -> list[Sample]:
    tasks = []
    interval = 1.0 / rate
    start = time.perf_counter()
    n = int(rate * seconds)
    for i in range(n):
        # Sleep until this request's scheduled send time, not "interval after the last reply".
        delay = start + i * interval - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        qid, text = queries[rng.randrange(len(queries))]
        tasks.append(asyncio.create_task(one_request(client, base, params, qid, text, phase, timeout_s)))
    return list(await asyncio.gather(*tasks))


def hook(cmd: str | None):
    if cmd:
        print(f"[chaos] $ {cmd}", flush=True)
        subprocess.run(cmd, shell=True, check=False)


async def main_async(args) -> dict:
    queries = load_queries(Path(args.queries))
    qrels = load_qrels(Path(args.qrels))
    judged = [q for q in queries if q[0] in qrels]
    rng = random.Random(args.seed)
    params = {"mode": args.mode, "rerank": str(args.rerank).lower(), "deadlineMs": args.deadline_ms, "k": 10}
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
    samples: list[Sample] = []
    async with httpx.AsyncClient(limits=limits) as client:
        for phase in ("baseline", "fault", "recovery"):
            if phase == "fault":
                hook(args.fault_start)
                await asyncio.sleep(args.settle_seconds)
            if phase == "recovery":
                hook(args.fault_stop)
                await asyncio.sleep(args.settle_seconds)
            print(f"[chaos] phase {phase}: {args.rate}/s for {args.phase_seconds}s", flush=True)
            samples += await run_phase(client, args.base_url, params, judged, phase,
                                       args.rate, args.phase_seconds, args.timeout_s, rng)
    by_phase = {p: summarize([s for s in samples if s.phase == p], qrels)
                for p in ("baseline", "fault", "recovery")}
    base_mrr = by_phase["baseline"]["mrr_at_10"]
    fault_mrr = by_phase["fault"]["mrr_at_10"]
    return {
        "experiment": args.name,
        "fault_start": args.fault_start,
        "fault_stop": args.fault_stop,
        "rate_per_s": args.rate,
        "phase_seconds": args.phase_seconds,
        "params": params,
        "phases": by_phase,
        "mrr_retained_under_fault": (fault_mrr / base_mrr) if base_mrr and fault_mrr is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--rate", type=float, default=50)
    ap.add_argument("--phase-seconds", type=float, default=30)
    ap.add_argument("--settle-seconds", type=float, default=2)
    ap.add_argument("--timeout-s", type=float, default=5)
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--rerank", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--deadline-ms", type=int, default=300)
    ap.add_argument("--fault-start")
    ap.add_argument("--fault-stop")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = asyncio.run(main_async(args))
    result["hardware"] = {"machine": platform.machine(), "system": platform.platform(),
                          "note": "macOS runs are dev-signal-only" if platform.system() == "Darwin" else ""}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
