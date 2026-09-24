"""A/B + interleaving report from an event log in the ``/api/events`` schema.

    .venv/bin/python -m hybridsearch.clicks.analyze_events \
        --events data/events --out results/m9/events_report

Runs unchanged on the broker's real log (``data/events/*.jsonl``) and on simulated logs
(which carry ``"simulated": "<click model>"`` on every event; the report says so).

Per experiment:
* ``ab`` / ``aa``: per-variant CTR, clicks@1, abandonment, MRR of first click, the
  treatment-minus-control delta with a bootstrap 95% CI and Welch p-value, and a
  sample-ratio-mismatch check on sessions per arm (expected 50/50).
* ``interleave``: wins / losses / ties by team-draft credit, Delta_AB with bootstrap CI and
  sign-test p-value, plus a team-balance check (the team owning rank 1 should be A half
  the time; a skew means the coin or the logging is broken).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hybridsearch.clicks import ab as abstats
from hybridsearch.clicks.events import Page, click_matrix, pages_from_events, read_events
from hybridsearch.clicks.interleave import team_draft_credit
from hybridsearch.data.io import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "broker/HybridSearch.Broker/experiments.json"


def load_config(path) -> dict[str, dict]:
    try:
        with open(path) as f:
            return {e["id"]: e for e in json.load(f).get("experiments", [])}
    except (OSError, ValueError, KeyError):
        return {}


def _kind(exp_id: str, pages: list[Page], config: dict) -> str:
    if exp_id in config and config[exp_id].get("kind"):
        return config[exp_id]["kind"]
    return "interleave" if any(p.interleaved for p in pages) else "ab"


def analyse(events: list[dict], config: dict | None = None, seed: int = 0, k: int = 10,
            n_boot: int = 2000) -> dict:
    config = config or {}
    rng = np.random.default_rng(seed)
    pages, dq = pages_from_events(events)
    sims = Counter(p.simulated for p in pages if p.simulated)
    by_exp: dict[str | None, list[Page]] = defaultdict(list)
    for p in pages:
        by_exp[p.experiment_id].append(p)

    report = {
        "n_events": len(events),
        "n_pages": len(pages),
        "n_sessions": len({p.session_id for p in pages}),
        "data_quality": dq,
        "simulated": dict(sims),
        "label": ("SIMULATED users: " + ", ".join(f"{k_} ({v} pages)" for k_, v in sims.items())
                  + "; relevance from TREC DL graded qrels") if sims else "logged traffic",
        "experiments": {},
    }
    if None in by_exp:
        m = abstats.impression_metrics(click_matrix(by_exp[None], k))
        report["not_enrolled"] = {"pages": len(by_exp[None]), **{kk: float(v.mean()) for kk, v in m.items()}}

    for exp_id, pgs in sorted(((e, p) for e, p in by_exp.items() if e is not None)):
        kind = _kind(exp_id, pgs, config)
        entry: dict = {"kind": kind, "pages": len(pgs)}
        if kind == "interleave":
            outcomes, rank1_a, rank1_n = [], 0, 0
            for p in pgs:
                ranks = sorted(p.docs)
                teams = [p.teams.get(r, "") for r in ranks]
                pos = {r: i for i, r in enumerate(ranks)}
                ca, cb, w = team_draft_credit(teams, [pos[r] for r in p.clicked_ranks if r in pos])
                outcomes.append(1 if w == "A" else -1 if w == "B" else 0)
                if 1 in p.teams:
                    rank1_n += 1
                    rank1_a += p.teams[1] == "A"
            entry["interleaving"] = abstats.interleave_report(np.array(outcomes), rng, n_boot)
            from scipy.stats import binomtest
            entry["team_balance_rank1"] = {
                "a_share": rank1_a / rank1_n if rank1_n else None, "n": rank1_n,
                "p": float(binomtest(rank1_a, rank1_n, 0.5).pvalue) if rank1_n else None,
            }
        else:
            arms: dict[str, list[Page]] = defaultdict(list)
            for p in pgs:
                arms[p.variant or "?"].append(p)
            names = sorted(arms)
            ctrl = "control" if "control" in arms else names[0]
            trt = "treatment" if "treatment" in arms else (names[1] if len(names) > 1 else None)
            sessions = {a: len({p.session_id for p in arms[a]}) for a in names}
            entry["variants"] = {a: {"pages": len(arms[a]), "sessions": sessions[a]} for a in names}
            entry["srm"] = abstats.srm_check([sessions[a] for a in names], [1.0] * len(names)) \
                if len(names) > 1 else None
            if trt is not None:
                entry["control"], entry["treatment"] = ctrl, trt
                entry["ab"] = abstats.ab_report(click_matrix(arms[ctrl], k), click_matrix(arms[trt], k), rng, n_boot)
        report["experiments"][exp_id] = entry
    return report


def to_markdown(rep: dict) -> str:
    L = [f"# Event-log report", "", f"**Source label:** {rep['label']}", "",
         f"events {rep['n_events']}, result pages {rep['n_pages']}, sessions {rep['n_sessions']}",
         f"data-quality counters: {rep['data_quality'] or 'none'}", ""]
    for eid, e in rep["experiments"].items():
        L.append(f"## {eid} ({e['kind']}, {e['pages']} pages)")
        if "interleaving" in e:
            r = e["interleaving"]
            L.append(f"wins A {r['wins_a']}, wins B {r['wins_b']}, ties {r['ties']}; "
                     f"Delta_AB {r['delta_ab']:+.4f} (95% CI {r['delta_ci95'][0]:+.4f}..{r['delta_ci95'][1]:+.4f}); "
                     f"sign test p = {r['p_sign']:.3g}")
            tb = e["team_balance_rank1"]
            if tb["n"]:
                L.append(f"team balance at rank 1: A share {tb['a_share']:.3f} (n={tb['n']}, p={tb['p']:.3g})")
        if e.get("srm"):
            s = e["srm"]
            L.append(f"SRM check: sessions {s['counts']}, chi2 {s['chi2']:.2f}, p = {s['p']:.3g} "
                     f"-> {'SAMPLE RATIO MISMATCH, do not trust the metrics' if s['srm'] else 'ok'}")
        if "ab" in e:
            L.append("")
            L.append(f"| metric | {e['control']} | {e['treatment']} | delta | 95% CI | p (Welch) |")
            L.append("|---|---|---|---|---|---|")
            for m, v in e["ab"]["metrics"].items():
                L.append(f"| {m} | {v['a']:.4f} | {v['b']:.4f} | {v['delta']:+.4f} | "
                         f"{v['delta_ci95'][0]:+.4f}..{v['delta_ci95'][1]:+.4f} | {v['p_welch']:.3g} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default=str(REPO_ROOT / "data/events"))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(REPO_ROOT / "results/m9/events_report"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    p = Path(a.events)
    if not p.exists():
        print(f"no event log at {p}; nothing to analyse", file=sys.stderr)
        return 1
    events = read_events(p)
    rep = analyse(events, load_config(a.config), seed=a.seed)
    rep["source"] = str(p)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(rep, indent=2) + "\n")
    Path(a.out + ".md").write_text(to_markdown(rep) + "\n")
    print(to_markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
