#!/usr/bin/env python
"""M9 headline: sensitivity of interleaving vs A/B under SIMULATED users.

    .venv/bin/python scripts/m9_sensitivity.py                 # full study
    .venv/bin/python scripts/m9_sensitivity.py --quick         # smoke run (small pools)

Rankers: every real ranker with DL19+DL20 runs in data/runs/reference (plus RRF of BM25
and dense when both exist) and synthetic noisy-oracle rankers with controlled gaps.
Click models: PBM, Cascade, DBN x {perfect, navigational, informational}.
Relevance: TREC DL 2019+2020 graded qrels. Users: simulated. See
py/hybridsearch/clicks/sensitivity.py for the method.

Writes results/m9/sensitivity.{json,md,png}, sensitivity_n80.png, aa.{json,md}, each
with a .meta.json sidecar.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # shared laptop: run via scripts/bg.sh, 1 thread

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import xxhash
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from hybridsearch.clicks import rankers as R  # noqa: E402
from hybridsearch.clicks import sensitivity as S  # noqa: E402
from hybridsearch.clicks.models import ALL_MODELS, make_model  # noqa: E402
from hybridsearch.rerank.meta import write_meta  # noqa: E402

OUT = ROOT / "results/m9"
RAW = ROOT / "data/m9"
CELLS = ROOT / "data/cache/m9/sensitivity_cells"   # per-cell cache (keyed by every input)
LABEL = "simulated users (click model {m}, relevance from TREC DL 2019+2020 graded qrels)"
METHODS = ["tdi", "bi", "pi"] + [f"ab:{m}" for m in S.AB_METRICS]
METHOD_NAMES = {"tdi": "team-draft interleaving", "bi": "balanced interleaving", "pi": "probabilistic interleaving",
                "ab:ctr": "A/B clicks per impression", "ab:clicks_at_1": "A/B clicks@1",
                "ab:abandonment": "A/B abandonment", "ab:rr_first": "A/B MRR of first click"}
COLORS = {"tdi": "#2a78d6", "bi": "#eb6834", "pi": "#1baf7a", "ab:ctr": "#eda100",
          "ab:clicks_at_1": "#e87ba4", "ab:abandonment": "#008300", "ab:rr_first": "#4a3aa7"}
SYN_SIGMAS = [1.0, 1.03, 1.1, 1.25, 1.5, 2.0]


def paired_randomization_p(x, y, rng, n=20000):
    d = np.asarray(x) - np.asarray(y)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def world(name: str) -> str:
    """Rankers are only compared within one document collection ("world"): the dense
    runs cover the 1M subset, Anserini's BM25 the full 8.8M collection."""
    if name.startswith("synthetic"):
        return "synthetic"
    return "1m" if "1m" in name else "full"


def load_rankers(include_real=True):
    qrels, prov = R.load_dl_qrels()
    runs, sources = {}, {}
    if include_real:
        real = R.discover_real_runs()
        for name, run in real.items():
            runs[name] = run
            sources[name] = ("run built by scripts/m9_make_runs.py (data/runs/m9/); BGE vectors encoded by M9 "
                             "for BM25 top-100 only" if name in ("bm25-bge-rerank100", "rrf-bm25-bge100")
                             else "HybridSearch's own BM25 engine run (data/runs/lexical/)" if name.startswith("hs-")
                             else "run file " + name + ".{dl19,dl20}.trec under data/runs/")
        bm25 = next((n for n in real if "bm25" in n and "1m" not in n), None)
        dense = next((n for n in real if "dense" in n and "rrf" not in n), None)
        if bm25 and dense and world(dense) == "1m" and not any("bm25" in n and "1m" in n for n in real):
            # put BM25 in the dense run's world: keep only 1M-subset passages, order unchanged
            from hybridsearch.data.io import DATA, read_u64bin
            keep = {str(x) for x in read_u64bin(DATA / "subset/1m/docids.u64bin")}
            name = f"{bm25}-filtered-1m"
            runs[name] = {q: {d: s for d, s in r.items() if d in keep} for q, r in real[bm25].items()}
            sources[name] = f"{bm25} run restricted to the 1M-subset passages (built here)"
            bm25 = name
        elif bm25 and dense:
            bm25 = next((n for n in real if "bm25" in n and world(n) == world(dense)), bm25)
        if bm25 and dense:
            name = f"rrf({bm25},{dense})"
            runs[name] = R.rrf([runs[bm25], real[dense]], k=60)
            sources[name] = "reciprocal rank fusion k=60 of the two runs, built here"
    for s in SYN_SIGMAS:
        # common random numbers: every sigma scales the SAME noise draw, so a larger sigma
        # is a strictly noisier version of the same ranker
        runs[f"synthetic-oracle-sigma{s}"] = R.synthetic_noisy_oracle(qrels, s, seed=1000)
        sources[f"synthetic-oracle-sigma{s}"] = f"SYNTHETIC: DL grade + {s}*N(0,1), shared noise seed 1000"
    return qrels, prov, runs, sources


def choose_pairs(names, ndcg):
    """Real rankers: within each collection world, sort by offline nDCG@10 and compare
    neighbours (the hard, realistic comparisons) plus the worst against the best.
    Synthetic: sigma1.0 against every noisier version (a controlled range of gaps)."""
    pairs = []
    worlds = {}
    for n in names:
        if not n.startswith("synthetic"):
            worlds.setdefault(world(n), []).append(n)
    for w, rs in sorted(worlds.items()):
        rs = sorted(rs, key=lambda n: -ndcg[n].mean())
        pairs += [(rs[i], rs[i + 1]) for i in range(len(rs) - 1)]
        if len(rs) > 2:
            pairs.append((rs[0], rs[-1]))
    base = "synthetic-oracle-sigma1.0"
    pairs += [(base, f"synthetic-oracle-sigma{s}") for s in SYN_SIGMAS[1:]]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=float, default=4e6, help="simulated impressions per arm/method pool")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--aa-n", type=int, default=2000)
    ap.add_argument("--aa-trials", type=int, default=1000)
    args = ap.parse_args()
    if args.quick:
        args.pool, args.trials, args.aa_trials = 2e5, 300, 200
    M = int(args.pool)
    rng = np.random.default_rng([args.seed, 1])  # A/A and validation stream
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    qrels, prov, runs, sources = load_rankers(not args.no_real)
    table = R.build_table(runs, qrels, k=10)
    ndcg = {n: table.ndcg(n) for n in runs}
    print(f"{len(table.qids)} topics; rankers: " + ", ".join(f"{n} nDCG@10={ndcg[n].mean():.4f}" for n in runs))
    pairs = choose_pairs(list(runs), ndcg)
    ns = S.n_grid(1, 8, 20)
    models = [make_model(k, p) for k, p in ALL_MODELS]

    def sub_rng(*keys):
        """Independent stream per (pair, model) / (ranker, model): a cell's result does not
        depend on which other pairs are in the study, so cells can be cached and reused."""
        return np.random.default_rng([args.seed] + [xxhash.xxh32_intdigest(str(k).encode()) for k in keys])

    CELLS.mkdir(parents=True, exist_ok=True)

    def cached(key, fn):
        path = CELLS / (xxhash.xxh64_hexdigest(key.encode()) + ".json")
        if path.exists():
            return json.loads(path.read_text())
        val = fn()
        path.write_text(json.dumps(val))
        return val

    def ab_pool(r, model):
        def run():
            d = S.pool_ab(table.grades(r), model, M, sub_rng("ab", r, model.name))
            return {mt: [x.values.tolist(), x.probs.tolist()] for mt, x in d.items()}
        raw = cached(f"ab|{r}|{model.name}|{M}|{args.seed}|{table_sig(r)}", run)
        return {mt: S.Dist(np.array(v), np.array(p_), M) for mt, (v, p_) in raw.items()}

    def table_sig(r):
        return xxhash.xxh64_hexdigest(json.dumps(table.lists[r]).encode())

    results, skipped = [], []
    for a, b in pairs:
        pair = None
        delta = float(ndcg[a].mean() - ndcg[b].mean())
        same = sum(la == lb for la, lb in zip(table.lists[a], table.lists[b]))
        if same == len(table.qids):
            skipped.append({"a": a, "b": b, "reason": "identical top-10 on every topic (no online difference possible)"})
            print(f"skip {a} vs {b}: identical top-10 lists", flush=True)
            continue
        a_better = delta > 0
        off_p = paired_randomization_p(ndcg[a], ndcg[b], sub_rng("offline", a, b))
        for model in models:
            key = f"cell|{a}|{b}|{model.name}|{M}|{args.trials}|{args.seed}|{table_sig(a)}|{table_sig(b)}"
            path = CELLS / (xxhash.xxh64_hexdigest(key.encode()) + ".json")
            if path.exists():
                results.append(json.loads(path.read_text()))
                print(f"cached: {a} vs {b} | {model.name}", flush=True)
                continue
            if pair is None:
                pair = S.build_pair(table, a, b, sub_rng("pair", a, b))
            rng = sub_rng("cell", a, b, model.name)
            t0 = time.time()
            rec = {"a": a, "b": b, "model": model.name, "label": LABEL.format(m=model.name),
                   "offline": {"ndcg10_a": float(ndcg[a].mean()), "ndcg10_b": float(ndcg[b].mean()),
                               "delta": delta, "better": a if a_better else b, "paired_randomization_p": off_p,
                               "topics_with_identical_top10": same},
                   "methods": {}}
            dists = {}
            for m in ("tdi", "bi", "pi"):
                dists[m] = S.pool_interleave(m, pair.entries[m], model, M, rng)
            da, db = ab_pool(a, model), ab_pool(b, model)
            for meth in METHODS:
                if meth.startswith("ab:"):
                    mt = meth[3:]
                    pc, pw = S.power_ab(mt, da[mt], db[mt], ns, args.trials, rng, a_better)
                    eff = da[mt].mean - db[mt].mean
                    se = np.sqrt(da[mt].var / M + db[mt].var / M)
                    good = 1 if S.abstats.HIGHER_IS_BETTER[mt] else -1
                    online_prefers_a = eff * good > 0
                    analytic = S.analytic_n80_ab(da[mt], db[mt])
                    extra = {"mean_a": da[mt].mean, "mean_b": db[mt].mean}
                else:
                    d = dists[meth]
                    pc, pw = S.power_interleave(meth, d, ns, args.trials, rng, a_better)
                    eff, se = d.mean, np.sqrt(d.var / M)
                    online_prefers_a = eff > 0
                    analytic = S.analytic_n80_one_sample(d)
                    extra = {"p_a_wins": float(d.probs[d.values > 0].sum()),
                             "p_b_wins": float(d.probs[d.values < 0].sum()),
                             "p_tie": float(d.probs[d.values == 0].sum())} if meth != "pi" else {}
                n80 = S.n_at_power(ns, pc)
                agree = bool(online_prefers_a == a_better)
                rec["methods"][meth] = {
                    "n80": n80, "analytic_n80": analytic if agree else None,
                    "pool_limited": bool(n80 is not None and n80 > M / 10),
                    "online_effect": float(eff), "online_effect_se": float(se),
                    "online_effect_z": float(eff / se) if se > 0 else None,
                    "online_agrees_with_offline": agree,
                    "wrong_direction_rate_at_max_power_grid": float(pw.max()),
                    "power": pc.tolist(), "power_wrong": pw.tolist(), **extra,
                }
            results.append(rec)
            path.write_text(json.dumps(rec))
            tdi = rec["methods"]["tdi"]["n80"]
            best_ab = min((v["n80"] for k_, v in rec["methods"].items() if k_.startswith("ab:") and v["n80"]),
                          default=None)
            print(f"[{time.time() - t_start:6.0f}s] {a} vs {b} | {model.name}: dNDCG={delta:+.4f} "
                  f"TDI N80={tdi and round(tdi)} best-A/B N80={best_ab and round(best_ab)} ({time.time() - t0:.1f}s)",
                  flush=True)

    # -------- A/A: fresh simulation per trial (no pool), same ranker on both sides
    aa_ranker = next((n for n in runs if "bm25" in n and not n.startswith("rrf")), "synthetic-oracle-sigma1.0")
    grades = table.grades(aa_ranker)
    aa_pair = S.build_pair(table, aa_ranker, aa_ranker, rng, methods=("tdi",))
    aa = {"ranker": aa_ranker, "n_impressions": args.aa_n, "trials": args.aa_trials, "alpha": 0.05, "models": {}}
    for model in models:
        entry = {}
        p, _ = S.direct_interleave_pvalues("tdi", aa_pair.entries["tdi"], model, args.aa_n, args.aa_trials, rng)
        entry["tdi"] = {"fpr": float(np.mean(p < 0.05)), "ks_p_uniform": float(stats.kstest(p, "uniform").pvalue)}
        abp = S.direct_ab_pvalues(grades, grades, model, args.aa_n, args.aa_trials, rng)
        for mt, (pp, _) in abp.items():
            entry["ab:" + mt] = {"fpr": float(np.mean(pp < 0.05)), "ks_p_uniform": float(stats.kstest(pp, "uniform").pvalue)}
        aa["models"][model.name] = entry
        print(f"A/A {model.name}: " + ", ".join(f"{k_} {v['fpr']:.3f}" for k_, v in entry.items()), flush=True)

    # -------- direct-simulation validation of pool-based N80 (no pool, fresh clicks per trial)
    validation = []
    vt = 300 if args.quick else 1000
    for rec in results:
        if rec["model"] not in ("dbn-navigational", "pbm-informational"):
            continue
        a, b = rec["a"], rec["b"]
        if not (a.startswith("synthetic") and b.endswith("1.25")) and a.startswith("synthetic"):
            continue
        pair = S.build_pair(table, a, b, rng, methods=("tdi",))
        model = make_model(*rec["model"].split("-"))
        a_better = rec["offline"]["delta"] > 0
        v = {"a": a, "b": b, "model": rec["model"], "trials": vt}
        n = rec["methods"]["tdi"]["n80"]
        if n and n * vt <= 3e8:
            p, d = S.direct_interleave_pvalues("tdi", pair.entries["tdi"], model, int(round(n)), vt, rng)
            v["tdi"] = {"n": int(round(n)), "power_direct": float(np.mean((p < 0.05) & (d * (1 if a_better else -1) > 0)))}
        cand = [(k_, x["n80"]) for k_, x in rec["methods"].items() if k_.startswith("ab:") and x["n80"]]
        if cand:
            mt, n = min(cand, key=lambda z: z[1])
            if n * vt <= 3e8:
                res = S.direct_ab_pvalues(pair.grades_a, pair.grades_b, model, int(round(n)), vt, rng)
                p, d = res[mt[3:]]
                good = 1 if S.abstats.HIGHER_IS_BETTER[mt[3:]] else -1
                v[mt] = {"n": int(round(n)),
                         "power_direct": float(np.mean((p < 0.05) & (d * good * (1 if a_better else -1) > 0)))}
        validation.append(v)
        print("validation", v, flush=True)

    common = {"seed": args.seed, "pool_impressions_per_arm_or_method": M, "trials_per_grid_point": args.trials,
              "n_grid": "logspace(1, 8, 141) rounded", "click_models": [m.describe() for m in models],
              "qrels": prov, "rankers": sources, "topics": len(table.qids),
              "label": "SIMULATED users; relevance from TREC DL 2019+2020 graded qrels; not user traffic",
              "runtime_s": round(time.time() - t_start, 1)}
    doc = {"meta": common, "n_grid": ns.tolist(),
           "rankers": {n: {"ndcg10_mean": float(ndcg[n].mean()), "source": sources[n]} for n in runs},
           "results": results, "skipped_pairs": skipped, "validation_direct": validation}
    # raw power curves (per grid point) go to data/m9/ (gitignored); results/ keeps a summary
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "sensitivity_raw.json").write_text(json.dumps(doc) + "\n")
    summary = {k: v for k, v in doc.items() if k not in ("results", "n_grid")}
    summary["raw_power_curves"] = "data/m9/sensitivity_raw.json (regenerate with this script)"
    summary["results"] = [{**{k: v for k, v in r.items() if k != "methods"},
                           "methods": {m: {k: v for k, v in x.items() if k not in ("power", "power_wrong")}
                                       for m, x in r["methods"].items()}} for r in doc["results"]]
    (OUT / "sensitivity.json").write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    write_meta(OUT / "sensitivity.json", **common)
    (OUT / "aa.json").write_text(json.dumps(aa, indent=1) + "\n")
    write_meta(OUT / "aa.json", **common, aa_note="fresh click simulation per trial; no pool")
    write_markdown(doc, aa)
    write_meta(OUT / "sensitivity.md", **common)
    plot(doc)
    write_meta(OUT / "sensitivity.png", **common)
    write_meta(OUT / "sensitivity_n80.png", **common)
    print(f"done in {time.time() - t_start:.0f}s")


def fmt_n(x, lim=False):
    if x is None:
        return "not reached"
    if x <= 10:
        return "<=10"  # grid floor
    s = f"{x:,.0f}" if x < 1e5 else f"{x:.2g}"
    return s + ("*" if lim else "")


def write_markdown(doc, aa):
    L = ["# M9 sensitivity study: interleaving vs A/B",
         "",
         "**All users are simulated** (click models below, relevance from TREC DL 2019+2020 graded qrels). "
         "No number here is user traffic. Generated by `scripts/m9_sensitivity.py`; raw data in `sensitivity.json`.",
         "",
         f"N80 = total query impressions for 80% power at two-sided p<0.05 **in the offline-correct direction** "
         f"(A/B splits N in half). Pool of {doc['meta']['pool_impressions_per_arm_or_method']:,} simulated impressions "
         f"per arm/method, {doc['meta']['trials_per_grid_point']} trials per grid point, 20 grid points per decade. "
         "`*` = N80 above pool/10 (pool-limited). 'not reached' = power never reached 0.8 up to 1e8 "
         "(including when the metric favours the offline-worse ranker).",
         "", "## Rankers (offline, DL19+DL20 topics, nDCG@10)", "", "| ranker | nDCG@10 | source |", "|---|---|---|"]
    for n, r in doc["rankers"].items():
        L.append(f"| {n} | {r['ndcg10_mean']:.4f} | {r['source']} |")
    for sk in doc.get("skipped_pairs", []):
        L.append(f"\nSkipped {sk['a']} vs {sk['b']}: {sk['reason']}.")
    by_pair = {}
    for rec in doc["results"]:
        by_pair.setdefault((rec["a"], rec["b"]), []).append(rec)
    L += ["", "## Summary: how many more impressions A/B needs than team-draft interleaving", "",
          "Per pair, over the 9 click models: ratio = (best A/B metric's N80) / (TDI N80), counting only "
          "cells where both reach 80% power in the offline-correct direction. 'TDI wrong' = cells where TDI's "
          "expected outcome favours the offline-worse ranker.", "",
          "Pairs whose offline difference is not significant (paired randomization p >= 0.05 over the 97 "
          "topics) have **no known quality ordering**: their 'correct direction' is just the sign of a tiny "
          "offline delta, so they are marked and are not evidence for or against either method.", "",
          "| pair | offline delta nDCG@10 | offline p | median TDI N80 | median best-A/B N80 | ratio median | "
          "ratio min-max | cells | TDI wrong | best-interleaving / TDI (median) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (a, b), recs in by_pair.items():
        ratios, tdis, abs_best, il_ratio = [], [], [], []
        wrong = sum(not r["methods"]["tdi"]["online_agrees_with_offline"] for r in recs)
        for r in recs:
            t = r["methods"]["tdi"]["n80"]
            ab_ = [r["methods"][m]["n80"] for m in METHODS if m.startswith("ab:") and r["methods"][m]["n80"]]
            il = [r["methods"][m]["n80"] for m in ("tdi", "bi", "pi") if r["methods"][m]["n80"]]
            if t:
                tdis.append(t)
            if ab_:
                abs_best.append(min(ab_))
            if t and ab_:
                ratios.append(min(ab_) / t)
            if t and il:
                il_ratio.append(min(il) / t)
        med = (lambda x: f"{np.median(x):,.0f}" if x else "n/a")
        op = recs[0]["offline"]["paired_randomization_p"]
        tag = "" if op < 0.05 else " (no known ordering)"
        L.append(f"| {a} vs {b}{tag} | {recs[0]['offline']['delta']:+.4f} | {op:.2g} | {med(tdis)} | {med(abs_best)} | "
                 + (f"{np.median(ratios):.1f}x | {min(ratios):.1f}x-{max(ratios):.1f}x" if ratios else "n/a | n/a")
                 + f" | {len(ratios)}/{len(recs)} | {wrong} | "
                 + (f"{np.median(il_ratio):.2f}" if il_ratio else "n/a") + " |")
    for (a, b), recs in by_pair.items():
        o = recs[0]["offline"]
        L += ["", f"## {a} vs {b}", "",
              f"offline nDCG@10 {o['ndcg10_a']:.4f} vs {o['ndcg10_b']:.4f} (delta {o['delta']:+.4f}, paired "
              f"randomization p={o['paired_randomization_p']:.3g}); offline-better: **{o['better']}**", "",
              "| click model | " + " | ".join(METHOD_NAMES[m] for m in METHODS) + " | best A/B / TDI |",
              "|---|" + "---|" * (len(METHODS) + 1)]
        for rec in recs:
            cells = []
            for m in METHODS:
                v = rec["methods"][m]
                c = fmt_n(v["n80"], v["pool_limited"])
                if not v["online_agrees_with_offline"]:
                    c += " (disagrees)"
                cells.append(c)
            tdi = rec["methods"]["tdi"]["n80"]
            abs_ = [rec["methods"][m]["n80"] for m in METHODS if m.startswith("ab:") and rec["methods"][m]["n80"]]
            ratio = f"{min(abs_) / tdi:.1f}x" if tdi and abs_ else "n/a"
            L.append(f"| {rec['model']} | " + " | ".join(cells) + f" | {ratio} |")
    L += ["", "## Agreement of the online verdict with offline nDCG@10", "",
          "Share of (pair, click model) cells where the method's large-pool expected effect favours the "
          "offline-better ranker.", "", "| method | agree | cells |", "|---|---|---|"]
    for m in METHODS:
        cells = [rec["methods"][m]["online_agrees_with_offline"] for rec in doc["results"]]
        L.append(f"| {METHOD_NAMES[m]} | {sum(cells)} | {len(cells)} |")
    L += ["", "## A/A false-positive rate (fresh simulation per trial, no pool)", "",
          f"Ranker `{aa['ranker']}` against itself, N={aa['n_impressions']} impressions per trial, "
          f"{aa['trials']} trials, alpha=0.05. KS = p-value of a Kolmogorov-Smirnov test of p-value uniformity "
          "(the sign test is discrete, so its p-values are conservative, not uniform).", "",
          "| click model | " + " | ".join(METHOD_NAMES[m] for m in ["tdi"] + METHODS[3:]) + " |",
          "|---|" + "---|" * 5]
    for mname, e in aa["models"].items():
        L.append(f"| {mname} | " + " | ".join(f"{e[m]['fpr']:.3f} (KS {e[m]['ks_p_uniform']:.2g})"
                                              for m in ["tdi"] + METHODS[3:]) + " |")
    if doc["validation_direct"]:
        L += ["", "## Direct-simulation check of pool-based N80", "",
              "At the pool-based N80, fresh click simulation per trial (no pool); power should be near 0.80.", "",
              "| pair | click model | method | N | direct power | trials |", "|---|---|---|---|---|---|"]
        for v in doc["validation_direct"]:
            for m in [k for k in v if k in METHOD_NAMES]:
                L.append(f"| {v['a']} vs {v['b']} | {v['model']} | {METHOD_NAMES[m]} | {v[m]['n']:,} | "
                         f"{v[m]['power_direct']:.3f} | {v['trials']} |")
    (OUT / "sensitivity.md").write_text("\n".join(L) + "\n")


def plot(doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = np.array(doc["n_grid"])
    recs = doc["results"]
    known = [r for r in recs if r["offline"]["paired_randomization_p"] < 0.05]
    real_pairs = [(r["a"], r["b"]) for r in known if not r["a"].startswith("synthetic")]
    head = real_pairs[0] if real_pairs else ("synthetic-oracle-sigma1.0", "synthetic-oracle-sigma1.1")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888", "axes.labelcolor": "#333",
                         "xtick.color": "#555", "ytick.color": "#555"})
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for ax, rec in zip(axes.flat, [r for r in recs if (r["a"], r["b"]) == head]):
        for m in METHODS:
            ax.plot(ns, rec["methods"][m]["power"], color=COLORS[m], lw=2,
                    ls="-" if not m.startswith("ab:") else "--", label=METHOD_NAMES[m])
        ax.axhline(0.8, color="#999", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(rec["model"], fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("query impressions N (A/B: N/2 per arm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("power (p<0.05, correct direction)")
    o = [r for r in recs if (r["a"], r["b"]) == head][0]["offline"]
    fig.suptitle(f"Power vs impressions: {head[0]} vs {head[1]} (offline nDCG@10 {o['ndcg10_a']:.3f} vs "
                 f"{o['ndcg10_b']:.3f})\nSIMULATED users, relevance from TREC DL 2019+2020 graded qrels", fontsize=11)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT / "sensitivity.png", dpi=150, facecolor="white")
    plt.close(fig)

    # N80 against offline gap, per click-model family (navigational profile + all)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, kind in zip(axes, ("pbm", "cascade", "dbn")):
        for m in METHODS:
            xs, ys = [], []
            for r in known:
                if not r["model"].startswith(kind + "-") or r["model"].endswith("perfect"):
                    continue
                v = r["methods"][m]
                if v["n80"] and v["online_agrees_with_offline"]:
                    xs.append(abs(r["offline"]["delta"]))
                    ys.append(v["n80"])
            ax.scatter(xs, ys, s=28, color=COLORS[m], marker="o" if not m.startswith("ab:") else "^",
                       edgecolor="white", linewidth=0.8, label=METHOD_NAMES[m], zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{kind} (navigational + informational)", fontsize=10)
        ax.set_xlabel("|offline delta nDCG@10|")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("N80 (impressions)")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Impressions needed for 80% power vs offline gap, ranker pairs with a significant offline "
                 "difference (SIMULATED users)", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(OUT / "sensitivity_n80.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
