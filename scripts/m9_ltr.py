#!/usr/bin/env python
"""M9 counterfactual learning to rank from SIMULATED logged clicks.

    .venv/bin/python scripts/m9_ltr.py                # real features if runs exist, else synthetic
    .venv/bin/python scripts/m9_ltr.py --synthetic    # synthetic stand-in only
    .venv/bin/python scripts/m9_ltr.py --quick

Setup (documented in docs/EXPERIMENTS.md):
* Logging (production) ranker: BM25 (Anserini run). It shows its top-K=20 candidates;
  simulated users click under PBM (navigational profile, eta_r = 1/r).
* Training queries: MS MARCO train_tune (2,000 train queries; sparse labels, label 1
  mapped to DL grade 2). Evaluation queries: TREC DL 2019+2020 (97 topics, graded).
  Evaluation queries are never used for training or for choosing any setting; the L2
  strength is fixed a priori (1e-3), not tuned.
* The learned ranker is linear over per-candidate features (see ltr/data.py) and
  reranks the logging ranker's top-20 on the held-out DL queries; metric nDCG@10.
* Clicks: every number here comes from simulated users; relevance from judgments.

Writes results/m9/ltr.{json,md}, ltr_learning_curve.png (+ .meta.json each).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")  # shared laptop: cap BLAS threads

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from hybridsearch.clicks.models import make_model  # noqa: E402
from hybridsearch.clicks.rankers import RUN_DIR, load_dl_qrels, load_dl_queries, ranked  # noqa: E402
from hybridsearch.data.io import DATA, read_qrels, read_queries, read_run, read_u64bin  # noqa: E402
from hybridsearch.ltr import data as D  # noqa: E402
from hybridsearch.ltr import logsim as LS  # noqa: E402
from hybridsearch.ltr import model as M  # noqa: E402
from hybridsearch.rerank.meta import write_meta  # noqa: E402

OUT = ROOT / "results/m9"
K = 20
LAM = 1e-3
CACHE = DATA / "cache/m9"


# ------------------------------------------------------------------ data
def find_run(patterns, split, exclude=("-1m",)):
    """First non-empty run matching a pattern; full-collection runs only (names with
    '-1m' are the 1M-subset world and are never mixed in silently)."""
    for pat in patterns:
        for p in sorted(RUN_DIR.glob(f"*{pat}*.{split}.trec")):
            if p.stat().st_size > 0 and not any(x in p.name for x in exclude):
                return p
    return None


def doc_lengths(pids: set[str]) -> dict[str, int]:
    """Whitespace token counts for the given passages (one scan of collection.tsv, cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "doclen.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    need = pids - set(have)
    if need:
        coll = DATA / "raw/msmarco/collection.tsv"
        with open(coll, encoding="utf-8") as f:
            for line in f:
                pid, text = line.split("\t", 1)
                if pid in need:
                    have[pid] = len(text.split())
        cache.write_text(json.dumps(have))
    return {p: have.get(p, 0) for p in pids}


def real_sets(subset_world: bool, no_dense: bool = False, n_train: int | None = None, seed: int = 0,
              dense_source: str = "candidates"):
    """Returns (train, eval, provenance) or None if the needed runs are missing."""
    bm25_tr, bm25_ev = find_run(["bm25"], "train_tune"), [find_run(["bm25"], s) for s in ("dl19", "dl20")]
    if not bm25_tr or not all(bm25_ev):
        return None
    dense_tr = find_run(["dense"], "train_tune", exclude=())
    dense_ev = [find_run(["dense"], s, exclude=()) for s in ("dl19", "dl20")]
    use_dense = bool(dense_tr and all(dense_ev)) and dense_source == "runs"
    prov = {"bm25_train": str(bm25_tr), "bm25_eval": [str(p) for p in bm25_ev], "dense": use_dense}
    bm_tr = read_run(bm25_tr)
    bm_ev = {**read_run(bm25_ev[0]), **read_run(bm25_ev[1])}
    de_tr, de_ev = {}, {}
    if use_dense:
        prov.update({"dense_train": str(dense_tr), "dense_eval": [str(p) for p in dense_ev]})
        de_tr = read_run(dense_tr)
        de_ev = {**read_run(dense_ev[0]), **read_run(dense_ev[1])}
        if subset_world:
            # dense runs cover the 1M subset only: restrict BM25 to the same documents so
            # every candidate has both features (the subset contains every judged-relevant
            # passage, so this world is easier than the full collection; stated)
            keep = {str(x) for x in read_u64bin(DATA / "subset/1m/docids.u64bin")}
            bm_tr = {q: {d: s for d, s in r.items() if d in keep} for q, r in bm_tr.items()}
            bm_ev = {q: {d: s for d, s in r.items() if d in keep} for q, r in bm_ev.items()}
            prov["world"] = "1M subset (BM25 run filtered to subset docs)"
    q_tr = read_queries(DATA / "subset/train_tune/queries.tsv")
    if n_train and n_train < len(q_tr):
        # seeded sample of the training queries (the bottleneck is encoding their
        # candidates on a shared CPU; click volume per query is what the study varies)
        keep_q = set(np.random.default_rng(seed).choice(sorted(q_tr, key=int), n_train, replace=False).tolist())
        q_tr = {q: t for q, t in q_tr.items() if q in keep_q}
        bm_tr = {q: r for q, r in bm_tr.items() if q in keep_q}
        prov["train_query_sample"] = f"{n_train} of 2000 train_tune queries, seed {seed}"
    qrels_tr = read_qrels(DATA / "subset/train_tune/qrels.tsv")
    qrels_ev, qprov = load_dl_qrels()
    q_ev = load_dl_queries()
    cands = {}
    for run in (bm_tr, bm_ev):
        for q, r in run.items():
            cands[q] = ranked(r)[:K]
    pids = set().union(*cands.values())
    dl = doc_lengths(pids)
    if not use_dense and not no_dense:
        # no dense run on disk: score just the candidates with BGE ourselves (CPU)
        from hybridsearch.ltr import dense_feature as DF
        texts = DF.passage_texts(pids)
        qtext = {**{q: t for q, t in q_tr.items() if q in cands}, **{q: t for q, t in q_ev.items() if q in cands}}
        vecs = DF.encode_needed(qtext, texts, log=lambda m: print(m, flush=True))
        cand_run = DF.dense_run_for_candidates(cands, vecs)
        de_tr = {q: cand_run[q] for q in bm_tr if q in cand_run}
        de_ev = {q: cand_run[q] for q in bm_ev if q in cand_run}
        use_dense = True
        prov["dense"] = ("candidate-only: BGE-base-en-v1.5 inner product for each BM25 top-20 candidate, "
                         "encoded by M9 on CPU (hybridsearch.dense.encode convention); dense_z / dense_rr are "
                         "z-score and reciprocal rank within the 20 candidates, not within a dense run")
        prov["dense_encode"] = vecs["info"]
    prov.setdefault("world", "full collection (Anserini BM25 top-20 candidates)")
    tr = D.build_from_runs("train_tune", sorted(q_tr, key=int), bm_tr, bm_tr, de_tr, qrels_tr, q_tr, dl, K,
                           label_map={1: 2})
    ev = D.build_from_runs("dl19+dl20", sorted(qrels_ev, key=int), bm_ev, bm_ev, de_ev, qrels_ev, q_ev, dl, K)
    if not use_dense:
        drop = [i for i, n in enumerate(D.FEATURES) if "dense" in n]
        for s in (tr, ev):
            s.data.X = np.delete(s.data.X, drop, axis=1)
            s.feature_names = [n for n in D.FEATURES if "dense" not in n]
    prov["qrels_eval"] = qprov
    prov["train_label_map"] = "MS MARCO label 1 -> DL grade 2; unjudged -> 0"
    return tr, ev, prov


# ------------------------------------------------------------------ experiment
def evaluate(ev: D.LTRSet, theta) -> float:
    orders = M.rank_lists(ev.data, theta)
    return float(np.mean([M.ndcg10(o, ev.grades[ev.data.offsets[q]:ev.data.offsets[q + 1]], ev.ideal[q])
                          for q, o in enumerate(orders)]))


def logging_ndcg(ev: D.LTRSet) -> float:
    return float(np.mean([M.ndcg10(np.arange(ev.data.offsets[q + 1] - ev.data.offsets[q]),
                                   ev.grades[ev.data.offsets[q]:ev.data.offsets[q + 1]], ev.ideal[q])
                          for q in range(ev.data.n_queries)]))


def candidate_oracle_ndcg(ev: D.LTRSet) -> float:
    out = []
    for q in range(ev.data.n_queries):
        g = ev.grades[ev.data.offsets[q]:ev.data.offsets[q + 1]]
        out.append(M.ndcg10(np.argsort(-g, kind="stable"), g, ev.ideal[q]))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--profile", default="navigational")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.05, help="fraction of sessions with a swap intervention")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-dense", action="store_true", help="BM25-only features (ablation)")
    ap.add_argument("--train-queries", type=int, default=200, help="seeded sample of train_tune queries")
    ap.add_argument("--dense-source", choices=["candidates", "runs"], default="candidates",
                    help="candidates: BGE score for the full-collection BM25 top-20 (encoded here); "
                         "runs: dense-flat-1m run files, BM25 restricted to the 1M subset")
    args = ap.parse_args()
    t0 = time.time()
    rng0 = np.random.default_rng(args.seed)

    sets = None if args.synthetic else real_sets(subset_world=True, no_dense=args.no_dense,
                                                 n_train=args.train_queries, seed=args.seed,
                                                 dense_source=args.dense_source)
    if sets is None:
        tr, ev = D.synthetic(2000, K, rng0, "synthetic-train"), D.synthetic(97, K, rng0, "synthetic-eval")
        prov = {"world": "SYNTHETIC stand-in (features generated from grades); real runs not available"}
    else:
        tr, ev, prov = sets
    D.standardise(tr, ev)
    model = make_model("pbm", args.profile, k=K, severity=args.severity)
    label = (f"simulated users (click model {model.name}, eta_r=(1/r)^{args.severity}, relevance from "
             f"{'MS MARCO train labels (train) / TREC DL 2019+2020 graded qrels (eval)' if sets else 'synthetic grades'})")
    print(label, "|", prov["world"], "| features:", tr.feature_names, flush=True)
    print(f"train {tr.data.n_queries} queries, eval {ev.data.n_queries} queries", flush=True)

    G = tr.grades_log_order()
    ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000]
    seeds = args.seeds
    if args.quick:
        ns, seeds = [1_000, 30_000, 1_000_000], 2

    base = {
        "logging_bm25": logging_ndcg(ev),
        "candidate_oracle": candidate_oracle_ndcg(ev),
        "oracle_true_grades": evaluate(ev, th_or := M.fit(tr.data, tr.grades.astype(float), LAM)),
    }
    # skyline: what IPS converges to (labels = alpha(grade), i.e. click prob. given examination)
    alpha_lab = model.alpha[tr.grades]
    base["skyline_alpha_labels"] = evaluate(ev, th_sky := M.fit(tr.data, alpha_lab, LAM))
    print({k: round(v, 4) for k, v in base.items()}, flush=True)

    harvest = "dense_z" in tr.feature_names
    methods = ["naive", "ips_true", "ips_swap_est", "ips_randtop_est"] + (["ips_harvest_est"] if harvest else [])
    curve = {m: np.full((seeds, len(ns)), np.nan) for m in methods}
    eta_err = {m: np.full((seeds, len(ns)), np.nan) for m in ["swap", "randtop"] + (["harvest"] if harvest else [])}
    if harvest:
        # second production ranker for intervention harvesting: the same candidates ordered
        # by the dense feature (standardisation is affine, so the order is the raw order)
        dz = tr.feature_names.index("dense_z")
        orders = np.full((2, tr.data.n_queries, K), -1, np.int64)
        for q in range(tr.data.n_queries):
            a, b = tr.data.offsets[q], tr.data.offsets[q + 1]
            orders[0, q, : b - a] = np.arange(b - a)
            orders[1, q, : b - a] = np.argsort(-tr.data.X[a:b, dz], kind="stable")
    eta_est_last = {}
    thetas = {}
    for s in range(seeds):
        for j, n in enumerate(ns):
            rng = np.random.default_rng([args.seed, s, j])
            logs = LS.simulate_logs(G, tr.data.offsets, model, n, rng, "swap", args.eps)
            logs_rt = LS.simulate_logs(G, tr.data.offsets, model, n, rng, "randtop", args.eps, randtop_n=10)
            est_sw = LS.monotone_propensity(LS.estimate_propensity(logs, "swap"), LS.propensity_support(logs, "swap"))
            est_rt = LS.monotone_propensity(LS.estimate_propensity(logs_rt, "randtop"),
                                            LS.propensity_support(logs_rt, "randtop"))
            eta_err["swap"][s, j] = float(np.mean(np.abs(est_sw[:10] - model.eta[:10])))
            eta_err["randtop"][s, j] = float(np.mean(np.abs(est_rt[:10] - model.eta[:10])))
            if harvest:
                ca_h, ses_h = LS.simulate_logs_multi(G, orders, tr.data.offsets, model, n, rng)
                raw_h, sup_h = LS.harvest_propensity(ca_h, ses_h, orders, tr.data.offsets)
                est_h = LS.monotone_propensity(raw_h, sup_h)
                eta_err["harvest"][s, j] = float(np.mean(np.abs(est_h[:10] - model.eta[:10])))
            th = {
                "naive": M.fit(tr.data, LS.ips_weights(logs.clicks_at, None), LAM),
                "ips_true": M.fit(tr.data, LS.ips_weights(logs.clicks_at, model.eta), LAM),
                "ips_swap_est": M.fit(tr.data, LS.ips_weights(logs.clicks_at, est_sw), LAM),
                "ips_randtop_est": M.fit(tr.data, LS.ips_weights(logs_rt.clicks_at, est_rt), LAM),
            }
            if harvest:
                th["ips_harvest_est"] = M.fit(tr.data, LS.ips_weights(logs.clicks_at, est_h), LAM)
            for m, t in th.items():
                curve[m][s, j] = evaluate(ev, t)
            if j == len(ns) - 1:
                thetas[s] = {m: t.tolist() for m, t in th.items()}
                eta_est_last[s] = {"swap": est_sw.tolist(), "randtop": est_rt.tolist(),
                                   **({"harvest": est_h.tolist()} if harvest else {})}
            print(f"[{time.time() - t0:5.0f}s] seed {s} n={n:>9,}: " +
                  " ".join(f"{m}={curve[m][s, j]:.4f}" for m in curve) +
                  " | eta MAE " + " ".join(f"{k} {v[s, j]:.3f}" for k, v in eta_err.items()), flush=True)

    # misestimation and clipping at a fixed log size
    n_fix = 1_000_000 if not args.quick else 100_000
    sev = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    clips = [1.5, 2.0, 5.0, 10.0, None]
    mis = np.full((seeds, len(sev)), np.nan)
    clip_res = {n_: np.full((seeds, len(clips)), np.nan) for n_ in (10_000, n_fix)}
    for s in range(seeds):
        rng = np.random.default_rng([args.seed, s, 99])
        logs = LS.simulate_logs(G, tr.data.offsets, model, n_fix, rng)
        for i, p in enumerate(sev):
            eta_p = (1.0 / np.arange(1, K + 1)) ** (p * args.severity)
            mis[s, i] = evaluate(ev, M.fit(tr.data, LS.ips_weights(logs.clicks_at, eta_p), LAM))
        for n_ in clip_res:
            lg = logs if n_ == n_fix else LS.simulate_logs(G, tr.data.offsets, model, n_, rng)
            for i, c in enumerate(clips):
                clip_res[n_][s, i] = evaluate(ev, M.fit(tr.data, LS.ips_weights(lg.clicks_at, model.eta, c), LAM))
        print(f"[{time.time() - t0:5.0f}s] seed {s}: misestimation " + " ".join(f"{p}:{v:.4f}" for p, v in zip(sev, mis[s])),
              flush=True)

    def ms(a):
        return {"mean": np.nanmean(a, 0).tolist(), "std": np.nanstd(a, 0, ddof=1).tolist() if a.shape[0] > 1 else None}

    doc = {
        "label": label, "provenance": prov, "features": tr.feature_names, "K_displayed": K, "lambda_l2": LAM,
        "click_model": model.describe(), "train_queries": tr.data.n_queries, "eval_queries": ev.data.n_queries,
        "seeds": seeds, "intervention_eps": args.eps, "baselines_ndcg10": base,
        "theta_oracle": th_or.tolist(), "theta_skyline": th_sky.tolist(),
        "learning_curve": {"n_sessions": ns, **{m: ms(v) for m, v in curve.items()}},
        "propensity_mae_top10": {"n_sessions": ns, **{m: ms(v) for m, v in eta_err.items()}},
        "eta_true": model.eta.tolist(), "eta_estimates_at_max_n": eta_est_last, "theta_at_max_n": thetas,
        "misestimation": {"n_sessions": n_fix, "assumed_severity_multiplier": sev, "ndcg10": ms(mis),
                          "note": "IPS with eta_r=(1/r)^(p*true_severity); p=1 is correct, p=0 is naive"},
        "clipping": {"clip": [c if c is not None else "none" for c in clips],
                     **{str(n_): ms(v) for n_, v in clip_res.items()}},
        "runtime_s": round(time.time() - t0, 1),
    }
    suffix = args.tag or ("" if sets else "_synthetic")
    jpath = OUT / f"ltr{suffix}.json"
    jpath.write_text(json.dumps(doc, indent=1) + "\n")
    meta = {"label": label, "seed": args.seed, "seeds": seeds, "provenance": prov}
    write_meta(jpath, **meta)
    write_md(doc, OUT / f"ltr{suffix}.md")
    write_meta(OUT / f"ltr{suffix}.md", **meta)
    plot(doc, OUT / f"ltr_learning_curve{suffix}.png")
    write_meta(OUT / f"ltr_learning_curve{suffix}.png", **meta)
    print(f"done in {time.time() - t0:.0f}s")


def write_md(doc, path):
    b = doc["baselines_ndcg10"]
    lc = doc["learning_curve"]
    L = ["# M9 counterfactual LTR from simulated clicks", "",
         f"**{doc['label']}.** World: {doc['provenance']['world']}. Train {doc['train_queries']} queries, "
         f"evaluate on {doc['eval_queries']} held-out queries (nDCG@10, reranking the logging ranker's top-{doc['K_displayed']}). "
         f"Features: {', '.join(doc['features'])}. {doc['seeds']} seeds; mean ± sd.", "",
         "| baseline | nDCG@10 |", "|---|---|"]
    for k, v in b.items():
        L.append(f"| {k} | {v:.4f} |")
    L += ["", "## Learning curve", "", "| logged sessions | " + " | ".join(k for k in lc if k != "n_sessions") + " |",
          "|---|" + "---|" * (len(lc) - 1)]
    for j, n in enumerate(lc["n_sessions"]):
        cells = []
        for k, v in lc.items():
            if k == "n_sessions":
                continue
            sd = f" ± {v['std'][j]:.4f}" if v["std"] else ""
            cells.append(f"{v['mean'][j]:.4f}{sd}")
        L.append(f"| {n:,} | " + " | ".join(cells) + " |")
    pm = doc["propensity_mae_top10"]
    L += ["", "## Position-bias estimation (mean absolute error of eta_r/eta_1 over ranks 1-10)", "",
          f"Interventions in {doc['intervention_eps']:.0%} of sessions.", "",
          "Harvesting (if present) uses no deliberate randomisation: two production rankers (BM25 order and "
          "dense order of the same candidates) share traffic 50/50, in a separate log of the same size.", "",
          "| logged sessions | " + " | ".join(k for k in pm if k != "n_sessions") + " |",
          "|---|" + "---|" * (len(pm) - 1)]
    for j, n in enumerate(pm["n_sessions"]):
        L.append(f"| {n:,} | " + " | ".join(f"{v['mean'][j]:.4f}" for k, v in pm.items() if k != "n_sessions") + " |")
    mi = doc["misestimation"]
    L += ["", f"## Propensity misestimation ({mi['n_sessions']:,} sessions)", "", mi["note"], "",
          "| p | " + " | ".join(str(p) for p in mi["assumed_severity_multiplier"]) + " |",
          "|---|" + "---|" * len(mi["assumed_severity_multiplier"]),
          "| nDCG@10 | " + " | ".join(f"{v:.4f}" for v in mi["ndcg10"]["mean"]) + " |"]
    cl = doc["clipping"]
    L += ["", "## Weight clipping (IPS with true eta, weight min(1/eta, tau))", "",
          "| sessions | " + " | ".join(f"tau={c}" for c in cl["clip"]) + " |", "|---|" + "---|" * len(cl["clip"])]
    for k, v in cl.items():
        if k == "clip":
            continue
        L.append(f"| {int(k):,} | " + " | ".join(f"{x:.4f}" for x in v["mean"]) + " |")
    path.write_text("\n".join(L) + "\n")


def plot(doc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lc = doc["learning_curve"]
    ns = np.array(lc["n_sessions"])
    b = doc["baselines_ndcg10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    names = {"naive": ("naive (click = relevant)", "#eb6834"), "ips_true": ("IPS, true propensities", "#2a78d6"),
             "ips_swap_est": ("IPS, propensities from swap interventions", "#1baf7a"),
             "ips_randtop_est": ("IPS, propensities from RandTop-10", "#4a3aa7"),
             "ips_harvest_est": ("IPS, propensities harvested from 2 production rankers", "#eda100")}
    for k, (lab, col) in names.items():
        if k not in lc:
            continue
        m, s = np.array(lc[k]["mean"]), np.array(lc[k]["std"] or np.zeros(len(ns)))
        ax.plot(ns, m, color=col, lw=2, marker="o", ms=5, label=lab)
        ax.fill_between(ns, m - s, m + s, color=col, alpha=0.12, lw=0)
    for k, lab, ls in (("oracle_true_grades", "oracle (trained on true grades)", "--"),
                       ("skyline_alpha_labels", "skyline (IPS limit: alpha(grade) labels)", ":"),
                       ("logging_bm25", "logging ranker (BM25)", "-.")):
        ax.axhline(b[k], color="#555", lw=1.2, ls=ls)
        ax.text(ns[0], b[k], " " + lab, va="bottom", fontsize=8, color="#333")
    ax.set_xscale("log")
    ax.set_xlabel("logged sessions (simulated)")
    ax.set_ylabel("nDCG@10 on held-out DL19+DL20" if "SYNTHETIC" not in doc["provenance"]["world"] else "nDCG@10 (synthetic eval)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.33))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax = axes[1]
    mi = doc["misestimation"]
    x = mi["assumed_severity_multiplier"]
    m, s = np.array(mi["ndcg10"]["mean"]), np.array(mi["ndcg10"]["std"] or np.zeros(len(x)))
    ax.errorbar(x, m, yerr=s, color="#2a78d6", marker="o", ms=5, lw=2, capsize=3)
    ax.axvline(1.0, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("assumed bias severity / true (0 = naive, 1 = correct)")
    ax.set_title(f"propensity misestimation, {mi['n_sessions']:,} sessions", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle("Counterfactual LTR from clicks: " + doc["label"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
