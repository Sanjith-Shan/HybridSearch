"""M4 hybrid retrieval study: tune fusion on train_tune only, evaluate on MS MARCO dev / DL19 /
DL20 (1M subset) and BEIR scifact / nfcorpus / fiqa, with paired randomization tests + Holm.

Inputs (see scripts/m4_all.sh for how they are produced):
  BM25   data/runs/m4/hs-bm25-1m.{train_tune,dev,dl19,dl20}.trec   (HybridSearch engine, Lucene BM25)
         data/runs/m4/hs-bm25.beir-{ds}.test.trec
  dense  data/runs/reference/dense-flat-1m.{split}.trec, dense-flat.beir-{ds}.test.trec (exact FAISS flat, BGE-base)
Outputs:
  data/runs/m4/{system}.{dataset}.trec, results/m4/fusion.{json,md} (+ .meta.json)

Usage: .venv/bin/python scripts/m4_fusion.py [--n-perm 20000] [--approx-dense dev=PATH ...]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "py"))

from hybridsearch.data.io import read_qrels, read_run  # noqa: E402
from hybridsearch.eval.metrics import evaluate, mrr_at_k, ndcg_at_k, parse_metric, recall_at_k  # noqa: E402
from hybridsearch.eval.results import write_result  # noqa: E402
from hybridsearch.fusion import fuse_runs, read_run_ranked, rrf, weighted, write_run_exact  # noqa: E402
from hybridsearch.fusion.fast import prepare, rrf_fast, weighted_fast  # noqa: E402
from hybridsearch.stats import compare  # noqa: E402

RUNS = REPO / "data/runs"
OUT_RUNS = RUNS / "m4"
DEPTH = 1000   # candidate depth per retriever (both inputs are top-1000 runs)
OUT_K = 1000   # fused list length
SEED = 20260923
RRF_KS = [10, 20, 40, 60, 100]
ALPHAS = [round(0.1 * i, 1) for i in range(1, 10)]
NORMS = ["minmax", "zscore"]

MSM = "MS MARCO (1M subset)"
DATASETS = {
    # name: (lex run, dense run, qrels, metrics, primary, recall metric, group)
    "dev": ("m4/hs-bm25-1m.dev.trec", "reference/dense-flat-1m.dev.trec", "data/subset/1m/qrels.dev.tsv",
            ["RR@10", "R@100", "R@1000"], "RR@10", "R@1000", MSM),
    "dl19": ("m4/hs-bm25-1m.dl19.trec", "reference/dense-flat-1m.dl19.trec", "data/subset/1m/qrels.dl19.tsv",
             ["nDCG@10", "R(rel=2)@1000"], "nDCG@10", "R(rel=2)@1000", MSM),
    "dl20": ("m4/hs-bm25-1m.dl20.trec", "reference/dense-flat-1m.dl20.trec", "data/subset/1m/qrels.dl20.tsv",
             ["nDCG@10", "R(rel=2)@1000"], "nDCG@10", "R(rel=2)@1000", MSM),
}
for _ds in ("scifact", "nfcorpus", "fiqa"):
    DATASETS[f"beir-{_ds}"] = (f"m4/hs-bm25.beir-{_ds}.test.trec", f"reference/dense-flat.beir-{_ds}.test.trec",
                               f"data/raw/beir/{_ds}/qrels/test.trec", ["nDCG@10", "R@100"], "nDCG@10", "R@100", "BEIR (out of domain)")
TUNE = ("m4/hs-bm25-1m.train_tune.trec", "reference/dense-flat-1m.train_tune.trec", "data/subset/train_tune/qrels.tsv")


def metric_value(m, ranked, qrel):
    if m.kind == "RR":
        return mrr_at_k(ranked, qrel, m.k, m.rel)
    if m.kind == "nDCG":
        return ndcg_at_k(ranked, qrel, m.k)
    return recall_at_k(ranked, qrel, m.k, m.rel)


def load(lex_path, dense_path, qrels_path):
    lex = read_run_ranked(RUNS / lex_path)
    dense = read_run_ranked(RUNS / dense_path)
    qrels = read_qrels(REPO / qrels_path)
    return lex, dense, qrels


def sweep(cands, qrels, metric_names):
    """All grid configs via the fast path -> {config: {metric: mean}}."""
    ms = [parse_metric(x) for x in metric_names]
    need = max(m.k for m in ms)  # only the top `need` ids affect these metrics
    configs = [("rrf", k, None) for k in RRF_KS] + [("wsum", a, n) for n in NORMS for a in ALPHAS]
    out = {}
    for kind, p, n in configs:
        vals = {m.name: [] for m in ms}
        for q, qrel in qrels.items():
            qc = cands.get(q)
            if qc is None:
                ranked = []
            else:
                r = rrf_fast(qc, p, need) if kind == "rrf" else weighted_fast(qc, p, n, need)
                ranked = [d for d, _ in r]
            for m in ms:
                vals[m.name].append(metric_value(m, ranked, qrel))
        out[cfg_name(kind, p, n)] = {k: float(np.mean(v)) for k, v in vals.items()}
    return out


def cfg_name(kind, p, n):
    return f"rrf(k={p})" if kind == "rrf" else f"wsum({n},alpha={p})"


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=20_000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--approx-dense", nargs="*", default=[],
                    help="DATASET=RUN (path relative to repo) of an approximate dense run, e.g. dev=results/vector/runs/x.trec")
    ap.add_argument("--approx-label", default="DiskANN")
    ap.add_argument("--out-name", default="m4/fusion", help="results/<out-name>.{json,md}")
    a = ap.parse_args(argv)
    gc.disable()  # millions of small tuples, no reference cycles: cyclic GC passes dominated the runtime
    t0 = time.time()
    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    command = "python scripts/m4_fusion.py " + " ".join(argv if argv is not None else sys.argv[1:])

    # ------------------------------------------------------------- 1. tuning on train_tune only
    lex, dense, qrels = load(*TUNE)
    cands = {q: prepare(lex.get(q, []), dense.get(q, []), DEPTH) for q in qrels}
    grid_tune = sweep(cands, qrels, ["RR@10", "R@1000"])
    del cands
    rrf_grid = {k: grid_tune[cfg_name("rrf", k, None)]["RR@10"] for k in RRF_KS}
    best_k = max(RRF_KS, key=lambda k: (rrf_grid[k], -RRF_KS.index(k)))
    w_cfgs = [(n, al) for n in NORMS for al in ALPHAS]
    best_n, best_alpha = max(w_cfgs, key=lambda c: (grid_tune[cfg_name("wsum", c[1], c[0])]["RR@10"], -w_cfgs.index(c)))
    tune_single = {}
    for name, run in (("bm25", read_run(RUNS / TUNE[0])), ("dense", read_run(RUNS / TUNE[1]))):
        agg, _ = evaluate(run, qrels, ["RR@10", "R@1000"])
        tune_single[name] = agg
    print(f"[tune] best RRF k={best_k} ({rrf_grid[best_k]:.4f}); best weighted {best_n} alpha={best_alpha} "
          f"({grid_tune[cfg_name('wsum', best_alpha, best_n)]['RR@10']:.4f}); {time.time() - t0:.0f}s", flush=True)

    systems = {
        "rrf_tuned": ("rrf", best_k, None),
        "rrf_k60": ("rrf", 60, None),
        "wsum_tuned": ("wsum", best_alpha, best_n),
    }
    duplicate_rrf = best_k == 60

    def fuser(kind, p, n):
        if kind == "rrf":
            return lambda l_, d_: rrf([l_, d_], k=p, depth=DEPTH, out_k=OUT_K)
        return lambda l_, d_: weighted(l_, d_, alpha=p, norm=n, depth=DEPTH, out_k=OUT_K)

    # ------------------------------------------------------------- 2. held-out evaluation
    results, per_query, grids, sanity = {}, {}, {}, {}
    for ds, (lp, dp, qp, metrics, primary, recall_m, group) in DATASETS.items():
        lex, dense, qrels = load(lp, dp, qp)
        res, pq = {}, {}
        for name, path in (("bm25", lp), ("dense", dp)):
            agg, per = evaluate(read_run(RUNS / path), qrels, metrics)
            res[name], pq[name] = agg, per
        qids = list(qrels)
        cands = {q: prepare(lex.get(q, []), dense.get(q, []), DEPTH) for q in qids}
        mismatches = 0
        for name, (kind, p, n) in systems.items():
            fused = fuse_runs(lex, dense, fuser(kind, p, n), qids=qids)
            # cross-check the sweep path against the reference implementation on every query
            for q in qids:
                fast = rrf_fast(cands[q], p, OUT_K) if kind == "rrf" else weighted_fast(cands[q], p, n, OUT_K)
                mismatches += fast != fused[q]
            scores = write_run_exact(OUT_RUNS / f"{name}.{ds}.trec", fused, f"m4-{name}")
            agg, per = evaluate(scores, qrels, metrics)
            res[name], pq[name] = agg, per
        if mismatches:
            raise AssertionError(f"{ds}: fast path disagrees with core on {mismatches} query-configs")
        grids[ds] = sweep(cands, qrels, [primary])
        del cands
        union = [len({d for d, _ in lex.get(q, [])[:DEPTH]} | {d for d, _ in dense.get(q, [])[:DEPTH]}) for q in qids]
        sanity[ds] = {"n_queries": len(qids), "queries_missing_from_bm25": sum(1 for q in qids if not lex.get(q)),
                      "queries_missing_from_dense": sum(1 for q in qids if not dense.get(q)),
                      "mean_candidate_union": float(np.mean(union)), "fast_vs_core_mismatches": mismatches}
        results[ds], per_query[ds] = res, pq
        print(f"[eval] {ds}: " + ", ".join(f"{s} {res[s][primary]:.4f}" for s in res) + f"; {time.time() - t0:.0f}s", flush=True)

    # Anserini cross-check of the BM25 input on MS MARCO (same 1M subset)
    for ds in ("dev", "dl19", "dl20"):
        _, _, qp, metrics, *_ = DATASETS[ds]
        agg, _ = evaluate(read_run(RUNS / f"reference/anserini-bm25-default-1m.{ds}.trec"), read_qrels(REPO / qp), metrics)
        sanity[ds]["anserini_bm25_1m"] = agg

    # ------------------------------------------------------------- 3. significance
    hybrids = ["rrf_tuned", "wsum_tuned"] + ([] if duplicate_rrf else ["rrf_k60"])
    families = {}
    for fam, idx in (("primary", 4), ("recall", 5)):
        fam_in, meta = {}, {}
        for ds, spec in DATASETS.items():
            m = spec[idx]
            for h in hybrids:
                for single in ("bm25", "dense"):
                    key = f"{ds} | {h} vs {single} | {m}"
                    fam_in[key] = (per_query[ds][h][m], per_query[ds][single][m])
                    meta[key] = (ds, h, single, m)
        comps = compare(fam_in, metric=fam, n_perm=a.n_perm, n_boot=a.n_boot, alpha=0.05, seed=SEED)
        families[fam] = [dict(c.to_dict(), dataset=meta[c.name][0], hybrid=meta[c.name][1],
                              baseline=meta[c.name][2], metric=meta[c.name][3],
                              holm_p_lt_0_01=bool(c.p_holm < 0.01), holm_p_lt_0_05=bool(c.p_holm < 0.05))
                         for c in comps]
        print(f"[sig] {fam}: {len(comps)} comparisons; {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------- 4. approximate dense (optional)
    approx = {}
    for spec in a.approx_dense:
        ds, path = spec.split("=", 1)
        lp, dp, qp, metrics, primary, recall_m, group = DATASETS[ds]
        lex, dense_exact, qrels = load(lp, dp, qp)
        dense_ap = read_run_ranked(REPO / path)
        depth_ap = int(np.median([len(v) for v in dense_ap.values()])) if dense_ap else 0
        qids = list(qrels)
        entry = {"run": path, "median_depth": depth_ap,
                 "queries_in_run": len(dense_ap), "queries_in_qrels": len(qids)}
        agg, per_ap = evaluate(read_run(REPO / path), qrels, metrics)
        entry["dense_approx"] = agg
        entry["dense_exact"] = results[ds]["dense"]
        per_query_approx = {}
        fam_in = {f"{ds} | dense {a.approx_label} vs exact | {primary}": (per_ap[primary], per_query[ds]["dense"][primary])}
        for name, (kind, p, n) in systems.items():
            fused = fuse_runs(lex, dense_ap, fuser(kind, p, n), qids=qids)
            scores = write_run_exact(OUT_RUNS / f"{name}-{a.approx_label.lower()}.{ds}.trec", fused, f"m4-{name}-approx")
            agg, per = evaluate(scores, qrels, metrics)
            entry[f"{name}_approx"] = agg
            entry[f"{name}_exact"] = results[ds][name]
            fam_in[f"{ds} | {name}: {a.approx_label} vs exact dense | {primary}"] = (per[primary], per_query[ds][name][primary])
            per_query_approx[name] = per[primary]
        # depth-matched control: exact dense cut to the approximate run's depth, so the approximation
        # error is separated from the shallower dense list (BM25 stays at depth 1000 in both)
        dense_cut = {q: r[:depth_ap] for q, r in dense_exact.items()}
        for name, (kind, p, n) in systems.items():
            fused = fuse_runs(lex, dense_cut, fuser(kind, p, n), qids=qids)
            agg, per = evaluate(write_run_exact(OUT_RUNS / f"{name}-exactcut{depth_ap}.{ds}.trec", fused, f"m4-{name}-exactcut"),
                                qrels, metrics)
            entry[f"{name}_exact_cut"] = agg
            fam_in[f"{ds} | {name}: {a.approx_label} vs exact dense cut to depth {depth_ap} | {primary}"] = (
                per_query_approx[name], per[primary])
        entry["significance"] = [c.to_dict() for c in compare(fam_in, metric=primary, n_perm=a.n_perm, n_boot=a.n_boot, seed=SEED)]
        approx[ds] = entry
        print(f"[approx] {ds}: done; {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------- 5. write results
    payload = {
        "protocol": {
            "tuning_queries": "data/subset/train_tune (2,000 seeded MS MARCO *train* queries, 1M subset)",
            "tuning_metric": "MRR@10 (RR@10) on train_tune",
            "never_tuned_on": ["dev", "dl19", "dl20", "beir-scifact", "beir-nfcorpus", "beir-fiqa"],
            "grid": {"rrf_k": RRF_KS, "alpha_on_dense": ALPHAS, "normalisation": NORMS},
            "grid_tie_break": "first in grid order",
            "candidate_depth": DEPTH, "fused_list_length": OUT_K,
            "missing_doc_rule": "a doc absent from one retriever's top-depth list gets that list's minimum normalised score (RRF: no contribution)",
            "fused_tie_break": "lower doc id",
            "bm25": "HybridSearch engine, hs_lex_search --algo exhaustive --model lucene (k1=0.9 b=0.4), top-1000",
            "dense": "exact inner-product search (FAISS flat) over Waterloo's precomputed BGE-base-en-v1.5 vectors (MS MARCO); BEIR encoded locally with BGE-base on MPS",
            "significance": {"test": "paired randomization (sign flip), two-sided", "n_perm": a.n_perm,
                             "bootstrap": "paired percentile bootstrap 95% CI", "n_boot": a.n_boot, "seed": SEED,
                             "correction": "Holm-Bonferroni within each family"},
        },
        "tuning": {"grid_train_tune": grid_tune, "single_retrievers_train_tune": tune_single,
                   "chosen": {"rrf_k": best_k, "wsum_norm": best_n, "wsum_alpha": best_alpha}},
        "results": results,
        "diagnostic_grid_heldout_primary_metric_not_used_for_selection": grids,
        "significance": families,
        "sanity": sanity,
        "approx_dense": approx or None,
        "rrf_tuned_equals_k60": duplicate_rrf,
    }
    write_result(f"{a.out_name}.json", payload, command)
    write_result(f"{a.out_name}.md", render_md(payload, a), command)
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


def render_md(P, a) -> str:
    ch = P["tuning"]["chosen"]
    R = P["results"]
    L = []
    L.append("# M4: hybrid retrieval (BM25 + dense), tuned on train queries only\n")
    L.append("Generated by `scripts/m4_all.sh` (driver `scripts/m4_fusion.py`). Every number below is read from "
             "`results/m4/fusion.json`, which the same run wrote.\n")
    L.append(headline(P))
    L.append("## Tuning protocol (no tuning on dev, DL19, DL20 or BEIR)\n")
    L.append("- Fusion parameters are chosen **only** on `data/subset/train_tune`: 2,000 seeded MS MARCO *train* queries "
             "whose relevant passage is in the 1M subset. Selection metric: MRR@10 on those queries.")
    L.append("- **Nothing was tuned on dev, DL19, DL20, or any BEIR set.** Those are used once, for evaluation, with the "
             "parameters fixed from train_tune. BEIR gets the MS MARCO-tuned parameters unchanged (a transfer test).")
    L.append(f"- Grid: RRF k ∈ {RRF_KS}; weighted fusion alpha (weight on dense) ∈ {ALPHAS} × normalisation ∈ {NORMS}. "
             "Ties in the grid go to the first config in grid order.")
    L.append(f"- Candidate depth {DEPTH} per retriever; fused list length {OUT_K}. A document missing from one retriever's "
             "list gets that list's minimum normalised score (0 for min-max); RRF gives it no contribution. Fused ties break by lower doc id.")
    L.append(f"- **Chosen:** RRF k = **{ch['rrf_k']}**; weighted = **{ch['wsum_norm']}, alpha = {ch['wsum_alpha']}**. "
             "The untuned RRF default k = 60 is reported too" + (" (it is the tuned value, so the two rows coincide)." if P["rrf_tuned_equals_k60"] else ".") + "\n")
    g = P["tuning"]["grid_train_tune"]
    ts = P["tuning"]["single_retrievers_train_tune"]
    L.append(f"train_tune MRR@10: BM25 {fmt(ts['bm25']['RR@10'])}, dense {fmt(ts['dense']['RR@10'])}.\n")
    L.append("| RRF k | " + " | ".join(str(k) for k in RRF_KS) + " |")
    L.append("|---|" + "---|" * len(RRF_KS))
    L.append("| train_tune MRR@10 | " + " | ".join(fmt(g[f'rrf(k={k})']['RR@10']) for k in RRF_KS) + " |\n")
    L.append("| alpha (dense weight) | " + " | ".join(str(x) for x in ALPHAS) + " |")
    L.append("|---|" + "---|" * len(ALPHAS))
    for n in NORMS:
        L.append(f"| {n} MRR@10 | " + " | ".join(fmt(g[f'wsum({n},alpha={x})']['RR@10']) for x in ALPHAS) + " |")
    L.append("")

    names = {"bm25": "BM25 (HybridSearch engine)", "dense": "Dense (BGE-base, exact flat)",
             "rrf_tuned": f"RRF, tuned (k={ch['rrf_k']})", "rrf_k60": "RRF, default (k=60)",
             "wsum_tuned": f"Weighted, tuned ({ch['wsum_norm']}, α={ch['wsum_alpha']})"}
    order = ["bm25", "dense", "rrf_tuned", "rrf_k60", "wsum_tuned"]
    L.append("## Results\n")
    L.append("### MS MARCO, 1M-passage subset (in-domain for BGE)\n")
    L.append("The subset keeps every passage judged relevant for dev/DL19/DL20 plus a seeded uniform sample of the rest, "
             "so absolute numbers are far higher than on the full 8.8M collection (BM25 dev MRR@10 is 0.184 there). "
             "Compare systems within a table, not against published full-collection numbers.\n")
    L.append("| system | dev MRR@10 | dev R@100 | dev R@1000 | DL19 nDCG@10 | DL19 R@1000 (rel≥2) | DL20 nDCG@10 | DL20 R@1000 (rel≥2) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in order:
        d, d19, d20 = R["dev"][s], R["dl19"][s], R["dl20"][s]
        L.append(f"| {names[s]} | {fmt(d['RR@10'])} | {fmt(d['R@100'])} | {fmt(d['R@1000'])} | {fmt(d19['nDCG@10'])} | "
                 f"{fmt(d19['R(rel=2)@1000'])} | {fmt(d20['nDCG@10'])} | {fmt(d20['R(rel=2)@1000'])} |")
    L.append(f"\nQueries: dev {P['sanity']['dev']['n_queries']}, DL19 {P['sanity']['dl19']['n_queries']}, DL20 {P['sanity']['dl20']['n_queries']}.\n")
    L.append("### BEIR (out of domain for a model fine-tuned on MS MARCO)\n")
    L.append("Honesty rule 5: BGE-base was trained with MS MARCO among its data, so the MS MARCO numbers above flatter the "
             "dense retriever. BEIR is the out-of-domain check. (We make no claim about which BEIR training splits, if any, "
             "were in BGE's training mix; only the test queries/qrels are used here.) BEIR BM25 is HybridSearch's engine on "
             "title + text, the same passage text the dense encoder saw.\n")
    L.append("| system | scifact nDCG@10 | scifact R@100 | nfcorpus nDCG@10 | nfcorpus R@100 | fiqa nDCG@10 | fiqa R@100 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in order:
        row = [names[s]]
        for ds in ("beir-scifact", "beir-nfcorpus", "beir-fiqa"):
            row += [fmt(R[ds][s]["nDCG@10"]), fmt(R[ds][s]["R@100"])]
        L.append("| " + " | ".join(row) + " |")
    L.append("\nQueries: " + ", ".join(f"{ds[5:]} {P['sanity'][ds]['n_queries']}" for ds in ("beir-scifact", "beir-nfcorpus", "beir-fiqa"))
             + f". nfcorpus: {P['sanity']['beir-nfcorpus']['queries_missing_from_bm25']} test queries get no BM25 hit at all "
             "(every query term is out of the corpus vocabulary); they score 0 for BM25 and fusion falls back to the dense list.\n")

    for fam, title in (("primary", "Primary metric (dev MRR@10; DL19/DL20/BEIR nDCG@10)"),
                       ("recall", "Recall (MS MARCO R@1000, rel≥2 on DL; BEIR R@100)")):
        comps = P["significance"][fam]
        L.append(f"## Significance family: {title}\n")
        L.append(f"Family = every (hybrid system × single retriever × dataset) pair on this metric: {len(comps)} comparisons, "
                 f"Holm–Bonferroni across all {len(comps)}. Paired randomization test, two-sided, {P['protocol']['significance']['n_perm']:,} "
                 f"sign-flip permutations, seed {SEED}; Δ = hybrid − baseline with a paired bootstrap 95% CI "
                 f"({P['protocol']['significance']['n_boot']:,} resamples). The smallest attainable raw p is 1/(n_perm+1) = "
                 f"{1 / (P['protocol']['significance']['n_perm'] + 1):.1e}.\n")
        L.append("| dataset | hybrid | vs | metric | hybrid | baseline | Δ | 95% CI | raw p | Holm p | p<0.05 | p<0.01 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in comps:
            L.append(f"| {c['dataset']} | {c['hybrid']} | {c['baseline']} | {c['metric']} | {fmt(c['mean_a'])} | {fmt(c['mean_b'])} | "
                     f"{c['delta']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {fmt_p(c['p_value'])} | {fmt_p(c['p_holm'])} | "
                     f"{'yes' if c['holm_p_lt_0_05'] else 'no'} | {'yes' if c['holm_p_lt_0_01'] else 'no'} |")
        L.append("")
        L.append(summarise(comps))
        L.append("")

    L.append("## Diagnostic: the whole grid on the held-out sets (not used for selection)\n")
    L.append("Shown only to measure how far the train-chosen parameters are from the best held-out value in hindsight. "
             "Selection used train_tune alone.\n")
    L.append("| dataset (metric) | chosen RRF | best RRF in hindsight | chosen weighted | best weighted in hindsight |")
    L.append("|---|---|---|---|---|")
    for ds, grid in P["diagnostic_grid_heldout_primary_metric_not_used_for_selection"].items():
        m = DATASETS[ds][4]
        rr = {k: v[m] for k, v in grid.items() if k.startswith("rrf")}
        ww = {k: v[m] for k, v in grid.items() if k.startswith("wsum")}
        br, bw = max(rr, key=rr.get), max(ww, key=ww.get)
        cr, cw = f"rrf(k={ch['rrf_k']})", f"wsum({ch['wsum_norm']},alpha={ch['wsum_alpha']})"
        L.append(f"| {ds} ({m}) | {fmt(rr[cr])} | {fmt(rr[br])} `{br}` | {fmt(ww[cw])} | {fmt(ww[bw])} `{bw}` |")
    L.append("")

    L.append("## Approximate dense (DiskANN-style index) in the hybrid\n")
    if P["approx_dense"]:
        for ds, e in P["approx_dense"].items():
            m = DATASETS[ds][4]
            L.append(f"Run: `{e['run']}` ({e['queries_in_run']} queries, median depth {e['median_depth']}).\n")
            dp = e["median_depth"]
            L.append(f"The approximate run returns top-{dp} per query, so each hybrid is also fused with exact dense cut to "
                     f"top-{dp} (BM25 stays at depth {DEPTH}); the difference between those two columns is the approximation "
                     "error alone, the rest is the shallower dense list. Fusion parameters are the train_tune-tuned ones above.\n")
            L.append(f"| system | exact dense (depth {DEPTH}) {m} | exact dense cut to {dp} | {a.approx_label} (depth {dp}) | Δ {a.approx_label} − exact | Δ {a.approx_label} − exact cut |")
            L.append("|---|---|---|---|---|---|")
            L.append(f"| dense alone | {fmt(e['dense_exact'][m])} | {fmt(e['dense_exact'][m])} | {fmt(e['dense_approx'][m])} | "
                     f"{e['dense_approx'][m] - e['dense_exact'][m]:+.4f} | n/a |")
            for s in ("rrf_tuned", "rrf_k60", "wsum_tuned"):
                L.append(f"| {names[s]} | {fmt(e[s + '_exact'][m])} | {fmt(e[s + '_exact_cut'][m])} | {fmt(e[s + '_approx'][m])} | "
                         f"{e[s + '_approx'][m] - e[s + '_exact'][m]:+.4f} | {e[s + '_approx'][m] - e[s + '_exact_cut'][m]:+.4f} |")
            L.append("\n| comparison | Δ | 95% CI | raw p | Holm p (this family) |")
            L.append("|---|---|---|---|---|")
            for c in e["significance"]:
                L.append(f"| {c['name'].replace(' | ', ' · ')} | {c['delta']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {fmt_p(c['p_value'])} | {fmt_p(c['p_holm'])} |")
            L.append("")
    else:
        L.append("No approximate (DiskANN) dense run files existed when this was generated, so the hybrid uses exact "
                 "dense search only. Re-run `scripts/m4_all.sh` once they exist (the script picks them up).\n")

    L.append("## Sanity checks\n")
    for ds in ("dev", "dl19", "dl20"):
        s = P["sanity"][ds]
        m = DATASETS[ds][4]
        L.append(f"- {ds}: BM25 from HybridSearch's engine {fmt(R[ds]['bm25'][m])} vs Anserini's own 1M-subset run "
                 f"{fmt(s['anserini_bm25_1m'][m])} ({m}).")
    L.append("- Every fused run was produced twice, by the reference implementation (`fusion/core.py`) and the vectorised "
             "sweep path (`fusion/fast.py`); rankings agree on every query (mismatches: "
             + ", ".join(f"{ds} {P['sanity'][ds]['fast_vs_core_mismatches']}" for ds in P["sanity"]) + ").")
    L.append("- Fused run files are written so that an evaluator re-sorting by score (trec_eval's rule) reproduces the fused "
             "order exactly; `write_run_exact` checks this for every query.")
    return "\n".join(L) + "\n"


def headline(P) -> str:
    """Computed summary: for each dataset, each hybrid vs the stronger single retriever."""
    ch = P["tuning"]["chosen"]
    comps = {(c["dataset"], c["hybrid"], c["baseline"]): c for c in P["significance"]["primary"]}
    L = ["## Headline\n"]
    L.append("Hybrid vs the **stronger** single retriever on each dataset's primary metric (Holm-adjusted p over the "
             f"36-comparison primary family). Weighted = {ch['wsum_norm']}, α={ch['wsum_alpha']}; RRF k={ch['rrf_k']} (tuned) / 60.\n")
    L.append("| dataset | metric | BM25 | dense | weighted (tuned) | Δ vs best single, Holm p | RRF tuned | Δ, Holm p | RRF k=60 | Δ, Holm p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds, spec in DATASETS.items():
        m = spec[4]
        r = P["results"][ds]
        best = "dense" if r["dense"][m] >= r["bm25"][m] else "bm25"
        row = [ds, m, fmt(r["bm25"][m]), fmt(r["dense"][m])]
        for h in ("wsum_tuned", "rrf_tuned", "rrf_k60"):
            c = comps.get((ds, h, best))
            if c is None:  # rrf_k60 folded into rrf_tuned when k=60 was chosen
                c = comps[(ds, "rrf_tuned", best)]
            row += [fmt(r[h][m]), f"{c['delta']:+.4f}, {fmt_p(c['p_holm'])}"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    if ch["wsum_alpha"] == max(ALPHAS) or ch["wsum_alpha"] == min(ALPHAS):
        L.append(f"Note: the tuned α = {ch['wsum_alpha']} sits at the edge of the pre-registered grid {ALPHAS}. The grid was "
                 "not widened after seeing held-out results.\n")
    if ch["rrf_k"] == min(RRF_KS) or ch["rrf_k"] == max(RRF_KS):
        L.append(f"Note: the tuned RRF k = {ch['rrf_k']} sits at the edge of the pre-registered grid {RRF_KS}.\n")
    return "\n".join(L)


def summarise(comps) -> str:
    out = []
    for thr, key in (("0.01", "holm_p_lt_0_01"), ("0.05", "holm_p_lt_0_05")):
        up = [c for c in comps if c[key] and c["delta"] > 0]
        down = [c for c in comps if c[key] and c["delta"] < 0]
        out.append(f"- After Holm, at p<{thr}: **{len(up)}** significant lifts, **{len(down)}** significant drops, "
                   f"{len(comps) - len(up) - len(down)} not significant.")
    ns = [c for c in comps if not c["holm_p_lt_0_05"]]
    if ns:
        out.append("- Not significant at p<0.05 after correction: " + "; ".join(
            f"{c['dataset']} {c['hybrid']} vs {c['baseline']} (Δ {c['delta']:+.4f})" for c in ns) + ".")
    dn = [c for c in comps if c["holm_p_lt_0_05"] and c["delta"] < 0]
    if dn:
        out.append("- Hybrid is significantly **worse** than: " + "; ".join(
            f"{c['dataset']} {c['hybrid']} vs {c['baseline']} (Δ {c['delta']:+.4f})" for c in dn) + ".")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
