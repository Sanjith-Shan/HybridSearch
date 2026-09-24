"""BM25 parity against Anserini: MRR@10, per-query top-10 overlap, disagreements bucketed by cause.

Inputs are our raw runs (hs_lex_search --raw-out: qid, docid, rank, score, float bits) for the
Lucene model and the textbook model, Anserini's run (scores rounded to 1e-4 by its
ScoreTiesAdjusterReranker), and optionally the term dictionaries and norms of both indexes
(hs_lex_dump --terms, tools/lucene_ref indexstats) to attribute score differences.

Buckets, per query, comparing top-10 lists:
  identical            same docids in the same order
  tie_order            same set; every order difference is between docs whose scores agree
                       to Anserini's printed precision (1e-4): Lucene/Anserini break exact ties
                       by docid *string*, we by numeric id
  tie_cutoff           sets differ only by docs tied (to 1e-4) with the 10th-ranked score
  analyzer_vocab       a query term's df (or presence) differs between the two indexes
  analyzer_length      a document in either top-10 has a different Lucene norm byte
  length_quantisation  (textbook model only) the Lucene-model run agrees with Anserini here,
                       so the difference comes from using exact lengths
  unexplained          none of the above

Usage:
  python -m hybridsearch.lexical.parity --lucene RAW --textbook RAW --anserini RUN --qrels QRELS
      --queries TSV [--ours-terms TSV --anserini-terms TSV --ours-index DIR --anserini-norms BIN]
      --out-md results/lexical/parity.md --out-json results/lexical/parity.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..data.io import read_qrels
from ..eval.metrics import evaluate


def read_raw(path: str) -> dict[str, list[tuple[str, float]]]:
    runs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, doc, _rank, score, _bits = line.rstrip("\n").split("\t")
            runs[qid].append((doc, float(score)))
    return runs


def read_trec(path: str) -> dict[str, list[tuple[str, float]]]:
    runs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, _q0, doc, rank, score, _tag = line.split()
            runs[qid].append((doc, float(score)))
    for q in runs:  # file order is rank order, but be safe
        runs[q].sort(key=lambda x: -x[1])
    return runs


def base4(x: float) -> float:
    """Anserini prints round(score, 4) minus 1e-6 per tie; recover the rounded score."""
    return round(x, 4)


def as_run(r: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
    # Rank-preserving scores so evaluation sees exactly our order.
    return {q: {d: float(len(v) - i) for i, (d, _) in enumerate(v)} for q, v in r.items()}


def classify(ours: list[tuple[str, float]], ref: list[tuple[str, float]], k: int = 10) -> str:
    a, b = ours[:k], ref[:k]
    if [d for d, _ in a] == [d for d, _ in b]:
        return "identical"
    sa = {d: base4(s) for d, s in a}
    sb = {d: base4(s) for d, s in b}
    if set(sa) == set(sb):
        # same set: every doc must sit in a group of equal (rounded) scores with the same members
        ok = all(abs(sa[d] - sb[d]) <= 1.01e-4 for d in sa)
        return "tie_order" if ok else "score"
    cut_a = base4(a[-1][1]) if a else None
    cut_b = base4(b[-1][1]) if b else None
    only = (set(sa) - set(sb)) | (set(sb) - set(sa))
    common_ok = all(abs(sa[d] - sb[d]) <= 1.01e-4 for d in set(sa) & set(sb))
    tied = all(
        (d in sa and cut_a is not None and abs(sa[d] - cut_a) <= 1.01e-4)
        or (d in sb and cut_b is not None and abs(sb[d] - cut_b) <= 1.01e-4)
        for d in only
    )
    if common_ok and tied and cut_a is not None and cut_b is not None and abs(cut_a - cut_b) <= 1.01e-4:
        return "tie_cutoff"
    return "score"


def load_terms(path: str) -> dict[str, int]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            t, df, _cf = line.rstrip("\n").split("\t")
            out[t] = int(df)
    return out


def analyze(queries: dict[str, str], engine_bin: str) -> dict[str, list[str]]:
    lines = "".join(f"{q}\t{t}\n" for q, t in queries.items())
    out = subprocess.run([engine_bin], input=lines.encode(), capture_output=True, check=True).stdout.decode()
    res = {}
    for line in out.splitlines():
        qid, toks, _raw = line.split("\t")
        res[qid] = [t for t in toks.split("\x01") if t]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lucene", required=True)
    ap.add_argument("--textbook", required=True)
    ap.add_argument("--anserini", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--ours-terms")
    ap.add_argument("--anserini-terms")
    ap.add_argument("--ours-index")
    ap.add_argument("--anserini-norms")
    ap.add_argument("--analyze-bin", default="engine/build-lex/hs_lex_analyze")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    runs = {"lucene": read_raw(args.lucene), "textbook": read_raw(args.textbook)}
    ans = read_trec(args.anserini)
    qrels = read_qrels(args.qrels)
    queries = {}
    with open(args.queries) as f:
        for line in f:
            q, t = line.rstrip("\n").split("\t", 1)
            queries[q] = t

    metrics = {}
    for name, r in [("anserini", ans), *runs.items()]:
        agg, _ = evaluate(as_run(r), qrels, ["RR@10", "R@1000"])
        metrics[name] = agg

    have_diag = all([args.ours_terms, args.anserini_terms, args.ours_index, args.anserini_norms])
    if have_diag:
        ours_df = load_terms(args.ours_terms)
        ans_df = load_terms(args.anserini_terms)
        qterms = analyze(queries, args.analyze_bin)
        docids = np.fromfile(Path(args.ours_index) / "docids.u64bin", dtype="<u8")[1:]
        our_norms = np.fromfile(Path(args.ours_index) / "norms.u8", dtype=np.uint8)
        ans_norms = np.fromfile(args.anserini_norms, dtype=np.uint8)
        norm_by_pid = np.zeros(int(docids.max()) + 1, dtype=np.int16) - 1
        norm_by_pid[docids] = our_norms

    report = {}
    for name, r in runs.items():
        buckets = Counter()
        overlaps = []
        examples = defaultdict(list)
        for q in qrels:
            a, b = r.get(q, []), ans.get(q, [])
            overlaps.append(len({d for d, _ in a[:10]} & {d for d, _ in b[:10]}))
            c = classify(a, b)
            if c == "score":
                if name == "textbook" and classify(runs["lucene"].get(q, []), b) != "score":
                    c = "length_quantisation"
                elif have_diag:
                    if any(ours_df.get(t) != ans_df.get(t) for t in qterms.get(q, [])):
                        c = "analyzer_vocab"
                    elif any(
                        int(norm_by_pid[int(d)]) != int(ans_norms[int(d)])
                        for d in {d for d, _ in a[:10]} | {d for d, _ in b[:10]}
                    ):
                        c = "analyzer_length"
                    else:
                        c = "unexplained"
                else:
                    c = "unexplained"
            buckets[c] += 1
            if c not in ("identical",) and len(examples[c]) < 5:
                examples[c].append(q)
        ov = np.array(overlaps)
        report[name] = {
            "queries": len(overlaps),
            "buckets": dict(buckets),
            "top10_overlap_mean": float(ov.mean()),
            "top10_overlap_hist": {int(k): int(v) for k, v in sorted(Counter(overlaps).items())},
            "examples": dict(examples),
        }

    diag = {}
    if have_diag:
        diag = {
            "terms_ours": len(ours_df),
            "terms_anserini": len(ans_df),
            "terms_only_ours": sum(1 for t in ours_df if t not in ans_df),
            "terms_only_anserini": sum(1 for t in ans_df if t not in ours_df),
            "terms_df_differs": sum(1 for t, v in ours_df.items() if t in ans_df and ans_df[t] != v),
            "norms_compared": int(len(docids)),
            "norms_differ": int(np.sum(norm_by_pid[docids] != ans_norms[docids])),
        }
    out = {"metrics": metrics, "comparison_vs_anserini": report, "index_diagnostics": diag}
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    order = ["identical", "tie_order", "tie_cutoff", "length_quantisation", "analyzer_vocab", "analyzer_length", "unexplained"]
    md = ["| run | MRR@10 | R@1000 | " + " | ".join(order) + " | mean top-10 overlap |",
          "|---|---|---|" + "---|" * len(order) + "---|"]
    md.append(f"| Anserini (reference run) | {metrics['anserini']['RR@10']:.4f} | {metrics['anserini']['R@1000']:.4f} | "
              + " | ".join("" for _ in order) + " | |")
    for name in runs:
        b = report[name]["buckets"]
        md.append(f"| ours, {name} BM25 | {metrics[name]['RR@10']:.4f} | {metrics[name]['R@1000']:.4f} | "
                  + " | ".join(str(b.get(o, 0)) for o in order) + f" | {report[name]['top10_overlap_mean']:.3f} |")
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
