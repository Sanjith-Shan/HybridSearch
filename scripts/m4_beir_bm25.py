"""M4: BM25 runs on BEIR (scifact, nfcorpus, fiqa) with HybridSearch's own lexical engine.

For each dataset:
  1. corpus.jsonl -> data/indexes/lexical/beir-<ds>/input.tsv  as "int_id \\t title + ' ' + text"
     (the same passage text the dense encoder saw: f"{title} {text}".strip(), see
     py/hybridsearch/dense/beir.py). int_id = 0..n-1 in corpus-file order; the string id map is
     written to data/indexes/lexical/beir-<ds>/docids.map.tsv.
  2. hs_index_build --input TSV --out data/indexes/lexical/beir-<ds> --threads T --no-docstore
  3. test queries (those in qrels/test.tsv) -> TSV; hs_lex_search --algo exhaustive --model lucene --k 1000
  4. map int ids back to BEIR string ids -> data/runs/m4/hs-bm25.beir-<ds>.test.trec

Tabs/newlines inside texts are replaced by spaces (the TSV format cannot carry them).
Usage: .venv/bin/python scripts/m4_beir_bm25.py [--threads 7] [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BEIR = REPO / "data/raw/beir"
ENGINE = REPO / "engine/build-lex"
RUNS = REPO / "data/runs/m4"
DATASETS = ("scifact", "nfcorpus", "fiqa")


def clean(s: str) -> str:
    return " ".join(s.replace("\t", " ").replace("\r", " ").replace("\n", " ").split())


def build(ds: str, threads: int, force: bool) -> Path:
    idx = REPO / f"data/indexes/lexical/beir-{ds}"
    run_out = RUNS / f"hs-bm25.beir-{ds}.test.trec"
    if run_out.exists() and run_out.stat().st_size > 0 and not force:
        print(f"exists: {run_out}")
        return run_out
    work = RUNS / f"_beir_{ds}"
    work.mkdir(parents=True, exist_ok=True)
    tsv = work / "input.tsv"
    ids: list[str] = []
    with open(BEIR / ds / "corpus.jsonl", encoding="utf-8") as fin, open(tsv, "w", encoding="utf-8") as fout:
        for line in fin:
            o = json.loads(line)
            text = clean(f"{o.get('title', '') or ''} {o.get('text', '') or ''}".strip())
            fout.write(f"{len(ids)}\t{text}\n")
            ids.append(str(o["_id"]))
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{ds}: duplicate corpus ids")
    queries = {}
    with open(BEIR / ds / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            queries[str(o["_id"])] = o["text"]
    test_qids = []
    with open(BEIR / ds / "qrels/test.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            q = line.split("\t", 1)[0]
            if q not in test_qids:
                test_qids.append(q)
    missing = [q for q in test_qids if q not in queries]
    if missing:
        raise SystemExit(f"{ds}: {len(missing)} test qids without query text")
    qtsv = work / "queries.test.tsv"
    with open(qtsv, "w", encoding="utf-8") as f:
        f.writelines(f"{q}\t{clean(queries[q])}\n" for q in test_qids)

    if force and idx.exists():
        shutil.rmtree(idx)
    if not (idx / "meta.txt").exists():
        subprocess.run([str(ENGINE / "hs_index_build"), "--input", str(tsv), "--out", str(idx),
                        "--threads", str(threads), "--no-docstore", "--min-free-gb", "6", "--quiet"], check=True)
    (idx / "docids.map.tsv").write_text("".join(f"{i}\t{s}\n" for i, s in enumerate(ids)), encoding="utf-8")

    raw_run = work / "run.int.trec"
    subprocess.run([str(ENGINE / "hs_lex_search"), "--index", str(idx), "--queries", str(qtsv), "--out", str(raw_run),
                    "--algo", "exhaustive", "--model", "lucene", "--k", "1000", "--threads", str(threads),
                    "--tag", "hs-bm25"], check=True)
    tmp = str(run_out) + ".tmp"
    with open(raw_run, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            qid, q0, did, rank, score, tag = line.split()
            fout.write(f"{qid} {q0} {ids[int(did)]} {rank} {score} {tag}\n")
    Path(tmp).replace(run_out)
    shutil.rmtree(work)
    print(f"wrote {run_out} ({len(test_qids)} queries, {len(ids)} docs)")
    return run_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets:
        build(ds, a.threads, a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
