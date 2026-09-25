"""M0 data preparation: verify downloads against published MD5s and lay them out as
docs/ARCHITECTURE.md specifies.

    python -m hybridsearch.data.prepare            # verify + lay out MS MARCO and TREC DL

Downloads are expected in data/downloads/ (see scripts/data_download.sh). The large
tarball is deleted after successful extraction to save disk.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .io import DATA, read_qrels, write_queries

# url, filename, expected md5, source of the expected md5
MANIFEST = [
    ("https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
     "collectionandqueries.tar.gz", "31644046b18952c1386cd4564ba2ae69", "ir_datasets downloads.json; Anserini docs"),
    ("https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
     "msmarco-test2019-queries.tsv.gz", "eda71eccbe4d251af83150abe065368c", "ir_datasets downloads.json; server Content-MD5"),
    ("https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
     "msmarco-test2020-queries.tsv.gz", "00a406fb0d14ed3752d70d1e4eb98600", "ir_datasets downloads.json; server Content-MD5"),
    ("https://trec.nist.gov/data/deep/2019qrels-pass.txt",
     "2019qrels-pass.txt", "2f4be390198da108f6845c822e5ada14", "ir_datasets downloads.json"),
    ("https://trec.nist.gov/data/deep/2020qrels-pass.txt",
     "2020qrels-pass.txt", "0355ccee7509ac0463e8278186cdd8d1", "ir_datasets downloads.json"),
]

DL = DATA / "downloads"
MS = DATA / "raw" / "msmarco"
TDL = DATA / "raw" / "trec-dl"

EXPECTED_COUNTS = {
    "collection.tsv": 8_841_823,
    "queries.dev.small.tsv": 6_980,
}


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            n += b.count(b"\n")
    return n


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def require_free(path: Path, need_gb: float, headroom_gb: float = 3.0) -> None:
    free = free_gb(path)
    if free - need_gb < headroom_gb:
        raise SystemExit(f"refusing: {free:.1f} GB free, step needs {need_gb:.1f} GB + {headroom_gb} GB headroom")


def main() -> int:
    MS.mkdir(parents=True, exist_ok=True)
    TDL.mkdir(parents=True, exist_ok=True)
    record = {"files": [], "extracted": {}}
    verified_marker = DL / "collectionandqueries.verified"
    for url, name, md5, src in MANIFEST:
        p = DL / name
        if name == "collectionandqueries.tar.gz" and not p.exists() and verified_marker.exists():
            got = verified_marker.read_text().strip()
        else:
            got = md5sum(p)
        ok = got == md5
        print(f"{name}: md5 {got} expected {md5} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            return 1
        record["files"].append({"url": url, "file": name, "md5": got, "expected_md5_source": src})

    tar = DL / "collectionandqueries.tar.gz"
    if tar.exists():
        require_free(DATA, 3.2)
        subprocess.run(["tar", "xzf", str(tar), "-C", str(MS)], check=True)
        verified_marker.write_text(record["files"][0]["md5"] + "\n")
        tar.unlink()
        print("extracted and removed tarball")

    for f in sorted(MS.glob("*.tsv")):
        n = count_lines(f)
        record["extracted"][f"raw/msmarco/{f.name}"] = {"lines": n, "md5": md5sum(f)}
        if f.name in EXPECTED_COUNTS and n != EXPECTED_COUNTS[f.name]:
            print(f"{f.name}: {n} lines, expected {EXPECTED_COUNTS[f.name]}")
            return 1

    # TREC DL: keep only judged topics (43 / 54) in the canonical queries file.
    for year, tag, expect in (("2019", "dl19", 43), ("2020", "dl20", 54)):
        qrels_src = DL / f"{year}qrels-pass.txt"
        shutil.copyfile(qrels_src, TDL / f"{tag}.qrels")
        qrels = read_qrels(TDL / f"{tag}.qrels")
        with gzip.open(DL / f"msmarco-test{year}-queries.tsv.gz", "rt", encoding="utf-8") as fin:
            allq = [line.rstrip("\n").split("\t", 1) for line in fin if line.strip()]
        (TDL / f"{tag}.queries.all.tsv").write_text("".join(f"{q}\t{t}\n" for q, t in allq), encoding="utf-8")
        judged = [(q, t) for q, t in allq if q in qrels]
        if len(judged) != expect or len(qrels) != expect:
            print(f"{tag}: {len(judged)} judged queries, {len(qrels)} qrels topics, expected {expect}")
            return 1
        write_queries(TDL / f"{tag}.queries.tsv", judged)
        for fn in (f"{tag}.queries.tsv", f"{tag}.qrels", f"{tag}.queries.all.tsv"):
            record["extracted"][f"raw/trec-dl/{fn}"] = {"lines": count_lines(TDL / fn), "md5": md5sum(TDL / fn)}
        print(f"{tag}: {len(judged)} judged topics, {sum(len(v) for v in qrels.values())} judgments")

    (DATA / "raw" / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
