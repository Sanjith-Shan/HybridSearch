"""Stream Waterloo's precomputed BGE-base-en-v1.5 MS MARCO vectors (a tar of parquet
files, 26 GB, MD5 a55b3cb338ec4a1b1c36825bf0854648) from stdin straight into
passages.fbin + docids.u64bin, without ever storing the tar or the parquet files.

    curl -sL URL | tee >(md5sum > tar.md5) | python -m hybridsearch.bench.parquet_stream OUT_DIR

Each tar member is read into memory (one parquet shard), converted, appended, dropped.
The id and vector columns are auto-detected (id: docid|id|_id; vector: vector|
embedding|emb). NOT YET RUN against the real archive (it was never downloaded on the
laptop): the first shard's schema is printed and the script stops if it does not
recognise the columns. Rows are L2-normalised check-summed: the script reports the
min/max norm, and fails if any vector is not unit length within 1e-3 (the engine
assumes inner product == cosine).
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np


def main(out_dir: str) -> int:
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fb = open(out / "passages.fbin.partial", "wb")
    np.zeros(2, dtype=np.uint32).tofile(fb)  # header patched at the end
    ids_all: list[np.ndarray] = []
    n, dim = 0, None
    nmin, nmax = np.inf, 0.0
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".parquet"):
                continue
            table = pq.read_table(io.BytesIO(tar.extractfile(m).read()))
            cols = table.column_names
            idc = next((c for c in ("docid", "id", "_id") if c in cols), None)
            vc = next((c for c in ("vector", "embedding", "emb") if c in cols), None)
            if idc is None or vc is None:
                print(f"unrecognised schema in {m.name}: {table.schema}", file=sys.stderr)
                return 2
            ids = np.asarray(table.column(idc).to_pylist(), dtype=np.uint64)
            vecs = np.asarray(table.column(vc).to_pylist(), dtype=np.float32)
            if dim is None:
                dim = vecs.shape[1]
                print(f"schema: {table.schema}; dim={dim}", file=sys.stderr)
            norms = np.linalg.norm(vecs, axis=1)
            nmin, nmax = min(nmin, float(norms.min())), max(nmax, float(norms.max()))
            if abs(nmin - 1) > 1e-3 or abs(nmax - 1) > 1e-3:
                print(f"non-unit vectors in {m.name}: norm range [{nmin}, {nmax}]", file=sys.stderr)
                return 3
            vecs.tofile(fb)
            ids_all.append(ids)
            n += len(ids)
            print(f"{m.name}: +{len(ids)} -> {n}", file=sys.stderr, flush=True)
    fb.seek(0)
    np.array([n, dim], dtype=np.uint32).tofile(fb)
    fb.close()
    (out / "passages.fbin.partial").rename(out / "passages.fbin")
    ids = np.concatenate(ids_all)
    with open(out / "docids.u64bin", "wb") as f:
        np.array([len(ids)], dtype=np.uint64).tofile(f)
        ids.tofile(f)
    print(f"done: n={n} dim={dim} norms [{nmin:.5f}, {nmax:.5f}] unique ids={len(np.unique(ids))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
