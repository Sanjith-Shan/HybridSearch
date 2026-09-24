"""hnswlib and FAISS on the same vectors, queries and ground truth as the engine's
Vamana / DiskANN runs: recall@10 against queries/second.

    python -m hybridsearch.bench.ann_compare --base passages.fbin [--max-n N] \
        --queries queries.train_tune.fbin --gt gt.bin --out results/vector/ann_compare_1m.jsonl \
        [--systems hnswlib,faiss-hnsw,faiss-ivfpq,faiss-flat] [--build-threads 4]

Search runs single-threaded (hnswlib set_num_threads(1), faiss omp_set_num_threads(1)),
with the whole query set submitted as one batch, so there is no per-query Python
overhead; the engine's tools likewise loop over queries in C++ on one thread.
Every system uses inner product on the L2-normalised vectors.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np

from .io import read_fbin, read_gt, recall_at_k


def emit(out, row):
    line = json.dumps(row)
    print(line, flush=True)
    with open(out, "a") as f:
        f.write(line + "\n")


_LAST_CPU = [0.0]


def timed_search(fn, q):
    """Wall seconds for the batch; the calling thread's CPU seconds go to _LAST_CPU.
    With 1 search thread both libraries run the batch on the calling thread, so
    thread CPU time is the steadier cost measure on a loaded machine (same as the
    engine tools' cpu_us_per_query)."""
    c = time.thread_time()
    t = time.perf_counter()
    ids = fn(q)
    wall = time.perf_counter() - t
    _LAST_CPU[0] = time.thread_time() - c
    return ids, wall


def cpu_fields(nq):
    cpu = _LAST_CPU[0]
    return {"cpu_us_per_query": cpu / nq * 1e6, "cpu_qps": nq / cpu if cpu > 0 else None}


def run_hnswlib(x, q, gt, out, threads, M, efc, efs_list, tag):
    import hnswlib
    n, d = x.shape
    ix = hnswlib.Index(space="ip", dim=d)
    t = time.perf_counter()
    ix.init_index(max_elements=n, M=M, ef_construction=efc, random_seed=100)
    ix.set_num_threads(threads)
    for s in range(0, n, 100_000):
        ix.add_items(np.ascontiguousarray(x[s:s + 100_000]), np.arange(s, min(n, s + 100_000)))
    build = time.perf_counter() - t
    ix.set_num_threads(1)
    # hnswlib's level-0 link list holds 2*M ids; upper levels ~1/M of nodes.
    mem = n * (d * 4 + 2 * M * 4 + 4 + 8) + int(n / M) * M * 4
    for ef in efs_list:
        ix.set_ef(max(ef, 10))
        ids, secs = timed_search(lambda qq: ix.knn_query(qq, k=10)[0], q)
        emit(out, {"system": tag, "M": M, "efConstruction": efc, "efSearch": ef, "n": n, "nq": len(q),
                   "recall": recall_at_k(ids, gt), "qps": len(q) / secs, **cpu_fields(len(q)), "build_seconds": build,
                   "build_threads": threads, "index_bytes_est": mem})
    del ix
    gc.collect()


def run_faiss_hnsw(x, q, gt, out, threads, M, efc, efs_list, tag):
    import faiss
    n, d = x.shape
    faiss.omp_set_num_threads(threads)
    ix = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    ix.hnsw.efConstruction = efc
    t = time.perf_counter()
    for s in range(0, n, 100_000):
        ix.add(np.ascontiguousarray(x[s:s + 100_000]))
    build = time.perf_counter() - t
    faiss.omp_set_num_threads(1)
    for ef in efs_list:
        ix.hnsw.efSearch = ef
        ids, secs = timed_search(lambda qq: ix.search(qq, 10)[1], q)
        emit(out, {"system": tag, "M": M, "efConstruction": efc, "efSearch": ef, "n": n, "nq": len(q),
                   "recall": recall_at_k(ids, gt), "qps": len(q) / secs, **cpu_fields(len(q)), "build_seconds": build,
                   "build_threads": threads, "index_bytes_est": n * (d * 4 + 2 * M * 4)})
    del ix
    gc.collect()


def run_faiss_ivfpq(x, q, gt, out, threads, nlist, pq_m, nprobes, refine_factors, train_n, seed, tag):
    import faiss
    n, d = x.shape
    faiss.omp_set_num_threads(threads)
    rng = np.random.default_rng(seed)
    sample = np.ascontiguousarray(x[np.sort(rng.choice(n, size=min(n, train_n), replace=False))])
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFPQ(quant, d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
    t = time.perf_counter()
    ivf.train(sample)
    for s in range(0, n, 100_000):
        ivf.add(np.ascontiguousarray(x[s:s + 100_000]))
    build = time.perf_counter() - t
    faiss.omp_set_num_threads(1)
    mem = n * (pq_m + 8) + nlist * d * 4
    for nprobe in nprobes:
        ivf.nprobe = nprobe
        ids, secs = timed_search(lambda qq: ivf.search(qq, 10)[1], q)
        emit(out, {"system": tag, "nlist": nlist, "pq_M": pq_m, "nprobe": nprobe, "refine": 0, "n": n,
                   "nq": len(q), "recall": recall_at_k(ids, gt), "qps": len(q) / secs, **cpu_fields(len(q)), "build_seconds": build,
                   "build_threads": threads, "index_bytes_est": mem})
    # IVF-PQ + exact re-rank of the top k*factor from full vectors (RAM): the
    # closest FAISS analogue of DiskANN's full-precision rerank.
    if refine_factors:
        faiss.omp_set_num_threads(threads)
        flat = faiss.IndexFlatIP(d)
        for s in range(0, n, 100_000):
            flat.add(np.ascontiguousarray(x[s:s + 100_000]))
        ref = faiss.IndexRefine(ivf, flat)
        faiss.omp_set_num_threads(1)
        for f in refine_factors:
            ref.k_factor = f
            for nprobe in nprobes:
                ivf.nprobe = nprobe
                ids, secs = timed_search(lambda qq: ref.search(qq, 10)[1], q)
                emit(out, {"system": tag + "+refine", "nlist": nlist, "pq_M": pq_m, "nprobe": nprobe,
                           "refine": f, "n": n, "nq": len(q), "recall": recall_at_k(ids, gt),
                           "qps": len(q) / secs, **cpu_fields(len(q)), "build_seconds": build, "build_threads": threads,
                           "index_bytes_est": mem + n * d * 4})
        del ref, flat
    del ivf
    gc.collect()


def run_flat(x, q, gt, out, tag):
    import faiss
    n, d = x.shape
    faiss.omp_set_num_threads(4)
    ix = faiss.IndexFlatIP(d)
    for s in range(0, n, 100_000):
        ix.add(np.ascontiguousarray(x[s:s + 100_000]))
    faiss.omp_set_num_threads(1)
    ids, secs = timed_search(lambda qq: ix.search(qq, 10)[1], q)
    emit(out, {"system": tag, "n": n, "nq": len(q), "recall": recall_at_k(ids, gt), "qps": len(q) / secs, **cpu_fields(len(q)),
               "build_seconds": 0.0, "index_bytes_est": n * d * 4})
    del ix
    gc.collect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hybridsearch.bench.ann_compare")
    ap.add_argument("--base", required=True)
    ap.add_argument("--max-n", type=int)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", default="faiss-flat,hnswlib,faiss-hnsw,faiss-ivfpq")
    ap.add_argument("--build-threads", type=int, default=4)
    ap.add_argument("--hnsw-M", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", type=int, nargs="+", default=[10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--pq-M", type=int, default=96)
    ap.add_argument("--nprobe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--refine", type=int, nargs="*", default=[10])
    ap.add_argument("--train-n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260923)
    a = ap.parse_args(argv)

    x = read_fbin(a.base, a.max_n)
    q = np.ascontiguousarray(read_fbin(a.queries, a.max_queries, mmap=False))
    gt_ids, _ = read_gt(a.gt)
    gt_ids = gt_ids[: len(q)]
    systems = a.systems.split(",")
    print(f"# n={len(x)} nq={len(q)} systems={systems}", file=sys.stderr)
    if "faiss-flat" in systems:
        run_flat(x, q, gt_ids, a.out, "faiss-flat")
    if "hnswlib" in systems:
        for M in a.hnsw_M:
            run_hnswlib(x, q, gt_ids, a.out, a.build_threads, M, a.efc, a.ef, "hnswlib")
    if "faiss-hnsw" in systems:
        for M in a.hnsw_M:
            run_faiss_hnsw(x, q, gt_ids, a.out, a.build_threads, M, a.efc, a.ef, "faiss-hnsw")
    if "faiss-ivfpq" in systems:
        run_faiss_ivfpq(x, q, gt_ids, a.out, a.build_threads, a.nlist, a.pq_M, a.nprobe, a.refine,
                        a.train_n, a.seed, "faiss-ivfpq")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
