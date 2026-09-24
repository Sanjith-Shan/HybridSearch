"""M5 reranker. CPU threads are capped (the laptop is shared with other agents)."""
import os as _os

MAX_CPU_THREADS = int(_os.environ.get("HS_MAX_THREADS", "4"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "RAYON_NUM_THREADS"):
    _os.environ.setdefault(_v, str(MAX_CPU_THREADS))


def cap_threads() -> None:
    import torch

    torch.set_num_threads(MAX_CPU_THREADS)
    # Do NOT call faiss.omp_set_num_threads here: with torch already loaded, torch and faiss
    # each bring a libomp and the call aborts the process (docs/BUG_LOG.md, 2026-09-23).
    # FAISS honours OMP_NUM_THREADS, set above at package import.
