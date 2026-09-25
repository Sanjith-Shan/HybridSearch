"""Test-session setup that must run before torch, FAISS or ONNX Runtime are imported.

The suite loads FAISS (libomp) and torch (its own OpenMP) into one pytest process. With
multi-threaded OpenMP pools from two runtimes, a torch forward pass in test_export_parity
deadlocked in layer_norm when it ran after the FAISS tests (it passed on its own). One OpenMP
thread per runtime removes the contention; the tests are small, so nothing needs the threads.
See docs/BUG_LOG.md 2026-09-24.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
