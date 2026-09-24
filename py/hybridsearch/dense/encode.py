"""BGE-base-en-v1.5 encoding for HybridSearch (passages + queries).

Convention (docs/ARCHITECTURE.md): CLS pooling, L2-normalised, fp32; queries get the
BGE retrieval instruction; passages max 512 tokens, queries max 64 tokens.

    python -m hybridsearch.dense.encode queries   # dev, dl19, dl20, train_tune, train50k
    python -m hybridsearch.dense.encode passages  # 1M subset -> passages.fbin (resumable)
    python -m hybridsearch.dense.encode all       # queries then passages, then READY

Passages are written in place into a preallocated ``passages.fbin.partial`` (memmap),
in chunks of CHUNK rows; each finished chunk is flushed and appended to
``passages.progress``. A restart skips finished chunks. On completion the file is
renamed to ``passages.fbin``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "4")  # HF fast-tokenizer thread pool
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..data.io import DATA, read_queries, read_u64bin, write_fbin

MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIM = 768
PASSAGE_MAX_LEN = 512
QUERY_MAX_LEN = 64
CHUNK = 20_000
TOKEN_BUDGET = 32_768  # padded tokens per batch
MAX_BATCH = 256

OUT = DATA / "embeddings" / "1m"
LOCK = DATA / "locks" / "mps"

QUERY_SETS = {
    "dev": DATA / "raw" / "msmarco" / "queries.dev.small.tsv",
    "dl19": DATA / "raw" / "trec-dl" / "dl19.queries.tsv",
    "dl20": DATA / "raw" / "trec-dl" / "dl20.queries.tsv",
    "train_tune": DATA / "subset" / "train_tune" / "queries.tsv",
    "train50k": DATA / "subset" / "train50k" / "queries.tsv",
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder:
    def __init__(self, device: str | None = None, model_name: str = MODEL):
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device).eval()

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str], max_len: int) -> np.ndarray:
        b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
        cls = self.model(**b).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls.float(), p=2, dim=-1).cpu().numpy()

    def encode(self, texts: list[str], max_len: int) -> np.ndarray:
        """Encode in length-sorted, token-budgeted batches; returns rows in input order."""
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        lens = [min(len(x), max_len) for x in self.tok(texts, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]]
        order = np.argsort(lens, kind="stable")
        out = np.empty((len(texts), DIM), dtype=np.float32)
        i = 0
        while i < len(order):
            j = i
            # grow the batch while bs * maxlen_in_batch stays under budget
            while j < len(order) and (j - i + 1) <= MAX_BATCH and (j - i + 1) * lens[order[j]] <= TOKEN_BUDGET:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            out[idx] = self._encode_batch([texts[k] for k in idx], max_len)
            i = j
        return out

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.encode([QUERY_PREFIX + q for q in queries], QUERY_MAX_LEN)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self.encode(passages, PASSAGE_MAX_LEN)


def hardware() -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return None

    return {
        "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "cores": os.cpu_count(),
        "mem_bytes": int(sh(["sysctl", "-n", "hw.memsize"]) or 0),
        "os": f"{platform.system()} {platform.release()} ({platform.mac_ver()[0]})",
        "torch": torch.__version__,
    }


def acquire_lock(timeout_s: float = 6 * 3600) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            (LOCK / "owner").write_text(f"dense-encode pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            return
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"{LOCK} held for > {timeout_s}s")
            time.sleep(30)


def release_lock() -> None:
    try:
        (LOCK / "owner").unlink(missing_ok=True)
        os.rmdir(LOCK)
    except FileNotFoundError:
        pass


def encode_queries(enc: Encoder, sets=None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, path in QUERY_SETS.items():
        if sets and name not in sets:
            continue
        q = read_queries(path)
        qids = list(q)
        t = time.time()
        emb = enc.encode_queries([q[k] for k in qids])
        dt = time.time() - t
        write_fbin(OUT / f"queries.{name}.fbin", emb)
        (OUT / f"queries.{name}.qids.txt").write_text("".join(f"{k}\n" for k in qids))
        stats[name] = {"n": len(qids), "seconds": round(dt, 2), "queries_per_sec": round(len(qids) / dt, 1)}
        print(f"queries {name}: {len(qids)} in {dt:.1f}s", flush=True)
    return stats


def encode_passages(enc: Encoder) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / "passages.fbin"
    if final.exists():
        print("passages.fbin exists; skipping")
        return json.loads((OUT / "passages.stats.json").read_text()) if (OUT / "passages.stats.json").exists() else {}
    ids = read_u64bin(DATA / "subset" / "1m" / "docids.u64bin")
    n = len(ids)
    partial = OUT / "passages.fbin.partial"
    progress = OUT / "passages.progress"
    if not partial.exists():
        free = os.statvfs(OUT).f_bavail * os.statvfs(OUT).f_frsize / 1e9
        need = (8 + n * DIM * 4) / 1e9
        if free - need < 3.0:
            raise SystemExit(f"refusing: {free:.1f} GB free, need {need:.1f} GB + 3 GB headroom")
        with open(partial, "wb") as f:
            f.write(np.array([n, DIM], dtype="<u4").tobytes())
            f.truncate(8 + n * DIM * 4)
        progress.write_text("")
    done = {int(x) for x in progress.read_text().split()} if progress.exists() else set()
    mm = np.memmap(partial, dtype="<f4", mode="r+", offset=8, shape=(n, DIM))

    # read texts; verify order == docids.u64bin
    texts: list[str] = []
    with open(DATA / "subset" / "1m" / "collection.tsv", encoding="utf-8") as f:
        for i, line in enumerate(f):
            pid, text = line.rstrip("\n").split("\t", 1)
            if int(pid) != int(ids[i]):
                raise SystemExit(f"collection row {i} pid {pid} != docids {ids[i]}")
            texts.append(text)
    assert len(texts) == n

    nchunks = (n + CHUNK - 1) // CHUNK
    enc_seconds = 0.0
    enc_rows = 0
    for c in range(nchunks):
        if c in done:
            continue
        s, e = c * CHUNK, min(n, (c + 1) * CHUNK)
        t = time.time()
        mm[s:e] = enc.encode_passages(texts[s:e])
        mm.flush()
        dt = time.time() - t
        enc_seconds += dt
        enc_rows += e - s
        with open(progress, "a") as f:
            f.write(f"{c}\n")
            f.flush()
            os.fsync(f.fileno())
        with open(OUT / "passages.chunks.log", "a") as f:
            f.write(json.dumps({"chunk": c, "rows": e - s, "seconds": round(dt, 2), "ts": time.strftime("%H:%M:%S")}) + "\n")
        print(f"chunk {c + 1}/{nchunks}: {e - s} rows in {dt:.1f}s ({(e - s) / dt:.0f} p/s)", flush=True)
    del mm
    # sanity: every row unit-norm
    mm = np.memmap(partial, dtype="<f4", mode="r", offset=8, shape=(n, DIM))
    norms = np.linalg.norm(mm[:: max(1, n // 10000)], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"norms off: {norms.min()} {norms.max()}"
    del mm
    os.replace(partial, final)
    progress.unlink(missing_ok=True)
    # throughput from the chunk log (covers resumed runs too)
    rows = secs = 0
    for line in (OUT / "passages.chunks.log").read_text().splitlines():
        r = json.loads(line)
        rows += r["rows"]
        secs += r["seconds"]
    stats = {"n": n, "dim": DIM, "encode_seconds": round(secs, 1), "passages_per_sec": round(rows / secs, 1),
             "chunks_logged": rows // CHUNK}
    (OUT / "passages.stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["queries", "passages", "all"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch (shared machine)")
    args = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(args.threads)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so `finally` releases the lock
    acquire_lock()
    try:
        enc = Encoder(args.device)
        meta = {"model": MODEL, "device": enc.device, "dtype": "float32", "pooling": "cls", "normalize": True,
                "query_prefix": QUERY_PREFIX, "passage_max_len": PASSAGE_MAX_LEN, "query_max_len": QUERY_MAX_LEN,
                "hardware": hardware(), "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if args.what in ("queries", "all"):
            meta["queries"] = encode_queries(enc)
        if args.what in ("passages", "all"):
            meta["passages"] = encode_passages(enc)
        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["wall_seconds"] = round(time.time() - t0, 1)
        (OUT / f"encode.{args.what}.meta.json").write_text(json.dumps(meta, indent=2))
    finally:
        release_lock()
    if args.what in ("passages", "all") and (OUT / "passages.fbin").exists():
        (OUT / "READY").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
