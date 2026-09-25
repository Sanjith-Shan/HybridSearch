"""Fine-tune a MiniLM cross-encoder on MS MARCO train groups (1 positive + n mined negatives).

    python -m hybridsearch.rerank.train --config configs/rerank/mps_bm25.yaml [--resume]
    python -m hybridsearch.rerank.train --set strategy=dense-hard max_steps=100 out_dir=...

Base model: ``nreimers/MiniLM-L6-H384-uncased`` (6 layers, hidden 384, 22.7M params), a
general-purpose MiniLM distilled from UniLM, *not* trained on MS MARCO. Why L6 and not
microsoft/MiniLM-L12-H384-uncased: (1) it is the same architecture as the public baseline
cross-encoder/ms-marco-MiniLM-L6-v2, so the comparison isolates the training recipe
(data, negatives, loss, compute) instead of model size; (2) half the layers means twice the
steps in a fixed MPS or A100 budget and half the CPU serving latency per pair, which is the
constraint the cascade curve is about; (3) the public L6-v2 and L12-v2 cards report the same
dev MRR@10 (39.01 vs 39.02), so depth buys little on this task at this width.

Losses (``loss``):
- ``bce``      : pointwise binary cross-entropy on every (query, passage) logit, label 1/0.
- ``listwise`` : softmax cross-entropy over each group's 1+n logits, target = the positive
                 (a.k.a. localised contrastive estimation).

Checkpoints: ``<out_dir>/last/`` (weights + optimizer + scheduler + step, overwritten every
``save_every`` steps, for resume) and ``<out_dir>/best/`` (weights only, best validation
RR@10 on ``data/subset/train_tune``, never dev). ``--resume`` continues from ``last``; since
batches depend only on (seed, step), the resumed run sees the same data it would have.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hybridsearch.data.io import DATA, REPO_ROOT, read_qrels, read_queries
from hybridsearch.rerank.model import (TRAIN_MAX_LEN, clip_query, pair_batch, pick_device)
from hybridsearch.rerank.sampler import STRATEGIES, GroupSampler, needed_pids, read_mined

LOCK = DATA / "locks" / "mps"


@dataclass
class TrainConfig:
    base_model: str = "nreimers/MiniLM-L6-H384-uncased"
    strategy: str = "bm25-hard"
    loss: str = "listwise"
    n_neg: int = 7
    mix_bm25: float = 0.5
    batch_groups: int = 16          # groups per optimizer step (pairs = groups * (1 + n_neg))
    grad_accum: int = 1
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    max_steps: int = 1000
    max_len: int = TRAIN_MAX_LEN
    seed: int = 20260923
    device: str = "auto"
    amp: str = "none"               # none | bf16 | fp16 (fp16/bf16 on CUDA; MPS stays fp32)
    mps_mem_fraction: float = 0.35  # hard cap (~4.7 GB of 18 GB) on MPS memory (fraction of recommended max): OOM, not swap
    mined: str = "data/rerank/mined/train50k.jsonl"
    queries: str = "data/subset/train50k/queries.tsv"
    qrels: str = "data/subset/train50k/qrels.tsv"
    n_queries: int | None = None    # use only the first n mined queries (small runs)
    pool_depth: int = 50            # sample negatives from the top-N of each filtered mined pool
    val_run: str = "data/runs/reference/anserini-bm25-default.train_tune.trec"
    val_queries: str = "data/subset/train_tune/queries.tsv"
    val_qrels: str = "data/subset/train_tune/qrels.tsv"
    val_n: int = 500
    val_depth: int = 30
    eval_every: int = 250
    save_every: int = 250
    log_every: int = 25
    out_dir: str = "data/rerank/runs/debug"
    collection: str = "data/raw/msmarco/collection.tsv"
    label: str = ""                 # free text, e.g. "M3 Pro MPS"
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | None, overrides: list[str]) -> "TrainConfig":
        import yaml

        def read(p: Path) -> dict:
            y = yaml.safe_load(p.read_text()) or {}
            base = y.pop("base", None)
            return {**(read(p.parent / base) if base else {}), **y}

        d: dict = read(Path(path)) if path else {}
        types = {f.name: f.type for f in dataclasses.fields(cls)}
        for o in overrides:
            k, v = o.split("=", 1)
            if k not in types:
                raise KeyError(f"unknown config key {k}")
            d[k] = yaml_value(v)
        cfg = cls(**d)
        if cfg.strategy not in STRATEGIES:
            raise ValueError(f"strategy {cfg.strategy!r} not in {STRATEGIES}")
        if cfg.loss not in ("bce", "listwise"):
            raise ValueError("loss must be bce or listwise")
        return cfg


def yaml_value(v: str):
    import yaml

    return yaml.safe_load(v)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


# ------------------------------------------------------------------ losses
def group_loss(logits: torch.Tensor, group_size: int, kind: str) -> torch.Tensor:
    """``logits``: [G * group_size], each group ordered [pos, neg_1..neg_n]."""
    lg = logits.view(-1, group_size).float()
    if kind == "listwise":
        target = torch.zeros(lg.shape[0], dtype=torch.long, device=lg.device)
        return torch.nn.functional.cross_entropy(lg, target)
    labels = torch.zeros_like(lg)
    labels[:, 0] = 1.0
    return torch.nn.functional.binary_cross_entropy_with_logits(lg, labels)


def rng_state(device: str) -> dict:
    st = {"cpu": torch.get_rng_state()}
    if device == "cuda":
        st["cuda"] = torch.cuda.get_rng_state()
    if device == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st: dict | None, device: str) -> None:
    """Restore dropout RNG so a resumed run matches an uninterrupted one (exact on CPU)."""
    if not st:
        return
    torch.set_rng_state(st["cpu"])
    if device == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"])
    if device == "mps" and "mps" in st:
        torch.mps.set_rng_state(st["mps"])


# ------------------------------------------------------------------ lock
class MPSLock:
    """``mkdir data/locks/mps`` as a mutex with the data agent's encoder (and anyone else)."""

    def __init__(self, device: str, owner: str, wait: bool):
        self.active = device == "mps"
        self.owner, self.wait = owner, wait

    def __enter__(self):
        if not self.active:
            return self
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                LOCK.mkdir()
                break
            except FileExistsError:
                if not self.wait:
                    raise SystemExit(f"{LOCK} is held ({(LOCK / 'owner').read_text().strip() if (LOCK / 'owner').exists() else '?'}); rerun with --wait-lock")
                time.sleep(120)
        (LOCK / "owner").write_text(f"{self.owner} pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        return self

    def __exit__(self, *exc):
        if self.active:
            shutil.rmtree(LOCK, ignore_errors=True)


# ------------------------------------------------------------------ validation
@torch.no_grad()
def validate(model, tok, val: dict, device: str, max_len: int, batch: int = 128) -> dict:
    from hybridsearch.eval.metrics import evaluate

    model.eval()
    Q, P, keys = val["q"], val["p"], val["keys"]
    scores = np.empty(len(Q), np.float32)
    order = np.argsort([len(q) + len(p) for q, p in zip(Q, P)])
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        b = pair_batch(tok, [Q[i] for i in idx], [P[i] for i in idx], max_len).to(device)
        scores[idx] = model(**b.as_dict()).logits[:, 0].float().cpu().numpy()
    run: dict[str, dict[str, float]] = {}
    for (qid, pid), s in zip(keys, scores):
        run.setdefault(qid, {})[pid] = float(s)
    agg, _ = evaluate(run, val["qrels"], ["RR@10", "nDCG@10"])
    model.train()
    return agg


def build_val(cfg: TrainConfig, tok) -> tuple[dict, set[str]]:
    from hybridsearch.rerank.text import iter_run_groups

    queries = read_queries(resolve(cfg.val_queries))
    qrels_all = read_qrels(resolve(cfg.val_qrels))
    qids = sorted(q for q in queries if q in qrels_all)[: cfg.val_n]
    want = set(qids)
    keys, need = [], set()
    for qid, hits in iter_run_groups(resolve(cfg.val_run), max_hits=cfg.val_depth):
        if qid in want:
            for pid in hits:
                keys.append((qid, pid))
                need.add(pid)
    qrels = {q: qrels_all[q] for q in qids}
    return {"keys": keys, "qids": qids, "queries": queries, "qrels": qrels}, need


# ------------------------------------------------------------------ main loop
def train(cfg: TrainConfig, resume: bool = False, wait_lock: bool = False) -> dict:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from hybridsearch.rerank.meta import hardware, versions, write_meta
    from hybridsearch.rerank.text import load_passages

    from hybridsearch.rerank import cap_threads

    cap_threads()
    out = resolve(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(cfg.mps_mem_fraction)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    last = out / "last"
    src = str(last) if resume and (last / "trainer_state.pt").exists() else cfg.base_model
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1)
    model.to(device).train()

    # data
    t0 = time.time()
    mined = read_mined(resolve(cfg.mined), cfg.n_queries)
    for m in mined:
        m.bm25, m.dense, m.random = m.bm25[:cfg.pool_depth], m.dense[:cfg.pool_depth], m.random[:cfg.pool_depth]
    queries = read_queries(resolve(cfg.queries))
    judged = {q: set(d) for q, d in read_qrels(resolve(cfg.qrels)).items()}
    val, val_need = build_val(cfg, tok) if cfg.val_n else ({}, set())
    need = needed_pids(mined, cfg.strategy) | val_need
    passages = load_passages(need, resolve(cfg.collection))
    mined_qids = {m.qid for m in mined}
    queries = {q: clip_query(tok, t) for q, t in queries.items() if q in mined_qids}
    sampler = GroupSampler(mined, queries, passages, cfg.strategy, cfg.n_neg, cfg.seed,
                           cfg.mix_bm25, judged)
    if val:
        val["q"] = [val["queries"][q] for q, _ in val["keys"]]
        val["p"] = [passages[p] for _, p in val["keys"]]
    print(f"data: {len(sampler)} train queries, {len(passages)} passages, "
          f"{len(val.get('keys', []))} val pairs, {time.time() - t0:.0f}s", flush=True)

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * cfg.max_steps), cfg.max_steps)
    step, best, history = 0, -1.0, []
    train_seconds_resumed = 0.0
    if src == str(last):
        st = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, best, history = st["step"], st["best"], st["history"]
        train_seconds_resumed = st.get("train_seconds", 0.0)
        set_rng_state(st.get("rng"), device)
        print(f"resumed at step {step} (best val RR@10 {best:.4f})", flush=True)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp)
    scaler = torch.amp.GradScaler("cuda") if cfg.amp == "fp16" and device == "cuda" else None
    gsize = 1 + cfg.n_neg
    log_f = open(out / "log.jsonl", "a")
    t_train, pairs_since, t_since = time.time(), 0, time.time()
    train_seconds = train_seconds_resumed

    def save_last():
        tmp = out / "last.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
                    "best": best, "history": history, "train_seconds": train_seconds,
                    "rng": rng_state(device),
                    "config": dataclasses.asdict(cfg)},
                   tmp / "trainer_state.pt")
        shutil.rmtree(last, ignore_errors=True)
        tmp.rename(last)

    def run_eval():
        nonlocal best
        if not val:
            return
        agg = validate(model, tok, val, device, 512)
        rec = {"step": step, "val": agg, "train_seconds": round(train_seconds, 1)}
        history.append(rec)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"step {step}: train_tune RR@10 {agg['RR@10']:.4f} nDCG@10 {agg['nDCG@10']:.4f}", flush=True)
        if agg["RR@10"] > best:
            best = agg["RR@10"]
            tmp = out / "best.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)
            (tmp / "best.json").write_text(json.dumps(rec, indent=2) + "\n")
            shutil.rmtree(out / "best", ignore_errors=True)
            tmp.rename(out / "best")

    lock_owner = f"rerank-train {cfg.out_dir}"
    with MPSLock(device, lock_owner, wait_lock):
        if step == 0 and val:
            run_eval()
        while step < cfg.max_steps:
            t_step = time.time()
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for micro in range(cfg.grad_accum):
                groups = sampler.batch(step * cfg.grad_accum + micro, cfg.batch_groups)
                Q, P = [], []
                for qid, pos, negs in groups:
                    for pid in [pos, *negs]:
                        Q.append(queries[qid])
                        P.append(passages[pid])
                b = pair_batch(tok, Q, P, cfg.max_len).to(device)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(**b.as_dict()).logits[:, 0]
                loss = group_loss(logits, gsize, cfg.loss) / cfg.grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_acc += float(loss.detach())
                pairs_since += len(Q)
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            step += 1
            if device == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()  # return cached blocks each step: long batches OOM otherwise
            train_seconds += time.time() - t_step
            if not math.isfinite(loss_acc):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % cfg.log_every == 0:
                rate = pairs_since / (time.time() - t_since)
                rec = {"step": step, "loss": round(loss_acc, 5), "lr": sched.get_last_lr()[0],
                       "pairs_per_s": round(rate, 1), "topups": sampler.topups,
                       "loadavg_1m": round(os.getloadavg()[0], 1)}
                if device == "mps":
                    rec["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
                    torch.mps.empty_cache()
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(json.dumps(rec), flush=True)
                pairs_since, t_since = 0, time.time()
            if cfg.eval_every and step % cfg.eval_every == 0:
                run_eval()
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                save_last()
        if val and (not history or history[-1]["step"] != step):
            run_eval()
            save_last()

    groups_seen = step * cfg.grad_accum * cfg.batch_groups
    summary = {
        "label": cfg.label, "device": device, "steps": step,
        "groups_seen": groups_seen, "pairs_seen": groups_seen * gsize,
        "unique_train_queries": len(sampler),
        "epochs": round(groups_seen / max(len(sampler), 1), 3),
        "train_seconds": round(train_seconds, 1),
        "wall_seconds_this_session": round(time.time() - t_train, 1),
        "best_val_RR@10": best, "history": history, "negative_topups": sampler.topups,
        "config": dataclasses.asdict(cfg), "hardware": hardware(), "versions": versions(),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_meta(out / "train_summary.json", steps=step, examples_seen=summary["pairs_seen"],
               groups_seen=groups_seen, device=device, label=cfg.label)
    log_f.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wait-lock", action="store_true", help="poll data/locks/mps until free")
    ap.add_argument("--drop-optimizer", action="store_true",
                    help="after training, delete last/ (optimizer state) to save disk")
    args = ap.parse_args()
    cfg = TrainConfig.load(args.config, args.set)
    s = train(cfg, args.resume, args.wait_lock)
    print(json.dumps({k: v for k, v in s.items() if k not in ("history", "config", "hardware")}, indent=2))
    if args.drop_optimizer:
        shutil.rmtree(resolve(cfg.out_dir) / "last", ignore_errors=True)


if __name__ == "__main__":
    main()
