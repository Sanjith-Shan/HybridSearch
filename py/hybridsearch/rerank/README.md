# M5: the reranker

Code: `py/hybridsearch/rerank/` (training, mining, scoring, export, bench, cascade) and
`py/hybridsearch/dense/export_query_encoder.py`. Tests: `py/tests/test_rerank_*.py`,
`py/tests/test_export_parity.py`. Every number lives in `results/rerank/` with a
`.meta.json` sidecar. Results summary: `results/rerank/README.md`.

## Pipeline, in order

```bash
# 0. ONNX exports for the broker (CPU, minutes)
.venv/bin/python -m hybridsearch.dense.export_query_encoder              # data/models/query-encoder
.venv/bin/python -m hybridsearch.rerank.export --model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --out data/models/reranker-public --int8 --tag reranker-public         # public baseline

# 1. hard-negative mining (needs bm25.train50k.trec + the 1M passage vectors)
.venv/bin/python -m hybridsearch.rerank.mine positives   # BGE vectors of train50k positives (ORT CPU)
.venv/bin/python -m hybridsearch.rerank.mine mine        # data/rerank/mined/train50k.jsonl (+ .stats.json)

# 2. training (MPS; takes data/locks/mps while it runs)
.venv/bin/python -m hybridsearch.rerank.train --config py/hybridsearch/rerank/configs/mps_bm25.yaml --wait-lock
#   resume after an interruption: same command + --resume

# 3. evaluation, export, speed, cascade: see scripts/rerank_eval.sh
```

## Design decisions

- **Base model: `nreimers/MiniLM-L6-H384-uncased`** (not MS MARCO-trained). Same
  architecture as the public baseline `cross-encoder/ms-marco-MiniLM-L6-v2`, so the
  comparison isolates the training recipe; half the layers of L12 means twice the steps per
  GPU-hour and half the CPU serving cost per pair; the public L6-v2 and L12-v2 cards report the
  same dev MRR@10 (39.01 vs 39.02). See the `train.py` docstring.
- **Groups:** 1 positive + n negatives, one query per group. Loss `listwise` (softmax CE over the
  group, default) or `bce` (pointwise). Config key `loss`.
- **Negative strategies** (`strategy`): `random`, `bm25-hard`, `dense-hard`, `mixed`. Negatives
  are sampled uniformly from the top-`pool_depth` of the source's filtered pool (mined to 50;
  the MPS runs use 20 to keep the text cache small), not "always the top n".
- **No positive leaks:** every judged pid of the query is removed from every pool, and a passage
  with the same normalised text as a positive (MS MARCO has exact duplicates) is never a negative.
  Both are tested.
- **False-negative rule (both retrievers):** a candidate is dropped when dense puts it
  ≥ 0.05 cosine above the query's best positive **and** BM25 puts it ≥ 10% above the positive's
  BM25 score. If a positive is not in the BM25 top-1000, the list's lowest score stands in
  (an upper bound on the positive's score, so the rule drops fewer, never more). A candidate
  only one retriever can score is kept and counted as unverifiable. Margins were fixed before
  looking at any result and not tuned. Counts: `data/rerank/mined/train50k.stats.json`, copied to
  `results/rerank/mining_stats.json`.
- **Dense pool** = exact inner-product search (BLAS matmul + top-k, no ANN) over the 1M-subset BGE vectors;
  **BM25 pool** = BM25 over the full 8.8M collection. So dense negatives come from a 1M slice
  and BM25 negatives from the whole collection: a real asymmetry, stated wherever the ablation is.
- **Determinism:** the batch at step s depends only on (seed, s); checkpoints carry optimizer,
  scheduler and RNG state, so a resumed CPU run is bit-identical to an uninterrupted one (tested).
  MPS kernels are not guaranteed deterministic.
- **Checkpoint selection** by RR@10 on 500 `train_tune` queries (BM25 top-30), never dev.
- **Disk:** `last/` (weights + optimizer, ~270 MB for L6) is overwritten in place; `best/` is
  weights only (~90 MB). Pass `--drop-optimizer` to delete `last/` when a run is final.

## Full-scale run on a rented A100 (prepared, not run)

**Nothing here has been run on a GPU. No pod was rented.** Per
`campaign/specs/RUNPOD_SHARED.md`, only the owner session creates or destroys pods; this
script only runs *on* a pod someone else has provisioned.

Script: `scripts/full_scale/train_reranker_a100.sh <strategy>`; config:
`py/hybridsearch/rerank/configs/a100.yaml`. The header of the script has the exact
`rsync` / `ssh` commands (code + `collection.tsv` + the mined file up, `best/` + logs back).

| Setting | MPS ablation (measured here) | A100 run (prepared) |
|---|---|---|
| group | 1 + 7 | 1 + 15 |
| groups per step | 16 (128 pairs = 4 micro-batches of 32) | 64 (1,024 pairs) |
| precision | fp32 | bf16 autocast |
| max_len (train) | 192 (MPS memory, see BUG_LOG 2026-09-24) | 256 |
| negative pool | top-20 of each filtered pool | top-50 |
| mined queries | first 10k of train50k | all 50k (re-mine with the full `bm25.train50k.trec`) |
| steps | 1,000 | 5,000 (320k groups, 6.4 passes over 50k mined queries) |
| LR / warmup | 3e-5 / 10% | 5e-5 / 5% |

Memory: MiniLM-L6 at 1,024 pairs × 256 tokens in bf16 is far inside 80 GB; if a smaller
card is used, halve `batch_groups` and set `grad_accum: 2` (same effective batch).

**Expected hours: not a measurement.** The only throughput measured is MPS (see
`results/rerank/README.md`); no A100 number exists until the script is run. The script's
cost is bounded by `STEPS` and logs pairs/s every 100 steps, so the first minute on the pod
gives the real rate: hours = 5,000 × 1,024 / (measured pairs/s) / 3600. Budget the pod for
that plus ~10 minutes of setup and rsync, then terminate.

**The binding limit at full scale is data, not GPU:** only 50k train queries have a BM25 run
(`data/subset/train50k`). Scaling further means mining more queries first (run
`scripts/reference_bm25.sh` on a larger train sample and `mine` again), not more steps.
