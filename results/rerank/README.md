# M5 reranker results (measured 2026-09-23/24, Apple M3 Pro, macOS; laptop shared with other agents)

Every number below has its file and `.meta.json` (CPU, OS, torch/ORT versions, date, command,
load average). Candidates everywhere: **Anserini BM25 default (k1=0.9, b=0.4) over the full
8.8M collection, top-100**, reranked and scored through `hybridsearch.eval` (trec_eval parity).

## Headline: HybridSearch's own reranker is much weaker than the public model and worse than BM25 alone

| | DL19 nDCG@10 (43 q) | DL20 nDCG@10 (54 q) | dev RR@10, first 200 judged q |
|---|---|---|---|
| BM25 only (no rerank) | 0.5058 | 0.4796 | 0.1906 |
| public cross-encoder/ms-marco-MiniLM-L6-v2 (torch MPS) | **0.7267** | **0.6747** | **0.3780** |
| ours, bm25-hard, 500 steps (torch MPS) | 0.3950 | 0.2902 | 0.1014 |
| ours, ORT CPU fp32 | 0.3950 | not run | 0.1014 |
| ours, ORT CPU int8 | 0.3922 | not run | 0.1050 |

Files: `eval/public.bm25full.{dl19,dl20}.json`, `eval/public.bm25full-q200.dev.json`,
`eval/ours.bm25full.{dl19,dl20}.json`, `eval/ours.bm25full-q200.dev.json`,
`eval/ours-ort-{fp32,int8}.bm25full.dl19.json`, `eval/ours-ort-{fp32,int8}.bm25full-q200.dev.json`;
all rows in `summary.md`. The public model card's 0.3901 dev MRR@10 / 0.7430 DL19 are on
different candidate depths; our public DL19 through this harness at depth 100 is 0.7267.
**Full dev (6,980 queries) was not run**: the 200-query dev sample is the first 200 judged
dev.small queries by qid and is not comparable to 0.3901 beyond direction.

Why ours is weak: 500 optimizer steps × 128 pairs = **64,000 pairs (8,000 groups) seen** from a
base MiniLM never trained on MS MARCO, vs. the public model's far larger distillation run. At
this budget it has not learned to beat BM25's order (train_tune validation 0.1410 vs BM25 0.2063).

## Training runs (M3 Pro MPS, fp32; `train/`)

| run | LR | schedule | steps done | pairs seen | best train_tune RR@10 (500 q, BM25 top-30) |
|---|---|---|---|---|---|
| bm25-hard (`train/mps_bm25_1000_lr3e-5.*`) | 3e-5 | 1,000 | 1,000 (resumed once from step 250 after an MPS OOM; resumed loss matched to 5 decimals) | 128,000 | 0.1275 (step 1000) |
| LR probe (`train/lrprobe_1e-4.*`) | 1e-4 | 500 | 500 | 64,000 | 0.1203 |
| bm25-hard (`train/mps_bm25_2000_lr1e-4_stopped.log.jsonl`) **= exported model** | 1e-4 | 2,000 | ~950, stopped on deadline | ~121,600 | **0.1410 (step 500)** |
| BM25 order, no model | | | | | 0.2063 |

Throughput logged in the training logs: 69–122 pairs/s (fwd+bwd, max_len 192), load average
6–34 at the time; dev-signal-only.

## NOT done (stated plainly)

- **The negative-mining ablation (random vs bm25-hard vs dense-hard) was not completed.** Only
  bm25-hard was trained. Random and dense-hard runs at the 2,000-step budget were stopped
  unstarted/at step 0 on the wrap-up deadline. The mined pools for all three exist
  (`data/rerank/mined/train50k.jsonl`, stats in `mining_stats.json`) and
  `scripts/rerank_train_mps.sh random dense` runs them.
- **Cascade curve** (k ∈ {10,20,50,100,200}): code + tests exist (`hybridsearch.rerank.cascade`),
  not run; no `cascade.png`.
- Full dev (6,980 q), DL20 through ORT, 1M-subset / hybrid candidates: not run.

## int8 vs fp32 (ORT CPU)

Quality: DL19 nDCG@10 0.3950 → 0.3922 (−0.0028); dev-200 RR@10 0.1014 → 0.1050 (+0.0036): within
noise at these query counts. Logit deviation on 500 dev pairs: mean 0.086, max 0.90
(`export_reranker_parity.json`). Size 91 MB → 59 MB (embeddings stay fp32).

Speed (`bench_reranker.json`, 7 intra-op threads, 10 requests per batch size, one query + b BM25
candidates per request, `session.run` only). **Measured at load average 52: heavily contended,
dev-signal-only, not a claim.**

| batch | fp32 p50 ms | int8 p50 ms | fp32 pairs/s | int8 pairs/s |
|---|---|---|---|---|
| 1 | 7.7 | 5.0 | 111 | 157 |
| 16 | 310.5 | 97.9 | 56 | 158 |
| 64 | 1139.0 | 591.8 | 55 | 92 |
| 100 | 1809.0 | 1638.0 | 53 | 43 |

The batch-100 int8 row is slower than fp32, which under this load is noise, not a finding.
Re-measure on a quiet or pinned box before using any of these.

## Exports

- `query_encoder_parity.json`: BGE-base ONNX vs sentence-transformers, 500 dev queries, min cosine 0.99999988.
- `export_reranker-public_parity.json`: public reranker ONNX fp32 max |Δlogit| 7.6e-6; int8 mean 0.027.
- `export_reranker_parity.json`: ours, fp32 max |Δlogit| 6.4e-6; int8 mean 0.086.
