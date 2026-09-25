# HybridSearch

A search engine over MS MARCO's 8,841,823 passages, built from the index up. It has a
Lucene-exact BM25 engine with block-max WAND and a DiskANN-style vector index on SSD, both in
C++20. It fuses them, with the fusion weights tuned on training queries and every gain
significance-tested, then reranks with a cross-encoder served through ONNX Runtime. A C# broker
fans each query out to sharded gRPC servers with deadlines, hedged requests and a visible
degradation policy. The React front end is accessibility-tested: axe-core WCAG 2.2 AA rules, 0 violations, plus a keyboard-only end-to-end test. There is also a feedback loop: click
logging, team-draft interleaving, and counterfactual learning-to-rank.

Every number below links to the results file it came from. Every latency figure is labelled with
its hardware. Anywhere HybridSearch loses to the incumbent, the loss is published.

```
Browser (React + TypeScript)            web/        instant results, autocomplete, "why this result", experiments page
   │  REST + SSE, W3C traceparent
   ▼
Broker (C#, ASP.NET Core)               broker/     BGE query encoding (ONNX Runtime), fan-out with deadlines,
   │  gRPC (proto/)                                  budgeted hedging, fail-fast, weighted fusion, cross-encoder
   ▼                                                 rerank, degradation policy, interleaving, event log, OTel
Shard servers (C++20) × 4 slices × 2    engine/     lexical: Lucene-exact BM25, VByte/BP128, MaxScore/WAND/BMW
   ├── lexical index                                 vector: Vamana graph + full vectors on SSD, PQ codes in RAM
   └── vector index
Offline (Python)                        py/         eval harness, significance, fusion study, reranker training,
                                                    click models, interleaving sensitivity, counterfactual LTR
```

## Results

All MS MARCO dense and hybrid numbers are on a **1M-passage subset**. It contains every judged
passage plus a seeded random sample, so absolute scores run well above full-collection numbers,
and only the deltas transfer. The lexical parity gate runs on the **full 8.8M collection**.
Latencies come from an Apple M3 Pro shared with other workloads (**dev-signal only**; resume-grade
latency comes from a pinned Linux box, `scripts/full_scale/`).

### Lexical engine: exact parity with Anserini (M1, M2)
- **MS MARCO dev MRR@10 = 0.1840** over all 8,841,823 passages, against Anserini's 0.1840
  (0.183983 vs 0.183988). DL19 and DL20 nDCG@10 are **0.5058 / 0.4796**, identical to Anserini.
  Top-10 disagreements, bucketed: 6,799 of 6,980 queries identical, 181 differ only in tie order,
  0 unexplained. [`results/lexical/parity.md`](results/lexical/parity.md)
- **Analyzer: 0 token mismatches against Lucene itself.** That covers 10,000 MS MARCO passages
  (396,841 tokens) and every Unicode code point in 13 contexts. Lucene's UAX#29 tokenizer grammar is
  reimplemented as a DFA. The term dictionary is byte-identical to Anserini's prebuilt index, with
  2,660,824 terms.
- **Why textbook BM25 does not match:** Lucene stores document length in one lossy byte. The textbook
  formula changes the top-10 of 3,253 dev queries. Every one of those changes is explained by that
  quantisation.
- **Pruning, proven safe:** all MaxScore / WAND / block-max WAND × VByte / BP128 combinations return
  byte-identical top-1000 lists (IDs and float bits) to exhaustive search on all 6,980 dev queries
  over the full index. The check runs in CI, and a mutation test (bounds scaled 0.97×) fails it.
  **Block-max WAND scores 566× fewer documents** than exhaustive (1,633 vs 923,730 per query).
  [`results/lexical/pruning.md`](results/lexical/pruning.md)
- **Compression:** BP128 is 13.55 bits/posting against VByte's 18.90. The NEON decoder runs at
  3.6 ns/posting against 12.5 for VByte.
- **Distributed BM25:** shards are built with the collection's global statistics, so the merged
  top-k from 4 shards is identical to the single index (0 mismatches in 2,000 comparisons).
  [`results/lexical/sharding.md`](results/lexical/sharding.md)

### Vector index: Vamana + DiskANN on SSD (M3)
- **Merged top-10 from 4 shards equals global brute force on 2,000 of 2,000 queries** (exact mode).
  Approximate recall@10 is 0.957 / 0.987 / 0.992 at L = 20 / 100 / 200, with **96 MB of RAM for 1M
  768-d vectors** (PQ codes only; full vectors and graph stay on SSD in 4 KB blocks).
- **What the approximation costs on dev MRR@10:** −0.0191 at L=20, −0.0052 at L=100, −0.0027 at
  L=200, all against exact search. [`results/vector/`](results/vector/), [`docs/VECTOR.md`](docs/VECTOR.md)
- **Built the way the paper does it, with limited memory:** 5 overlapping k-means partitions per
  shard under a 0.4 GB budget, then merged. Construction is deterministic: rebuilds are
  byte-identical at any thread count.
- **Against the incumbents** (100K vectors, one thread, queries per CPU-second at matched recall):
  Vamana is level with FAISS HNSW (−8% at recall 0.95, +12% at 0.99) and 3–4× ahead of hnswlib on
  arm64. [`results/vector/compare_100k.*`](results/vector/)

### Hybrid retrieval, with statistics (M4)
Tuned **only on 2,000 MS MARCO train queries**. Holm correction runs over a 36-comparison family.
[`results/m4/fusion.md`](results/m4/fusion.md)

| dataset | BM25 | dense (BGE) | weighted hybrid | Δ vs dense [95% CI] | Holm p |
|---|---|---|---|---|---|
| MS MARCO dev MRR@10 | 0.4209 | 0.6538 | **0.6630** | +0.0092 [+0.0065, +0.0119] | 0.0018 |
| DL19 nDCG@10 | 0.6429 | 0.7955 | 0.8071 | +0.0116 [−0.0005, +0.0249] | 0.78 |
| DL20 nDCG@10 | 0.6525 | 0.7949 | 0.8004 | +0.0055 [−0.0038, +0.0150] | 1.0 |
| BEIR SciFact nDCG@10 | 0.6789 | 0.7404 | 0.7585 | +0.0181 [+0.0068, +0.0300] | 0.020 |
| BEIR NFCorpus nDCG@10 | 0.3217 | 0.3735 | 0.3791 | +0.0056 [+0.0015, +0.0100] | 0.096 |
| BEIR FiQA nDCG@10 | 0.2361 | 0.4062 | 0.4167 | +0.0105 [+0.0059, +0.0151] | 0.0018 |

- **Equal-weight RRF is significantly worse than dense alone on dev** (0.5864 vs 0.6538). BM25 is so
  much weaker than BGE in-domain that equal weighting drags dense down. The broker therefore serves
  weighted z-score fusion (α=0.8, tuned on train queries at the serving depth of 100;
  [`results/m4/serving_config.json`](results/m4/serving_config.json)). A shared golden fixture
  checks that its fusion is identical to the fusion that was evaluated.
- BGE was trained on MS MARCO, so the in-domain numbers flatter dense. BEIR sits next to them for
  that reason.

### Reranker (M5)
The cross-encoder is served through ONNX Runtime on CPU; ONNX output matches PyTorch to a max
logit difference of 7.6e-6. BM25 candidates come from the full 8.8M collection.
[`results/rerank/`](results/rerank/)

| DL19 / DL20 nDCG@10 | BM25 | + public MiniLM-L6 reranker | + HybridSearch's own MiniLM |
|---|---|---|---|
| rerank BM25 top-100 | 0.5058 / 0.4796 | **0.7267 / 0.6747** | 0.3950 / — |
| rerank BM25 top-200 | | **0.7407 / 0.7006** (model card: 0.7430 on DL19) | |

- **The cascade curve picks the operating point.** Reranking the top k gives DL19 nDCG@10 of
  0.554 / 0.643 / 0.698 / 0.727 / 0.741 at k = 10 / 20 / 50 / 100 / 200. On this laptop the added
  CPU time is 0.22 / 0.36 / 1.0 / 3.8 / 13 s. The broker serves **k=20**, which captures 58% of the
  full gain at 3% of the cost.
- **Our own reranker lost, and badly.** MiniLM trained from scratch on hard negatives mined from
  HybridSearch's own BM25, but for only 500 steps (64K pairs) on the laptop GPU. It scores
  **below BM25** (DL19 0.395), so the broker serves the public model. The training pipeline is done:
  hard-negative mining with leak and false-negative filters, a listwise loss, resumable
  checkpoints, and int8 export with a quality delta of −0.003. The full run and the
  random / BM25-hard / dense-hard ablation are scripted for a rented A100
  (`scripts/full_scale/train_reranker_a100.sh`) and not yet run.
- **int8 vs fp32:** quality changes by −0.0028 nDCG@10 and the model shrinks from 91 MB to 59 MB.
  Batch-16 latency drops from 311 ms to 98 ms (dev-signal only).

### Serving under failure (M6)
Every chaos experiment replays the **same** dev queries in its baseline, fault and recovery phases,
and reports **MRR@10 under failure** alongside latency (4 slices × 2 replicas, hybrid, 300 ms
budget). [`results/chaos/`](results/chaos/), [`bench/chaos/run_all.sh`](bench/chaos/run_all.sh)

| fault | MRR@10 retained (range over the post-fix runs) |
|---|---|
| one replica hung (SIGSTOP) | 0.95–1.00 |
| one replica +50 ms (hedging on / off) | 0.99–1.00 / 1.00 |
| one replica CPU-starved | 0.99–1.00 |
| a whole slice down (both replicas hung) | **0.71–0.77** (0.08 with range sharding, 0.58 before fail-fast) |

These come from a shared laptop under load average 10–25, with other jobs running. Run-to-run
variance is visible even in the baselines, which is why the table gives ranges and not a best run.
The quality figures are paired (the same queries in each phase). The latency effect of hedging did
not separate from noise here, so no p99 claim is made. That measurement belongs on the pinned
Linux box. Per-phase tables with load averages: `results/chaos/mod-hybrid/TABLE.md` (the last run),
with the earlier runs in git history.

Chaos testing found four real problems, each written up in [`docs/BUG_LOG.md`](docs/BUG_LOG.md):
- **Range sharding put 86% of dev-relevant passages on one shard,** because MS MARCO IDs are grouped
  by source split. Fixed by switching to `id mod N`.
- **A dead slice burned the whole latency budget.** Fixed by failing fast on slices whose every
  replica is unhealthy.
- **The degradation planner could lock itself into lexical-only.** Fixed with exploration probes.
- **The cluster restart had a port race.**

Hedging is budgeted at ≤5% extra shard load, and that extra load is a first-class metric.
Observability: OpenTelemetry traces run from the browser through the broker to every shard, plus
Prometheus metrics, a Grafana dashboard, and a written SLO with a burn-rate error budget
([`docs/SLO.md`](docs/SLO.md), [`deploy/`](deploy/)).

### Product (M7)
- **Search UI:** instant streamed results (lexical first, then the final list) and highlighted
  best-window snippets.
- **Autocomplete:** built over 808K real Bing queries from MS MARCO's training set, about 22 µs per
  lookup.
- **"Why this result" panel:** per-term BM25 contributions, dense similarity, and the result's rank
  at each stage.
- **Experiments dashboard.**
- **Accessibility, tested against WCAG 2.2 AA rules:** axe-core finds zero violations on every page and state, in light and dark mode.
  There is a keyboard-only end-to-end test, and a live region announces results. A manual VoiceOver
  pass has not been done, so this is not a full conformance claim ([`web/ACCESSIBILITY.md`](web/ACCESSIBILITY.md)).

### The feedback loop (M9)
The users here are **simulated**: click models (PBM, Cascade, DBN) whose relevance comes from real
TREC DL graded judgments. No number below is real user traffic. The event pipeline itself is real
and runs on the live UI. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- **Team-draft interleaving needs 1.4–4× fewer impressions than A/B** (median over 9 user models) to
  detect the better of two real rankers at p<0.05 with 80% power. It beats the best A/B metric in
  98 of 99 cells. The published "1–2 orders of magnitude" shows up only for small quality gaps.
  A/A false-positive rate: 3–7% at α=0.05.
- **Counterfactual learning to rank from clicks** (nDCG@10 on held-out DL19+DL20):

  | ranker | nDCG@10 |
  |---|---|
  | logging BM25 | 0.491 |
  | naive click-trained | 0.560 |
  | IPS with propensities estimated from swap interventions | **0.601** |
  | oracle | 0.609 |

  IPS recovers 84% of the gap between naive and oracle. Overestimating position bias is worse than
  not correcting at all.
- The C# broker and the Python analysis are held to shared golden fixtures for interleaving,
  experiment assignment and fusion.

## Run it

```bash
scripts/build_engine.sh engine/build-srv          # C++ engine + gRPC shard server (Release)
ctest --test-dir engine/build-srv                 # includes the pruning and sharding equality gates
~/.dotnet/dotnet test broker/HybridSearch.sln     # broker
(cd web && pnpm install && pnpm test && pnpm e2e) # UI, keyboard-only path, axe-core
.venv/bin/python -m pytest                        # evaluation, stats, fusion, reranker, click models

scripts/data_download.sh && scripts/data_subset.sh && scripts/dense_encode.sh   # data (checksummed)
scripts/build_serving_indexes.sh                  # 4 modulo shards: lexical (global stats) + DiskANN
scripts/run_cluster.sh start                      # 8 shard processes + broker on :8080
(cd web && pnpm dev)                              # UI on :5173
scripts/smoke_e2e.sh                              # one real query through every layer
bench/chaos/run_all.sh hybrid false               # every chaos experiment, latency + MRR@10
scripts/m4_all.sh                                 # hybrid fusion study with significance
scripts/m9_all.sh                                 # interleaving sensitivity + counterfactual LTR
docker compose -f deploy/docker-compose.yml up    # same system in containers, with Jaeger/Prometheus/Grafana
```

## Honesty notes
- **The embeddings are BGE-base-en-v1.5 vectors that HybridSearch encoded itself** for the 1M
  subset, on the laptop GPU in fp32. They match sentence-transformers at cosine ≥ 0.9999998.
  Waterloo's precomputed vectors are what the 8.8M run (`scripts/full_scale/`) uses; that run is
  scripted but not yet executed.
- **Latencies here are from a shared, unpinned macOS laptop** and are development signal only. The
  ratio against Lucene's query latency is queued for a quiet machine
  ([`results/lexical/TIMING_QUEUE.md`](results/lexical/TIMING_QUEUE.md)).
- **Losing to the incumbents is reported,** not hidden: equal-weight RRF vs dense, hnswlib vs FAISS,
  and the public reranker.
- MS MARCO and DiskANN are Microsoft's. That is why they were chosen, not a claim of any
  affiliation.
