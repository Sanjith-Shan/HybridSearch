# Bug log

Every non-obvious bug gets a dated entry: symptom, cause, how it was found, fix.
Append only. Real entries only — this is where "tell me about a hard bug" answers come from.


## 2026-09-23 — Harness R(rel=2)@k disagreed with ir_measures by ~0.01 (eval harness, M0)

- **Symptom:** `test_aggregate_matches_ir_measures` failed for `R(rel=2)@100` only: ours 0.2404 vs
  ir_measures 0.2284 on a synthetic graded run; every other metric matched to 1e-9 per query.
- **Cause:** the harness dropped queries that have qrels but no document at the relevance
  threshold from the recall mean. trec_eval (and so pytrec_eval / ir_measures) scores such a query 0
  and still counts it. This matters for TREC DL, where recall is taken at rel >= 2 and a topic can
  have grade-1 judgments only.
- **Found by:** the parity test against ir_measures on synthetic qrels with random grades 0..3.
- **Fix:** `recall_at_k` returns 0.0 for zero relevant docs, and every query in the qrels is counted.
  After the fix, DL19 BM25 R(rel=2)@1000 = 0.7501, which matches Anserini's regression page.

## 2026-09-23 — web: route-change focus landed on a hidden heading (keyboard users stranded)

- **Symptom:** the keyboard-only Playwright path (`web/e2e/keyboard.spec.ts`) failed: after
  choosing "Experiments" in the nav with Enter, the page's `<h1>` was never focused, so
  keyboard and screen-reader users stayed at the top of the old context.
- **Cause:** the search page stays mounted (hidden) while other views show, so the result
  list, scroll position and the click's dwell timer survive Back. The app-level route
  effect focused `main.querySelector('h1')`, which is the *hidden* search page's `h1`
  (first in DOM order). `focus()` on a `display:none` element silently does nothing.
- **Found by:** the e2e assertion `toBeFocused()` on the Experiments heading; passed by
  eye because the page *looked* right.
- **Fix:** each view that replaces the content focuses its own heading via a ref
  (`ExperimentsPage`, `DocPage`); no global DOM query.

## 2026-09-23 — web: `j`/`k` shortcuts dead whenever a checkbox or radio had focus

- **Symptom:** in the keyboard-only e2e path, after tabbing onto the "Rerank" switch,
  pressing `j` did not move to the first result.
- **Cause:** the global shortcut handler skipped any event whose target was an `<input>`,
  to avoid hijacking typing. Checkboxes, radios and switches are `<input>` too, so the
  shortcuts silently stopped working next to the controls users touch most.
- **Found by:** the Playwright keyboard-only test.
- **Fix:** only text-entry input types (and textarea/select/contenteditable) suppress
  single-key shortcuts.

## 2026-09-23 — ONNX query-encoder export shipped without vocab.txt (M5, rerank agent)
- **Symptom:** `data/models/query-encoder/` had `tokenizer.json` and `tokenizer_config.json`
  but no `vocab.txt`, which the C# broker's BERT tokenizer needs.
- **Cause:** transformers 5.x `save_pretrained` on a fast tokenizer writes only `tokenizer.json`;
  4.x also wrote `vocab.txt`. Nothing failed; the file was just absent.
- **Found:** listing the export directory before writing its README.
- **Fix:** `save_tokenizer()` in `dense/export_query_encoder.py` copies `vocab.txt` from the
  model snapshot when `save_pretrained` does not write it; `test_export_parity.py` asserts it exists.

## 2026-09-23 — tiny test tokenizer silently had a 5-token vocab (M5, rerank agent)
- **Symptom:** pair-format test decoded `what is the capital` as `[UNK] [UNK] [UNK] [UNK]`.
- **Cause:** transformers 5 renamed the constructor argument: `BertTokenizerFast(vocab_file=...)`
  is swallowed by `**kwargs` and the tokenizer is built with only the special tokens. No warning.
- **Found:** the pytest assertion on decoded query text (the test was written to catch exactly
  a tokenizer that does not round-trip).
- **Fix:** build with `BertTokenizer.from_pretrained(dir_with_vocab_txt)`. Any code in the repo
  that constructs a tokenizer from `vocab_file=` under transformers 5 has the same bug.

## 2026-09-23 — int8 quantisation pre-processing crashes on the traced BERT graph (M5, rerank agent)
- **Symptom:** `onnxruntime.quantization.shape_inference.quant_pre_process` raised
  `Exception: Incomplete symbolic shape inference` on the exported cross-encoder.
- **Cause:** ORT's symbolic shape inference cannot resolve the attention-mask subgraph the
  TorchScript exporter emits for transformers 5's masking utilities (dynamic seq axis).
- **Found:** first int8 export of the public reranker.
- **Fix:** `skip_symbolic_shape=True` (ONNX shape inference + graph optimisation still run);
  the quantised model's parity vs fp32 is measured and reported rather than assumed.

## 2026-09-23 — synthetic rankers of "known" quality disagreed with their own offline ordering (M9, clicks agent)
- **Symptom:** in the first sensitivity smoke run, team-draft interleaving never reached 80%
  power for `synthetic-oracle-sigma1.0` vs `sigma1.1` under any click model, while A/B on
  abandonment "won" in under 1,000 impressions. The pair was built so that sigma1.0 is better
  (nDCG@10 0.763 vs 0.743).
- **Cause:** each sigma drew its *own* noise (seeds 1000, 1001, ...). Over only 97 topics the
  realised lists differ by far more than the intended 0.1 sigma: the "worse" ranker happened to
  put more grade>=2 passages in its top 10 (6.81 vs 6.73 per query) and more relevant passages
  at rank 1 (clicks@1 0.854 vs 0.807 under cascade-perfect), so interleaving correctly preferred
  it, and nDCG@10 (which weights grade 3 heavily) disagreed. The ordering was only known in
  expectation over seeds, not for the realised pair.
- **Found by:** checking per-query top-10 grade sums and every A/B metric's pool mean for the pair
  after the "impossible" result.
- **Fix:** common random numbers: every sigma scales the same noise draw (seed 1000), so a larger
  sigma is a strictly noisier version of the same ranker. Lesson kept in EXPERIMENTS.md: an
  offline gap of 0.02 nDCG@10 on 97 topics is not a "known ordering" for click metrics.

## 2026-09-23 — probabilistic-interleaving power curves took 28 s each (M9, clicks agent)
- **Symptom:** one (pair, click model) cell of the sensitivity study took ~100 s in smoke mode;
  98% of it inside `power_interleave` for probabilistic interleaving.
- **Cause:** the marginalised PI outcome is a real number; a 200k-impression pool had 3,743
  distinct values, and every trial is a multinomial draw over the support (141 grid points x
  trials x 3,743 categories).
- **Found by:** timing each component separately.
- **Fix:** unbiased stochastic rounding of each outcome to a 0.01 grid (round up with probability
  equal to the remainder): expectation per impression unchanged, variance +<=0.000625, support
  capped at 201 values.

## 2026-09-23 — IPS with estimated propensities sometimes fell far below the logging ranker (M9, LTR)
- **Symptom:** in the BM25-feature LTR run, IPS with propensities estimated from swap
  interventions scored nDCG@10 0.411 on DL19+DL20 at 300k logged sessions (seed 2) while naive,
  true-propensity IPS and the logging ranker all sat at ~0.49; more data did not fix it
  monotonically (0.465 at 100k, 0.489 at 1M for another seed).
- **Cause:** the swap(1,k) estimator gives one ratio per rank k from the few intervened sessions
  for that k. At deep ranks (k up to 20) a ratio rests on a handful of clicks and can come out
  near 0 (floored at 1e-3), i.e. an IPS weight up to 1000 on a few clicks, which then dominates
  the listwise loss. The mean absolute error over ranks 1-10 looked fine (0.05) because the bad
  ranks were 11-20.
- **Found by:** the per-seed learning-curve log; the drop did not track the top-10 propensity MAE.
- **Fix:** `ltr.logsim.monotone_propensity`: weighted isotonic (non-increasing in rank) fit of the
  per-rank ratios, weights = intervention sessions behind each rank. Clipping (studied separately)
  is the other standard fix.
- **Follow-up (same day):** the first isotonic version still produced a 0.359 run (synthetic,
  30k sessions). A deep rank with ~70 intervened sessions and *zero* clicks gave a raw ratio of 0
  with non-trivial session weight, so the monotone fit dragged the whole tail to the 1e-3 floor.
  Weights are now the click counts behind each ratio (its log has variance ~1/c1 + 1/c2), and a
  ratio made of zero clicks is treated as unknown (carried forward), not as eta = 0.

## 2026-09-23 — Vamana graph split into islands on synthetic Gaussian blobs (vector, M3)

- **Symptom.** On 50K synthetic 768-d vectors (500 Gaussian blobs, σ=0.03 per dim,
  normalised) the in-memory Vamana index returned recall@10 = 0.000 at every L_search,
  with only ~100 distance computations per query regardless of L. hnswlib (M=16,
  efC=100) on the same file: recall@10 0.997.
- **Cause.** BFS from the medoid reached 92 of 50,000 nodes: the graph was ~500
  disconnected clusters. In high dimension, isotropic noise makes all points of a blob
  nearly equidistant (nearest kept neighbour at squared distance 0.73, farthest kept
  0.86, random pair 1.99). RobustPrune with α=1.2 on squared distance occludes p' only if
  1.2·d(p*,p') ≤ d(p,p'), i.e. 1.2·0.8 ≤ 0.8, which never holds inside a blob, so the R
  slots fill with same-blob neighbours (sorted first) before any cross-blob candidate is
  reached, and the random initial long edges are all pruned away. HNSW's α=1 heuristic
  still thins such a blob about half the time, and its upper layers link blobs.
- **How found.** The low, L-independent distance count, then a reachability BFS on the
  saved graph (a 10-line numpy script).
- **Fix / lesson.** Not a code bug; it is a data regime where Vamana degrades. The
  build tool now reports `reachable_from_medoid` after every build, and the unit test
  `Vamana.EveryNodeReachableFromMedoid` checks it. On real BGE embeddings the default build
  (R=64, L=100, alpha=1.2) reaches every node (20K and 100K subsets: 100%); smaller
  degrees leave a few stragglers at 100K (R=32: 99.95%, R=16: 99.53%;
  results/vector/build_sweep_100000.build.jsonl). Synthetic data for development must have
  low intrinsic dimension within clusters; isotropic 768-d blobs do not look like text
  embeddings.

## 2026-09-23 — I/O completion counter could be used after it went out of scope (vector, M3)

- **Symptom.** None observed; found by reading the code before the first run.
- **Cause.** `DiskIndex::search` hands W-1 reads to an I/O thread pool and waits on an
  `std::atomic<int>` counter declared on its stack. A worker decrements the counter to 0
  and then calls `notify_all()` on it. Between those two operations the waiting query
  can observe 0, return, and pop the frame, so `notify_all()` touches a dead object: a
  use-after-scope that only fires under exactly the wrong interleaving.
- **Fix.** The counters live in the per-thread search scratch (thread_local, alive for
  the thread's lifetime); a stale wake-up of the next query's wait is harmless because
  the waiter re-checks the count. Covered by `DiskIndex.ConcurrentQueriesAgreeWithSerial`
  (6 query threads sharing a 4-thread I/O pool).

## 2026-09-23 — broker: "hedges sent" over-counts the extra load shards actually see

- **Symptom.** The hedging test asserted `shard-observed Search calls == primaries + hedges sent`
  and failed with 626 expected vs 624 observed (600 primaries, 26 hedges).
- **Cause.** When the primary answers while a hedge's RPC is still being set up, the broker cancels
  the hedge; the cancellation can tear the HTTP/2 stream down before the request reaches the
  server. The broker counted the hedge as sent (and spent a budget token on it), but the shard
  never did any work for it. So the broker-side extra-load ratio is an upper bound on the load
  hedging really adds, not an exact figure.
- **How found.** Exact-equality accounting assertion in `FanOutTests` against FakeShard's own call counter.
- **Fix.** The test now asserts the bound (`primaries < shard calls ≤ primaries + hedges`, gap small);
  the hedging experiment reports both `extraLoadRatioBrokerCounted` and `extraLoadRatioShardObserved`;
  the README documents which one the `hs_hedges_sent_total` metric is.

## 2026-09-23 — macOS F_NOCACHE reads were served from the page cache: "cold" could mean RAM (vector, M3)

- **Symptom (caught before any result was recorded).** The plan was to measure the SSD
  index "cold" by opening it with `F_NOCACHE`, run after a "warm" (page-cache) sweep on the
  same file. A probe (`bench_vec_coldread`, 2,000 random 4 KB preads over a 1 GB file
  written with `F_NOCACHE`) gave p50 **271 us** with F_NOCACHE on the untouched file,
  **1.7 us** through the cache after warming, and **2.1 us** with F_NOCACHE *after warming*.
- **Cause.** On macOS, `F_NOCACHE` stops reads from *populating* the unified buffer cache
  but still serves pages that are already resident. Any cold run after a warm run (or
  after the build, had the build not written with F_NOCACHE) would have measured RAM and
  called it SSD.
- **How found.** Suspicion that `F_NOCACHE` is not `O_DIRECT`, then measuring it rather
  than assuming.
- **Fix.** Cold mode now evicts the index file's cached pages on open with
  `mmap` + `msync(MS_INVALIDATE)` (probe after eviction: p50 **232 us**, back to device
  latency); Linux uses `O_DIRECT` plus `posix_fadvise(DONTNEED)`. The build writes the
  index with `F_NOCACHE` so it does not leave it cached. Every disk result row also
  reports `us_per_io_round`, so a "cold" row with microsecond reads would stand out.

## 2026-09-23 — broker: an unwritable event-log directory stopped the whole broker

- **Symptom.** The Docker image started, logged `UnauthorizedAccessException: Access to the path '/data' is denied`,
  then shut down — search was unavailable because *click logging* could not create its directory.
- **Cause.** The container runs as the non-root `app` user and `/data` did not exist (no volume mounted).
  `EventLogWriter` (a BackgroundService) threw from `Directory.CreateDirectory`, and .NET's default
  `BackgroundServiceExceptionBehavior.StopHost` turns any background-service exception into a host shutdown.
  Every in-process test used a writable temp dir, so no test could see it.
- **How found.** Smoke-running the built container without a data volume.
- **Fix.** The writer now catches IO/permission errors, reports `eventLog.status = "unwritable: …"` in `/readyz`,
  and drains the queue counting each record in `hs_events_dropped_total{reason="write_error"}` (degrade, never silent).
  The image creates `/data/events` owned by `app`. Regression test: `EventLogFailureTests`.

## 2026-09-23 — lexical parity: Anserini's tie adjuster is not order-preserving, so a run file's MRR is not its ranking's MRR

- **Symptom.** The textbook-BM25 dev run scored MRR@10 **0.184510** when evaluated from its TREC file but
  **0.184593** when evaluated in the engine's own rank order. The Lucene-model run showed the same effect,
  smaller (0.183983 vs 0.183981).
- **Cause.** trec_eval (and our harness, which copies it) ignores the rank column and re-sorts by score, breaking
  ties by docid. To stop that, `hs_lex_search` formats scores exactly like Anserini's `ScoreTiesAdjusterReranker`:
  round to 1e-4, then subtract 1e-6 × (position in a tie run). But the tie test compares against the *already
  perturbed* previous score, so a long run of near-ties drifts down by more than 1e-4 and can drop below the next
  group; re-sorting by printed score then reorders the top 10. Counted directly: re-sorting our files by printed
  score changes the top-10 order of **4** of 6,980 dev queries (Lucene model) and **9** (textbook model).
  Anserini's own reference file has the same property.
- **How found.** Chasing a fourth-decimal disagreement between two evaluations of the same run while writing
  `results/lexical/parity.md`.
- **Fix.** Kept Anserini's formatting in run files, because the reference run uses it and the two files should be
  compared like for like. `py/hybridsearch/lexical/parity.py` also evaluates the raw (`--raw-out`) runs in true
  rank order, and parity.md reports both numbers.

## 2026-09-23 — lucene_ref oracle: sorting ties by docid string needs BinaryDocValues, not SortedDocValues

- **Symptom.** The Lucene reference scorer died with `IllegalStateException: unexpected docvalues type SORTED for
  field 'id' (expected=BINARY)` the first time it ran a query.
- **Cause.** To reproduce Anserini's ranking exactly the oracle sorts with Anserini's `BREAK_SCORE_TIES_BY_DOCID`
  (`SortField.Type.STRING_VAL` on `id`). `STRING_VAL` reads *binary* doc values; the oracle indexed the id as
  `SortedDocValuesField` (which `SortField.Type.STRING` would want).
- **How found.** First run of `tools/lucene_ref/run.sh score` on the test corpus.
- **Fix.** Index the id as `BinaryDocValuesField`. It also documents a parity fact used everywhere downstream:
  Lucene/Anserini break exact score ties by docid *string* ("10" < "9"), while HybridSearch's contract breaks them
  by numeric ID, so tied documents can legitimately appear in a different order.

## 2026-09-23 — `faiss.omp_set_num_threads` aborts the process when torch is loaded (M5, rerank agent)
- **Symptom:** `Fatal Python error: Aborted` inside `faiss/swigfaiss.py omp_set_num_threads`,
  no Python exception, whole pytest process killed.
- **Cause:** torch and faiss-cpu wheels each ship their own OpenMP runtime; once torch's
  is initialised, faiss's OMP call hits a duplicate-runtime abort.
- **Found:** adding a 4-thread CPU cap (the laptop is shared) and rerunning the mining test.
- **Fix:** cap threads through `OMP_NUM_THREADS` (set when `hybridsearch.rerank` is imported)
  and `torch.set_num_threads`, never `faiss.omp_set_num_threads` in a process that uses torch.

## 2026-09-24 — Range partitioning put 86% of the relevant passages on one shard

**Symptom.** Chaos experiment `slice-down`, on the 1M subset split into 4 shards × 2 replicas. Both
replicas of slice 3 were hung with SIGSTOP. Dev MRR@10 fell from 0.4119 to 0.0336, keeping only 8%.
Losing a quarter of the documents should have cost roughly a quarter of the quality.

**Cause.** The shards were contiguous passage-ID ranges. MS MARCO passage IDs are not random with
respect to the queries: passages are grouped by the split their source query came from, so the dev
split's passages cluster at high IDs. Of 7,433 dev-relevant passages, slice 0 held 466, slice 1 held
344, slice 2 held 222 and slice 3 held 6,401 (86.1%). That makes one shard both a quality single point
of failure and a load hot-spot, since it holds the documents that dev queries actually retrieve.

**How it was found.** The chaos harness reports quality alongside latency. A latency-only chaos
test would have shown a healthy-looking system: results still returned, with the failed slice
reported. A manual reproduction first appeared to contradict it. The cause of that was zsh's
1-based arrays in the reproduction shell, which stopped one replica of two different slices instead
of both replicas of one.

**Fix.** Modulo partitioning (`global id % N`) for both the lexical and vector shards, plus a broker
`Topology:Partitioning=modulo` option so the rare lookup that is not tied to a search reply routes
correctly. Global BM25 statistics keep the merged top-k identical to the single index under either
scheme. The range shards are kept as the comparison point, and results/chaos/ has both.

## 2026-09-24 — reranker training on MPS grew to 15 GB and pushed the laptop into swap (M5, rerank agent)
- **Symptom:** a 40-step throughput probe made no progress for 10 minutes; `top` showed the
  training process at 15 GB, system swap at 18 of 18.4 GB, the process in state `U` (paging).
- **Cause:** two effects stacked. (1) Activation memory: MiniLM-L6 on MPS runs SDPA through
  the math path and keeps the full attention matrices for backward. Measured with
  `torch.mps.current_allocated_memory()`: one forward of 32 pairs × 256 tokens holds 3.25 GB,
  so the configured 128-pair batch needed >10 GB. (2) No cap: MPS's default high-watermark lets
  one process take most of unified memory, so it swapped instead of failing.
- **Found:** `top -o mem` showed the hog was this process, not another agent's; then a
  forward/backward memory probe at fixed shapes.
- **Fix:** `torch.mps.set_per_process_memory_fraction(0.3)` (about 4 GB: OOM instead of swap),
  32-pair micro-batches with 4-step gradient accumulation (same 128-pair effective batch),
  training max_len 192 (eval stays 512), padding to multiples of 32 so MPS caches few graph
  shapes, `torch.mps.empty_cache()` at log steps, and MPS driver memory logged every
  `log_every` steps (steady at 3.1 GB).

## 2026-09-24 — The slice-down chaos run exposed two broker problems and one harness bug

1. **A dead slice burned the whole request budget.** When health checks had marked both replicas of a
   slice unhealthy, `PickPrimary` still fell back to one of them, and every request waited the full
   300 ms deadline for a reply that was never coming. The degradation planner then took that lost
   budget out of the dense path and the reranker. So slice-down cost 42% of MRR@10 instead of the
   ~25% that losing a quarter of the documents should cost, and p50 was 310 ms. The fix is to fail
   fast when every replica of a slice is known unhealthy (`Hedging:FailFastWhenSliceDown`, default
   on; recovery comes through the next successful health probe). Result on the 4×2 modulo cluster:
   MRR retained went from 0.576 to 0.770, and fault-phase p50 from 310 ms to 36 ms.
2. **The first baseline was taken before the planner had any latency samples.** It degraded 95% of
   those requests to lexical-only from its conservative priors. The chaos harness now warms up
   before measuring anything.
3. **`run_cluster.sh stop` returned before the processes had exited.** A restarted shard could lose
   the race for its port and exit, silently leaving its slice with one replica. The symptom: the
   "slow replica, no hedging" experiment showed no latency change, because the 50 ms had been
   injected into a replica that no longer existed. Stop now waits for exit, and start verifies that
   every shard logs "listening".
4. **The degradation planner could lock itself into lexical-only.** Stage costs are p90 over a
   60-second window. Once one slow window made dense look over budget, the planner skipped dense.
   A skipped stage records no new samples, so the estimate never recovered, and once the window
   emptied the conservative priors kept it tripped. One chaos baseline ran 100% lexical-only because
   of this. The fix is exploration: 5% of requests (`Degradation:ProbeFraction`) run the full plan
   regardless of estimates, and stage deadlines still bound them.
