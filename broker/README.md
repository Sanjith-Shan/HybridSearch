# HybridSearch broker (C#, ASP.NET Core, .NET 10)

The broker sits between the React UI and the C++ shards. For each query it: validates the
request → picks the experiment arm → plans degradation → encodes the query (ONNX Runtime) → fans
out to every slice with a deadline and hedged requests → merges the per-slice top-k lists → fuses
(RRF or weighted) → reranks with a cross-encoder (ONNX Runtime) → fetches snippets → responds (or
streams). It also serves autocomplete, "did you mean", event ingestion, and experiment results.
The contract is `docs/ARCHITECTURE.md` plus `proto/hybridsearch/v1/shard.proto`.

```
broker/
  HybridSearch.sln
  HybridSearch.Contracts/     C# generated from ../proto (client + server stubs), shared hashing embedding
  HybridSearch.Broker/        the web app (minimal APIs)
    Api/            endpoints, DTOs, request validation, ULID request ids
    Search/         SearchService (query path), Fusion, Degradation policy + cost model, UTF-8→UTF-16 highlights
    Shards/         topology, fan-out + hedging, rolling latency histogram, hedge token bucket, deadlines
    Models/         ONNX query encoder (BGE, CLS pooling) and cross-encoder reranker, BERT WordPiece
    Autocomplete/   compact radix trie with top-K per node, normaliser, SymSpell-style speller
    Experiments/    config + assignment, team-draft interleaving + credit, event log, analyser, statistics
    Observability/  Prometheus metrics (names are a contract), OpenTelemetry spans
  HybridSearch.FakeShard/     test double speaking shard.proto (tiny BM25 + hashing vectors, fault injection)
  HybridSearch.Broker.Tests/  xUnit: unit, property, FakeShard-backed and WebApplicationFactory tests
  HybridSearch.Broker.Bench/  BenchmarkDotNet suites + an open-loop hedging experiment
  Dockerfile
```

## Run

.NET 10 SDK: `export PATH="$HOME/.dotnet:$PATH" DOTNET_ROOT="$HOME/.dotnet"`.

```bash
# tests (from the repo root)
dotnet test broker/HybridSearch.sln

# two fake shards (contiguous halves of the built-in 32-passage corpus, or --corpus <pid\ttext tsv>)
dotnet run --project broker/HybridSearch.FakeShard -- --port 50051 --slice 0 --num-slices 2
dotnet run --project broker/HybridSearch.FakeShard -- --port 50052 --slice 1 --num-slices 2 --latency-ms 5 --tail-prob 0.02 --tail-ms 200
#   more knobs: --jitter-ms --fail-rate --partial-rate --no-vector --ignore-budget --dim --max-docs

# broker on :8080 against them; hashing-dev makes dense mode work without a model (clearly labelled, see below)
Broker__Topology__Slices__0__Id=0 Broker__Topology__Slices__0__Replicas__0=http://localhost:50051 \
Broker__Topology__Slices__1__Id=1 Broker__Topology__Slices__1__Replicas__0=http://localhost:50052 \
Broker__Models__QueryEncoderKind=hashing-dev \
dotnet run --project broker/HybridSearch.Broker

curl 'localhost:8080/api/search?q=capital%20of%20peru&k=5&explain=true'
curl -N 'localhost:8080/api/search/stream?q=paris'
curl 'localhost:8080/api/suggest?prefix=how%20to&k=8'
curl 'localhost:8080/readyz'; curl localhost:8080/metrics

# benchmarks (write results/broker/*.json + .meta.json)
dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- trie-stats
dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- hedging 100 30
dotnet run -c Release --project broker/HybridSearch.Broker.Bench -- bdn --filter '*' --job short

# container (context is broker/, proto passed as a named context so data/ is never sent)
docker build -f broker/Dockerfile --build-context proto=proto -t hybridsearch-broker broker
```

## Configuration

`HybridSearch.Broker/appsettings.json`, section `Broker` (override with env vars `Broker__...`).
Relative data paths resolve against `Broker:DataRoot` (default `../../data` from the project dir,
`/data` in the container).

| Key | Default | Meaning |
|---|---|---|
| `Topology:Slices[i]:Id`, `:Replicas[j]` | slice 0 at :50051 | one entry per disjoint doc-ID slice; ≥1 replica each |
| `Search:DefaultDeadlineMs` | 300 | per-request budget (`deadlineMs` overrides, 10–10000) |
| `Search:CandidateDepth` / `RerankDepth` | 100 / 50 | per-retriever candidates; cross-encoder depth |
| `Search:Fusion`, `RrfK`, `Alpha`, `Normalization` | rrf, 60, 0.5, minmax | fusion; α weights dense |
| `Search:DenseBeamWidth` / `DenseBeamWidthShrunk` | 0 (shard default) / 32 | beam before / after degradation step 2 |
| `Search:ShardGraceMs` | 3 | kept back from each shard's budget so a partial reply beats the gRPC deadline |
| `Hedging:Quantile`, `BudgetRatio`, `BurstTokens`, `MinSamples` | 0.95, 0.05, 10, 50 | hedge at rolling p95; ≤5% extra load |
| `Degradation:InFlightThresholds` | [64,128,256] | load-shedding floors for levels 1/2/3 |
| `Degradation:Default*Ms` | see `BrokerOptions.cs` | stage-cost priors until enough samples |
| `Models:QueryEncoderDir` / `RerankerDir` | `models/query-encoder`, `models/reranker` | `model.onnx` (+ `model.int8.onnx`) + `vocab.txt` |
| `Models:QueryEncoderKind` | onnx | `hashing-dev` = feature-hashing stand-in for local UI work only |
| `Autocomplete:QueryLogPath` | `raw/msmarco/queries.train.tsv` | built in the background at startup |
| `Events:Directory`, `QueueCapacity`, `MaxBatch` | `events`, 10000, 100 | JSONL event log |
| `Experiments:ConfigPath` | `experiments.json` | see ARCHITECTURE.md |
| `Otel:Exporter` | otlp | `none` disables export; endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT` |

If a model directory is missing, nothing fails: the capability is reported as unavailable in
`/readyz`, and every affected response records it (`lexical_only:encoder_unavailable`,
`reranker_skipped:model_unavailable`). **As of this writing the models under `data/models/` do not
exist yet, so the ONNX paths have not run against the real BGE / MiniLM exports.** The ONNX code
handles both `last_hidden_state` [1,L,d] and pooled [1,d] encoder outputs, and [B,1] or [B,2]
reranker heads. It picks the input names (`input_ids`, `attention_mask`, `token_type_ids`) by
substring, but none of this has been exercised yet.

## Design notes

### Deadlines
A request's budget becomes an absolute monotonic deadline. The degradation planner reserves the
estimated fetch cost (plus the rerank cost if the reranker is planned), and the shard phase gets
what is left. Each attempt sends `deadline_budget_us = remaining − grace` in the request, and
`remaining` as the gRPC deadline. The grace lets a shard that stops at its budget and returns a
partial top-k get its reply back before the transport deadline kills the call. Without the grace,
the two deadlines race, and partial answers show up as failures. Snippet fetches get at least
`MinFetchMs` even after the budget is spent, because a result without text is useless. If a fetch
still fails, the response records `snippets_unavailable:N` among the degradation steps.

### Hedging, and why it has a budget
Per slice, a lock-free rolling histogram (log-linear buckets, ~3% error, 6 sub-windows over
60 s) tracks attempt latency. If the primary has not answered by the slice's p95, the broker sends
a duplicate to another replica, takes the first success, and cancels the loser. The cancel is
real: gRPC cancellation reaches the loser's server, and the FakeShard tests see it arrive.

The budget matters because an unbudgeted hedge policy is a load amplifier at the worst moment.
When a shard is slow for everyone (GC storm, cold cache, noisy neighbour), every request passes
p95 and every request gets hedged. That doubles the load on a fleet that is already struggling,
which makes it slower, which drives more hedging. A token bucket fixes this. Every primary
deposits `BudgetRatio` tokens (0.05), a hedge spends one, and the bucket starts empty, so at all
times hedges ≤ 5% of primaries. That bound is tested under concurrency, and in an incident test
where all requests become slow. Refused hedges are counted (`hs_hedges_denied_total`), so the
budget shows how often it bit.

The extra load is exposed as `hs_hedges_sent_total / hs_shard_primary_requests_total` and as the
gauge `hs_hedge_extra_load_ratio`. One accounting detail showed up in testing: "hedges sent"
slightly overstates the load the shards actually see. A hedge cancelled because its primary won
can be torn down before its request leaves the client. The experiment therefore reports both the
broker-counted and the shard-observed ratios (`results/broker/hedging_fakeshard.json`).

Two further details. A primary that loses to a hedge is recorded at its elapsed time when
cancelled. That sample is right-censored, but it is ≥ the hedge delay, which is ≥ p95, so it keeps
the tail mass in the histogram. Dropping it would bias the observed p95 down, fire hedges earlier,
and drain the budget. And a fast failure (Unavailable) gets one immediate failover to another
replica. Failovers are not budgeted, because the failed attempt did no work.

### Degradation order: skip reranker → shrink dense beam → lexical-only
The planner is a pure function (`DegradationPolicy.Plan`). From rolling stage-cost estimates (p90
of recent encode / shard / rerank-per-doc / fetch times, with priors until 20 samples), it picks
the least-degraded level whose estimated cost fits the remaining budget. An in-flight-count floor
adds load shedding. The order reflects cost per unit of quality lost:
1. **Skip the reranker.** The cross-encoder is the most expensive stage on CPU (tens of ms for
   50 pairs), and it only reorders candidates the retrievers already found.
2. **Shrink the dense beam.** A smaller `L_search` cuts SSD reads roughly in proportion and loses
   a little recall at the tail of the candidate list.
3. **Lexical-only.** Dropping dense retrieval loses the most quality: on MS MARCO dev, BM25's
   MRR@10 (0.184) is about half of BGE's (0.358).

Budget-driven steps are cumulative. Capability-driven steps are recorded independently: with no
query encoder, the reranker still runs on BM25 candidates if the budget allows. A post-fan-out
check can still skip the reranker (`reranker_skipped:deadline`), and an ONNX run that overruns is
terminated through `RunOptions.Terminate`. Every step appears in `degradation.steps`
(`kind:reason`) and in `hs_degradation_total{step}`. None are silent.

### Fusion
The functions are pure and live in `Search/Fusion.cs`. RRF: Σ 1/(k + rank). Weighted:
α·norm(dense) + (1−α)·norm(bm25), normalised by min-max or z-score. A doc missing from a list gets
that list's minimum normalised value (0 for min-max), because "not retrieved" must never be a
bonus. Every output follows the project order: score descending, then lower doc ID. BM25 scores
merged across slices are comparable only if the shards use collection-wide statistics. That is the
engine's concern, and it is noted here.

### Autocomplete
The completion trie is a radix trie over UTF-8 bytes, stored as struct-of-arrays. Distinct queries
are sorted and concatenated into one blob, and edge labels are slices of it. Each node's children
are contiguous and binary-searched, and each node keeps min(K, subtree) precomputed completions
(count desc, then text). A lookup walks at most |prefix| bytes and decodes k strings. It is tested
against brute force on random logs, including multi-byte characters and prefixes that end inside
a compressed edge. The build runs in a BackgroundService: `/readyz` is 503 while it runs, and
`/api/suggest` returns 503 until it finishes.

An honest caveat about the data: the MS MARCO train log has 808,731 queries but 808,647 distinct
normalised ones (`results/broker/autocomplete_build.json`). Nearly every count is 1, so the
ranking by count is mostly a tie broken alphabetically. The structure is right, but this query
log is a weak popularity signal.

"Did you mean" is a SymSpell-style delete index (edit distance ≤ 2 on a 7-character prefix,
verified with optimal string alignment) over the query-log vocabulary (words seen ≥ 3 times). It
corrects unknown words independently, and never touches a query that appears in the log verbatim.

### Experiments (M9)
Assignment, team-draft interleaving, credit, and the results shape are specified in
docs/ARCHITECTURE.md. Implementation choices:
- The broker logs its own `served` record for every search with a `sessionId`. Credit comes from
  what was actually served, not from what the client echoes back.
- Ingestion never blocks. `POST /api/events` validates each event, stamps server time, and calls
  `TryWrite` on a bounded channel. If the channel is full, the event is dropped and counted
  (`hs_events_dropped_total{reason="queue_full"}`). A single background writer appends to hourly
  JSONL files. No IP address, user agent, or header is ever written, and a test asserts the log
  record type has no such field.
- The analysis unit is the query impression. The bootstrap resamples sessions, the randomisation
  unit. A/B results include a sample-ratio-mismatch check. Interleaving reports
  Δ = (wins − losses)/(wins + losses + ties) with a session-bootstrap CI and an exact two-sided
  sign test. Impressions with no clicks carry no preference: they are reported as `noClicks` and
  left out of Δ.
- **The users are simulated** (spec honesty rule). The click simulator sets `simulated: true` and
  `clickModel` on its events, and results carry `simulated` and `clickModels`. Nothing computed
  here is user traffic unless real people used the demo.

### Observability
The spans are `broker.search` → `broker.encode`, `shard.search` (tags `shard`, `replica`),
`shard.hedge`, `broker.fuse`, `broker.rerank`, and `shard.fetch`. ASP.NET Core instrumentation
continues the browser's `traceparent`. The broker writes W3C context into gRPC metadata itself and
turns off HttpClient's automatic header injection, so each shard sees exactly one `traceparent`.
Tests check that the trace id reaches the shard. Prometheus metric names match the contract in
`deploy/prometheus/slo_rules.yml`. Labeled series are pre-created at zero, so `rate()` works before
the first event.

### The hashing-dev encoder
`QueryEncoderKind=hashing-dev` feature-hashes query tokens into 64 dimensions, the same function
FakeShard applies to its passages. It exists so the web UI's dense and hybrid modes can be
exercised against FakeShard before the real encoder is exported. `/readyz` and the logs report it
as "NOT a real model", and it must never produce a reported number.

## Numbers
Every timing number is in `results/broker/*.json`, with a `.meta.json` sidecar (CPU, OS, .NET,
date, git SHA, command, load average). Timings from this laptop are **dev-signal-only**: macOS
cannot pin threads, and the machine was shared with other builds while they ran (load average in
the sidecars).
