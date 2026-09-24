# HybridSearch architecture and contracts

This file is the source of truth for every interface between components. If you
change a contract, change it here first.

```
Browser (React + TypeScript, web/)
   │  REST + Server-Sent Events, W3C traceparent header
   ▼
Broker (C#, ASP.NET Core, broker/)                      :8080
   │  query embedding (ONNX Runtime), fan-out with deadline, hedging,
   │  fusion, rerank cascade (ONNX Runtime), degradation policy,
   │  autocomplete, experiments + interleaving, event ingest
   ▼  gRPC (proto/hybridsearch/v1/shard.proto)
Shard servers (C++20, engine/) × N                      :50051, :50052, ...
   ├── lexical: compressed postings, BM25, MaxScore / WAND / block-max WAND
   └── vector: Vamana graph on SSD, PQ codes in RAM (DiskANN style)
Offline (Python, py/hybridsearch/)
   ├── data prep, subset, reference runs, evaluation harness, significance
   ├── reranker training (PyTorch) → ONNX export (fp32 + int8)
   └── click simulation, interleaving sensitivity, counterfactual LTR (M9)
```

## Repository layout

| Path | Owner | What |
|---|---|---|
| `engine/` | C++ | `include/hs/{common,lexical,vector,server}`, `src/...`, `tests/...`, `tools/*.cc` (each file = one CLI), `tools/server/*.cc`, `bench/*.cc` |
| `proto/hybridsearch/v1/shard.proto` | shared | broker ↔ shard contract |
| `broker/` | C# | `HybridSearch.Broker` (web app), `HybridSearch.Broker.Tests` (xUnit), `HybridSearch.Broker.Bench` (BenchmarkDotNet), `HybridSearch.FakeShard` (test double speaking shard.proto) |
| `web/` | TS | Vite + React + TypeScript, Vitest, Playwright, axe-core |
| `py/hybridsearch/` | Python | `data/`, `eval/`, `stats/`, `dense/`, `rerank/`, `clicks/`, `ltr/`, `bench/` |
| `py/tests/` | Python | pytest |
| `scripts/` | shared | one entry-point script per reproducible result |
| `bench/load`, `bench/chaos` | shared | k6 constant-arrival load, chaos experiments |
| `deploy/` | shared | docker-compose, OTel collector, Prometheus, Grafana, Azure VM scripts |
| `results/` | shared | every result table / plot, each with a `.meta.json` naming hardware + command |
| `docs/` | shared | this file, `BUG_LOG.md`, `REPORT.md`, `SLO.md`, `EXPERIMENTS.md` |
| `data/` | gitignored | see below |

Build commands:
- Engine: `scripts/build_engine.sh` (Release, Homebrew gRPC) then `ctest --test-dir engine/build`
- Python: `.venv/bin/python -m pytest` (venv at repo root, created from `pyproject.toml`; do not pip install into it concurrently from several processes)
- Broker: `~/.dotnet/dotnet test broker/HybridSearch.sln`
- Web: `cd web && pnpm test && pnpm e2e`

## Conventions

- **Document IDs.** Everywhere outside an index's internals, a document is its global
  MS MARCO passage ID (`uint64`, the first column of `collection.tsv`). Indexes use
  dense local ordinals internally and each index directory ships `docids.u64bin`
  mapping ordinal → global ID.
- **Ranking order.** Score descending, ties broken by lower global doc ID. Exhaustive
  and pruned lexical search must agree on IDs *and* scores under this order.
- **Vector similarity.** Embeddings are L2-normalised; similarity is inner product;
  higher is better.
- **Query encoder.** BAAI/bge-base-en-v1.5, 768-d, CLS pooling, L2-normalised. Queries
  get the BGE retrieval instruction prefix
  `"Represent this sentence for searching relevant passages: "`; passages get none.
  Same convention as Anserini's BGE regressions. Max length 512 for passages, 64 for queries.
- **Run files.** TREC format: `qid Q0 docid rank score tag`, one line per hit, ranks from 1.
- **Hardware labels.** Every timing result writes a sidecar `.meta.json` with CPU model,
  core count, OS, whether threads were pinned, warm/cold cache, compiler + flags, git SHA,
  command line. macOS timings are labelled `dev-signal-only` (cannot pin).
- **No tuning on dev.** Anything with a knob is tuned on a sample of MS MARCO *train*
  queries (`data/subset/train_tune/`) and evaluated on dev / DL19 / DL20 / BEIR.

## Data layout (`data/`, never committed)

```
data/raw/msmarco/collection.tsv               8,841,823 passages: pid \t text
data/raw/msmarco/queries.dev.small.tsv        6,980 queries: qid \t text
data/raw/msmarco/qrels.dev.small.tsv          qid 0 pid 1
data/raw/msmarco/queries.train.tsv            ~808K train queries (also the autocomplete log)
data/raw/msmarco/qrels.train.tsv
data/raw/trec-dl/dl19.queries.tsv, dl19.qrels  43 judged topics, graded 0..3
data/raw/trec-dl/dl20.queries.tsv, dl20.qrels  54 judged topics, graded 0..3
data/raw/beir/{scifact,nfcorpus,fiqa}/        BEIR format (corpus.jsonl, queries.jsonl, qrels/test.tsv)
data/runs/reference/anserini-bm25-default.dev.trec   Pyserini's own run (the gate: MRR@10 0.1840)
data/subset/1m/collection.tsv                 1,000,000 passages, global IDs kept, sorted by ID;
                                              every passage judged relevant for dev/DL19/DL20 is included,
                                              the rest a seeded uniform sample (seed 20260923)
data/subset/1m/docids.u64bin                  the same IDs, in file order
data/subset/1m/qrels.{dev,dl19,dl20}.tsv      qrels restricted to subset docs (unchanged: all relevant docs are in)
data/subset/train_tune/queries.tsv, qrels.tsv 2,000 seeded train queries whose relevant passage is in the subset
data/embeddings/1m/passages.fbin              n × 768 float32, row i = docids.u64bin[i]
data/embeddings/1m/queries.{dev,dl19,dl20,train_tune}.fbin + .qids.txt
data/indexes/lexical/{full,1m}/               engine lexical index (see engine/include/hs/lexical/)
data/indexes/vector/1m/                       engine vector index (see engine/include/hs/vector/)
data/models/                                  onnx query encoder, reranker fp32/int8
data/events/                                  click / impression logs (parquet), M9
```

Disk is the binding constraint on the laptop (≈16 GB free at start). Every step that
writes more than 1 GB checks free space first and refuses below 3 GB headroom.

## Engine library API (what the shard server links against)

Lexical (`hs::lexical`, header `hs/lexical/index.hpp`):
- `Analyzer` — `std::vector<std::string> analyze(std::string_view)`; Lucene EnglishAnalyzer parity.
- `LexicalIndex::open(dir)`, `num_docs()`, `global_id(ordinal)`.
- `search(const std::vector<std::string>& terms, const SearchOptions&, hs::Deadline&) -> LexicalResult`
  where options carry `k, algorithm, k1, b, explain` and the result carries sorted `hs::ScoredDoc`s,
  optional per-term contributions, and counters (`docs_scored`, `postings_decoded`).
- `DocStore::open(dir)`, `text(ordinal)`; `select_snippet(text, query_terms, max_chars) -> {window, highlights}`.

Vector (`hs::vector`, header `hs/vector/index.hpp`):
- `VamanaIndex` (in-memory) and `DiskIndex` (SSD layout + PQ in RAM), both exposing
  `search(const float* q, const VectorSearchOptions&, hs::Deadline&) -> VectorResult`
  with sorted `hs::ScoredDoc`s (inner product) and counters (`distance_computations`, `ssd_reads`).
- `global_id(ordinal)`.

Common (`hs::`): `Deadline` (`hs/common/deadline.hpp`), `TopK` / `ScoredDoc` / `ranks_before`
(`hs/common/topk.hpp`), `.fbin` / `.u64bin` IO (`hs/common/fbin.hpp`).

## Broker REST API (what the web app calls)

All JSON is camelCase. Every response carries header `x-request-id` and, if tracing is on,
`traceparent`.

### `GET /api/search`

Query params: `q` (required), `k` (default 10, max 100), `mode` = `hybrid` (default) |
`lexical` | `dense`, `rerank` = `true` (default) | `false`, `explain` = `false` (default) |
`true`, `deadlineMs` (default 300), `sessionId` (opaque random ID from the browser, used only
for experiment bucketing).

```jsonc
{
  "requestId": "01J…",
  "query": "what is the capital of peru",
  "didYouMean": null,                    // string when a spelling correction is suggested
  "mode": "hybrid",
  "results": [
    {
      "docId": 7067032,
      "rank": 1,
      "text": "…best-matching window of the passage…",
      "highlights": [{ "start": 12, "end": 16 }],   // UTF-16 code unit offsets into text
      "isSnippet": true,
      "team": null,                        // "A" | "B" when this result came from an interleaved experiment
      "scores": {
        "bm25": 14.2, "bm25Rank": 3,       // null when that retriever did not return the doc
        "dense": 0.81, "denseRank": 1,
        "fused": 0.032, "fusedRank": 1,
        "rerank": 7.9                     // null when the reranker did not run
      },
      "terms": [{ "term": "peru", "score": 9.1, "tf": 2, "df": 1402 }]   // only with explain=true
    }
  ],
  "degradation": {
    "level": 0,                            // 0 none, 1 reranker skipped, 2 dense beam shrunk, 3 lexical only
    "steps": [],                           // e.g. ["reranker_skipped:deadline"]
    "partialShards": [],                   // shards that hit their deadline and returned partial top-k
    "failedShards": [],                    // shards that errored or timed out entirely
    "hedgedShards": []                     // shards whose hedge request won
  },
  "timings": { "totalMs": 41.2, "encodeMs": 6.1, "shardsMs": 18.0, "fuseMs": 0.1, "rerankMs": 14.9, "fetchMs": 2.0 },
  "experiment": { "id": "rerank-depth", "variant": "treatment", "interleaved": false }   // null if none
}
```

### `GET /api/search/stream`

Same params. `text/event-stream`. Events, in order, each with the same JSON shape as
`/api/search`: `event: lexical` (lexical-only first page as soon as shards answer),
`event: final` (fused + reranked). An `event: error` carries `{ "message": … }`.

### `GET /api/suggest?prefix=…&k=8`

```json
{ "prefix": "how to", "suggestions": [{ "text": "how to lose weight", "count": 312 }], "micros": 3.1 }
```
Built from the MS MARCO train query log (≈808K real Bing queries), normalised
(lowercase, collapsed whitespace), counted, top-k precomputed at each trie node.

### `GET /api/doc/{docId}?q=…`

Full passage with highlights for query `q`: `{ "docId", "text", "highlights": [...] }`.

### `POST /api/events`

Batch of UI interaction events, M9. Body `{ "events": [Event, …] }`, max 100 per batch,
returns 202. The broker validates, stamps server time, and appends to the event log.

```jsonc
Event = {
  "type": "impression" | "click" | "dwell" | "query" | "abandon",
  "sessionId": "b2c…",       // random per browser tab, no user identity, no IP stored
  "requestId": "01J…",       // ties the event to the served result list
  "query": "…",
  "docId": 7067032,          // click / dwell / impression
  "rank": 1,
  "dwellMs": 12000,          // dwell only
  "experimentId": "…", "variant": "…", "team": "A",
  "clientTs": "2026-09-23T20:10:11.123Z"
}
```

### Experiments

- `GET /api/experiments` → list of configured experiments with status.
- `GET /api/experiments/{id}/results` → per-variant metrics with confidence intervals
  (A/B: CTR, clicks@1, abandonment, MRR of first click; interleaving: wins / losses / ties,
  Δ preference with bootstrap CI and sign-test p-value; A/A: same, expected null).
- Config lives in `broker/HybridSearch.Broker/experiments.json`:

```jsonc
{
  "experiments": [
    {
      "id": "hybrid-vs-lexical",
      "kind": "interleave",               // "ab" | "interleave" | "aa"
      "allocation": 0.2,                  // fraction of sessions enrolled
      "salt": "2026-09-23-a",
      "control":   { "mode": "lexical", "rerank": false },
      "treatment": { "mode": "hybrid",  "rerank": true, "rerankDepth": 50, "fusion": "rrf", "rrfK": 60 }
    }
  ]
}
```
Assignment: `bucket = xxhash64(salt + sessionId) mod 10000`, enrolled if
`bucket < allocation*10000`; A/B arm from a second, independent hash. Deterministic and
sticky per session. Interleaving uses **team-draft interleaving** (Radlinski et al. 2008).

Broker specifics (broker/README.md has the rationale):
- The arm hash is `xxhash64(salt + sessionId, seed = 0x5EED_A5B1_0000_0001) mod 2` (0 = control).
  A session is in at most one running experiment: the first in file order that enrolls it.
  Optional experiment fields: `"status": "running" | "paused"`, `"description"`, and for
  `"kind": "aa"` a `"method": "ab" | "interleave"` (A/A always uses `control` for both arms).
- Interleaving: team A = control, team B = treatment. The interleaving seed is
  `xxhash64(salt ␟ sessionId ␟ query)`, so a reload shows the same page. `team` is set on results
  only for event logging.
- Event log: JSON Lines, one file per UTC hour, `data/events/events-yyyyMMdd-HH.jsonl` (not
  parquet). Besides validated UI events, the broker writes its own `"kind": "served"` record for
  every search that carries a `sessionId` (request id, experiment, variant, and the served doc ids,
  ranks and teams). Credit is attributed from the served record, not from the team the client echoes.
- Events may carry optional `"simulated": true` and `"clickModel": "pbm" | "cascade" | "dbn" | …`.
  Results report `simulated` if any attributed click was simulated.

`GET /api/experiments`:

```jsonc
{ "experiments": [ { "id", "kind", "status", "allocation", "control", "treatment", "description"? } ] }
```

`GET /api/experiments/{id}/results` (unit of analysis: query impression; bootstrap resamples
sessions, the randomisation unit):

```jsonc
{
  "id": "hybrid-vs-lexical", "kind": "interleave", "simulated": false, "confidenceLevel": 0.95,
  "updatedAt": "2026-09-23T20:10:11Z",
  "variants": [                                  // A/B, A/A: "control", "treatment"; interleaving: one "interleaved"
    { "name": "control", "sessions": 120, "queries": 480,
      "metrics": { "ctr": { "value", "ciLow", "ciHigh" }, "clicksAt1": {…}, "abandonment": {…}, "mrrFirstClick": {…} } }
  ],
  "interleaving": {                              // null unless the experiment interleaves
    "wins": 40, "losses": 25, "ties": 5,         // wins = impressions where treatment's team got more clicks
    "deltaPreference": { "value", "ciLow", "ciHigh" },   // (wins − losses) / (wins + losses + ties)
    "pValue": 0.08                               // two-sided exact sign test, ties excluded
  }
  // additive fields: "queries", "clickEvents", "clickModels", "differences" (treatment − control, A/B),
  // "srm" (sample-ratio-mismatch chi-square), "interleaving.noClicks", "note"
}
```
Metric definitions: `ctr` = distinct clicked results per impression; `clicksAt1` = share of
impressions with a click at rank 1; `abandonment` = share of impressions with no click;
`mrrFirstClick` = mean 1/rank of the highest clicked result (0 if none).

### Operations

- `GET /healthz` (liveness), `GET /readyz` (≥1 shard healthy per slice), `GET /metrics` (Prometheus).
- OpenTelemetry: OTLP gRPC to `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4317`).
  Spans: `broker.search` → `broker.encode`, `shard.search[shard=i]` (and `shard.hedge`),
  `broker.fuse`, `broker.rerank`, `shard.fetch`. The browser starts the trace and sends
  `traceparent`; the broker forwards it to shards in gRPC metadata.

## Ports

| Component | Port |
|---|---|
| shard i | 50051 + i |
| broker | 8080 |
| web dev server | 5173 |
| OTel collector (OTLP gRPC / HTTP) | 4317 / 4318 |
| Jaeger UI | 16686 |
| Prometheus | 9090 |
| Grafana | 3000 |
