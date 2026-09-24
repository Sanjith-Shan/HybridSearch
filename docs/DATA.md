# Data manifest

Everything under `data/` is gitignored and reproducible from the commands below. Every
download is checked against a published MD5; every derived file's MD5/SHA is recorded
in `data/raw/manifest.json` and `data/subset/1m/manifest.json`.

## Downloads

| File | URL | MD5 (verified) | Expected MD5 source |
|---|---|---|---|
| `collectionandqueries.tar.gz` (1,057,717,952 B) | https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz | `31644046b18952c1386cd4564ba2ae69` | ir_datasets `downloads.json`, Anserini docs |
| `msmarco-test2019-queries.tsv.gz` | https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz | `eda71eccbe4d251af83150abe065368c` | ir_datasets; server `Content-MD5` |
| `msmarco-test2020-queries.tsv.gz` | https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz | `00a406fb0d14ed3752d70d1e4eb98600` | ir_datasets; server `Content-MD5` |
| `2019qrels-pass.txt` | https://trec.nist.gov/data/deep/2019qrels-pass.txt | `2f4be390198da108f6845c822e5ada14` | ir_datasets |
| `2020qrels-pass.txt` | https://trec.nist.gov/data/deep/2020qrels-pass.txt | `0355ccee7509ac0463e8278186cdd8d1` | ir_datasets |
| BEIR `scifact.zip` (2,816,079 B) | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip | `5f7d1de60b170fc8027bb7898e2efca1` | ir_datasets |
| BEIR `nfcorpus.zip` (2,448,432 B) | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip | `a89dba18a62ef92f7d323ec890a0d38d` | ir_datasets |
| BEIR `fiqa.zip` (17,948,027 B) | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip | `17918ed23cd04fb15047f73e6c3bd9d9` | ir_datasets |
| Pyserini prebuilt `msmarco-v1-passage` Lucene index (2,170,758,745 B, temporary) | https://huggingface.co/datasets/castorini/prebuilt-indexes-msmarco-v1/resolve/main/passage/original/lucene-inverted/tf/lucene-inverted.msmarco-v1-passage.20221004.252b5e.tar.gz | `678876e8c99a89933d553609a0fd8793` | Pyserini `prebuilt_index_info` |
| `BAAI/bge-base-en-v1.5` | Hugging Face hub (`~/.cache/huggingface`) | n/a | |

The msmarco tarball was downloaded as 8 parallel byte ranges (a single stream ran at
~1 MB/s) and concatenated before the MD5 check. The prebuilt index was streamed through
`md5` and `tar` at once (no tarball on disk). It was deleted on 2026-09-24 (by the lead,
to free disk) after the dev/DL19/DL20/train_tune runs were written.

Commands: `scripts/data_download.sh` (download + `python -m hybridsearch.data.prepare`),
then `scripts/data_subset.sh`.

## Layout and counts (`data/raw`)

| Path | Lines | MD5 |
|---|---|---|
| `raw/msmarco/collection.tsv` | 8,841,823 | `31e42eeac3a3ea1b75c689c483be4484` |
| `raw/msmarco/queries.dev.small.tsv` | 6,980 | `4621c583f1089d223db228a4f95a05d1` |
| `raw/msmarco/qrels.dev.small.tsv` | 7,437 | `38a80559a561707ac2ec0f150ecd1e8a` |
| `raw/msmarco/queries.train.tsv` | 808,731 | `33c9e2a484f1f5a8cc041f05497e639e` |
| `raw/msmarco/qrels.train.tsv` | 532,761 (502,939 queries) | `733fb9fe12d93e497f7289409316eccf` |
| `raw/msmarco/queries.dev.tsv`, `queries.eval{,.small}.tsv` | 101,093 / 101,092 / 6,837 | see manifest |
| `raw/trec-dl/dl19.queries.tsv` | 43 (judged topics only) | `c83e3dda130c7a191357b0287d854ddb` |
| `raw/trec-dl/dl19.qrels` | 9,260 (= NIST file) | `2f4be390198da108f6845c822e5ada14` |
| `raw/trec-dl/dl20.queries.tsv` | 54 (judged topics only) | `19773395d1d74b86a14f447bc5efecf3` |
| `raw/trec-dl/dl20.qrels` | 11,386 (= NIST file) | `0355ccee7509ac0463e8278186cdd8d1` |
| `raw/trec-dl/dl{19,20}.queries.all.tsv` | 200 each (all released test queries) | see manifest |
| `raw/beir/{scifact,nfcorpus,fiqa}/` | corpus 5,183 / 3,633 / 57,638; test queries 300 / 323 / 648 | BEIR zips above |

`raw/beir/{ds}/qrels/test.trec` is a TREC-format copy of `qrels/test.tsv`, written by
`hybridsearch.dense.beir` so the harness can read it.

## The 1M subset (`data/subset/1m`), seed 20260923

Built by `python -m hybridsearch.data.subset` (22 s):

1. R = every pid with grade > 0 in dev-small, DL19 or DL20 qrels: **15,127** pids.
2. The remaining 8,826,696 pids, ascending; `np.random.default_rng(20260923).choice(C, 984,873, replace=False)`.
3. Subset = sorted(R ∪ sample) = 1,000,000 pids (min 0, max 8,841,814).

| File | What | Checksum |
|---|---|---|
| `collection.tsv` | 1,000,000 lines, `pid \t text`, ascending pid, global IDs kept | MD5 `d7cf59404eab9b2d17c23b64a8eec260` |
| `docids.u64bin` | uint64 n=1,000,000 then the pids in file order (`hs/common/fbin.hpp`) | SHA-256 of the id payload `f68ed924d610763552abca14093b9c6db51965d9acd0b2c9625773ad8cc9f6a4`; file MD5 `96da468269935c7c0d2fbec7ae361844` |
| `qrels.dev.tsv` | 6,980 topics, 7,437 judgments (all relevant; unchanged from full) | |
| `qrels.dl19.tsv` | 43 topics; all 4,102 grade>0 judgments kept; grade-0 judgments kept only if the doc is in the subset (4,691 of 9,260 total) | |
| `qrels.dl20.tsv` | 54 topics; all 3,606 grade>0 kept (4,487 of 11,386 total) | |
| `READY` | written last; contains the docids SHA-256 | |

`data/subset/train_tune/` (2,000 queries): judged train queries whose every relevant pid is
in the subset (54,502 eligible), drawn with the same rng stream (second draw), sorted by
qid. `queries.tsv`, `qrels.tsv`.

`data/subset/train50k/` (50,000 queries, for hard-negative mining): judged train queries
excluding train_tune, drawn with `default_rng(20260924)`, sorted by qid. **Not** restricted
to the subset; their BM25 run is over the full 8.8M collection.

## Dense embeddings (`data/embeddings/1m`)

BAAI/bge-base-en-v1.5, encoded here on an Apple M3 Pro GPU (MPS) in fp32, CLS pooling,
L2-normalised, passages max 512 tokens, queries max 64 tokens with the prefix
`"Represent this sentence for searching relevant passages: "`. **These are encoded by
HybridSearch, not Waterloo's precomputed vectors** (those are 26 GB for the full collection
and do not fit the laptop); the full-scale runs may use Waterloo's.

| File | Format |
|---|---|
| `passages.fbin` | `.fbin`: uint32 n=1,000,000, uint32 dim=768, float32 row-major; row i = `docids.u64bin[i]` |
| `queries.{dev,dl19,dl20,train_tune,train50k}.fbin` + `.qids.txt` | `.fbin`; row i = line i of `.qids.txt` (queries-file order) |
| `gt.dev.top100.u64bin` | `.u64bin` with n = 6,980 × 100, row-major [query][rank], **global pids**, query order = `queries.dev.qids.txt`; exact inner-product top-100 over the subset, ties by lower pid |
| `gt.dev.top100.scores.fbin` | `.fbin` 6,980 × 100 float32 inner products matching the ids above |
| `READY` | written when `passages.fbin` is complete |
| `encode.*.meta.json`, `passages.stats.json`, `passages.chunks.log` | hardware, timings, per-chunk throughput |

Resumable: passages are written into a preallocated `passages.fbin.partial` in 20,000-row
chunks, each recorded in `passages.progress` after flush. While encoding runs, the directory
lock `data/locks/mps` (atomic `mkdir`) is held; it is removed on exit.

Parity: `python -m hybridsearch.dense.parity` encodes a seeded sample of 200 passages and 200
dev queries with plain `sentence-transformers` on CPU and compares them to the stored rows
(`results/dense/parity.json`).

BEIR: `data/embeddings/beir/{ds}/corpus.fbin` + `corpus.docids.txt`, `queries.test.fbin` +
`queries.test.qids.txt`. Passage text is `f"{title} {text}".strip()`.

## Reference runs (`data/runs/reference`)

| Run | Produced by |
|---|---|
| `anserini-bm25-default.{dev,dl19,dl20,train_tune}.trec` | Pyserini 2.4.0 BM25 k1=0.9 b=0.4, top 1000, prebuilt `msmarco-v1-passage` index (`scripts/reference_bm25.sh`) |
| `bm25.train50k.first10k.trec` | same, for the **first 10,000** of the 50,000 train50k queries (`data/subset/train50k/queries.first10k.tsv` + `qrels.first10k.tsv`, lines 1–10,000 of `queries.tsv`). The remaining 40,000 were dropped when the machine rebooted and the prebuilt index was deleted to save disk; the full train50k BM25 run is to come from HybridSearch's own lexical engine. |
| `anserini-bm25-default-1m.{dev,dl19,dl20,train_tune}.trec` | Pyserini BM25 over a Lucene index of the 1M subset at `data/indexes/anserini/1m` (`scripts/reference_bm25_1m.sh`) |
| `dense-flat-1m.{dev,dl19,dl20,train_tune}.trec` | exact FAISS `IndexFlatIP` over `passages.fbin`, top 1000 (`python -m hybridsearch.dense.flat`) |
| `dense-flat.beir-{scifact,nfcorpus,fiqa}.test.trec` | exact `IndexFlatIP` over the BEIR corpus embeddings |

Scores: `results/reference/reference_runs.{md,json}` (`python -m hybridsearch.eval.reference`).

## Coordination files

- `data/subset/1m/READY`: the subset and its qrels are final.
- `data/embeddings/1m/READY`: `passages.fbin` is complete.
- `data/locks/mps/`: present while a process owns the Apple GPU. Create it with `mkdir`
  (atomic) and remove it when done; `owner` inside names the holder.
