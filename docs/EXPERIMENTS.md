# Experimentation system (M9: the feedback loop)

This document is the design of HybridSearch's online-evaluation and learning-from-clicks
system, and the results of the simulation studies that justify its choices.

> **Read this first.** HybridSearch has no real users. Every click number in this document
> comes from **simulated users**: standard click models (PBM, Cascade, DBN) whose notion
> of relevance is the real graded NIST judgments of TREC DL 2019 + 2020 (and, for the LTR
> training set, MS MARCO train labels). The *pipeline* is real: the broker logs events in
> the schema below, and `hybridsearch.clicks.analyze_events` runs unchanged on those
> logs. The *numbers* describe what the methods do under textbook user models, not what
> any person did. Every result file says which click model produced it.

Code: `py/hybridsearch/clicks/` (click models, interleaving, A/B statistics, SRM, event
log analysis, sensitivity study), `py/hybridsearch/ltr/` (counterfactual LTR). Scripts:
`scripts/m9_*.py`. Results: `results/m9/` (every file has a `.meta.json` sidecar with
hardware, date, command, seeds and trial counts).

## 1. Event logging

The web UI batches events to `POST /api/events` (schema in `docs/ARCHITECTURE.md`):
`impression` (result in viewport), `click`, `dwell` (ms on the passage view), `query`,
`abandon` (page left with no click). Each event carries `sessionId` (random per tab, no
identity, no IP), `requestId` (ties it to the served result list), `rank`, `docId`, and,
for enrolled sessions, `experimentId`, `variant`, and for interleaved pages the `team`
that contributed the result. The broker stamps server time and appends JSON Lines under
`data/events/`.

The analyser reconstructs one *page* per `requestId` (ranks → docIds, teams, clicked
ranks, dwell) and counts data-quality problems instead of dropping them silently (events
with an unknown type, missing `requestId`, clicks with no matching impression).
Simulated logs carry an extra `"simulated": "<click model>"` field on every event, and
the report prints "SIMULATED users" whenever it sees one.

## 2. Assignment

Mirrors the broker (`docs/ARCHITECTURE.md`, Experiments), implemented in Python in
`clicks/events.py` so simulated traffic goes through the same code path:

* **Enrolment:** `bucket = xxhash64(salt + sessionId) mod 10000`; enrolled iff
  `bucket < allocation * 10000`.
* **Arm:** a second, independent hash of the same bytes: `xxhash64(salt + sessionId,
  seed = 0x5EEDA5B100000001) mod 2` (0 = control), exactly as the broker's
  `ExperimentAssigner`. Independence matters: using the same hash for enrolment and arm
  would put all low buckets in one arm whenever allocation < 1, and raising the
  allocation later would move enrolled sessions between arms.
* Deterministic and sticky per session; a new salt re-randomises everyone (use it for
  every new experiment so carry-over effects from a previous experiment do not line up
  with the new split).
* The unit of randomisation is the session, but the unit of most metrics is the query
  impression. Impressions within a session are correlated, so a per-impression t-test
  slightly overstates confidence for real traffic. The simulation draws impressions
  independently, so it does not have this problem; on real logs, the right fix is a
  session-level bootstrap (not implemented yet; see "What is not done").

## 3. Sample-ratio mismatch (SRM)

`clicks.ab.srm_check`: chi-square goodness-of-fit of observed sessions per arm against
the configured split, flagged at **p < 0.0005**. The strict threshold is conventional
because SRM is checked on every experiment every day, so a 0.05 threshold would raise
false alarms constantly (Fabijan et al., KDD 2019). An SRM means the arms are no longer
comparable (a bot filter that hits one arm, a crash in one arm that drops its events, a
redirect), and the report says "do not trust the metrics" rather than printing
significance stars next to them. `py/tests/test_clicks_analyze_events.py` injects a
logging bug that loses 15% of treatment sessions and checks that SRM fires.

In the simulated event log (`results/m9/events_report_simulated.md`) the A/A
experiment's arm counts were 944 vs 1,059 sessions, chi-square p = 0.010. Nothing is wrong
with that split (the hash is fair); at a 0.05 threshold it would have been a false alarm,
which is exactly why the threshold is strict.

For interleaving there are no arms, so the analyser checks **team balance** instead: the
team owning rank 1 should be A half of the time (binomial test). A skew means the coin or
the logging is broken.

## 4. Metrics

Per result-page impression (then averaged):

| metric | definition | better |
|---|---|---|
| clicks per impression ("CTR") | total clicks / impressions | higher |
| clicks@1 | share of impressions with a click on rank 1 | higher |
| abandonment | share of impressions with no click | lower |
| MRR of first click | mean of 1/rank of the first click (0 if none) | higher |
| interleaving Δ_AB | P(A wins) + P(tie)/2 − 1/2 (Chapelle et al. 2012) | > 0 prefers A |

Tests: Welch two-sample test (normal reference) for A/B metrics; exact two-sided sign
test on wins vs losses (ties dropped) for team-draft and balanced interleaving; one-sample
test on the mean marginalised outcome for probabilistic interleaving; percentile
bootstrap for CIs.

**Guardrails.** Any real experiment also tracks metrics it is *not* trying to move:
p95 latency and degradation level from the search response (a better ranker that
triggers `reranker_skipped:deadline` more often is not better), error rate, and
abandonment as a guardrail even when it is not the primary metric. A guardrail
regression blocks a launch regardless of the primary metric. The analyser computes the
click metrics; latency guardrails come from the broker's Prometheus metrics
(`docs/SLO.md`).

## 5. Interleaving vs A/B

**Team-draft interleaving** (Radlinski, Kurup & Joachims 2008) merges both rankers'
lists into one page. Each round a coin decides which team picks first; each team adds its
highest-ranked result not already shown; a click credits the team that contributed the
result; the team with more clicks wins the impression. The exact variant (when coins are
consumed, what happens when one list runs out) is pinned by
`tests/golden/interleaving.json`, which the Python and C# implementations both test
against.

Trade-offs:

* **Sensitivity.** Every user sees both rankers and makes a *relative* judgment on the
  same query, so query difficulty and user click propensity cancel. A/B compares absolute
  click rates between two different populations of queries and users, and the variance
  from those differences dominates. That is the published reason interleaving needs far
  fewer impressions (Chapelle et al. 2012 report 1 to 2 orders of magnitude on Bing and
  Yahoo! traffic). Section 8 measures how much in simulation.
* **What it can answer.** Interleaving answers "which ranking do users prefer", a
  pairwise preference. It cannot measure absolute effects (did abandonment go down 2%?),
  cannot test changes outside the result list (snippets, latency, UI), and its outcome is
  not a business metric. A/B is still needed to size a launch and for guardrails. The
  usual practice is interleaving to screen many ranker candidates, then A/B on the one
  that survives.
* **Bias.** Team-draft is fair under random clicks (tested:
  `test_team_draft_is_fair_under_random_clicks`), but credit can be biased when two
  rankers return near-identical lists that differ only in order. Probabilistic
  interleaving with marginalised credit (Hofmann et al. 2011) addresses that at the cost
  of sometimes showing worse lists. Balanced interleaving (Joachims 2002) is simple and
  sensitive but known to be biased in some duplicate-heavy cases.

## 6. What the simulation can and cannot tell you

It **can** tell you: whether the statistics are calibrated (A/A false-positive rate
≈ α), whether the pipeline end to end produces the right verdict when the truth is
known, how the *relative* sensitivity of methods depends on the user model, and how
position bias corrupts naive learning from clicks.

It **cannot** tell you: the absolute number of impressions a real experiment needs (real
users are noisier than any of these models: they click on snippets, not relevance; they
reformulate; they share sessions; there are bots), whether real users agree with NIST
assessors (click models map judgments to clicks by assumption), anything about
novelty effects, learning effects, or query-mix shift, or anything about the long tail
of queries (the impressions here are drawn uniformly over 97 DL topics; real traffic is
Zipfian). The three user profiles and three model families are there to show how much
the answer depends on the assumptions. Where the conclusion changes across them, the
honest summary is "it depends on the user model".

## 7. Learning from clicks: counterfactual LTR design

Clicks are biased toward whatever the production ranker put on top (position bias), so
"clicked = relevant" teaches a new ranker to copy the old one. The standard fix is
inverse propensity scoring (IPS): weight each click by 1/η_r, the probability the user
examined rank r (Joachims, Swaminathan & Schnabel, WSDM 2017).

**Setup** (`scripts/m9_ltr.py`, `py/hybridsearch/ltr/`):

* Logging ranker: Anserini BM25 (full collection). It shows its top 20; simulated users
  click under PBM (navigational profile, η_r = 1/r).
* Features per candidate: BM25 z-score and reciprocal rank, BGE-base inner product
  z-score and reciprocal rank *within the 20 candidates*, log passage length, and
  query-length interactions. The BGE vectors for the candidates were encoded by M9 code
  on CPU (`ltr/dense_feature.py`, project encoder convention), because no dense run over
  the collection existed when this ran. Said so in every result file.
* Model: linear, listwise softmax cross-entropy, L2 λ = 1e-3 fixed a priori (not tuned).
* **Query split, never mixed:** training clicks come from a seeded sample of 200 MS MARCO
  `train_tune` queries (sparse labels: label 1 → DL grade 2, unjudged → 0).
  Evaluation is on the 97 TREC DL 2019+2020 topics with their graded qrels (nDCG@10 of
  the reranked top 20). No evaluation topic is used for training or for any choice.
* Estimators: naive (click count), IPS with the true η, IPS with η estimated from
  (a) swap(1,k) interventions in 5% of sessions, (b) RandTop-10 shuffles in 5% of
  sessions, (c) intervention harvesting from two production rankers sharing traffic
  (BM25 order and BGE order of the same candidates; Agarwal et al., WSDM 2019), no
  deliberate randomisation. Raw per-rank ratios are smoothed by a click-weighted
  monotone (isotonic) fit, see `docs/BUG_LOG.md` for why.
* Baselines: logging ranker, an oracle trained on the true grades, a skyline trained on
  α(grade) (the click-probability labels IPS converges to), and the candidate-set upper
  bound (perfect reordering of the 20 candidates).
* Also: propensity misestimation (IPS with η_r = (1/r)^p for p = 0 … 2 when the truth is
  p = 1) and weight clipping (τ = 1.5 … ∞).

## 8. Results: interleaving vs A/B sensitivity (simulated users)

Source: `results/m9/sensitivity.md` (all tables), `sensitivity.json` (summary; raw power
curves in `data/m9/sensitivity_raw.json`), `sensitivity.png`, `sensitivity_n80.png`,
`aa.json`. Command: `.venv/bin/python scripts/m9_sensitivity.py --pool 2e6 --trials 1000`
(seed 20260923). **Simulated users** under 9 click models (PBM / Cascade / DBN ×
perfect / navigational / informational), relevance from TREC DL 2019+2020 graded qrels,
queries uniform over the 97 DL topics.

**Method.** N80 = total query impressions needed for 80% power at two-sided p < 0.05,
*in the offline-correct direction* (A/B splits N in half between arms). For each
(pair, click model) we simulate 2M impressions per arm/method, tabulate the discrete
per-impression outcome, and run 1,000 multinomial trials at each of 141 log-spaced N
between 10 and 1e8 (see `clicks/sensitivity.py`). The resampling shortcut is checked
against fresh click simulation at N80 (28 checks: 7 known-ordering pairs × 2 models × TDI and
the best A/B metric): direct power 0.759 to 0.828 (1,000 trials each; target 0.80).

**Rankers.** Real, 1M-subset world (the data agent's dense flat run covers the 1M subset):
BGE-base flat dense retrieval 0.7952, RRF(BM25-1M, dense) 0.7661, Anserini BM25 on the
subset 0.6483. The subset keeps every judged-relevant passage, so its scores are higher than
full-collection scores and the two worlds are never compared with each other.
Real, full-collection world: BM25 top-100 re-ordered by BGE-base similarity
0.6681; RRF(BM25, that BGE ordering) 0.6161; Anserini BM25 0.4912; HybridSearch's own
BM25 engine, Lucene-quantised lengths 0.4912 and textbook lengths 0.4906. The two BGE
rankers only reorder BM25's top 100 (no dense retrieval over the collection existed
when this ran; the BGE vectors were encoded by M9 code). Synthetic: "noisy oracle"
rankers (grade + σ·noise, shared noise), 0.4624 to 0.7634, so the gap is controlled.

**Headline (median over 9 click models; ratio = best A/B metric's N80 / team-draft N80):**

| pair | Δ nDCG@10 | TDI N80 | best A/B N80 | ratio (range) |
|---|---|---|---|---|
| dense vs RRF(BM25, dense) (real, 1M) | +0.029 | 8,293 | 26,328 | 4.0x (1.4–27x) |
| RRF(BM25, dense) vs BM25 (real, 1M) | +0.118 | 285 | 582 | 2.2x (1.6–7.0x) |
| dense vs BM25 (real, 1M) | +0.147 | 337 | 461 | 1.4x (1.0–3.3x) |
| BGE-rerank vs RRF (real) | +0.052 | 3,600 | 9,866 | 2.4x (1.4–11.6x) |
| RRF vs BM25 (real) | +0.125 | 151 | 313 | 2.5x (1.3–4.8x) |
| BGE-rerank vs BM25-textbook (real) | +0.177 | 164 | 214 | 1.6x (1.1–2.9x) |
| synthetic σ1.0 vs σ1.03 | +0.014 | 8,443 | 164,254 | 16.4x (8.7–154x) |
| synthetic σ1.0 vs σ1.1 | +0.038 | 1,251 | 17,059 | 10.5x (6.2–103x) |
| synthetic σ1.0 vs σ1.25 | +0.097 | 229 | 626 | 4.1x (2.4–31x) |
| synthetic σ1.0 vs σ1.5 | +0.180 | 74 | 180 | 2.7x (2.0–13x) |
| synthetic σ1.0 vs σ2.0 | +0.301 | 35 | 73 | 2.3x (1.8–4.9x) |

**What this says, honestly.**

* Team-draft interleaving needed fewer impressions than the best A/B metric in 98 of 99
  (pair, model) cells with a known ordering (ratio 0.96x to 154x; the one exception is
  dense vs BM25 on the subset, a large gap, where the best A/B metric was 4% cheaper, 0.96x). The advantage **grows as the gap shrinks**:
  about 2x for large gaps, about 10–16x (median) for gaps of 0.01–0.04 nDCG@10, and 100x or
  more only in the best case (cascade/DBN "perfect" users, who click only relevant results).
* The published finding (Chapelle et al. 2012: interleaving 1–2 orders of magnitude more
  sensitive on real Bing/Yahoo! traffic) is reproduced **only for small gaps and only for
  some user models**. For the real ranker pairs here the advantage is 1.0–27x: about
  1.4–2.5x (median) for the large gaps (0.12–0.18), 2.4–4x for the smaller ones
  (dense vs RRF, +0.029; BGE-rerank vs RRF, +0.052). Real A/B tests are also noisier than this
  simulation (heterogeneous users, sessions, bots, query mix), which hurts A/B more than
  interleaving, so these ratios are probably a lower bound on the real one. That last
  point is an argument, not a measurement.
* **Balanced interleaving was at least as sensitive as team-draft in 96 of 99 cells, and
  probabilistic interleaving in 69 of 99** (median best-interleaving/TDI N80 = 0.06–0.88
  per pair). Team-draft pays for its
  fairness guarantees with variance: its credit comes from team labels, which are random
  per impression.
* **Which A/B metric matters a lot.** Clicks per impression ("CTR") often pointed the
  *wrong* way: it agreed with offline nDCG in only 36 of 54 real-pair cells and 29 of 45
  synthetic ones. Under cascade/DBN users a better ranking satisfies users sooner and so
  produces *fewer* clicks. Clicks@1 and MRR of first click agreed in all 99 cells,
  abandonment in 97 of 99, and every interleaving method in all 99.
* **Pairs with no known ordering.** Anserini BM25 vs HybridSearch's own BM25 (identical top
  10 on 95 of 97 topics) and Lucene-length vs textbook-length BM25 (offline p = 0.3) are in
  the tables but are not evidence either way. Online methods still "detect" a difference
  there at 1e5 to 5e6 impressions, because the rankings differ even though nDCG@10 does not.
  That is a reminder that an online preference and an offline metric are different
  quantities.

**A/A calibration** (`aa.json`; Anserini BM25 against itself, 2,000 impressions per trial,
1,000 fresh-simulation trials per click model): false-positive rate at α = 0.05 is
0.032–0.067 for team-draft and for all four A/B metrics in all 9 models (binomial 95%
band for a true 5% is about 0.037–0.064), with one exception: abandonment under
cascade-informational at 0.002. Under that model almost every page gets a click, so
abandonment is nearly constant and the normal-approximation test is very conservative.
The sign test is discrete, so its p-values are conservative rather than uniform.

## 9. Results: counterfactual LTR (simulated users)

Source: `results/m9/ltr.{md,json}`, `ltr_learning_curve.png` (5 seeds, mean ± sd).
Command: `.venv/bin/python scripts/m9_ltr.py` (seed 20260923). **Simulated users**: PBM
navigational, η_r = 1/r. Clicks come from MS MARCO train labels on 200 train_tune
queries. Evaluation is on the held-out DL19+DL20 topics with graded qrels, reranking
Anserini BM25's top 20.

| model | nDCG@10 (DL19+DL20) |
|---|---|
| logging ranker (BM25) | 0.4912 |
| naive, click = relevant (3M sessions) | 0.5598 ± 0.0000 |
| IPS, true η (3M sessions) | 0.6012 ± 0.0005 |
| IPS, η from swap(1,k) interventions (3M) | 0.6010 ± 0.0013 |
| IPS, η from RandTop-10 (3M) | 0.5938 ± 0.0003 |
| IPS, η harvested from 2 production rankers (3M) | 0.6010 ± 0.0005 |
| skyline: labels α(grade), the IPS limit | 0.6014 |
| oracle: trained on true grades | 0.6089 |
| best possible reordering of the 20 candidates | 0.7104 |

* **Naive learning is biased toward the logging policy.** More clicks do not help it
  (0.5566 at 1k sessions, 0.5598 at 3M). Its largest weight is on BM25 reciprocal rank,
  i.e. the logging ranker's own position (0.50, against 0.05 for IPS and 0.15 for the
  oracle; `theta_at_max_n` in `ltr.json`).
* **IPS recovers 84% of the naive-to-oracle gap** ((0.6012 − 0.5598) / (0.6089 − 0.5598)),
  and reaches the α-label skyline, which is its theoretical limit (the rest of the gap is
  the click model's α mapping and the sparse train labels, not bias). With the true η it
  gets there by about 10k sessions. With estimated η it needs the estimate to be good first:
  swap interventions catch up at about 300k sessions, harvesting at about 30k.
* **Position-bias estimation.** Mean absolute error of η_r/η_1 over ranks 1–10 at 3M sessions:
  swap 0.016, RandTop-10 0.004, harvesting 0.001 (the harvesting log is a *separate*
  simulated log in which BM25 and BGE orderings of the same candidates split traffic, so
  it has two production rankers where the others have one ranker plus 5% interventions).
  RandTop-10 is accurate on ranks 1–10 but *cannot identify ranks 11–20*, which are
  carried flat. That overstates deep-rank examination, and IPS with it plateaus at 0.594,
  below the others. A randomisation window must cover every rank you learn from.
* **Misestimation** (1M sessions): assuming severity p × true gives 0.5596 (p = 0,
  naive), 0.5685 (0.5), 0.5840 (0.75), **0.6020 (1, correct)**, 0.5955 (1.25), 0.5081 (1.5),
  0.3975 (2.0). Underestimating the bias degrades gracefully toward naive.
  Overestimating it is worse than not correcting at all: at p ≥ 1.5 the model falls to or
  below the logging ranker, because a few deep clicks get huge weights.
* **Clipping** min(1/η, τ) trades bias for variance. Here variance is not the problem (20
  ranks, η ≥ 0.05), so every clip costs quality: τ = 1.5 → 0.564, 2 → 0.570,
  5 → 0.584, 10 → 0.594, none → 0.602 (1M sessions; the same ordering at 10k).
* **Ablations.** With BM25-only features (`ltr_bm25only.md`) there is nothing to learn:
  every method, including the oracle (0.4889), stays at the logging ranker's 0.49. The
  gain above comes from the dense feature. On the synthetic stand-in (`ltr_synthetic.md`)
  the same pattern holds: naive 0.8165, IPS 0.8632, oracle 0.8599.

## 10. What is not done

* Real traffic: the analyser (`python -m hybridsearch.clicks.analyze_events`) is ready for
  `data/events/*.jsonl`. It has only been run on the simulated log
  (`results/m9/events_report_simulated.md`).
* Session-level (cluster) bootstrap for A/B on real logs, where impressions within a
  session are correlated. The simulation draws independent impressions.
* Dense retrieval pairs in the sensitivity study are in the 1M-subset world (the dense
  flat run covers the subset only); a full-collection dense run (DiskANN) can be added by
  dropping its DL19/DL20 run files into data/runs/ and rerunning the script (cells are
  cached, only new pairs are simulated). The M5 cross-encoder reranker's runs are not
  included (not on disk when this ran).
* The broker's C# team-draft implementation should run `tests/golden/interleaving.json`
  (the fixture exists; wiring it into the xUnit tests is the broker owner's job).
