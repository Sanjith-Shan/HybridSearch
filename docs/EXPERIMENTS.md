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
against fresh click simulation at N80 for 6 pairs × 2 models: direct power 0.775 to 0.826
(1,000 trials each; target 0.80).

**Rankers.** Real (DL19+DL20 nDCG@10): BM25 top-100 re-ordered by BGE-base similarity
0.6681; RRF(BM25, that BGE ordering) 0.6161; Anserini BM25 0.4912; HybridSearch's own
BM25 engine, Lucene-quantised lengths 0.4912 and textbook lengths 0.4906. The two BGE
rankers only reorder BM25's top 100 (no dense retrieval over the collection existed
when this ran; the BGE vectors were encoded by M9 code). Synthetic: "noisy oracle"
rankers (grade + σ·noise, shared noise), 0.4624 to 0.7634, so the gap is controlled.

**Headline (median over 9 click models; ratio = best A/B metric's N80 / team-draft N80):**

| pair | Δ nDCG@10 | TDI N80 | best A/B N80 | ratio (range) |
|---|---|---|---|---|
| BGE-rerank vs RRF (real) | +0.052 | 3,600 | 9,866 | 2.4x (1.4–11.6x) |
| RRF vs BM25 (real) | +0.125 | 151 | 313 | 2.5x (1.3–4.8x) |
| BGE-rerank vs BM25-textbook (real) | +0.177 | 164 | 214 | 1.6x (1.1–2.9x) |
| synthetic σ1.0 vs σ1.03 | +0.014 | 8,443 | 164,254 | 16.4x (8.7–154x) |
| synthetic σ1.0 vs σ1.1 | +0.038 | 1,251 | 17,059 | 10.5x (6.2–103x) |
| synthetic σ1.0 vs σ1.25 | +0.097 | 229 | 626 | 4.1x (2.4–31x) |
| synthetic σ1.0 vs σ1.5 | +0.180 | 74 | 180 | 2.7x (2.0–13x) |
| synthetic σ1.0 vs σ2.0 | +0.301 | 35 | 73 | 2.3x (1.8–4.9x) |

**What this says, honestly.**

* Team-draft interleaving needed fewer impressions than the best A/B metric in all 72
  (pair, model) cells with a known ordering (ratio 1.06x to 154x). The advantage **grows as the gap shrinks**:
  about 2x for large gaps, about 10–16x (median) for gaps of 0.01–0.04 nDCG@10, and 100x or
  more only in the best case (cascade/DBN "perfect" users, who click only relevant results).
* The published finding (Chapelle et al. 2012: interleaving 1–2 orders of magnitude more
  sensitive on real Bing/Yahoo! traffic) is reproduced **only for small gaps and only for
  some user models**. For the real ranker pairs here, whose gaps are large (0.05–0.18),
  the advantage is 1.1–12x, median about 2x. Real A/B tests are also noisier than this
  simulation (heterogeneous users, sessions, bots, query mix), which hurts A/B more than
  interleaving, so these ratios are probably a lower bound on the real one. That last
  point is an argument, not a measurement.
* **Balanced interleaving was at least as sensitive as team-draft in all 72 cells, and
  probabilistic interleaving in 61 of 72** (median best-interleaving/TDI N80 = 0.06–0.88
  per pair). Team-draft pays for its
  fairness guarantees with variance: its credit comes from team labels, which are random
  per impression.
* **Which A/B metric matters a lot.** Clicks per impression ("CTR") often pointed the
  *wrong* way: it agreed with offline nDCG in only 19 of 27 real-pair cells and 29 of 45
  synthetic ones. Under cascade/DBN users a better ranking satisfies users sooner and so
  produces *fewer* clicks. Clicks@1, MRR of first click and abandonment agreed in 27 of 27
  and 45 of 45 cells; every interleaving method agreed in all cells.
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
