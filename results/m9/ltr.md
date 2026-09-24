# M9 counterfactual LTR from simulated clicks

**simulated users (click model pbm-navigational, eta_r=(1/r)^1.0, relevance from MS MARCO train labels (train) / TREC DL 2019+2020 graded qrels (eval)).** World: full collection (Anserini BM25 top-20 candidates). Train 200 queries, evaluate on 97 held-out queries (nDCG@10, reranking the logging ranker's top-20). Features: bm25_z, dense_z, bm25_rr, dense_rr, log_len, qlen_bm25, qlen_dense. 5 seeds; mean ± sd.

| baseline | nDCG@10 |
|---|---|
| logging_bm25 | 0.4912 |
| candidate_oracle | 0.7104 |
| oracle_true_grades | 0.6089 |
| skyline_alpha_labels | 0.6014 |

## Learning curve

| logged sessions | naive | ips_true | ips_swap_est | ips_randtop_est | ips_harvest_est |
|---|---|---|---|---|---|
| 1,000 | 0.5566 ± 0.0064 | 0.5865 ± 0.0264 | 0.5566 ± 0.0064 | 0.5754 ± 0.0143 | 0.5772 ± 0.0146 |
| 3,000 | 0.5553 ± 0.0049 | 0.5915 ± 0.0142 | 0.5575 ± 0.0065 | 0.5833 ± 0.0149 | 0.5826 ± 0.0099 |
| 10,000 | 0.5590 ± 0.0020 | 0.5997 ± 0.0032 | 0.5672 ± 0.0053 | 0.5903 ± 0.0067 | 0.5931 ± 0.0034 |
| 30,000 | 0.5598 ± 0.0016 | 0.5989 ± 0.0017 | 0.5716 ± 0.0061 | 0.5922 ± 0.0085 | 0.5971 ± 0.0023 |
| 100,000 | 0.5592 ± 0.0003 | 0.5996 ± 0.0015 | 0.5914 ± 0.0050 | 0.5954 ± 0.0015 | 0.5987 ± 0.0009 |
| 300,000 | 0.5601 ± 0.0004 | 0.6009 ± 0.0011 | 0.5976 ± 0.0032 | 0.5932 ± 0.0034 | 0.6009 ± 0.0014 |
| 1,000,000 | 0.5597 ± 0.0003 | 0.6008 ± 0.0011 | 0.6012 ± 0.0016 | 0.5941 ± 0.0009 | 0.6012 ± 0.0008 |
| 3,000,000 | 0.5598 ± 0.0000 | 0.6012 ± 0.0005 | 0.6010 ± 0.0013 | 0.5938 ± 0.0003 | 0.6010 ± 0.0005 |

## Position-bias estimation (mean absolute error of eta_r/eta_1 over ranks 1-10)

Interventions in 5% of sessions.

Harvesting (if present) uses no deliberate randomisation: two production rankers (BM25 order and dense order of the same candidates) share traffic 50/50, in a separate log of the same size.

| logged sessions | swap | randtop | harvest |
|---|---|---|---|
| 1,000 | 0.7071 | 0.3193 | 0.1327 |
| 3,000 | 0.5717 | 0.1404 | 0.0446 |
| 10,000 | 0.3494 | 0.0643 | 0.0251 |
| 30,000 | 0.2426 | 0.0361 | 0.0096 |
| 100,000 | 0.0613 | 0.0158 | 0.0055 |
| 300,000 | 0.0357 | 0.0116 | 0.0037 |
| 1,000,000 | 0.0203 | 0.0055 | 0.0021 |
| 3,000,000 | 0.0155 | 0.0039 | 0.0011 |

## Propensity misestimation (1,000,000 sessions)

IPS with eta_r=(1/r)^(p*true_severity); p=1 is correct, p=0 is naive

| p | 0.0 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.5596 | 0.5685 | 0.5840 | 0.6020 | 0.5955 | 0.5081 | 0.3975 |

## Weight clipping (IPS with true eta, weight min(1/eta, tau))

| sessions | tau=1.5 | tau=2.0 | tau=5.0 | tau=10.0 | tau=none |
|---|---|---|---|---|---|
| 10,000 | 0.5628 | 0.5676 | 0.5837 | 0.5918 | 0.5980 |
| 1,000,000 | 0.5637 | 0.5699 | 0.5843 | 0.5941 | 0.6020 |
