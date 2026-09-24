# M9 counterfactual LTR from simulated clicks

**simulated users (click model pbm-navigational, eta_r=(1/r)^1.0, relevance from MS MARCO train labels (train) / TREC DL 2019+2020 graded qrels (eval)).** World: full collection (Anserini BM25 top-20 candidates). Train 200 queries, evaluate on 97 held-out queries (nDCG@10, reranking the logging ranker's top-20). Features: bm25_z, bm25_rr, log_len, qlen_bm25. 5 seeds; mean ± sd.

| baseline | nDCG@10 |
|---|---|
| logging_bm25 | 0.4912 |
| candidate_oracle | 0.7104 |
| oracle_true_grades | 0.4889 |
| skyline_alpha_labels | 0.4905 |

## Learning curve

| logged sessions | naive | ips_true | ips_swap_est | ips_randtop_est |
|---|---|---|---|---|
| 1,000 | 0.4910 ± 0.0004 | 0.4845 ± 0.0098 | 0.4910 ± 0.0004 | 0.4909 ± 0.0023 |
| 3,000 | 0.4901 ± 0.0007 | 0.4897 ± 0.0043 | 0.4909 ± 0.0009 | 0.4907 ± 0.0009 |
| 10,000 | 0.4899 ± 0.0009 | 0.4911 ± 0.0019 | 0.4907 ± 0.0009 | 0.4901 ± 0.0007 |
| 30,000 | 0.4900 ± 0.0005 | 0.4896 ± 0.0005 | 0.4900 ± 0.0006 | 0.4892 ± 0.0002 |
| 100,000 | 0.4895 ± 0.0004 | 0.4893 ± 0.0009 | 0.4895 ± 0.0006 | 0.4895 ± 0.0005 |
| 300,000 | 0.4895 ± 0.0005 | 0.4895 ± 0.0006 | 0.4894 ± 0.0005 | 0.4891 ± 0.0010 |
| 1,000,000 | 0.4899 ± 0.0001 | 0.4899 ± 0.0008 | 0.4891 ± 0.0004 | 0.4895 ± 0.0004 |
| 3,000,000 | 0.4899 ± 0.0000 | 0.4898 ± 0.0005 | 0.4894 ± 0.0007 | 0.4892 ± 0.0005 |

## Position-bias estimation (mean absolute error of eta_r/eta_1 over ranks 1-10)

Interventions in 5% of sessions.

Harvesting (if present) uses no deliberate randomisation: two production rankers (BM25 order and dense order of the same candidates) share traffic 50/50, in a separate log of the same size.

| logged sessions | swap | randtop |
|---|---|---|
| 1,000 | 0.7071 | 0.3193 |
| 3,000 | 0.5717 | 0.1404 |
| 10,000 | 0.3494 | 0.0643 |
| 30,000 | 0.2426 | 0.0361 |
| 100,000 | 0.0613 | 0.0158 |
| 300,000 | 0.0357 | 0.0116 |
| 1,000,000 | 0.0203 | 0.0055 |
| 3,000,000 | 0.0155 | 0.0039 |

## Propensity misestimation (1,000,000 sessions)

IPS with eta_r=(1/r)^(p*true_severity); p=1 is correct, p=0 is naive

| p | 0.0 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.4898 | 0.4891 | 0.4901 | 0.4901 | 0.4451 | 0.3329 | 0.2699 |

## Weight clipping (IPS with true eta, weight min(1/eta, tau))

| sessions | tau=1.5 | tau=2.0 | tau=5.0 | tau=10.0 | tau=none |
|---|---|---|---|---|---|
| 10,000 | 0.4900 | 0.4899 | 0.4903 | 0.4901 | 0.4899 |
| 1,000,000 | 0.4899 | 0.4898 | 0.4902 | 0.4895 | 0.4901 |
