# M9 counterfactual LTR from simulated clicks

**simulated users (click model pbm-navigational, eta_r=(1/r)^1.0, relevance from MS MARCO train labels (train) / TREC DL 2019+2020 graded qrels (eval)).** World: full collection, BM25-only features (dense runs not available). Train 2000 queries, evaluate on 97 held-out queries (nDCG@10, reranking the logging ranker's top-20). Features: bm25_z, bm25_rr, log_len, qlen_bm25. 5 seeds; mean ± sd.

| baseline | nDCG@10 |
|---|---|
| logging_bm25 | 0.4912 |
| candidate_oracle | 0.7104 |
| oracle_true_grades | 0.4909 |
| skyline_alpha_labels | 0.4890 |

## Learning curve

| logged sessions | naive | ips_true | ips_swap_est | ips_randtop_est |
|---|---|---|---|---|
| 1,000 | 0.4910 ± 0.0009 | 0.4840 ± 0.0074 | 0.4910 ± 0.0009 | 0.4904 ± 0.0027 |
| 3,000 | 0.4904 ± 0.0009 | 0.4900 ± 0.0037 | 0.4905 ± 0.0009 | 0.4911 ± 0.0010 |
| 10,000 | 0.4903 ± 0.0006 | 0.4885 ± 0.0049 | 0.4903 ± 0.0006 | 0.4869 ± 0.0070 |
| 30,000 | 0.4903 ± 0.0005 | 0.4895 ± 0.0011 | 0.4898 ± 0.0006 | 0.4891 ± 0.0007 |
| 100,000 | 0.4903 ± 0.0009 | 0.4879 ± 0.0004 | 0.4840 ± 0.0109 | 0.4886 ± 0.0008 |
| 300,000 | 0.4901 ± 0.0007 | 0.4878 ± 0.0013 | 0.4651 ± 0.0328 | 0.4882 ± 0.0004 |
| 1,000,000 | 0.4902 ± 0.0005 | 0.4880 ± 0.0010 | 0.4887 ± 0.0008 | 0.4890 ± 0.0006 |
| 3,000,000 | 0.4902 ± 0.0005 | 0.4889 ± 0.0007 | 0.4882 ± 0.0010 | 0.4889 ± 0.0003 |

## Position-bias estimation (mean absolute error of eta_r/eta_1 over ranks 1-10)

Interventions in 5% of sessions.

| logged sessions | swap(1,k) | RandTop-10 |
|---|---|---|
| 1,000 | 0.7071 | 0.2198 |
| 3,000 | 0.7071 | 0.1587 |
| 10,000 | 0.6102 | 0.0700 |
| 30,000 | 0.2313 | 0.0408 |
| 100,000 | 0.0900 | 0.0148 |
| 300,000 | 0.0536 | 0.0119 |
| 1,000,000 | 0.0321 | 0.0056 |
| 3,000,000 | 0.0135 | 0.0029 |

## Propensity misestimation (1,000,000 sessions)

IPS with eta_r=(1/r)^(p*true_severity); p=1 is correct, p=0 is naive

| p | 0.0 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.4903 | 0.4898 | 0.4903 | 0.4882 | 0.4187 | 0.3184 | 0.2713 |

## Weight clipping (IPS with true eta, weight min(1/eta, tau))

| sessions | tau=1.5 | tau=2.0 | tau=5.0 | tau=10.0 | tau=none |
|---|---|---|---|---|---|
| 10,000 | 0.4903 | 0.4905 | 0.4896 | 0.4893 | 0.4878 |
| 1,000,000 | 0.4897 | 0.4894 | 0.4894 | 0.4886 | 0.4882 |
