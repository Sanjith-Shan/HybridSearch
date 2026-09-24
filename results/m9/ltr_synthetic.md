# M9 counterfactual LTR from simulated clicks

**simulated users (click model pbm-navigational, eta_r=(1/r)^1.0, relevance from synthetic grades).** World: SYNTHETIC stand-in (features generated from grades); real runs not available. Train 2000 queries, evaluate on 97 held-out queries (nDCG@10, reranking the logging ranker's top-20). Features: bm25_z, dense_z, bm25_rr, dense_rr, log_len, qlen_bm25, qlen_dense. 5 seeds; mean ± sd.

| baseline | nDCG@10 |
|---|---|
| logging_bm25 | 0.6994 |
| candidate_oracle | 1.0000 |
| oracle_true_grades | 0.8599 |
| skyline_alpha_labels | 0.8633 |

## Learning curve

| logged sessions | naive | ips_true | ips_swap_est | ips_randtop_est | ips_harvest_est |
|---|---|---|---|---|---|
| 1,000 | 0.8149 ± 0.0031 | 0.8570 ± 0.0079 | 0.8266 ± 0.0165 | 0.8544 ± 0.0108 | 0.8542 ± 0.0048 |
| 3,000 | 0.8157 ± 0.0013 | 0.8618 ± 0.0012 | 0.8405 ± 0.0135 | 0.8611 ± 0.0021 | 0.8605 ± 0.0010 |
| 10,000 | 0.8156 ± 0.0006 | 0.8619 ± 0.0012 | 0.8612 ± 0.0040 | 0.8606 ± 0.0011 | 0.8627 ± 0.0011 |
| 30,000 | 0.8158 ± 0.0007 | 0.8626 ± 0.0003 | 0.8618 ± 0.0017 | 0.8610 ± 0.0012 | 0.8631 ± 0.0009 |
| 100,000 | 0.8162 ± 0.0004 | 0.8629 ± 0.0007 | 0.8614 ± 0.0022 | 0.8600 ± 0.0006 | 0.8630 ± 0.0008 |
| 300,000 | 0.8163 ± 0.0003 | 0.8628 ± 0.0003 | 0.8637 ± 0.0014 | 0.8604 ± 0.0009 | 0.8630 ± 0.0008 |
| 1,000,000 | 0.8164 ± 0.0004 | 0.8631 ± 0.0003 | 0.8623 ± 0.0015 | 0.8607 ± 0.0002 | 0.8631 ± 0.0002 |
| 3,000,000 | 0.8165 ± 0.0003 | 0.8632 ± 0.0003 | 0.8628 ± 0.0009 | 0.8608 ± 0.0002 | 0.8630 ± 0.0003 |

## Position-bias estimation (mean absolute error of eta_r/eta_1 over ranks 1-10)

Interventions in 5% of sessions.

Harvesting (if present) uses no deliberate randomisation: two production rankers (BM25 order and dense order of the same candidates) share traffic 50/50, in a separate log of the same size.

| logged sessions | swap | randtop | harvest |
|---|---|---|---|
| 1,000 | 0.4893 | 0.0645 | 0.0242 |
| 3,000 | 0.4021 | 0.0357 | 0.0149 |
| 10,000 | 0.0720 | 0.0178 | 0.0083 |
| 30,000 | 0.0470 | 0.0132 | 0.0044 |
| 100,000 | 0.0286 | 0.0073 | 0.0027 |
| 300,000 | 0.0181 | 0.0047 | 0.0013 |
| 1,000,000 | 0.0120 | 0.0020 | 0.0007 |
| 3,000,000 | 0.0060 | 0.0014 | 0.0004 |

## Propensity misestimation (1,000,000 sessions)

IPS with eta_r=(1/r)^(p*true_severity); p=1 is correct, p=0 is naive

| p | 0.0 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| nDCG@10 | 0.8157 | 0.8480 | 0.8657 | 0.8634 | 0.8310 | 0.7776 | 0.6309 |

## Weight clipping (IPS with true eta, weight min(1/eta, tau))

| sessions | tau=1.5 | tau=2.0 | tau=5.0 | tau=10.0 | tau=none |
|---|---|---|---|---|---|
| 10,000 | 0.8264 | 0.8323 | 0.8483 | 0.8592 | 0.8622 |
| 1,000,000 | 0.8264 | 0.8318 | 0.8482 | 0.8606 | 0.8634 |
