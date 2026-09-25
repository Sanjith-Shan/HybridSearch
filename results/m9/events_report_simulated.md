# Event-log report

**Source label:** SIMULATED users: dbn-navigational (23950 pages); relevance from TREC DL graded qrels

events 319066, result pages 23950, sessions 12000
data-quality counters: none

## sim-aa (aa, 3952 pages)
SRM check: sessions [944, 1059], chi2 6.60, p = 0.0102 -> ok

| metric | control | treatment | delta | 95% CI | p (Welch) |
|---|---|---|---|---|---|
| ctr | 1.0832 | 1.0798 | -0.0034 | -0.0459..+0.0398 | 0.879 |
| clicks_at_1 | 0.5096 | 0.5280 | +0.0185 | -0.0130..+0.0499 | 0.246 |
| abandonment | 0.1442 | 0.1413 | -0.0029 | -0.0245..+0.0182 | 0.795 |
| rr_first | 0.6391 | 0.6542 | +0.0151 | -0.0097..+0.0407 | 0.229 |

## sim-ab (ab, 3983 pages)
SRM check: sessions [1044, 977], chi2 2.22, p = 0.136 -> ok

| metric | control | treatment | delta | 95% CI | p (Welch) |
|---|---|---|---|---|---|
| ctr | 1.1000 | 1.1539 | +0.0539 | +0.0128..+0.0905 | 0.00636 |
| clicks_at_1 | 0.5233 | 0.7702 | +0.2468 | +0.2168..+0.2743 | 4.22e-64 |
| abandonment | 0.1393 | 0.0338 | -0.1055 | -0.1232..-0.0881 | 4.73e-34 |
| rr_first | 0.6486 | 0.8597 | +0.2110 | +0.1895..+0.2326 | 5.2e-87 |

## sim-interleave (interleave, 3800 pages)
wins A 1081, wins B 2140, ties 579; Delta_AB -0.1393 (95% CI -0.1536..-0.1258); sign test p = 5.6e-79
team balance at rank 1: A share 0.503 (n=3800, p=0.709)

