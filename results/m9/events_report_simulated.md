# Event-log report

**Source label:** SIMULATED users: dbn-navigational (23950 pages); relevance from TREC DL graded qrels

events 319011, result pages 23950, sessions 12000
data-quality counters: none

## sim-aa (aa, 3952 pages)
SRM check: sessions [997, 1006], chi2 0.04, p = 0.841 -> ok

| metric | control | treatment | delta | 95% CI | p (Welch) |
|---|---|---|---|---|---|
| ctr | 1.1023 | 1.0601 | -0.0422 | -0.0839..+0.0005 | 0.0572 |
| clicks_at_1 | 0.5215 | 0.5164 | -0.0050 | -0.0356..+0.0272 | 0.752 |
| abandonment | 0.1397 | 0.1458 | +0.0061 | -0.0163..+0.0279 | 0.586 |
| rr_first | 0.6501 | 0.6434 | -0.0066 | -0.0308..+0.0168 | 0.598 |

## sim-ab (ab, 3983 pages)
SRM check: sessions [1035, 986], chi2 1.19, p = 0.276 -> ok

| metric | control | treatment | delta | 95% CI | p (Welch) |
|---|---|---|---|---|---|
| ctr | 1.1054 | 1.1322 | +0.0268 | -0.0152..+0.0657 | 0.169 |
| clicks_at_1 | 0.5355 | 0.7696 | +0.2341 | +0.2038..+0.2620 | 6.81e-58 |
| abandonment | 0.1392 | 0.0371 | -0.1021 | -0.1183..-0.0844 | 1.92e-31 |
| rr_first | 0.6581 | 0.8568 | +0.1987 | +0.1772..+0.2188 | 1.13e-76 |

## sim-interleave (interleave, 3800 pages)
wins A 1081, wins B 2140, ties 579; Delta_AB -0.1393 (95% CI -0.1530..-0.1257); sign test p = 5.6e-79
team balance at rank 1: A share 0.503 (n=3800, p=0.709)

