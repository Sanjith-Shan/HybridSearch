| experiment | phase | requests | error_rate | p50_ms | p99_ms | mrr_at_10 | degraded_rate | partial_rate | failed_shard_rate | hedged_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| replica-hung | baseline | 400 | 0.0225 | 63.8361 | 400.8584 | 0.6202 | 0.0225 | 0.0075 | 0.0000 | 0.0050 |
| replica-hung | fault | 400 | 0.0350 | 58.8605 | 564.1016 | 0.5859 | 0.0725 | 0.0100 | 0.0150 | 0.0275 |
| replica-hung | recovery | 400 | 0.0000 | 47.6118 | 222.3986 | 0.6438 | 0.0050 | 0.0000 | 0.0000 | 0.0550 |
| replica-hung | **MRR retained under fault** | 0.945 | | | | | | | | |
| replica-slow50-hedging | baseline | 400 | 0.0000 | 40.1518 | 218.4688 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0275 |
| replica-slow50-hedging | fault | 400 | 0.0050 | 87.8868 | 354.2925 | 0.6388 | 0.0000 | 0.0075 | 0.0050 | 0.0150 |
| replica-slow50-hedging | recovery | 400 | 0.0025 | 42.5595 | 393.4571 | 0.6313 | 0.0175 | 0.0100 | 0.0100 | 0.0050 |
| replica-slow50-hedging | **MRR retained under fault** | 0.992 | | | | | | | | |
| replica-slow50-no-hedging | baseline | 400 | 0.0000 | 34.4774 | 161.5818 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | fault | 400 | 0.0000 | 44.8374 | 224.9135 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | recovery | 400 | 0.0175 | 42.5839 | 425.7071 | 0.6275 | 0.0050 | 0.0075 | 0.0050 | 0.0000 |
| replica-slow50-no-hedging | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-starved | baseline | 400 | 0.0625 | 57.5203 | 568.0011 | 0.5932 | 0.0100 | 0.0150 | 0.0150 | 0.0200 |
| replica-starved | fault | 400 | 0.0025 | 57.1023 | 358.5755 | 0.6385 | 0.0025 | 0.0250 | 0.0400 | 0.0675 |
| replica-starved | recovery | 400 | 0.0000 | 29.1795 | 108.4415 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0125 |
| replica-starved | **MRR retained under fault** | 1.076 | | | | | | | | |
| slice-down | baseline | 400 | 0.0050 | 82.9062 | 381.6356 | 0.6397 | 0.0075 | 0.0000 | 0.0000 | 0.0225 |
| slice-down | fault | 400 | 0.0625 | 112.9022 | 452.1069 | 0.4508 | 0.0275 | 0.0325 | 0.9375 | 0.0400 |
| slice-down | recovery | 400 | 0.0075 | 31.9107 | 315.8422 | 0.6357 | 0.0100 | 0.0025 | 0.0000 | 0.0050 |
| slice-down | **MRR retained under fault** | 0.705 | | | | | | | | |
