| experiment | phase | requests | error_rate | p50_ms | p99_ms | mrr_at_10 | degraded_rate | partial_rate | failed_shard_rate | hedged_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| replica-hung | baseline | 400 | 0.0000 | 34.3539 | 164.5602 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-hung | fault | 400 | 0.0000 | 37.1009 | 325.0049 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0400 |
| replica-hung | recovery | 400 | 0.0000 | 34.7227 | 188.4265 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0300 |
| replica-hung | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-slow50-hedging | baseline | 400 | 0.0000 | 40.4166 | 135.0669 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0150 |
| replica-slow50-hedging | fault | 400 | 0.0000 | 42.3130 | 122.7346 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0350 |
| replica-slow50-hedging | recovery | 400 | 0.0000 | 35.4686 | 344.5947 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0175 |
| replica-slow50-hedging | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-slow50-no-hedging | baseline | 400 | 0.0000 | 16.0296 | 98.1867 | 0.4119 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | fault | 400 | 0.0175 | 36.9306 | 362.0054 | 0.4834 | 0.8025 | 0.0100 | 0.0325 | 0.0000 |
| replica-slow50-no-hedging | recovery | 400 | 0.0075 | 58.3308 | 323.1604 | 0.6319 | 0.1075 | 0.0075 | 0.0025 | 0.0000 |
| replica-slow50-no-hedging | **MRR retained under fault** | 1.174 | | | | | | | | |
| replica-starved | baseline | 400 | 0.0000 | 35.6946 | 136.0723 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0125 |
| replica-starved | fault | 400 | 0.0000 | 41.1824 | 316.7177 | 0.6388 | 0.0000 | 0.0100 | 0.0100 | 0.0600 |
| replica-starved | recovery | 400 | 0.0200 | 80.0594 | 397.6216 | 0.6185 | 0.0250 | 0.0250 | 0.0175 | 0.0350 |
| replica-starved | **MRR retained under fault** | 0.992 | | | | | | | | |
| slice-down | baseline | 400 | 0.1475 | 259.1109 | 605.2370 | 0.4711 | 0.1650 | 0.0700 | 0.0700 | 0.0400 |
| slice-down | fault | 400 | 0.0125 | 17.6161 | 398.0464 | 0.3405 | 0.8450 | 0.0350 | 0.9875 | 0.0100 |
| slice-down | recovery | 400 | 0.0000 | 10.1947 | 561.0230 | 0.4119 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| slice-down | **MRR retained under fault** | 0.723 | | | | | | | | |
