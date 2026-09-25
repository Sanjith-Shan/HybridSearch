| experiment | phase | requests | error_rate | p50_ms | p99_ms | mrr_at_10 | degraded_rate | partial_rate | failed_shard_rate | hedged_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| replica-hung | baseline | 400 | 0.0000 | 39.7553 | 489.6312 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-hung | fault | 400 | 0.0000 | 23.6867 | 130.5353 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0050 |
| replica-hung | recovery | 400 | 0.0325 | 44.1130 | 2449.6424 | 0.3934 | 0.0000 | 0.0000 | 0.0350 | 0.0075 |
| replica-hung | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-slow50-hedging | baseline | 400 | 0.0000 | 43.6510 | 180.6077 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0050 |
| replica-slow50-hedging | fault | 400 | 0.0000 | 58.8765 | 153.8145 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0175 |
| replica-slow50-hedging | recovery | 400 | 0.0025 | 31.3682 | 939.6407 | 0.4117 | 0.0000 | 0.0000 | 0.0025 | 0.0525 |
| replica-slow50-hedging | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-slow50-no-hedging | baseline | 400 | 0.0100 | 53.9505 | 674.9117 | 0.4101 | 0.0000 | 0.0025 | 0.0125 | 0.0000 |
| replica-slow50-no-hedging | fault | 400 | 0.0000 | 75.4729 | 187.9557 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | recovery | 400 | 0.0000 | 35.1923 | 166.3367 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | **MRR retained under fault** | 1.004 | | | | | | | | |
| replica-starved | baseline | 400 | 0.0000 | 36.2581 | 198.1297 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0050 |
| replica-starved | fault | 400 | 0.0050 | 64.2466 | 633.4844 | 0.3988 | 0.0000 | 0.0050 | 0.0650 | 0.0525 |
| replica-starved | recovery | 400 | 0.0000 | 57.8871 | 1710.9091 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0125 |
| replica-starved | **MRR retained under fault** | 0.968 | | | | | | | | |
| slice-down | baseline | 400 | 0.0075 | 27.0757 | 573.0588 | 0.4031 | 0.0000 | 0.0025 | 0.0075 | 0.0050 |
| slice-down | fault | 400 | 0.0425 | 320.4085 | 1456.2486 | 0.2879 | 0.0000 | 0.0000 | 0.9575 | 0.0250 |
| slice-down | recovery | 400 | 0.1975 | 45.0651 | 5055.4889 | 0.3176 | 0.0000 | 0.0025 | 0.0350 | 0.0150 |
| slice-down | **MRR retained under fault** | 0.714 | | | | | | | | |
