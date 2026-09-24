| experiment | phase | requests | error_rate | p50_ms | p99_ms | mrr_at_10 | degraded_rate | partial_rate | failed_shard_rate | hedged_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| replica-hung | baseline | 400 | 0.0000 | 28.6955 | 187.2317 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0175 |
| replica-hung | fault | 400 | 0.0000 | 26.0554 | 157.1922 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0100 |
| replica-hung | recovery | 400 | 0.0000 | 23.7093 | 103.7518 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-hung | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-slow50-hedging | baseline | 400 | 0.0000 | 27.6954 | 2016.7542 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0325 |
| replica-slow50-hedging | fault | 400 | 0.0000 | 93.7987 | 1882.5176 | 0.4119 | 0.0000 | 0.0000 | 0.0025 | 0.0375 |
| replica-slow50-hedging | recovery | 400 | 0.0000 | 25.6634 | 122.4019 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-hedging | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-slow50-no-hedging | baseline | 400 | 0.0000 | 33.1348 | 670.7467 | 0.4126 | 0.0000 | 0.0000 | 0.0050 | 0.0000 |
| replica-slow50-no-hedging | fault | 400 | 0.0075 | 332.4276 | 5026.5980 | 0.4026 | 0.0000 | 0.0025 | 0.0750 | 0.0000 |
| replica-slow50-no-hedging | recovery | 400 | 0.0000 | 550.8322 | 7554.9366 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | **MRR retained under fault** | 0.976 | | | | | | | | |
| replica-starved | baseline | 400 | 0.0000 | 26.3645 | 94.0144 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-starved | fault | 400 | 0.0000 | 117.4771 | 345.2366 | 0.4123 | 0.0000 | 0.0000 | 0.0625 | 0.0725 |
| replica-starved | recovery | 400 | 0.0000 | 22.7204 | 117.9244 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0075 |
| replica-starved | **MRR retained under fault** | 1.001 | | | | | | | | |
| slice-down | baseline | 400 | 0.0000 | 28.1968 | 116.1458 | 0.4119 | 0.0000 | 0.0000 | 0.0000 | 0.0075 |
| slice-down | fault | 400 | 0.0000 | 317.3873 | 427.7285 | 0.0336 | 0.0000 | 0.0000 | 1.0000 | 0.0100 |
| slice-down | recovery | 400 | 0.0000 | 40.6317 | 2103.4691 | 0.4041 | 0.0000 | 0.0000 | 0.0625 | 0.0250 |
| slice-down | **MRR retained under fault** | 0.081 | | | | | | | | |
