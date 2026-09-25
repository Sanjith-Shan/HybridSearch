| experiment | phase | requests | error_rate | p50_ms | p99_ms | mrr_at_10 | degraded_rate | partial_rate | failed_shard_rate | hedged_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| replica-hung | baseline | 400 | 0.0075 | 22.6676 | 392.8287 | 0.4220 | 0.9475 | 0.0125 | 0.0075 | 0.0000 |
| replica-hung | fault | 400 | 0.0000 | 12.4630 | 52.6149 | 0.4119 | 1.0000 | 0.0000 | 0.0000 | 0.0200 |
| replica-hung | recovery | 400 | 0.0025 | 6.8307 | 316.2086 | 0.4565 | 0.8325 | 0.0075 | 0.0025 | 0.0350 |
| replica-hung | **MRR retained under fault** | 0.976 | | | | | | | | |
| replica-slow50-hedging | baseline | 400 | 0.0050 | 65.2500 | 344.3593 | 0.6351 | 0.0075 | 0.0025 | 0.0000 | 0.0200 |
| replica-slow50-hedging | fault | 400 | 0.0150 | 107.8689 | 348.4599 | 0.6216 | 0.0175 | 0.0200 | 0.0175 | 0.0175 |
| replica-slow50-hedging | recovery | 400 | 0.0000 | 48.7670 | 338.0526 | 0.6432 | 0.0050 | 0.0150 | 0.0125 | 0.0175 |
| replica-slow50-hedging | **MRR retained under fault** | 0.979 | | | | | | | | |
| replica-slow50-no-hedging | baseline | 400 | 0.0000 | 38.2893 | 320.3095 | 0.6438 | 0.0000 | 0.0050 | 0.0025 | 0.0000 |
| replica-slow50-no-hedging | fault | 400 | 0.0000 | 36.5803 | 310.9196 | 0.6438 | 0.0000 | 0.0050 | 0.0050 | 0.0000 |
| replica-slow50-no-hedging | recovery | 400 | 0.0000 | 33.3247 | 154.1636 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-slow50-no-hedging | **MRR retained under fault** | 1.000 | | | | | | | | |
| replica-starved | baseline | 400 | 0.0000 | 37.0552 | 100.8711 | 0.6438 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| replica-starved | fault | 400 | 0.0050 | 50.5554 | 500.0920 | 0.6263 | 0.0000 | 0.0175 | 0.0500 | 0.0600 |
| replica-starved | recovery | 400 | 0.0025 | 90.1994 | 325.5992 | 0.6408 | 0.0050 | 0.0100 | 0.0050 | 0.0425 |
| replica-starved | **MRR retained under fault** | 0.973 | | | | | | | | |
| slice-down | baseline | 400 | 0.0050 | 165.4886 | 334.8689 | 0.6359 | 0.0025 | 0.0250 | 0.0075 | 0.0275 |
| slice-down | fault | 400 | 0.0000 | 309.6495 | 346.1948 | 0.3665 | 0.8225 | 0.0000 | 1.0000 | 0.0000 |
| slice-down | recovery | 400 | 0.0000 | 6.1536 | 54.3935 | 0.4119 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| slice-down | **MRR retained under fault** | 0.576 | | | | | | | | |
