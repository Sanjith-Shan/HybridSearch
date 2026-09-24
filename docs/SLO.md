# Service level objectives

Written before any load test so the tests measure against a target instead of
deriving the target from whatever the tests showed.

## Scope

The `/api/search` endpoint of the broker, measured at the broker. Autocomplete has its
own objective because it runs per keystroke.

## Objectives (rolling 30 days)

| SLI | Good event | Objective |
|---|---|---|
| Search availability + latency | response is 2xx **and** completes in ≤ 300 ms | **99.5%** of requests |
| Search quality under failure | MRR@10 on the dev replay set with any single shard degraded | ≥ 90% of the healthy-system MRR@10 |
| Suggest latency | `/api/suggest` completes in ≤ 20 ms | 99.9% |
| Degradation is visible | degraded responses carry `degradation.level > 0` | 100% (tested, not sampled) |

A degraded answer that meets the deadline counts as **good** for the latency SLI and is
tracked separately (`hs:search_degraded_ratio:rate5m`). This is deliberate: the degradation
policy exists to spend quality instead of availability, and the quality SLI is what
stops it from quietly spending too much.

## Error budget

0.5% of requests per 30 days. Alerting uses multi-window burn rates
(`deploy/prometheus/slo_rules.yml`):

- **Page**: burn rate > 14.4× over both 1 h and 5 m (2% of the monthly budget in one hour).
- **Ticket**: hedging adds > 5% extra shard load for 10 minutes. Hedging is a latency tool
  paid for in load, and the budget on that load is part of the SLO, not an afterthought.

## Error budget policy

- Budget remaining > 50%: ship freely, experiments may run at up to 20% allocation.
- 10–50%: ranker experiments limited to interleaving (lower exposure per decision).
- < 10%: freeze ranker and serving changes except reliability fixes.

## How it is measured

- Latency percentiles come from constant-arrival-rate load (`bench/load/`), never
  closed-loop, so a slow response cannot suppress the requests queued behind it
  (coordinated omission).
- Numbers that go anywhere public come only from the pinned Linux box, labelled with
  CPU, cores, pinning, and warm/cold cache. macOS numbers are development signal.
