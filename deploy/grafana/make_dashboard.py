"""Generates grafana/dashboards/hybridsearch.json.

The dashboard is generated rather than hand-edited so panel queries stay in
one reviewable place and match prometheus/slo_rules.yml.
"""
import json
from pathlib import Path

PANELS = [
    # (title, unit, [(legend, expr)], width)
    ("Search p50 / p95 / p99", "s", [
        ("p50", 'histogram_quantile(0.50, sum by (le) (rate(hs_search_duration_seconds_bucket[1m])))'),
        ("p95", 'histogram_quantile(0.95, sum by (le) (rate(hs_search_duration_seconds_bucket[1m])))'),
        ("p99", 'histogram_quantile(0.99, sum by (le) (rate(hs_search_duration_seconds_bucket[1m])))'),
    ], 12),
    ("Requests / s by outcome", "reqps", [
        ("{{outcome}}", 'sum by (outcome) (rate(hs_search_requests_total[1m]))'),
    ], 12),
    ("Bad-event ratio vs SLO (0.5%)", "percentunit", [
        ("bad ratio 5m", 'hs:search_bad_ratio:rate5m'),
        ("bad ratio 1h", 'hs:search_bad_ratio:rate1h'),
        ("SLO budget", 'vector(0.005)'),
    ], 12),
    ("Error budget remaining (30d)", "percentunit", [
        ("remaining", '1 - (sum(increase(hs_search_requests_total{outcome!="ok"}[30d])) '
                      '+ sum(increase(hs_search_duration_seconds_count[30d])) '
                      '- sum(increase(hs_search_duration_seconds_bucket{le="0.3"}[30d]))) '
                      '/ clamp_min(sum(increase(hs_search_requests_total[30d])) * 0.005, 1e-9)'),
    ], 12),
    ("Shard p99 by shard", "s", [
        ("shard {{shard}}", 'histogram_quantile(0.99, sum by (le, shard) (rate(hs_shard_duration_seconds_bucket[1m])))'),
    ], 12),
    ("Shard failures and partial results", "short", [
        ("fail {{shard}} {{reason}}", 'sum by (shard, reason) (rate(hs_shard_failures_total[1m]))'),
        ("partial {{shard}}", 'sum by (shard) (rate(hs_shard_partial_total[1m]))'),
    ], 12),
    ("Hedging: extra load and win rate", "percentunit", [
        ("extra load", 'hs:hedge_extra_load:ratio5m'),
        ("hedge win rate", 'sum(rate(hs_hedges_won_total[5m])) / clamp_min(sum(rate(hs_hedges_sent_total[5m])), 1e-9)'),
    ], 12),
    ("Degradation steps / s", "short", [
        ("{{step}}", 'sum by (step) (rate(hs_degradation_total[1m]))'),
    ], 12),
    ("Stage latency p99", "s", [
        ("encode", 'histogram_quantile(0.99, sum by (le) (rate(hs_encode_duration_seconds_bucket[1m])))'),
        ("rerank", 'histogram_quantile(0.99, sum by (le) (rate(hs_rerank_duration_seconds_bucket[1m])))'),
        ("suggest", 'histogram_quantile(0.99, sum by (le) (rate(hs_suggest_duration_seconds_bucket[1m])))'),
    ], 12),
    ("Feedback events / s (M9)", "short", [
        ("ingested {{type}}", 'sum by (type) (rate(hs_events_ingested_total[1m]))'),
        ("dropped {{reason}}", 'sum by (reason) (rate(hs_events_dropped_total[1m]))'),
    ], 12),
]


def panel(i, title, unit, targets, width):
    return {
        "id": i + 1,
        "type": "timeseries",
        "title": title,
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "gridPos": {"x": (i % 2) * 12, "y": (i // 2) * 8, "w": width, "h": 8},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "max"]}},
        "targets": [{"refId": chr(65 + j), "expr": expr, "legendFormat": legend}
                    for j, (legend, expr) in enumerate(targets)],
    }


def main():
    dash = {
        "title": "HybridSearch — serving and SLO",
        "uid": "hybridsearch-slo",
        "schemaVersion": 39,
        "time": {"from": "now-30m", "to": "now"},
        "refresh": "5s",
        "templating": {"list": [{"name": "DS_PROMETHEUS", "type": "datasource", "query": "prometheus"}]},
        "panels": [panel(i, *p) for i, p in enumerate(PANELS)],
    }
    out = Path(__file__).parent / "dashboards" / "hybridsearch.json"
    out.write_text(json.dumps(dash, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
