"""Summarise a k6 --out json file into exact percentiles.

Percentiles are computed from every raw sample (no bucketing), and the run is
rejected if k6 reported dropped iterations, because then the arrival rate was
not actually constant and the tail is understated.

    python bench/load/summarize.py results/load/raw.json --label hedging-on > results/load/hedging-on.json
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    durations, failed, dropped = [], 0, 0
    with open(args.raw) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "Point":
                continue
            metric, data = rec["metric"], rec["data"]
            if metric == "http_req_duration":
                durations.append(data["value"])
            elif metric == "http_req_failed" and data["value"]:
                failed += 1
            elif metric == "dropped_iterations":
                dropped += int(data["value"])

    if dropped:
        print(f"refusing: {dropped} dropped iterations, arrival rate was not constant", file=sys.stderr)
        sys.exit(2)
    d = np.asarray(durations)
    out = {
        "label": args.label,
        "requests": int(d.size),
        "failed": failed,
        "p50_ms": float(np.percentile(d, 50)),
        "p90_ms": float(np.percentile(d, 90)),
        "p99_ms": float(np.percentile(d, 99)),
        "p999_ms": float(np.percentile(d, 99.9)),
        "max_ms": float(d.max()),
    }
    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
