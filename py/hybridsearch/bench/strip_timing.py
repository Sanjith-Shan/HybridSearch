"""Drop timing fields from jsonl result rows measured while timing was not allowed
(shared machine under background QoS): keeps recall, hops, SSD read counts, RAM.

    python -m hybridsearch.bench.strip_timing IN.jsonl OUT.jsonl
"""
import json
import sys

TIMING = ("qps", "cpu_", "mean_us", "p50_us", "p90_us", "p99_us", "io_us", "us_per_io_round", "wall_",
          "open_seconds")


def main(src: str, dst: str) -> int:
    with open(src) as f, open(dst, "a") as g:
        for line in f:
            row = json.loads(line)
            row = {k: v for k, v in row.items() if not k.startswith(TIMING)}
            row["timing"] = "not measured (background QoS window)"
            g.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
