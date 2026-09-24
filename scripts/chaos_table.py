"""Print chaos experiment results (results/chaos/*.json) as one table."""
import json, sys
from pathlib import Path

cols = ["requests", "error_rate", "p50_ms", "p99_ms", "mrr_at_10", "degraded_rate", "partial_rate", "failed_shard_rate", "hedged_rate"]
print("| experiment | phase | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 2))
for f in sys.argv[1:] or sorted(Path("results/chaos").glob("*.json")):
    d = json.loads(Path(f).read_text())
    for phase, m in d["phases"].items():
        vals = [f"{m[c]:.4f}" if isinstance(m[c], float) else str(m[c]) for c in cols]
        print(f"| {d['experiment']} | {phase} | " + " | ".join(vals) + " |")
    print(f"| {d['experiment']} | **MRR retained under fault** | {d['mrr_retained_under_fault']:.3f} |" + " |" * (len(cols) - 1))
