#!/usr/bin/env bash
# Every chaos experiment, reported as latency AND MRR@10 under failure.
# Runs against the native cluster (scripts/run_cluster.sh start); each experiment
# replays the same dev query sequence in its baseline, fault and recovery phases.
#
#   bench/chaos/run_all.sh [mode] [rerank] [rate] [phase-seconds]
#   bench/chaos/run_all.sh hybrid true 20 30
#
# Faults (shard processes are 4 slices x 2 replicas, see run_cluster.sh):
#   replica-hung     one replica SIGSTOPped: a hung node that accepts connections and never answers
#   replica-slow50   one replica +50 ms per request via its chaos control file
#   replica-starved  one replica demoted to background QoS (efficiency cores, lowest priority)
#   slice-down       BOTH replicas of one slice hung: a quarter of the collection is unreachable
# replica-slow50 runs twice, with hedging on and off, to measure what hedging buys and costs.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$here"
mode="${1:-hybrid}" rerank="${2:-true}" rate="${3:-20}" secs="${4:-30}"
chaos="data/cluster/chaos"
suffix=""; [ "$rerank" = true ] && suffix="-rerank"
out="results/chaos/${HS_PARTITION:-mod}-${mode}${suffix}"
cluster_args="${HS_CLUSTER_ARGS:-}"  # e.g. --no-vector
mkdir -p "$out"

pids() { P=($(cat data/cluster/pids)); }
run() {
  .venv/bin/python -m hybridsearch.serving.chaos \
    --queries data/raw/msmarco/queries.dev.small.tsv --qrels data/subset/1m/qrels.dev.tsv \
    --rate "$rate" --phase-seconds "$secs" --mode "$mode" --rerank "$rerank" --deadline-ms 300 "$@" >/dev/null
}
# Warm the broker before measuring: its degradation planner starts from conservative stage-cost
# priors, and until it has latency samples it may shed the dense path. A baseline taken during
# that window is not a baseline (found in the first hybrid run: 95% of baseline requests degraded).
warm() {
  .venv/bin/python -m hybridsearch.serving.chaos --name warmup \
    --queries data/raw/msmarco/queries.dev.small.tsv --qrels data/subset/1m/qrels.dev.tsv \
    --rate "$rate" --phase-seconds 5 --mode "$mode" --rerank "$rerank" --deadline-ms 300 --seed 1 >/dev/null
}
record_load() { printf '{"load_avg": "%s", "host": "%s"}\n' "$(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)" "$(uname -m) $(uname -s)" >"$out/$1.load.json"; }

scripts/run_cluster.sh start $cluster_args >/dev/null
pids
warm
# Slot order in the pid file: shard0a shard0b shard1a shard1b shard2a shard2b shard3a shard3b broker
record_load replica_hung
run --name replica-hung --fault-start "kill -STOP ${P[2]}" --fault-stop "kill -CONT ${P[2]}" --out "$out/replica_hung.json"
record_load replica_slow50
run --name replica-slow50-hedging --fault-start "echo latency_ms=50 > $chaos/shard2a" --fault-stop ": > $chaos/shard2a" --out "$out/replica_slow50_hedging.json"
record_load replica_starved
run --name replica-starved --fault-start "taskpolicy -b -p ${P[4]}" --fault-stop "taskpolicy -B -p ${P[4]}" --out "$out/replica_starved.json"
record_load slice_down
run --name slice-down --fault-start "kill -STOP ${P[6]} ${P[7]}" --fault-stop "kill -CONT ${P[6]} ${P[7]}" --out "$out/slice_down.json"

scripts/run_cluster.sh start --no-hedge $cluster_args >/dev/null
pids
warm
record_load replica_slow50_nohedge
run --name replica-slow50-no-hedging --fault-start "echo latency_ms=50 > $chaos/shard2a" --fault-stop ": > $chaos/shard2a" --out "$out/replica_slow50_nohedge.json"
scripts/run_cluster.sh start $cluster_args >/dev/null

.venv/bin/python scripts/chaos_table.py $(ls "$out"/*.json | grep -v '\.load\.json$') | tee "$out/TABLE.md"
