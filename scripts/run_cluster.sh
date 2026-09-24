#!/usr/bin/env bash
# Run the distributed system as native processes: 4 slices x 2 replicas of the
# C++ shard server plus the C# broker. Same topology as deploy/docker-compose.yml,
# without the image builds, so experiments can start in seconds.
#
#   scripts/run_cluster.sh start [--no-vector] [--no-hedge]
#   scripts/run_cluster.sh stop
#
# Each shard process reads its chaos control file from $HS_CLUSTER_DIR/chaos/shard<i><r>
# (write "latency_ms=50" into one to slow that replica down at runtime).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dir="${HS_CLUSTER_DIR:-$here/data/cluster}"
# HS_PARTITION=mod (default): shard i holds docs with id % 4 == i.
# HS_PARTITION=range: shard i holds the i-th contiguous ID range (kept for comparison;
# see docs/BUG_LOG.md 2026-09-24 for why range partitioning is wrong on MS MARCO).
partition="${HS_PARTITION:-mod}"
if [ "$partition" = mod ]; then
  lex_root="$here/data/indexes/lexical/1m-4shards-mod"
  vec_root="$here/data/indexes/vector/1m-4shards-mod"
  broker_partitioning=modulo
else
  lex_root="$here/data/indexes/lexical/1m-4shards"
  vec_root="$here/data/indexes/vector/1m-4shards"
  broker_partitioning=range
fi
broker_port="${HS_BROKER_PORT:-8080}"
base_port=50051

stop() {
  if [ -f "$dir/pids" ]; then
    while read -r pid; do kill "$pid" 2>/dev/null || true; done <"$dir/pids"
    rm -f "$dir/pids"
  fi
}

case "${1:-}" in
  stop) stop; echo "cluster stopped"; exit 0 ;;
  start) shift ;;
  *) echo "usage: $0 start [--no-vector] [--no-hedge] | stop" >&2; exit 2 ;;
esac

use_vector=1 hedge=true
for a in "$@"; do
  case "$a" in
    --no-vector) use_vector=0 ;;
    --no-hedge) hedge=false ;;
  esac
done

stop
mkdir -p "$dir/chaos" "$dir/logs"
: >"$dir/pids"
broker_args=()
for s in 0 1 2 3; do
  broker_args+=("--Broker:Topology:Slices:$s:Id=$s")
  for r in 0 1; do
    port=$((base_port + s * 2 + r))
    name="shard${s}$( [ $r = 0 ] && echo a || echo b )"
    : >"$dir/chaos/$name"
    args=(--port "$port" --shard-id "$s" --lexical "$lex_root/shard$s" --threads 8
          --chaos-file "$dir/chaos/$name" --otlp "http://localhost:4318")
    if [ "$use_vector" = 1 ] && [ -f "$vec_root/shard$s/disk.index" ]; then
      args+=(--vector-disk "$vec_root/shard$s")
    fi
    "$here/engine/build-srv/hs_shard" "${args[@]}" >"$dir/logs/$name.log" 2>&1 &
    echo $! >>"$dir/pids"
    broker_args+=("--Broker:Topology:Slices:$s:Replicas:$r=http://127.0.0.1:$port")
  done
done

reranker="models/reranker"
[ -f "$here/data/models/reranker/model.onnx" ] || reranker="models/reranker-public"
(cd "$here/broker" && DOTNET_ROOT="$HOME/.dotnet" exec "$HOME/.dotnet/dotnet" run --project HybridSearch.Broker \
  -c Release --no-launch-profile --urls "http://127.0.0.1:$broker_port" -- "${broker_args[@]}" \
  --Broker:Models:RerankerDir="$reranker" --Broker:Hedging:Enabled="$hedge" --Broker:Topology:Partitioning="$broker_partitioning" --Broker:Otel:Exporter=none) \
  >"$dir/logs/broker.log" 2>&1 &
echo $! >>"$dir/pids"

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$broker_port/api/search?q=warmup&k=1&deadlineMs=5000" >/dev/null 2>&1; then
    echo "cluster up: broker :$broker_port, 8 shard processes (partition=$partition, vector=$use_vector, hedging=$hedge), reranker=$reranker"
    exit 0
  fi
  sleep 1
done
echo "cluster did not come up; see $dir/logs" >&2
exit 1
