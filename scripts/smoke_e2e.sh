#!/usr/bin/env bash
# End-to-end smoke test on real data: one C++ shard over a lexical index, the C#
# broker with the ONNX encoder and reranker, and a real query through /api/search
# and /api/search/stream. Fails loudly if any layer is missing or misbehaves.
#
#   scripts/smoke_e2e.sh [lexical-index-dir] [vector-index-dir]
#
# Defaults to the full 8.8M-passage lexical index with no vector index, in which
# case the response must say so (degradation lexical_only:vector_index_unavailable).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
lex="${1:-$here/data/indexes/lexical/full}"
vec="${2:-}"
shard_port=50071 broker_port=8090
log="$(mktemp -d)"
cleanup() { kill "${shard_pid:-}" "${broker_pid:-}" 2>/dev/null || true; }
trap cleanup EXIT

shard_args=(--port "$shard_port" --shard-id 0 --lexical "$lex" --threads 4)
[ -n "$vec" ] && shard_args+=(--vector-disk "$vec")
"$here/engine/build-srv/hs_shard" "${shard_args[@]}" >"$log/shard.log" 2>&1 &
shard_pid=$!

reranker="models/reranker"
[ -f "$here/data/models/reranker/model.onnx" ] || reranker="models/reranker-public"
(cd "$here/broker" && DOTNET_ROOT="$HOME/.dotnet" "$HOME/.dotnet/dotnet" run --project HybridSearch.Broker -c Release \
  --urls "http://127.0.0.1:$broker_port" -- \
  --Broker:Topology:Slices:0:Replicas:0="http://127.0.0.1:$shard_port" \
  --Broker:Models:RerankerDir="$reranker" --Broker:Otel:Exporter=none) >"$log/broker.log" 2>&1 &
broker_pid=$!

for _ in $(seq 1 90); do curl -sf "http://127.0.0.1:$broker_port/healthz" >/dev/null && break; sleep 2; done

# /healthz is liveness only; the broker marks a shard healthy after its first health probe.
resp=""
for _ in $(seq 1 30); do
  resp="$(curl -sf "http://127.0.0.1:$broker_port/api/search?q=what+is+the+capital+of+peru&k=5&explain=true&deadlineMs=10000" || true)"
  [ -n "$resp" ] && break
  sleep 2
done
[ -n "$resp" ] || { echo "search never succeeded; broker log:"; tail -20 "$log/broker.log"; exit 1; }
python3 - "$resp" "${vec:+vector}" <<'EOF'
import json, sys
d = json.loads(sys.argv[1]); has_vec = sys.argv[2] == "vector"
r = d["results"]
steps = d["degradation"]["steps"]
print("degradation:", d["degradation"]["level"], steps)
print("timings (ms):", d["timings"])
assert len(r) == 5, f"expected 5 results, got {len(r)}"
assert all(x["highlights"] for x in r), "every result should have highlights"
assert all(x["terms"] for x in r), "explain=true must return per-term BM25 contributions"
assert r[0]["scores"]["rerank"] is not None, "reranker did not run"
if not has_vec:
    assert "lexical_only:vector_index_unavailable" in steps, f"missing degradation step: {steps}"
else:
    assert any(x["scores"]["dense"] is not None for x in r), "dense scores missing"
print("top-1:", r[0]["docId"], "|", r[0]["text"][:80])
EOF
events="$(curl -sfN --max-time 30 "http://127.0.0.1:$broker_port/api/search/stream?q=inca+empire&k=3" | grep '^event:' | tr '\n' ' ')"
[ "$events" = "event: lexical event: final " ] || { echo "unexpected SSE events: $events"; exit 1; }
echo "SSE: $events"
echo "smoke_e2e: OK"
