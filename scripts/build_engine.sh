#!/usr/bin/env bash
# Build the C++ engine. Usage: scripts/build_engine.sh [build-dir] [extra cmake args...]
# Homebrew's gRPC is put first on the prefix path so an Anaconda protoc on PATH
# cannot mix protobuf versions into the build.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$here/engine/build}"; shift || true
prefix="/opt/homebrew"
[ -d "$prefix" ] || prefix="/usr/local"
cmake -S "$here/engine" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix" -DProtobuf_PROTOC_EXECUTABLE="$prefix/bin/protoc" -Wno-dev "$@"
cmake --build "$build" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
