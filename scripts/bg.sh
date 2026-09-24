#!/usr/bin/env bash
# Run a command as a polite background job on a shared laptop.
#
# macOS background QoS (taskpolicy -b) keeps the process on the efficiency
# cores and behind every normal-priority process, nice 19 on top, and every
# math library is held to one thread. It refuses to start when free disk is
# under HS_MIN_FREE_GB (default 12), which protects headroom other projects on
# this machine have reserved.
#
#   scripts/bg.sh .venv/bin/python -m hybridsearch.dense.encode all
set -euo pipefail
min_free_gb="${HS_MIN_FREE_GB:-12}"
free_gb=$(df -g "$HOME" | awk 'NR==2 {print $4}')
if [ "$free_gb" -lt "$min_free_gb" ]; then
  echo "bg.sh: refusing, ${free_gb} GB free < ${min_free_gb} GB reserve" >&2
  exit 3
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  DOTNET_PROCESSOR_COUNT="${DOTNET_PROCESSOR_COUNT:-1}"
if command -v taskpolicy >/dev/null; then
  exec taskpolicy -b nice -n 19 "$@"
else
  exec nice -n 19 "$@"
fi
