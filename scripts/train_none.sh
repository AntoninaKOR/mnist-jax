#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Runs training without gradient accumulation (effective batch = BATCH_SIZE)."
  echo "For throughput comparison, use ./scripts/benchmark.sh instead."
  exit 0
fi

parse_gpu_and_args "$@"

run_mnist --accum-method none --epochs "$EPOCHS" --warmup-epochs "$WARMUP_EPOCHS" "${EXTRA_ARGS[@]}"
