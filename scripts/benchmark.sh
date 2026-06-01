#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Benchmarks none, multisteps, and microbatch with matched workload."
  echo "Uses shared ACCUM_STEPS, BENCHMARK_STEPS, and WARMUP_STEPS from lib.sh."
  exit 0
fi

parse_gpu_and_args "$@"

run_mnist \
  --benchmark \
  --accum-steps "$ACCUM_STEPS" \
  --benchmark-steps "$BENCHMARK_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  "${EXTRA_ARGS[@]}"
