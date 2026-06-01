#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Runs training with microbatching.microbatch (effective batch = 32 * ACCUM_STEPS)."
  exit 0
fi

parse_gpu_and_args "$@"

run_mnist --accum-method microbatch --accum-steps "$ACCUM_STEPS" --epochs "$EPOCHS" --warmup-epochs "$WARMUP_EPOCHS" "${EXTRA_ARGS[@]}"
