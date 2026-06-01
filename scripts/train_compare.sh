#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Runs full training for none, multisteps, and microbatch."
  echo "Prints accuracy and FPS for each method."
  exit 0
fi

parse_gpu_and_args "$@"

run_mnist \
  --compare-all \
  --accum-steps "$ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  "${EXTRA_ARGS[@]}"
