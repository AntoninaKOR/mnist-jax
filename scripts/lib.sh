#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"

# Shared defaults for comparable runs across scripts.
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"

usage() {
  cat <<EOF
Usage:
  GPU=<id> $0 [mnist.py args...]
  $0 <gpu_id> [mnist.py args...]

Environment:
  GPU              GPU index (default: 0)
  BATCH_SIZE       Effective batch without accumulation (default: 32)
  ACCUM_STEPS      Micro-batches per effective batch when accumulating (default: 4)
  BENCHMARK_STEPS  Timed optimizer-equivalent steps in benchmark (default: 100)
  WARMUP_STEPS     Warmup steps before timing in benchmark (default: 10)
  PYTHON           Python interpreter (default: python)

Comparable setup:
  none:            batch = BATCH_SIZE
  multisteps/microbatch:
                   microbatch = BATCH_SIZE / ACCUM_STEPS
                   effective batch = BATCH_SIZE

For FPS comparison use:
  ./scripts/benchmark.sh

Examples:
  GPU=0 ./scripts/benchmark.sh
  BATCH_SIZE=32 ACCUM_STEPS=4 ./scripts/benchmark.sh
  GPU=1 ./scripts/train_multisteps.sh
EOF
}

parse_gpu_and_args() {
  local gpu="${GPU:-}"

  if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    gpu="$1"
    shift
  fi

  gpu="${gpu:-0}"
  export CUDA_VISIBLE_DEVICES="$gpu"

  if (( BATCH_SIZE % ACCUM_STEPS != 0 )); then
    echo "Error: BATCH_SIZE ($BATCH_SIZE) must be divisible by ACCUM_STEPS ($ACCUM_STEPS)" >&2
    exit 1
  fi

  echo "GPU: $gpu (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
  echo "BATCH_SIZE=$BATCH_SIZE, ACCUM_STEPS=$ACCUM_STEPS, microbatch=$((BATCH_SIZE / ACCUM_STEPS))" >&2
  EXTRA_ARGS=("$@")
}

run_mnist() {
  "$PYTHON" mnist.py --batch-size "$BATCH_SIZE" "$@"
}
