#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENPI_ROOT="${OPENPI_ROOT:-$REPO_ROOT/../openpi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/zac-models/wire-insert-jul16}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
PATH="$CUDA_HOME/bin:$PATH"

export CUDA_HOME PATH

echo "Starting the policy server only; this process has no robot connection."
echo "OpenPI root:       $OPENPI_ROOT"
echo "Checkpoint:        $CHECKPOINT_DIR"
echo "Policy endpoint:   $HOST:$PORT"
echo "Execution horizon: $EXECUTION_HORIZON"
echo "CUDA toolkit:      $CUDA_HOME"
echo "ptxas:             $(command -v ptxas)"

exec env \
  OPENPI_ROOT="$OPENPI_ROOT" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  EXECUTION_HORIZON="$EXECUTION_HORIZON" \
  HOST="$HOST" \
  PORT="$PORT" \
  "$REPO_ROOT/policy_serving/launch_openpi_wire.sh"
