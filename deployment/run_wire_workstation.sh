#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
CONFIG="${CONFIG:-$REPO_ROOT/config.yaml}"
TASK="${TASK:-Insert the USB-C plug into the USB-C port.}"
FPS="${FPS:-30}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Workstation Python is not executable: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Rig config does not exist: $CONFIG" >&2
  exit 1
fi

echo "Opening the workstation deployment UI."
echo "This script does not arm or start policy rollout."
echo "Policy movement starts only if the UI or a configured handle button sets policy_running=true."
echo "Config:            $CONFIG"
echo "Task:              $TASK"
echo "Camera/UI rate:    $FPS Hz"
echo "Execution horizon: $EXECUTION_HORIZON"
echo "Async prefetch:    disabled"

exec "$PYTHON" -m workstation.lerobot_recorder.deploy_main \
  --config "$CONFIG" \
  --task "$TASK" \
  --fps "$FPS" \
  --execution-horizon "$EXECUTION_HORIZON" \
  --no-async \
  "$@"
