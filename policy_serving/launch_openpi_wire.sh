#!/usr/bin/env bash
set -euo pipefail

I2RT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$I2RT_ROOT/../openpi" && pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/zac-models/wire-insert-jul16}"
CONFIG_NAME="${CONFIG_NAME:-pi05_wire_insertion_success}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
CONTROL_HZ=30
MODEL_HORIZON=50
PROMPT="${PROMPT:-Insert the USB-C plug into the USB-C port.}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PYTHON="${PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
MANIFEST="${MANIFEST:-$I2RT_ROOT/deployment_logs/wire_policy_$RUN_ID.json}"

"$PYTHON" "$OPENPI_ROOT/scripts/validate_yam_checkpoint.py" \
  --config-name "$CONFIG_NAME" \
  --checkpoint-dir "$CHECKPOINT_DIR"
"$PYTHON" -m yam_policy.deployment_manifest \
  --openpi-root "$OPENPI_ROOT" \
  --i2rt-root "$I2RT_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --config-name "$CONFIG_NAME" \
  --prompt "$PROMPT" \
  --model-horizon "$MODEL_HORIZON" \
  --execution-horizon "$EXECUTION_HORIZON" \
  --control-hz "$CONTROL_HZ" \
  --output "$MANIFEST"

echo "contract:          yam_bimanual_v1"
echo "checkpoint:        $CHECKPOINT_DIR"
echo "prompt:            $PROMPT"
echo "model horizon:     $MODEL_HORIZON"
echo "execution horizon: $EXECUTION_HORIZON"
echo "control rate:      $CONTROL_HZ Hz"
echo "policy endpoint:   $HOST:$PORT"
echo "compute hostname:  $(hostname)"
echo "manifest:          $MANIFEST"

exec "$PYTHON" -m yam_policy.serve \
  --policy yam_policy.policies.openpi_policy:OpenPiPolicy \
  --config "config_name=$CONFIG_NAME" \
  --config "checkpoint_dir=$CHECKPOINT_DIR" \
  --config "execution_horizon=$EXECUTION_HORIZON" \
  --config "default_prompt=$PROMPT" \
  --host "$HOST" \
  --port "$PORT"
