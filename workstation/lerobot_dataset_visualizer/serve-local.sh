#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3000}"
export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$HOME/lerobot_data}"

if [[ ! -d "$LEROBOT_DATA_ROOT" ]]; then
  echo "Dataset root does not exist: $LEROBOT_DATA_ROOT" >&2
  exit 1
fi

echo "i2rt LeRobot visualizer"
echo "  datasets: $LEROBOT_DATA_ROOT"
echo "  listening: http://$HOST:$PORT"
echo
echo "From your laptop, create an SSH tunnel:"
echo "  ssh -L $PORT:127.0.0.1:$PORT <user>@<this-host>"
echo "Then open http://127.0.0.1:$PORT"
echo

exec npm run dev -- --hostname "$HOST" --port "$PORT"
