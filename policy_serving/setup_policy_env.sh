#!/usr/bin/env bash
#
# Reproducible setup of the POLICY SERVER environment (unrestricted).
#
# Creates a uv venv (default: policy_serving/.venv) and installs yam-policy. This
# env is fully independent — add your model's deps here
# (torch / JAX / CUDA, lerobot, or openpi from its repo).
#
#   bash policy_serving/setup_policy_env.sh
#
# Env overrides:  POLICY_PY=3.12
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${POLICY_PY:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] creating venv (python $PY) ..."
uv venv --python "$PY" .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e .

if [ -f ../lerobot/pyproject.toml ]; then
    echo "[setup] installing the repository-pinned LeRobot + supported policy extras ..."
    # Some hosts export SETUPTOOLS_USE_DISTUTILS=stdlib. Python 3.12 removed
    # stdlib distutils, but SmolVLA's num2words dependency still pulls the old
    # docopt sdist. Force setuptools' maintained compatibility copy while the
    # optional dependencies are built.
    SETUPTOOLS_USE_DISTUTILS=local \
        uv pip install -e '../lerobot[multi_task_dit,pi,smolvla,diffusion]'
else
    echo "[setup] local LeRobot submodule not found; install LeRobot before running yam-lerobot-serve" >&2
fi

python -c "import yam_policy, lerobot; print('LeRobot policy env ready')"

cat <<EOF

[setup] done.  Activate: source $HERE/.venv/bin/activate
  Serve LeRobot: yam-lerobot-serve --checkpoint /path/to/pretrained_model --device cuda
  Serve MTD RTC: yam-lerobot-serve --checkpoint /path/to/pretrained_model --device cuda \\
                   --rtc --num-inference-steps 20
EOF
