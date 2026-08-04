#!/usr/bin/env bash
#
# Reproducible setup of the WORKSTATION environment — conda + uv.
#
# conda owns the environment (so you can also `pip install` other policy repos into
# it), and uv does the fast installs for THIS repo. Installs i2rt (portal client),
# yam-policy (websocket client for the bridge), and the LeRobot recorder deps.
#
#   bash workstation/setup_workstation_env.sh
#
# Env overrides:  YAM_WS_ENV=yam_ws  WS_PY=3.11
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV="${YAM_WS_ENV:-yam_ws}"
PY="${WS_PY:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] conda not found — install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[setup] conda env '$ENV' (python $PY) ..."
conda activate "$ENV" 2>/dev/null || { conda create -y -n "$ENV" python="$PY"; conda activate "$ENV"; }

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

# ffmpeg: LeRobot v3.0 video encoding. build-essential/cmake: ruckig (pinned in
# pyproject.toml) is sdist-only and must be compiled from source — without a C++
# toolchain the uv step below dies with "Failed to build ruckig".
echo "[setup] system deps (ffmpeg, C++ toolchain for source builds) ..."
sudo apt-get install -y ffmpeg build-essential cmake ||
    echo "  (skip apt; install ffmpeg + build-essential + cmake manually if missing)"

# See workstation/build-constraints.txt — pins scikit-build-core for ruckig's build.
BUILD_CONSTRAINTS="$REPO/workstation/build-constraints.txt"

echo "[setup] uv-installing i2rt + yam-policy + recorder deps into conda env '$ENV' ..."
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e .  # uv targets the active conda env
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e policy_serving
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -r workstation/lerobot_recorder/requirements.txt

# Optional: abcdl data layer (recorder `format: abcdl` + per-frame RL signals). Pulled
# from GitHub; skip-on-failure so a network hiccup doesn't break the core setup.
echo "[setup] abcdl data layer (optional; for format: abcdl + rl_features) ..."
uv pip install --build-constraint "$BUILD_CONSTRAINTS" -e '.[abcdl]' ||
    echo "  (abcdl skipped; only needed for abcdl format / rl_features)"

# Fetch just the rules file (a full librealsense clone into /tmp is slow, and /tmp gets
# reaped — leaving a later `cp` staring at a path that no longer exists). Staged next to
# this script so a failed `sudo` can be retried by hand without re-downloading.
echo "[setup] RealSense udev rules (USB permissions) ..."
RULES_URL="https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules"
RULES_TMP="$REPO/workstation/99-realsense-libusb.rules"
if [ ! -e /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    if curl -fsSL -o "$RULES_TMP" "$RULES_URL" 2>/dev/null ||
        wget -q -O "$RULES_TMP" "$RULES_URL" 2>/dev/null; then
        # udev rules only take effect on re-enumeration — replug the cameras afterwards.
        if sudo cp "$RULES_TMP" /etc/udev/rules.d/ &&
            sudo udevadm control --reload-rules && sudo udevadm trigger; then
            echo "  (rules installed — REPLUG the cameras for them to take effect)"
        else
            echo "  (sudo failed; retry: sudo cp $RULES_TMP /etc/udev/rules.d/)"
        fi
    else
        echo "  (could not fetch udev rules; see workstation/lerobot_recorder/README.md)"
    fi
fi

python -c "import i2rt, yam_policy, lerobot, pyrealsense2; print('workstation env ready')" ||
    echo "  (verify deps; pyrealsense2/lerobot may need a moment)"

cat <<EOF

[setup] done — conda env: $ENV
  Activate:  conda activate $ENV
  Run:       workstation/yam-data record    (auto-activates '$ENV')
  Another policy repo in the SAME env:
             conda activate $ENV && pip install -e /path/to/policy_repo   # or: uv pip install -e ...
EOF
