# I2RT Python API

A Python client library for interacting with [I2RT](https://i2rt.com/) products — designed for learning-based robotics, teleoperation, and real-world deployment.

[![I2RT](https://github.com/user-attachments/assets/025ac3f0-7af1-4e6f-ab9f-7658c5978f92)](https://i2rt.com/)

## Features

- Plug-and-play Python interface for YAM arms and Flow Base
- Real-time robot control via CAN bus (DM series motors)
- MuJoCo gravity compensation, simulation, and URDF/MJCF models
- Gripper force control and auto-calibration
- Bimanual teleoperation and trajectory record & replay
- Policy-deployment ready — works with standard robot learning pipelines

## 🚀 Quick Start — what to run

The YAM setup spans **two machines** on the same LAN:

- 🤖 **robot** — the YAM arms on CAN; runs the **portal robot server**. uv-managed, so `robot/yam` uses `uv run` and there is **nothing to activate**.
- 💻 **workstation** — RealSense cameras + LeRobot; connects to the robot over **portal / plain TCP**. Lives in a conda env (`yam_ws`).

Two launchers pick the right env for you: **`robot/yam`** on the robot,
**`workstation/yam-data`** on the workstation.

One **`config.yaml`** at the repo root holds everything shared — robot host, camera serials,
control gains, recorder defaults. Every tool finds it on its own: no `--config`, no env var, from
any directory.

### 0 · One-time setup

```bash
# 🤖 robot — create the uv env (optional; robot/yam auto-resolves it) + fix CAN names
bash robot/setup_robot_env.sh      # optional: pre-create .venv + install i2rt
bash robot/setup_can_ids.sh        # plug the 4 CAN adapters one-by-one -> persistent names (once)
```

```bash
# 💻 workstation — conda env yam_ws + uv installs (you can pip-install other policy repos into it)
bash workstation/setup_workstation_env.sh # conda create yam_ws + uv pip install + RealSense udev
conda activate yam_ws
workstation/yam-data cams                 # list connected RealSense serials
workstation/yam-data tune                 # live sliders to match camera brightness
```

Then edit [`config.yaml`](config.yaml) at the repo root — at minimum the robot address and camera serials:

```yaml
robot:   { host: 192.168.0.42, port: 11331 }
cameras: { agentview: "<D455>", wrist_left: "<D405>", wrist_right: "<D405>" }
```

### A · Bimanual teleop only (no recording)

```bash
# 🤖 robot
robot/yam canup                       # bring up the 4 CAN interfaces (after each boot)
robot/yam teleop
# lift both gellos to engage; bring both home to stop & auto-return.
```

### B · Data collection (teleop + LeRobot recorder) — the main flow

```bash
# 🤖 robot — teleop server (serves state / action / engage-gate over portal)
robot/yam canup
robot/yam teleop
```

```bash
# 💻 workstation — recorder GUI (host + serials come from config.yaml)
workstation/yam-data record
```

The recorder opens on a **Setup page**:

1. Confirm `repo_id` / `root` / `task` and the **source** (teleop / dagger / eval).
   The dataset lands in `<root>/<name>`, where *name* is the last segment of `repo_id` —
   `~/lerobot_data` + `hello/pick_and_place` → `~/lerobot_data/pick_and_place`.
   The status line shows which cameras were detected and whether that dataset already exists.
2. To add to an existing dataset, tick **Continue collecting** (resume/append). Otherwise **START** creates it fresh — and if the folder exists it asks twice before overwriting.
3. **START** connects the robot, opens cameras + dataset, and (with `auto_arm: true`) arms collection immediately.

**Past demonstration overlay** (after START) plays a recorded episode's three camera views
under the live ones, so you can match the scene before starting a take. Available in both
`yam-data record` and `yam-data deploy`.

| Control | What it does |
|---|---|
| **Overlay dataset** | any dataset folder under `root`, including one you are not recording into |
| **Live camera opacity** | blend live against the reference; `100%` hides the reference |
| **Resume reference** | a selected episode starts paused on frame 1 and advances only after this |

- The first saved demonstration is selected automatically.
- **Preview only** — saved frames and policy inputs are always the unmodified camera images.
- Overlay-source datasets other than the one being recorded are **read-only**.
- Recording into an existing dataset still needs **Continue collecting** ticked *before* START;
  starting fresh deletes that dataset's episodes after the two overwrite confirmations.

Then teleoperate — **lift both gellos** to start recording, **bring both home** to end the episode:

`review_before_save` decides what happens at the end of each episode: `true` holds it for
**Keep** / **Delete**, `false` (what `config.yaml` ships) **auto-saves**.

**Leader handle buttons**

| Button | Action |
|---|---|
| left upper | toggle **fine-grained control** (2.5:1 with the checked-in `fine_grained_scale: 0.4`) |
| left lower | **success** |
| right lower | **fail** |
| right upper | **discard** — force-home, no save |

Leaving fine-grained mode pauses recording and teleoperation while the follower holds and the
leader realigns.

Close the window when done: that calls `finalize()`, which completes the dataset. Cameras run on
their own capture thread, so the live view and saving never stall on a slow frame.

> **Dry run (no hardware):** `workstation/yam-data record --mock` exercises the whole pipeline with synthetic teleop + fake frames — no robot, cameras, or lerobot needed.

### C · Deployment / DAgger (policy + human takeover)

```bash
robot/yam deploy                                         # 🤖 robot
python -m yam_policy.serve                               # 🧠 policy host (:8000)
workstation/yam-data deploy --repo-id user/yam_pick --prompt "pick up the cube"  # 💻 workstation UI
# Left upper starts/stops rollout, or toggles fine control during intervention.
# Other handle buttons toggle intervention and keep/discard + home.
# Past demonstration overlay is inherited from the recorder UI and remains preview-only.
```

**openpi and LeRobot checkpoints both deploy through the same client.** openpi's is the wire
protocol; a LeRobot checkpoint runs through an adapter on our side, with no conversion step:

```bash
# 🧠 a LeRobot checkpoint (local dir or a Hub repo id)
python -m yam_policy.serve \
    --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
    --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
    --config device=cuda
```

- The client configures itself from the handshake: camera names, image size, chunk length.
- The deploy UI **names what answered** (`LeRobot · act`, `ACRFT · pi05_yam_lego_taxi`) — they
  share a wire but not an observation format.
- **Before training on this recorder's data, drop `observation.leader`**, or the policy is handed
  the answer as an input. Details: [`policy_serving/README.md`](policy_serving/README.md).

### D · Replay a dataset onto the robot

Replay is deployment with the actions read from a dataset, so it is **not a separate tool** — it
is the `dataset` **mode** of `yam-data deploy` (no `wrapper` server, no policy server, no separate
GUI):

```bash
robot/yam canup && robot/yam deploy   # 🤖 robot — the deploy server, as usual
workstation/yam-data deploy           # 💻 workstation — set mode = dataset, pick an episode
```

- Set `mode = dataset`, Start, then pick the episode on the run page's past-demonstration panel —
  the runner builds an **in-process `DatasetPolicy`** (reads the actions from the parquet) so
  takeover, e-stop, the follower smoother and the link-loss watchdog all apply.
- The rollout button is **Start Replay → Pause / Resume**: pause is a send-gate (the robot holds,
  nothing on the robot side toggles, so the gripper is never snapped shut on resume).
- The overlay shows the episode's **first frame** as a scene reference (the arm is the moving
  reference). Watch-only; nothing is recorded.
- `speed` / `loop` live on `DatasetPolicy`; see [`policy_serving/README.md`](policy_serving/README.md).

> Full hardware bring-up checklist: [`docs/hardware-checklist.md`](docs/hardware-checklist.md).

### E · View & edit a recorded dataset (no robot)

Use **[`jellyho/hf-utils`](https://github.com/jellyho/hf-utils)** — a local web app that
replaced the PyQt editor that used to live here. It works on *any* LeRobot v3.0 dataset, not
only YAM recordings, so it is a separate tool rather than part of this repo. Nothing here
imports it and nothing needs installing.

```bash
git clone https://github.com/jellyho/hf-utils && cd hf-utils && ./run.sh
# open http://127.0.0.1:8000 -> LeRobot tab -> point it at a dataset folder
```

Per episode: **scrub / play** every camera in sync, plot `action` vs `observation.state`,
**set the task**, **delete** episodes (with re-indexing), split, merge, render to MP4/GIF,
and annotate subtasks. It reads this recorder's `outcomes.jsonl` when present, so
success / fail / discard still show up per episode and survive a delete. Everything
destructive backs up first.

Two things stayed here because they encode YAM specifics a general tool should not have:
**homing annotation** (below) and `yam-data check-videos`, which can re-encode an mp4 that
the recorder's encoder truncated — hf-utils detects that but only repairs the metadata side.

**Homing annotation** — every episode ends with the arms returning home and the
gripper closing; that tail isn't useful for training. New recordings are tagged live
(`observation.control_mode = homing`). For datasets collected earlier, auto-annotate
it from the gripper:

```bash
workstation/yam-data mark-homing --dataset user/yam_pick --dry-run   # preview
workstation/yam-data mark-homing --dataset user/yam_pick             # apply (or --all)
```

It's non-destructive (only relabels the `control_mode` column — no frames removed, no
re-encode). At train time, drop the tail by filtering `control_mode == homing`.

> If a GPU/streaming video encode dropped an episode's trailing frame (a camera's
> video ends up 1 frame shorter than the recorded length), LeRobot would otherwise
> refuse to delete a shared-file episode. The editor detects this and **auto-repairs
> the metadata** before retrying, and the recorder now runs the same check at
> `finalize()` so freshly collected datasets stay consistent (with a warning if it
> recurs — consider `recorder.streaming_encoding: false` or `vcodec: h264`).

---

# 📚 Library & API Reference

Everything below documents using **i2rt as a Python library** — driving motors over CAN, grippers, kinematics, the Flow Base, and the serving stack.

> **Running the YAM teleop / recording rig?** The [Quick Start](#-quick-start--what-to-run) is
> all you need — its setup scripts already create the envs and install everything
> (`robot/setup_robot_env.sh`, `workstation/setup_workstation_env.sh`).
>
> The manual install below is only for using i2rt standalone as a library.

## Installation (manual / standalone library)

```bash
git clone https://github.com/i2rt-robotics/i2rt.git && cd i2rt
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate
```

```bash
sudo apt update
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

## CAN Bus Setup

```bash
# Check detected CAN devices
ls -l /sys/class/net/can*

# Bring up interface at 1 Mbit/s
sudo ip link set can0 up type can bitrate 1000000

# Auto-enable on boot
sudo sh devices/install_devices.sh

# Reset unresponsive adapter
bash robot/reset_all_can.sh
```

## YAM Arm

### Zero-gravity mode

```bash
python i2rt/robots/motor_chain_robot.py --channel can0 --gripper linear_4310
```

### Python API

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
import numpy as np

robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310)

# Read joint positions (radians)
q = robot.get_joint_pos()   # shape: (6,)

# Command a target configuration
robot.command_joint_pos(np.zeros(6))
```

### Leader-follower teleoperation

```bash
# Follower arm
python examples/minimum_gello/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can0 --bilateral-kp 0.2

# Leader arm (teaching handle)
python examples/minimum_gello/minimum_gello.py --gripper yam_teaching_handle --mode leader --can-channel can1 --bilateral-kp 0.2
```

- **Top button (press once):** enable synchronisation — follower tracks leader
- **Top button (press again):** disable synchronisation
- `--bilateral-kp` controls resistance felt on the leader (0.1–0.2 recommended)

To inspect leader arm output:

```bash
python robot/run_yam_leader.py --channel $CAN_CHANNEL
```

### MuJoCo visualiser

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

## Gripper Types

| Gripper | Motor | Notes |
|---------|-------|-------|
| `crank_4310` | DM4310 | Zero-linkage crank — minimises gripper width |
| `linear_3507` | DM3507 | Lightweight linear; start closed or run calibration |
| `linear_4310` | DM4310 | Standard linear; slightly more force than 3507 |
| `yam_teaching_handle` | — | Leader arm handle with trigger + 2 buttons. |

The linear grippers require calibration because their motor travels more than 2π radians over the full stroke — either start with the gripper fully closed, or run the calibration routine.

## Flow Base

```bash
# Joystick demo
python i2rt/flow_base/flow_base_controller.py
```

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient

client = FlowBaseClient(host="172.6.2.20")
client.set_target_velocity([0.1, 0.0, 0.0], frame="local")
```

## Examples

| Example | Location |
|---------|----------|
| Bimanual lead-follower | `examples/bimanual_lead_follower/` |
| Record & replay trajectory | `examples/record_replay_trajectory/` |
| Single motor PD control | `examples/single_motor_position_pd_control/` |
| MuJoCo control interface | `examples/control_with_mujoco/` |

## Networking & deployment

The YAM rig is exposed to a workstation over **`portal`** (plain TCP); policy
inference is a separate **websocket** link (openpi-compatible). Bimanual by default
(2 leaders + 2 followers).

```bash
source .venv/bin/activate                       # robot env (uv; robot/setup_robot_env.sh)
python -m i2rt.serving.run_robot_server teleop  --sim   # auto home/engage teleop
python -m i2rt.serving.run_robot_server deploy  --sim   # HG-DAgger: policy + button takeover
python -m i2rt.serving.run_robot_server wrapper --sim   # followers track an external command

# …or the shortcut launcher (activates the env for you):
robot/yam teleop --sim                        # also: dagger / wrapper / can / canup
```

Targets are rate-limited and gravity compensation is always on, so policy↔human
takeovers ramp smoothly.

### Data collection & deployment stack

| Subsystem | Path | What it is |
|-----------|------|------------|
| Robot serving (portal) | [`i2rt/serving/`](i2rt/serving/README.md) | teleop / DAgger / wrapper servers + `RobotClient`; snapshot contract; safety (e-stop, joint/effort limits, link-loss watchdog), EEF FK + safe resolved-rate OSC |
| Policy serving (websocket) | [`policy_serving/`](policy_serving/README.md) | openpi-compatible `WebsocketPolicyServer`/`Client` + `serve.py`; deploys **openpi and LeRobot checkpoints** through the same client |
| Workstation tools | [`workstation/lerobot_recorder/`](workstation/lerobot_recorder/README.md) | LeRobot recorder (teleop/dagger/eval), replay+overlay, policy bridge; modern themed GUI with status banner, health, live stats, audio cues, success/fail/discard labeling |

Quick CLIs (workstation): `workstation/yam-data {record\|replay\|bridge\|cams\|tune\|doctor}`.

**One config for everything**: edit the tracked [`config.yaml`](config.yaml) at the
**repo root** — robot host/port, control gains/limits, camera serials, recorder
defaults, tasks, and the policy endpoint. Every tool auto-discovers `<repo>/config.yaml`
(no env var, no matter the directory); no `--config` needed. Precedence:
**CLI flag > `config.yaml` > default**.

**Envs**: the **robot** is uv-managed — `robot/yam …` uses `uv run`, nothing to
activate. The **workstation** is a **conda** env (`workstation/setup_workstation_env.sh`)
with this repo installed via uv, so you can `pip install` other policy repos into the
same env. The **policy server** env is unconstrained (its own conda/uv).

**Safety & ops highlights**: network E-STOP, per-joint position + optional effort
(collision) limits, command-staleness watchdog (link loss → hold), async dataset
writer (collect while saving), camera-disconnect auto-reconnect, disk-space guard,
and a per-episode `outcomes.jsonl` (`yam-data doctor` summarizes success rates).

**Verify on hardware**: follow [`docs/hardware-checklist.md`](docs/hardware-checklist.md)
— an ordered, runnable confirmation list for every feature.

## Advanced: Motor Configuration

### Safety timeout

The factory default is a **400 ms timeout** — motors enter damping mode if no command is received within 400 ms.

```bash
# Disable timeout (advanced users only — run twice)
python i2rt/motor_config_tool/set_timeout.py --channel can0
python i2rt/motor_config_tool/set_timeout.py --channel can0

# Re-enable timeout
python i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout
```

> ⚠️ Without the timeout, a failed gravity-compensation loop can produce uncontrolled torque. If you disable it, always initialise with a PD target:
> ```python
> robot = get_yam_robot(channel="can0", zero_gravity_mode=False)
> ```

### Zero motor offsets

```bash
python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1
```

Run for each motor ID (1–6 for a standard YAM).

## Contributing

Pull requests welcome. Open an issue to request examples or report bugs.

## License

MIT License — see [LICENSE](LICENSE).

## Support

- Email: support@i2rt.com
- Sales: sales@i2rt.com

## Acknowledgments

- [TidyBot++](https://github.com/jimmyyhwu/tidybot2) — Flow Base hardware and control inspired by TidyBot++
- [GELLO](https://github.com/wuphilipp/gello_software) — Teleoperation design inspired by GELLO
