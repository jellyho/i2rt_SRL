# YAM ↔ LeRobot Recorder & Replay

Record and replay [LeRobot](https://github.com/huggingface/lerobot) **v3.0**
datasets for the bimanual YAM teleop rig. Runs on the **workstation** (a
different machine / env than the robot): it connects to the YAM **robot server
over portal** (plain TCP), reads three RealSense cameras **locally**,
and **auto-starts/stops each episode from the teleop gate** — no manual record
button per episode.

Two tools (both have a PyQt GUI):

| Tool | Module | What it does |
|------|--------|--------------|
| **Recorder** | `workstation.lerobot_recorder` | teleop-gated LeRobot capture + review/delete |
| **Replay**   | `workstation.lerobot_recorder.replay_main` | play a dataset back onto the robot |

## Recording video backend

The workstation recorder supports two encoding implementations through
`recorder.encoding_backend` in the repository `config.yaml`:

- `torchcodec` (default): prefers TorchCodec's `VideoEncoder`. At writer startup
  and again before each camera encode,
  it falls back to PyAV with CPU H.264 if TorchCodec is unavailable, GPU memory
  cannot be measured, or primary-GPU usage is at least
  `recorder.torchcodec_max_used_vram_gb` (5 GiB by default).
- `pyav`: always uses LeRobot's existing PyAV encoder.

TorchCodec 0.10 accepts a complete episode tensor rather than incremental frames,
so `streaming_encoding` is disabled while TorchCodec is effective. Cameras are
encoded sequentially and tensors remain in host memory to keep peak VRAM small.
PyAV continues to honor `streaming_encoding: true`.

Benchmark both implementations on the workstation with:

```bash
conda run -n yam_ws python -m workstation.lerobot_recorder.benchmark_video_encoders \
  --frames 1800 --output-json /tmp/encoder-benchmark.json
```

## How recording is triggered (the key idea)

`i2rt.serving.run_robot_server teleop` (on the robot machine) reports a
`teleop_state` (`HOMING` / `IDLE` / `ENGAGED`) in its snapshot. The recorder polls
it over portal:

```
 press "Start collection"      ──▶  gate armed
 both gellos lifted → ENGAGED  ──▶  episode STARTS recording   ◀── auto
 …teleoperate…       (ENGAGED)      every frame recorded
 gellos home → HOMING          ──▶  still recording (return included)
 homing done → IDLE            ──▶  episode ends → REVIEW       ◀── auto
   review playback → Keep / Delete
```

One episode = **ENGAGED → HOMING → IDLE**. The action stored is the robot's
**`applied`** (the rate-limited command actually sent) → reproducible.

`config.yaml` tunes the flow: **`auto_arm`** arms on START (skip "Start collection"),
**`review_before_save: false`** auto-saves each episode (skip Keep/Delete), and a
leader **button** can end+label in one press (see *Labeling*). The engage/release
thresholds (`control.engage_thr` / `release_thr` / `dwell` / `gate_joints`) live in
`config.yaml` too and are applied by the robot server.

That gate is the `teleop` source. `recorder.source` picks which one is used:

| source | what starts/stops an episode | action stored | dataset |
|---|---|---|---|
| `teleop` | the engage gate above (ENGAGED → IDLE) | `applied` | yes |
| `dagger` | one complete policy rollout | `executed` | yes |
| `eval` | Start collection → Stop collection, continuously | `executed` | yes |
| `deploy` | nothing is recorded | — | **no** |

`yam-data record` uses `teleop`. The deploy UI picks between the other three from its
**mode** + **record** choices — you never set the source by hand there. In `deploy` the
cameras, robot link, live view, takeover and e-stop all behave the same; no writer is
opened and no frames are buffered. See *C. Deployment*.

## Dataset schema (LeRobot v3.0)

| Key | Shape | Source |
|-----|-------|--------|
| `observation.images.wrist_left`  | (H,W,3) uint8 | D405 left wrist |
| `observation.images.wrist_right` | (H,W,3) uint8 | D405 right wrist |
| `observation.images.agentview`   | (H,W,3) uint8 | D455 scene |
| `observation.state` | (42,) float32 | both arms × [pos(7), vel(7), eff(7)] |
| `action`            | (14,) float32 | both arms × `applied`(7) |
| task                | string | the language instruction |

**Rate: 30 fps**, set by `config.yaml` and matched to the camera streams. (`RecorderConfig`
defaults to 60; the config pins both down because three 640x480 streams at 60 overrun one
USB 2.0 bus.)

Written through the official v3.0 API — `create`, `add_frame` with a `task` key,
`save_episode`, `clear_episode_buffer`, **`finalize`**. The version-sensitive calls are all in
`dataset_writer.py`.

### Training on one of these: pass `--tolerance_s=1e-3`

```bash
lerobot-train --dataset.root=~/lerobot_data/<name> ... --tolerance_s=1e-3
```

Without it, training dies partway through with `FrameTimestampError: One or several query
timestamps unexpectedly violate the tolerance (tensor([0.0001]) > tolerance_s=0.0001)`.

Nothing is wrong with the dataset, and re-recording will not help. It is arithmetic:

- v3.0 timestamps are `float32`, and v3.0 packs many episodes into one mp4 — so the
  file-relative query time climbs into the hundreds of seconds.
- Past **t = 1024 s**, adjacent `float32` values are 1.22e-4 s apart, already wider than the
  1e-4 default tolerance. No frame can satisfy it, however well the data was collected.

Any sufficiently long v3.0 dataset hits this.

`1e-3` is still 3% of a frame at 30 fps, so it rejects a genuinely wrong frame just as well.

---

# One-time setup

### [robot machine] — YAM robot server (uv; nothing to activate)

```bash
bash robot/setup_robot_env.sh          # optional: pre-create .venv + install i2rt
bash robot/setup_can_ids.sh            # persistent CAN names (once)
```

You don't need to activate anything — `robot/yam` uses **`uv run`**, which resolves
(and on first run creates) the env automatically. Already inside a conda/venv? set
`YAM_NO_UV=1` and it uses plain `python`.

### [workstation] — conda env + uv

conda owns the env (so you can also `pip install` other policy repos into it); uv
does the fast installs for this repo:

```bash
bash workstation/setup_workstation_env.sh     # conda create yam_ws + uv pip install + udev rules
conda activate yam_ws
```

<details><summary>What it does (manual equivalent)</summary>

```bash
conda create -y -n yam_ws python=3.11      # any Python >= 3.10
conda activate yam_ws
sudo apt install -y ffmpeg                  # LeRobot v3.0 video encoding
uv pip install -e .                         # i2rt (portal RobotClient) — uv targets the conda env
uv pip install -e policy_serving            # yam-policy (websocket client for the bridge)
uv pip install -r workstation/lerobot_recorder/requirements.txt
# another policy repo in the SAME env:  pip install -e /path/to/policy_repo   (or uv pip install)
```
</details>

The `yam-data` launcher activates the **conda** env for you (default `yam_ws`,
override with `YAM_WS_ENV=...`). The robot host/port come from `config.yaml` (or
`--robot-host`/`--robot-port`, default
`127.0.0.1:11331`) — both machines just need to be on the same network.

### [workstation] — RealSense cameras

The pip `pyrealsense2` wheel ships the SDK bindings but **not** the udev rules, so
without them the cameras open only as root / fail with permission errors (the setup
script installs them; manual version):

```bash
git clone --depth 1 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense
sudo cp /tmp/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# (optional) realsense-viewer for live preview: sudo apt install -y librealsense2-utils
```

Plug the cameras into **USB 3** ports (three 640x480 streams do not fit on one USB 2.0 bus —
one camera starves with "stopped delivering frames"), then map each
**serial → role**. Each camera's serial is a permanent firmware ID — no udev
renaming needed (unlike CAN), you just record which serial is which physical view:

```bash
workstation/yam-data cams        # lists connected RealSense name + serial
```

To find which serial is which view, plug them in **one at a time** and re-run
`cams` (or use `realsense-viewer`). Then pass them in the fixed order
`wrist_left,wrist_right,agentview` at launch (`--serials A,B,C`), or hard-code them
in [`config.py`](config.py) `default_cameras()`.

## Exposure / brightness matching

Auto-exposure makes brightness drift mid-episode and differ between cameras. Lock it
per camera in `config.yaml` with an `options:` map (applied on open **and** on every
reconnect):

```yaml
cameras:
  agentview:
    serial: "246322303794"
    options:
      enable_auto_exposure: 0
      exposure: 300            # D455 RGB: units of 100us -> 30 ms
      gain: 64                 # D455 RGB gain range 0..128
```

To pick the values interactively — live feeds, a spinbox + slider per control, and a
mean-luma readout with the delta between cameras:

```bash
workstation/yam-data tune        # adjust, then "Write to config.yaml" (keeps a .bak)
```

Type exact numbers into the spinbox; the slider is for coarse sweeps (its full travel
is only ~190 px, so one pixel is tens or hundreds of exposure units). **link
same-model cameras** (on by default) mirrors every change onto other cameras of the
same model, so the two D405 wrists stay identical.

Cameras of the same model should share one block. A YAML anchor keeps them literally
identical, so editing one edits both:

```yaml
  wrist_left:
    serial: "352122271652"
    options: &d405_options
      enable_auto_exposure: 0
      exposure: 16000
      gain: 16
  wrist_right:
    serial: "409122274199"
    options: *d405_options     # same values, by reference
```

Caveat: "Write to config.yaml" writes literal values, so it **expands the anchor**.
With linking on the two blocks stay identical in value, but the `&`/`*` reference is
replaced by two copies — re-add the anchor by hand if you want to keep editing one
place. (A `.bak` of the previous file is always written alongside.)

**Exposure values are NOT portable between D405 and D455.** A D455 has a real `RGB
Camera` sensor whose exposure counts in **100 us** steps (range `1..10000`). A D405
has *no* RGB sensor — its color stream comes off the `Stereo Module`, which counts in
**1 us** steps (range `1..165000`). The same number therefore means a 100x different
exposure time on the two models, so tune each camera against the luma readout rather
than copying numbers. (`gain` ranges differ too: 0..128 on D455 RGB, 16..248 on
stereo.) The tuner shows which sensor it is driving under each feed.

Watch the `clip NN%` badge: it is the share of pixels at/near pure white. Clipped
highlights are unrecoverable detail, so match cameras **down** to the darkest
well-exposed one rather than up into saturation.

---

# Runbook — scenarios (run top to bottom)

Commands are tagged **[robot]**, **[workstation]**, or **[policy]**. Replace
`<ROBOT_IP>` / `<POLICY_IP>` with the machine addresses (use `127.0.0.1` if a
component runs locally).

## A. Bimanual teleop only (no recording)

```bash
# [robot]
robot/yam canup                 # bring up the 4 CAN interfaces (after boot)
robot/yam teleop
# lift both gellos to engage; bring both home to stop & auto-return.
```

## B. Data collection (teleop + LeRobot recorder)  ← main flow

```bash
# 1. [robot]   start the teleop server (serves state/action/gate over portal)
robot/yam canup
robot/yam teleop

# 2. [workstation]   start the recorder GUI
workstation/yam-data record \
    --robot-host <ROBOT_IP> \
    --repo-id user/yam_pick --root ~/lerobot_data \
    --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn>
```

The recorder opens on a **Setup page**:

1. Confirm `repo_id` / `root` / `task` and the **source** (teleop / dagger / eval).
   Closing the window saves the filled setup fields locally; the next Record or
   DAgger Deploy session restores its own last-used values without pressing START.
   The dataset is written to **`<root>/<name>`** (name = last segment of `repo_id`,
   e.g. `~/lerobot_data` + `hello/pick_and_place` → `~/lerobot_data/pick_and_place`).
   The status line shows cameras detected and whether that dataset already exists.
2. Tick **Continue collecting** to resume/append; otherwise **START** creates it
   fresh (and asks **twice** before overwriting an existing folder).
3. **START** connects the robot, opens cameras + dataset, and — with `auto_arm` —
   arms collection immediately.

The **Past demonstration overlay** panel plays a recorded episode's three camera views under
the live ones, so the scene can be matched before a take.

| Control | Behaviour |
|---|---|
| **Overlay dataset** | any sibling folder under the session `root` — the same list as the setup-page picker. Changing it re-reads that folder's `meta/episodes`. |
| **Live camera opacity** | `100%` live only · `0%` past episode only · anything between blends |
| **Resume reference** / **Pause reference** | selection always starts paused on frame 1; resume plays, pause freezes the comparison frame |
| **Refresh demonstrations** | picks up saves encoded since the panel last read the folder |

- One row per demonstration (`episode_index`), even when many share an MP4 container.
- The first saved demonstration is selected automatically. There is no Off entry —
  `100%` live opacity already hides the reference.
- The player reads **completed** MP4s, so it coexists safely with an append session.
- Overlay-source folders other than the one being recorded are **read-only**. For the
  recording folder itself, tick **Continue collecting**; starting fresh removes its episodes
  after the overwrite confirmations.

**It is display only.** Dataset frames and policy observations are the untouched RealSense
arrays, which is why the same panel serves `yam-data record` and `yam-data deploy` without
either robot-side server having to decode video.

Then teleoperate: **lift both gellos** → records; **bring both home** → episode ends.

- `review_before_save: true` — held in the **review panel** for **Keep** (S/F) / **Delete** (D).
- `review_before_save: false` — **auto-saves** each engage→idle.
- A **leader handle button** ends + labels in one press (see *Labeling* below).
- Close the window when done: that calls `finalize()`, which is what completes the dataset.

Quick dry run with no robot/cameras/lerobot:

```bash
# [workstation]
workstation/yam-data record --mock
```

## C. Deployment (run a policy on the robot)

Deployment always needs **three processes on (usually) three machines**. Steps 1 and 2 are
identical whether or not you record; only step 3 differs.

```
[robot]                        [workstation]                      [policy]
run_robot_server deploy  ◀─portal─▶  deploy UI  ◀──websocket+msgpack──▶  yam_policy.serve
  owns the arms,                     owns cameras,                       owns the network
  takeover, homing, e-stop           builds obs, sends action chunks
```

The robot server is the **source of truth** for whether the policy is allowed to drive:
the workstation only streams actions while the robot reports `policy_running` and not
intervention / homing / e-stop. So a takeover or an e-stop stops the policy no matter what
the UI is doing.

**Why `deploy` and not `teleop` for the robot server** — the two differ in who drives the
follower and what the leader is for:

| | `run_robot_server teleop` | `run_robot_server deploy` |
|---|---|---|
| drives the follower | the human, through the leader (gello) arm | the **policy**, via `set_policy_action` from the workstation |
| accepts policy actions | no | yes |
| what the leader is | the **input device** (free by default; `bilateral_kp` > 0 adds force feel) | an **override handle**: mirrors the follower while the policy drives, goes free when you take over |
| episode boundary | the engage gate (IDLE → ENGAGED → HOMING → IDLE) | a policy rollout (start/stop by button or UI) |
| handle buttons | label the episode (success / fail / discard) | drive the rollout state machine (start/stop, takeover, keep/discard + home) |
| used by | `yam-data record` | `yam-data deploy` (GUI or `--headless`) |

(The `deploy` controller was called `dagger` until it also started serving plain
deployment — HG-DAgger is one *use* of it, not what it is. `robot/yam dagger` still
works as an alias, and an un-updated robot server reporting the old name is accepted.)

### Which policy server (step 2)

**openpi and LeRobot checkpoints both work, through the same client and the same wire.**
openpi's protocol is what the client speaks; a LeRobot checkpoint runs behind an adapter on
this side, so there is nothing to convert.

```bash
python -m yam_policy.serve                                    # zero-model smoke test ("hold pose")

python -m yam_policy.serve \
    --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
    --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
    --config device=cuda                                      # a LeRobot checkpoint
```

Nothing on the workstation changes — the client reads the camera names, image size and chunk
length out of the handshake.

It also **names what answered** (`LeRobot · act`, `ACRFT · pi05_yam_lego_taxi`) next to the
connection dot and in `--headless` logs. That matters because the wrong server still returns a
well-formed chunk: they share a wire but not an observation format.

The cameras line up on their own: this recorder writes `observation.images.<role>`, so a
policy trained on its data already names its image features after `wrist_left` /
`wrist_right` / `agentview`. A checkpoint trained elsewhere needs one `--config camera_map=…`.

> **Before training on one of these datasets, exclude `observation.leader`.** LeRobot's
> trainer takes every column as a policy input, and on a teleop dataset the leader pose *is*
> the action — so the policy can copy the answer, the loss looks better for it, and the
> checkpoint then fails on the robot. Full details and the exact invocation:
> [`policy_serving/README.md`](../../policy_serving/README.md).

A `teleop` server ignores `set_policy_action` entirely, so deployment against it does
nothing — hence the startup check below.

There is a third mode, `wrapper`: the followers track an external command with **no
leader arms at all**, which is what `yam-data replay` drives. So the robot server has
three controllers — `teleop`, `deploy`, `wrapper` — plus `robot/yam can` / `canup`,
which are CAN-bus utilities rather than modes.

```bash
# 1. [robot]    dagger server (policy drives followers; handle button = takeover)
robot/yam canup
robot/yam deploy

# 2. [policy]   serve your policy (own env; openpi-compatible websocket)
#               see policy_serving/README.md
python -m yam_policy.serve --policy <module>:<Class> --config k=v     # :8000
```

```bash
# 3. [workstation]
workstation/yam-data deploy \
    --robot-host <ROBOT_IP> --policy-host <POLICY_IP> \
    --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn> \
    --repo-id user/yam_pick --prompt "pick up the cube"
```

### Two choices in the UI

Everything else is on the setup page, so plain `yam-data deploy` is enough. A run has two
**independent** axes — recording a deployment is not the same thing as doing DAgger:

**mode** — what you are here to do. This is what decides *leader mirroring*:

* `deploy` — watch the policy work. The leader hangs **free**, so the handles do not fly
  around while the arm moves.
* `dagger` — correct the policy. The leader **mirrors** the follower, so grabbing a handle
  to take over starts from the arm's current pose.

**record** — whether this run lands in a dataset. Off hides the dataset fields entirely;
no dataset is created, opened, or resumed and no frames are buffered.

| mode | record | source | what it is |
|---|---|---|---|
| deploy | off | `deploy` | just run a checkpoint and watch |
| deploy | on | `eval` | log the run — Start/Stop collection bounds one episode |
| dagger | on | `dagger` | one episode per rollout, ended with keep/discard |
| dagger | off | `deploy` | practise takeovers, save nothing |

Mirroring follows the mode automatically, and the collect-page checkbox overrides it live
(it is the one setting worth changing mid-rollout). `--mode` / `--no-record` /
`--no-leader-mirror` only set the initial values; `robot/yam deploy --no-leader-mirror` makes
"no mirroring" the robot's own default. The mirror setting is latched workstation-side and
re-applied on reconnect, so the robot never silently reverts mid-session.

> **The trade-off is on takeover.** Mirroring exists so the leader is already where the
> follower is. With it off, the moment you take over, the follower travels — rate-limited
> by the smoother, not instantly — to wherever the leader happens to be. Park the handles
> near the arm pose before intervening, or switch to `dagger` mode first.

### If you started the wrong robot server

The robot modes are mutually exclusive and launched separately, and a mismatch used to fail
*silently* — a `teleop` server just ignores policy actions, so the policy looks connected
and the arms never move. Both GUIs now check the mode the robot reports and refuse to start:

```
The robot server at 10.0.0.5:11331 is running in 'teleop' mode, but this needs 'dagger'.

On the robot machine, restart it with:  robot/yam deploy
```

### Operating it

| | UI button | handle button |
|---|---|---|
| start/stop the rollout | Start/Stop Policy | left upper |
| human takeover on/off | Human Intervention | left lower |
| end the run + home | Keep + Home / Discard + Home (recording), Stop + Home (not) | right lower / right upper |
| fine-grained control | — | left upper, while intervening |

The prompt sent to the policy is the **task** field, so switching task in the UI switches
the instruction the policy is conditioned on.

### No screen on the workstation?

`yam-data deploy --headless` runs the same loop without Qt: same observation, same policy
client, same robot-mode check — and unlike the old `yam-data bridge` it can record, because
it drives the very same `Recorder`.

```bash
workstation/yam-data deploy --headless --mode deploy --no-record
```

It starts the policy on launch (there is no button to press), so **the arms begin moving
when you run it** — `--no-autostart` waits for the handle button instead. Ctrl-C stops the
policy, closes any in-flight episode, and finalizes the dataset.

`yam-data bridge` was a second, independent copy of this loop and has been removed: the
duplicated code decided *what the policy receives*, so the two could drift into feeding the
same policy different observations.

This closes the loop: train → deploy (`--no-record`) to sanity-check → DAgger to collect
where it fails → retrain.

## D. Replay a dataset onto the robot

Replay is deployment with the actions read from a dataset, so it runs on **the deploy stack** —
same robot server, same UI, same safeguards. There is no `wrapper` server and no separate GUI.

```bash
# 1. [policy]  serve the episode
python -m yam_policy.serve \
    --policy yam_policy.policies.dataset_policy:DatasetPolicy \
    --config root=~/lerobot_data/yam_pick --config episode=3

# 2. [robot]
robot/yam canup && robot/yam deploy

# 3. [workstation]
workstation/yam-data deploy --robot-host <ROBOT_IP> --policy-host <POLICY_IP>
```

- The **past-demonstration overlay** follows the replay automatically: same episode, at the rate
  it was recorded, paused whenever the rollout is not streaming.
- Everything deployment has applies — takeover, e-stop, the follower smoother (so there is no
  separate ramp to the first frame), the link-loss watchdog. Replay is not a path around them.
- `--config speed=2` / `speed=0.5` / `loop=true`; see
  [`policy_serving/README.md`](../../policy_serving/README.md).

The standalone `workstation/yam-data replay` GUI is still there for scrubbing a dataset with no
robot (`--mock` for no robot at all).

## E. Render the candidate fan offline

For a run recorded with `deploy --num-samples N` against a server started with the same N, the
`action_samples` column can be drawn back onto the recorded frames:

```bash
workstation/yam-data render-samples \
    --repo-id my_deploy_run --episode 0 --arm left \
    --horizon 30 --candidates 8 --out .scratch/fan.mp4
```

Offline on purpose — a live per-tick overlay was built and dropped, because watching the spread
go past at 30 fps was not worth much; the analysis belongs with the dataset.

No extra checkout and no `matplotlib`: the fan drawing is vendored (PIL only), so this needs
nothing the recorder env does not already have.

## F. Calibrate agentview's extrinsic

`render-samples`' fan is drawn on the wrist view because that camera's pose is already known
(published extrinsic + FK). agentview's is not — nothing ties it to the robot base by default.
`calibrate-agentview` gets it from a ChArUco board left sitting on the desk (never moved, never
attached to the robot), using a wrist camera's known pose as the bridge:

```bash
workstation/yam-data calibrate-agentview --out ~/lerobot_data/calibration/agentview_extrinsic.json
workstation/yam-data calibrate-agentview --mock     # GUI shell only, no hardware
```

Print a ChArUco board (default 8x6 squares, 30mm/22mm — pass `--squares-x/-y`,
`--square-length-m`, `--marker-length-m` to match a different one; **measure the printed squares,
don't trust the page-fit scale**) and set it on the desk in view of agentview. Both wrist cameras
bridge by default (YAM is bimanual — `--arms left` to use only one); move either arm so its wrist
camera also sees the board, **Capture**, then move to a few more poses and capture again at each
— the board itself never moves. **Solve** pools every capture (whichever wrist it came from) and
reports how much they disagree with each other in mm/degrees, which is the calibration's own
confidence number rather than a separate validation step; **Save** writes the result next to the
intrinsics it was solved against.

Camera intrinsics come from the RealSense devices themselves (factory calibration), not a
separate checkerboard sweep. Needs `opencv-contrib-python>=4.7` (`cv2.aruco`'s
`CharucoDetector`/`matchImagePoints` API) — see requirements.txt.

---

## Notes

- **finalize**: closing the recorder window (or `recorder.shutdown()`) calls
  `LeRobotDataset.finalize()`. Skipping it leaves parquet files incomplete.
- The record loop is clocked at `recorder.fps` (30 as checked in). Cameras grab on their **own capture thread**
  and cache the latest frame; the loop and GUI read that cache **non-blocking**, so a
  slow frame or a pipe re-open never stalls recording or freezes the view. Each tick
  pairs the latest cached frame with the latest robot state/action polled over portal.
- **Camera fps fallback**: the requested fps (60) is auto-reduced to the highest the
  device supports (e.g. 30 on USB 2.0) — no config edit needed. Over **USB 2.0** a
  640×480 stream caps at 30 fps and can drop frames under 3-camera load, so use the
  **USB 3 cables** before raising the stream rate (a USB-2 cable downgrades the link even on a
  USB-3 port; `config.yaml` shows how to check every camera's negotiated speed).
- **Single instance**: the recorder takes a lock so a second instance can't fight
  over the cameras; starting a second one reports a clear error instead of flapping.
- `lerobot`'s API can shift between releases — the version-sensitive calls are
  isolated in `dataset_writer.py` and `dataset_reader.py`.
- **Dataset location**: `root` is a **parent dir**; the dataset lives at
  `<root>/<name>` (name = last segment of `repo_id`), so several datasets can sit
  side by side under one `root`. The reader/replay resolve the same path.
- **Outcome labels**: **Keep (success)** / **Keep (fail)** tag each episode; the
  label + task + frame count are appended to `outcomes.jsonl` **inside the dataset
  folder** (a sidecar, since LeRobot has no per-episode label slot).
- **Resume**: tick **Continue collecting** in the GUI (or `--resume`) to append to
  the existing dataset at `<root>/<name>` instead of creating a new one (episode
  indices continue).
- **Doctor**: `workstation/yam-data doctor --root ~/lerobot_data [--repo-id ...]`
  prints episode counts, success rate, and per-task stats from `outcomes.jsonl`
  (and validates the LeRobot dataset if `--repo-id` is given). The replay episode
  list is annotated with ✓/✗ from the same sidecar.
- **Safety**: E-STOP button in both GUIs (holds the followers); optional collision
  soft-stop (`control_config.FOLLOWER_EFFORT_LIMIT`); disk-space guard refuses to
  save below `min_free_gb`.
- **Always-on provenance (fixed schema)**: every frame carries
  `observation.state(42)`, `observation.leader(12)`, `observation.eef(14)` (FK from
  the company `Kinematics`; zeros if no model), `observation.control_mode(1)`
  (teleop/policy/intervention), and `action(14)`. The schema is **predefined from the
  robot's known outputs** (no runtime probe).
- **Async writer**: a finished episode is queued and saved by a background worker
  (one at a time), so LeRobot's per-trajectory encoding never blocks the next
  collection. The GUI shows the pending `queue` depth.
- **Labeling**: in the review panel use the mouse, **keyboard** ([S] keep success,
  [F] keep fail, [D] delete, [space] toggle collection), or the **leader handle
  buttons**. The button→outcome map is **per-(side, index)** and configurable in
  `config.yaml` under `recorder.buttons` (keyed `<side>.<index>`, upper=0/lower=1).
  This is also the robot's source for end-of-episode homing, so there is no
  separate home-button list to keep synchronized.
  Default: **left upper = fine-grained toggle, left lower = success, right lower =
  fail, right upper = discard**. A label button also
  starts homing, so one press ends + labels + saves (records through homing).
- **Safe fine-mode exit**: toggling fine-grained control off freezes the follower
  arm and gripper, pauses dataset appends without closing the episode, and slowly
  aligns the leader. Recording and normal 1:1 control resume after a stable
  alignment; timeout leaves the follower held and the leader gravity-comp free.
- **Operator UI**: a big color status **banner** (IDLE/ARMED/REC/REVIEW/fault), a
  **health strip** (robot link · cameras · save queue), **live stats** (kept ✓/✗,
  discarded, success rate), **audio cues** (start, keep/fail/delete, fault), and a
  review **scrubber** — so you can collect while watching the robot.
- **Past demonstration overlay**: after START, select any completed demonstration in the active
  dataset to blend its three synchronized camera videos into the matching live
  previews. It starts paused and plays only after **Resume reference**. The opacity
  setting persists separately for Record and Deploy.
- **Task templates**: `--tasks "pick the cube; stack the blocks; open the drawer"`
  gives a quick-switch dropdown; the active task **persists until you change it**
  (editable — type a new one on the fly).
- **Eval rollouts**: `--source eval` records a continuous policy rollout from
  Start to Stop (action = the executed command, labeled policy/intervention) — for
  saving evaluation episodes as datasets.
- **DAgger rollouts**: `--source dagger` records the complete policy run as one
  episode, including periods with no takeover. `action` is always the command
  executed by the robot and `observation.control_mode` marks policy (1) versus
  human intervention (2). Keep/Discard applies to the entire rollout.
- **Camera fault tolerance**: a faulted RealSense shows a red ⚠ warning, recording
  pauses (no garbage frames), and the capture thread auto-reconnects in the
  background (logging only the down/recovered transitions, not every retry).
- **Replay overlay**: tick **Overlay live** to blend an episode's first frame with
  the live agentview, so you can place objects to match the dataset before Play.
- The robot link is the snapshot contract in
  [`i2rt/serving/README.md`](../../i2rt/serving/README.md); the policy link is in
  [`policy_serving/README.md`](../../policy_serving/README.md).
