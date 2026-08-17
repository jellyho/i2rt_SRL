# yam-policy — policy serving (openpi-compatible)

A small, **dependency-light** websocket policy layer. The robot/workstation side
holds only this package (`numpy`, `msgpack-numpy`, `websockets`, `pillow`); the
**policy server** runs in its own unrestricted env with whatever the model needs
(torch / JAX / CUDA), on this machine or a remote GPU box.

The wire protocol is **identical to openpi** (`openpi_client`), so a real openpi
checkpoint served by `openpi` works against our client, and a policy written
against `BasePolicy` can be served by openpi.

**openpi and LeRobot checkpoints both deploy here**, through the same client and the same wire.
openpi's is the protocol; LeRobot's models arrive through an adapter on this side
([`policies/lerobot_policy.py`](yam_policy/policies/lerobot_policy.py)) that mirrors LeRobot's own
inference path — see [Deploying a LeRobot checkpoint](#deploying-a-lerobot-checkpoint). Neither
upstream is modified or vendored.

```
workstation (policy bridge)              policy server (this package)
  WebsocketClientPolicy  ──ws+msgpack──▶  WebsocketPolicyServer(policy)
  + ActionChunkBroker    ◀────actions───  policy.infer(obs) -> {"actions": (H, D)}
```

## Contract

```python
obs = {
    "observation/state":            np.ndarray,   # proprioception, unnormalized
    "observation/images/<cam>":     np.ndarray,   # HxWx3 uint8 (e.g. 224x224)
    "prompt":                       "do the task" # optional, language-conditioned
}
action_chunk = client.infer(obs)["actions"]       # (action_horizon, action_dim)
```

## Install (policy server env — unrestricted)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd policy_serving
uv venv                      # any Python >= 3.10
source .venv/bin/activate
uv pip install -e .          # + your model deps, e.g. uv pip install -e /path/to/openpi
```

## Run a server

```bash
# zero-model smoke test (returns a "hold pose" chunk):
python -m yam_policy.serve

# a LeRobot checkpoint (see "Deploying a LeRobot checkpoint" below):
python -m yam_policy.serve \
    --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
    --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
    --config device=cuda

# a real openpi checkpoint (template — needs openpi installed here):
python -m yam_policy.serve \
    --policy yam_policy.policies.openpi_policy:OpenPiPolicy \
    --config config_name=pi0_fast_droid --config checkpoint_dir=/abs/ckpt
```

## Deploying a LeRobot checkpoint

Train with LeRobot, point the server at the checkpoint, run deploy. There is no conversion step
and nothing to hand-match on the client.

```bash
# in the policy-server env
uv pip install -e "policy_serving[lerobot]"

python -m yam_policy.serve \
    --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
    --config pretrained_path=outputs/train/my_act/checkpoints/last/pretrained_model \
    --config device=cuda
```

`pretrained_path` takes a local checkpoint directory or a Hub repo id. Then start deploy as
usual — the client reads the camera names, image size and chunk length off the handshake.

**Why the cameras line up on their own.** This recorder writes `observation.images.<role>` for
each of its cameras (`wrist_left`, `wrist_right`, `agentview`), so a policy trained on its data
already names its image features after them. The adapter hands those names back as `image_keys`
and the client sends each camera to the key the checkpoint reads. A checkpoint trained elsewhere
needs one line:

```bash
--config camera_map=agentview=observation.images.top,wrist_left=observation.images.wrist
```

A `camera_map` naming an image the policy does not have fails at startup, where it is a typo,
rather than at the first rollout, where it is a robot.

**Both key conventions are accepted.** openpi checkpoints read `observation/state`, LeRobot ones
read `observation.state`, and the client sends both — the dotted one wins, because they are
different vectors (openpi's 14 joint targets vs. this stack's 42-value record) that would
otherwise collapse onto the same name.

**A missing camera is refused, not zero-filled.** Deploy reports the error and holds, instead of
driving from a black frame and reporting nothing wrong.

**What is served is the whole predicted chunk** (`chunk_size`), not the `n_action_steps` a
closed-loop LeRobot rollout consumes before replanning — this stack replans on its own schedule.
Use `--config actions_per_chunk=N` to serve less.

### Before training: drop `observation.leader`

**LeRobot's trainer takes every column in the dataset as a policy input**, and this recorder
writes more columns than a policy should see. On a teleop dataset the leader pose *is* the action
— measured at **2e-4 rad** apart on `yam_cable_tie_v4` — so a policy handed `observation.leader`
can copy the answer straight out of its input. Training loss looks excellent for exactly that
reason, and the checkpoint is useless at deploy, where the leader is either hanging free or
mirroring the follower. `observation.control_mode` is worth dropping too: it is constant within a
teleop dataset, so it teaches nothing.

Name the inputs you want in the training config. The dataset is not touched — LeRobot only infers
features when `input_features` is empty (LeRobot's own `policies/factory.py`:
`if not cfg.input_features`), so setting it wins:

```bash
lerobot-train ... --tolerance_s=1e-3 \
    --policy.input_features='{
        "observation.state":               {"type": "STATE",  "shape": [42]},
        "observation.images.wrist_left":   {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.wrist_right":  {"type": "VISUAL", "shape": [3, 480, 640]},
        "observation.images.agentview":    {"type": "VISUAL", "shape": [3, 480, 640]}}'
```

List every input you *do* want — this replaces the inferred set rather than subtracting from it.
`output_features` is always derived from the dataset, so `action` needs no mention. Add
`observation.eef` if you want it; it is a real signal and available at deploy.

The server warns at load if a checkpoint reads a leaking column, since nothing later can: such a
policy trains beautifully and simply behaves badly on the robot. The deploy client does send all
of these columns, so an already-trained checkpoint still runs — it just runs with a leader signal
that no longer means what it meant during collection.

The inference path mirrors LeRobot's own `async_inference.policy_server` rather than being
invented, because two of its steps fail **silently**:

- the pre/post processors need an explicit **device override**, or the batch stays on the CPU
  while the model sits on the GPU;
- the postprocessor unnormalises one `(B, action_dim)` step at a time. Handing it a whole chunk
  does not raise — it broadcasts against the wrong axis and returns plausible wrong numbers.

`tests/test_lerobot_policy.py` therefore checks the actions against known stats, not just shapes.

Nothing in LeRobot is modified or vendored; the adapter is one file on this side.

## Replaying a recorded episode

Replay is deployment with the actions read from a dataset instead of a model. The normal way to
run it is the deploy GUI's **`dataset` mode**, which builds `DatasetPolicy` **in-process** — no
server to start: set `mode = dataset` in `workstation/yam-data deploy` and pick the episode on the
run page (see `workstation/lerobot_recorder/README.md` §D).

`DatasetPolicy` is also an ordinary servable policy, if you want to drive a plain deploy client
from another machine:

```bash
python -m yam_policy.serve \
    --policy yam_policy.policies.dataset_policy:DatasetPolicy \
    --config root=~/lerobot_data/yam_cable_tie_v4 --config episode=3

robot/yam deploy                 # the SAME robot server deployment uses -- not `wrapper`
workstation/yam-data deploy      # connect as a live policy
```

| `--config` | |
|---|---|
| `root` | the dataset directory (the one holding `meta/` and `data/`) |
| `episode` | which `episode_index` to replay |
| `speed` | `>1` drops frames, `<1` repeats them — the client ticks at a fixed rate, so changing the stream is what changes the speed |
| `loop` | start again at frame 0 instead of holding at the end |
| `chunk` | actions per reply (default 30) |

What replay inherits, none of which the old standalone replay had: the follower smoother and
joint-speed clamp (so there is no hand-rolled ramp to the first frame), human takeover on a handle
button, the network e-stop, the link-loss watchdog, and pause/resume as a send-gate.

The deploy GUI shows the replayed episode's **first frame** as a scene reference (the overlay
decoder streams forward and can't seek, so it is not run as a moving ghost that would drift against
the arm's real speed — the arm itself is the moving reference).

**Only the action column is read**, straight from the parquet: no video decoding, no `lerobot`
dependency, and a 100-episode dataset opens in about a tenth of a second.

At the end the final pose is held rather than the stream stopping — the client is driving a
robot and needs something to hold — and `replay_done` marks which frames those are.

## Add your own policy

Subclass `BasePolicy` and implement `infer(obs) -> {"actions": (H, D)}`. See
[`yam_policy/policies/dummy.py`](yam_policy/policies/dummy.py) for the simplest
example and [`lerobot_policy.py`](yam_policy/policies/lerobot_policy.py) /
[`openpi_policy.py`](yam_policy/policies/openpi_policy.py) for real-model templates.
Then serve it with `--policy your.module:YourPolicy`.

## Client side (used by the workstation bridge)

```python
from yam_policy import WebsocketClientPolicy, ActionChunkBroker
client = WebsocketClientPolicy(host="policy-host", port=8000)
policy = ActionChunkBroker(client)      # chunk size comes from the policy's own reply
action = policy.infer(obs)["actions"]   # one (action_dim,) step per call
```

- **ActionChunkBroker** holds the chunk and hands out one step per control tick, re-inferring
  when it runs out — so a 30-step chunk means one inference per second at 30 Hz rather than
  thirty. It takes the chunk size from what the policy returns, so a checkpoint trained at 30
  cannot be silently truncated by a client configured for 16.
- There is no prefetching variant. One existed, starting the next inference two steps early on
  the observation available then; measured, its freshness came out a wash against the
  synchronous version — it started early by the same margin it was stale by — and it cost the
  guarantee that a chunk was computed from the observation the caller actually handed over.
  The price is a stall at each chunk boundary, which is what openpi's and LeRobot's own
  brokers do as well.
- **Metadata-driven config**: declare `action_horizon` (and optionally an
  `obs_spec` dict with `image_keys` / `image_size`) on your policy; `serve.py`
  puts them in the server metadata and the bridge auto-configures from
  `get_server_metadata()` — no need to hand-match the bridge to the policy.

## Which server answered

Deploy shows what is behind the policy port — `LeRobot · act`, `ACRFT · pi05_yam_lego_taxi`,
`openpi · pi0_fast_droid` — next to the connection dot.

It earns a field of its own because the failure is otherwise invisible:

- all three stacks speak this wire, and none reads the others' observations;
- every one of them answers a mismatched observation with a well-formed chunk of the right shape.

Nothing raises. The robot just moves wrongly. Naming the server at the handshake turns that from
a rollout symptom into something readable before starting.

A policy declares it by exposing `policy_info`, which `serve.py` merges into the metadata:

```python
self.policy_info = {"framework": "lerobot", "policy_type": "act", "checkpoint": path}
```

`framework` is the only key that matters; `policy_name` / `policy_type` fill in the second half of
the label. A server that declares nothing is described from what it does advertise and shown with
a trailing `?` — an inferred name is marked as inferred rather than asserted, because a confident
wrong one is worse than none.

## Extra per-step data (`extra_features`)

A policy can return more than actions — a critic's value for each candidate, the candidate
chunks themselves, anything worth keeping — and have it recorded automatically. Neither side
hard-codes a name: the server declares them once, at handshake.

```python
metadata = {
    "extra_features": {
        "critic_q":       [8],      # a Q per candidate,   arrives as (X, 8)
        "action_samples": [8, 14],  # the candidates,      arrives as (X, 8, 14)
    }
}
```

**The shape you declare is one step's.** The leading axis is left out on purpose: the chunk
length `X` is adaptive — whatever a reply happens to carry, and free to differ from one replan
to the next — so it is not something either side can declare. The per-step shape is fixed, and
is exactly what a dataset column needs.

**Do not declare `action_horizon`.** This client never reads it: the chunk length comes from
`actions.shape[0]` on every reply, which is what lets the horizon be adaptive at all. Serving
it is harmless (openpi's own broker takes it as a constructor argument) but it is not part of
this contract, and with a varying chunk it is not a well-defined number.

**The rule.** An extra array's leading axis is `X`, the same as `actions`. That is what makes
it per-step, so step `i` gets row `i` — the broker slices by that rule rather than by a list of
known keys, and so never needs to learn a new one. An array whose leading axis is *not* `X` is
passed through whole instead of mis-indexed, because slicing it returns a different thing
entirely and still looks like data.

**What you get.** Each declared feature becomes a dataset column at its declared shape, one row
per frame:

```python
dataset[i]["critic_q"]        # (8,)
dataset[i]["action_samples"]  # (8, 14)     — not flattened
```

Shapes are kept rather than flattened so the dataset describes itself; an anonymous 112-vector
would leave every reader to know the layout out of band. A malformed declaration is dropped
with a warning rather than trusted — a wrong shape here becomes a wrong column in every episode
— and a policy that declares nothing produces exactly the dataset it produced before.

The deploy stack is unchanged by any of this: one chunk, adaptive length, one action per tick.

## Visualisation (`yam_policy.viz`)

Draw what a policy predicted, on the frames it predicted from. Packaged here so a policy repo
can import it and build its own views:

```python
from yam_policy.viz import WristCameraGeometry, CameraIntrinsics, overlay_samples

geometry = WristCameraGeometry(mjcf_path)        # the arm+gripper model the robot runs
path     = geometry.chunk_to_path(chunk)         # [T, joints] -> [T, 3] in the arm's base frame
pixels   = geometry.project(path, q_now, intrinsics)
frame    = overlay_samples(frame, geometry, samples, q_now, intrinsics)
```

- **The paths are in metres.** YAM actions are joint targets, so FK over a chunk gives the real
  path. Overlays built on end-effector *deltas* cannot — openpi's RoboCasa one rescales each
  replan to a legible length and says so — which means a bundle that looks tight here IS tight.
- **The wrist extrinsic is published, not calibrated.** i2rt ships the arm with its D405 mount
  as one model whose body chain ends in a `camera` optical frame, so `T_GRIPPER_CAMERA` is the
  manufacturer's transform. Composing that chain reproduces the three figures its header states
  (pos, quat, and a 25° cant), which is what makes it safe to carry the matrix instead of the
  model and its meshes.
- **Projection flags points behind the lens** rather than returning their pixel: the divide is
  happy to produce a finite coordinate for those, and the mirrored path that results reads as a
  confident wrong prediction rather than a bug.
- `yam_policy.viz.sample_log` reads and writes the chunk sets a run recorded beside its dataset,
  so a render can show what the policy predicted AT THE TIME rather than what the current
  checkpoint would predict now.

### Plugging into a HUD

`WristProjector` matches the projector interface ACRFT's `examples/robocasa/deploy_hud.py`
expects — `chunks[N, H, A] -> list of [H, 2]` — so it drops in where its `SketchProjector`
sits. That one integrates two action dims into a corner minimap because the fan had to be
visible "before camera calibration/FK exist"; both exist now.

```python
from yam_policy.viz import WristCameraGeometry, CameraIntrinsics, WristProjector

proj = WristProjector(WristCameraGeometry(mjcf), intrinsics, arm_slice=slice(0, 7))
rec  = HudRecorder(mode="bon", horizon=H, projector=proj)
...
proj.set_pose(state[0:7])        # this arm's joints, THIS replan — the camera is on the wrist
rec.add(agent_rgb, wrist_rgb, response, step=t)
```

The pose has to be set each replan and cannot be inferred from the chunk: the camera rides on
the wrist, so where a future position lands on screen depends on where the arm is now.

`CriticSelectPolicy` does not put its choice first, so a recorded run stores `chosen` (and the
critic's `scores`) alongside the candidates — `yam_policy.viz.row_at` returns all three. Drawing
candidate 0 as the executed one produces a picture that looks right and is not.

`mujoco`/`mink` (kinematics) and `pillow` (drawing) are imported lazily and declared as the
`viz` extra, so a policy server never loads them:

```bash
pip install -e "policy_serving[viz]"
```
