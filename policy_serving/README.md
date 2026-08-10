# yam-policy — policy serving (openpi-compatible)

A small, **dependency-light** websocket policy layer. The robot/workstation side
holds only this package (`numpy`, `msgpack-numpy`, `websockets`, `pillow`); the
**policy server** runs in its own unrestricted env with whatever the model needs
(torch / JAX / CUDA), on this machine or a remote GPU box.

The wire protocol is **identical to openpi** (`openpi_client`), so a real openpi
checkpoint served by `openpi` works against our client, and a policy written
against `BasePolicy` can be served by openpi.

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

# a real LeRobot policy (template — adapt policies/lerobot_policy.py):
python -m yam_policy.serve \
    --policy yam_policy.policies.lerobot_policy:LeRobotPolicy \
    --config pretrained_path=/abs/path --config device=cuda

# a real openpi checkpoint (template — needs openpi installed here):
python -m yam_policy.serve \
    --policy yam_policy.policies.openpi_policy:OpenPiPolicy \
    --config config_name=pi0_fast_droid --config checkpoint_dir=/abs/ckpt
```

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
