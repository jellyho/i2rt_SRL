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

## Install (LeRobot policy-server environment)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git submodule update --init lerobot
bash policy_serving/setup_policy_env.sh
source policy_serving/.venv/bin/activate
```

## Serve a LeRobot checkpoint

```bash
# ACT, Diffusion Policy, MultiTaskDiT, pi0.5, SmolVLA, etc. are selected
# automatically from pretrained_model/config.json.
yam-lerobot-serve \
    --checkpoint /abs/path/to/pretrained_model \
    --device cuda

# MultiTaskDiT flow-matching checkpoint with native RTC (the same command also
# supports diffusion checkpoints through training-free diffusion-to-flow RTC):
yam-lerobot-serve \
    --checkpoint /abs/path/to/pretrained_model \
    --device cuda \
    --rtc \
    --num-inference-steps 20

# pi0.5 and SmolVLA already default to 10 flow integration steps. Omit the
# solver override unless you intentionally want to tune it:
yam-lerobot-serve --checkpoint /abs/path/to/pi05_or_smolvla --device cuda --rtc
```

``--num-inference-steps`` is solver-agnostic. For flow-matching MultiTaskDiT it
overrides ``num_integration_steps``. For diffusion MultiTaskDiT with RTC, it
selects trained diffusion timesteps approximately uniformly in the converted
flow time; without RTC it is the number of DDIM denoising steps. For pi0.5 it overrides
``num_inference_steps`` and for SmolVLA it overrides ``num_steps``. Omitting it
preserves the checkpoint's policy-specific default.

The older generic ``yam-serve`` entry point remains available for dummy and
custom ``BasePolicy`` adapters. It is not needed for normal LeRobot deployment.

## RTC request flow

The workstation keeps the 30 Hz robot clock and owns the action queue. It sends
a new request after ``rtc.min_execute_steps``, including consecutive observation
history, the unexecuted normalized action tail, and a conservative inference
delay. The server performs guided denoising and returns both normalized
``model_actions`` and executable ``actions``. The workstation continues the old
chunk while inference is running, then drops the elapsed prefix and swaps to the
new chunk immediately.

The broker enforces the RTC feasibility condition ``d <= s <= H - d``. The
initial synchronous prediction is treated as a warmup and is not added to the
latency window. If measured inference exceeds half of the action horizon, or a
chunk expires before its replacement is ready, the broker raises instead of
blocking the control loop or repeating an action whose safety it cannot infer.

The server advertises ``rtc_enabled``, ``n_obs_steps``, policy type, solver, and
action horizon in its connection metadata. ``yam-data deploy --rtc`` fails fast
if it connects to a server that was not started with ``--rtc``.

## Add your own policy

Subclass `BasePolicy` and implement `infer(obs) -> {"actions": (H, D)}`. See
[`yam_policy/policies/dummy.py`](yam_policy/policies/dummy.py) for the simplest
example and [`lerobot_policy.py`](yam_policy/policies/lerobot_policy.py) /
[`openpi_policy.py`](yam_policy/policies/openpi_policy.py) for real-model templates.
Then serve it with `--policy your.module:YourPolicy`.

## Client side (used by the workstation bridge)

```python
from yam_policy import WebsocketClientPolicy, AsyncActionChunkBroker
client = WebsocketClientPolicy(host="policy-host", port=8000)
policy = AsyncActionChunkBroker(client, action_horizon=16)   # prefetches the next chunk
action = policy.infer(obs)["actions"]   # one (action_dim,) step per call, re-queries every 16
```

- **AsyncActionChunkBroker** fetches the next chunk in a background thread so the
  per-chunk inference latency doesn't stall the control loop (use
  `ActionChunkBroker` for the simple synchronous version).
- **Metadata-driven config**: declare `action_horizon` (and optionally an
  `obs_spec` dict with `image_keys` / `image_size`) on your policy; `serve.py`
  puts them in the server metadata and the bridge auto-configures from
  `get_server_metadata()` — no need to hand-match the bridge to the policy.
