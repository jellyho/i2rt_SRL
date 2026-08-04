# Run MultiTaskDiT with RTC

Run these commands from `/home/rllab2/i2rt_rllab-mtd-rtc`.

## 1. Policy server (GPU machine)

First-time setup:

```bash
bash policy_serving/setup_policy_env.sh
```

Start the policy server:

```bash
policy_serving/yam-serve \
  --checkpoint /home/rllab2/zac-models/mtd_90_eth_insertion_full2/checkpoints/040000/pretrained_model \
  --device cuda \
  --host 0.0.0.0 \
  --port 8000 \
  --rtc \
  --num-inference-steps 20
```

The server reads the MultiTaskDiT objective from the checkpoint. For a
flow-matching checkpoint, RTC guides the native flow integration directly; for
a diffusion checkpoint, it uses the training-free diffusion-to-flow adapter.

## 2. Robot machine

```bash
robot/yam canup
robot/yam dagger --mirror-kp 0.2
```

## 3. Workstation

Replace the IP addresses and camera serial numbers:

```bash
workstation/yam-data deploy \
  --rtc \
  --rtc-min-execute-steps 4 \
  --robot-host <ROBOT_IP> \
  --policy-host <POLICY_SERVER_IP> \
  --policy-port 8000 \
  --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn> \
  --task "Insert the Ethernet cable."
```

The checkpoint plans 32 actions and exposes a 24-action execution chunk. At
30 Hz, the measured 20-step RTC inference delay was about 3 control ticks;
the broker starts at 4 execute steps and then calibrates from live latency.
Keep `--num-inference-steps 20`: it controls native flow integration steps (or
converted-flow steps for a diffusion checkpoint), not chunk overlap.

## Inference diagnostics

The Ethernet-insertion checkpoint was benchmarked on an NVIDIA RTX 5090 at a
30 Hz control rate (one control tick is 33.33 ms). These are local model-side
timings; the live broker measures end-to-end request time, including transport.

| Inference path | Mean | p95 | Maximum | Delay at 30 Hz |
| --- | ---: | ---: | ---: | ---: |
| Non-RTC diffusion, 20 steps | 40.58 ms | — | — | about 2 ticks |
| RTC converted flow, 20 steps | 90.15 ms | 95.24 ms | 96.24 ms | 3 ticks |
| RTC converted flow, 100 steps | 356.15 ms | 385.64 ms | 388.93 ms | 11–12 ticks |

Checkpoint loading took about 2.615 seconds. The first RTC inference took about
197 ms, while CUDA kernels and caches were warming up, and peak allocated CUDA
memory was about 1,184.5 MiB. The broker performs this first prediction
synchronously and excludes it from the rolling latency estimate.

With the recommended 20 steps, the measured delay is `d = 3`. Using
`min_execute_steps = 4` gives `s = 4` and satisfies RTC's feasibility condition
`d <= s <= H - d` for the 24-action execution horizon. This replans after 4
executed actions, leaving 20 overlapping actions: 3 have full prefix weight, 17
use exponential blending, and the final 4 are unconstrained new actions.

The 100-step setting is not recommended for deployment. Its p95 and maximum
latencies round up to `d = 12`; with `H = 24`, this leaves only `s = 12` and no
timing margin. The live broker starts from the configured minimum and then uses
the conservative maximum of its 10 most recent end-to-end latency measurements,
so network or server slowdowns are included automatically.
