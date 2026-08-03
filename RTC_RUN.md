# Run MultiTaskDiT with RTC

Run these commands from `/home/rllab2/i2rt_rllab-mtd-rtc`.

## 1. Policy server (GPU machine)

First-time setup:

```bash
bash policy_serving/setup_policy_env.sh
```

Start the policy server:

```bash
source policy_serving/.venv/bin/activate

yam-lerobot-serve \
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
