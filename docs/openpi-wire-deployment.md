# OpenPI wire-insertion deployment

The live boundary is `yam_bimanual_v1`: the workstation sends a 14D unnormalized
position state, three 224×224 RGB `uint8` images, and a prompt. OpenPI remains
installed only in the GPU policy environment. The robot and workstation install
I2RT plus the lightweight `yam-policy` package.

## One-time GPU environment setup

```bash
cd /home/rllab3/openpi
uv pip install -e /home/rllab3/i2rt_rllab/policy_serving
```

## Launch order

GPU policy host:

```bash
cd /home/rllab3/i2rt_rllab
CHECKPOINT_DIR=$HOME/zac-models/wire-insert-jul16 \
  policy_serving/launch_openpi_wire.sh
```

Robot machine, simulation first:

```bash
robot/yam dagger --sim --max-joint-speed 0.2 --command-timeout 0.25
```

Workstation, synchronous and unarmed first:

```bash
python -m workstation.policy_bridge \
  --config configs/wire_insertion_policy.example.yaml \
  --mock --no-async \
  --prompt "Insert the USB-C plug into the USB-C port."
```

The headless bridge stays `READY` and sends no action until restarted with
`--arm`. The DAgger deployment UI uses its Start Policy control as the arm gate.
For a real workstation configuration, use the same command with a local config
containing the robot endpoint and camera serials, without `--mock`.

If the policy runs on a remote compute node, bind it to `0.0.0.0:8000` and expose
it only through an SSH local forward:

```bash
ssh -N -L 8000:<compute-node>:8000 <login-host>
```

Tunnel loss, invalid actions, inference errors, camera loss/staleness, and e-stop
all stop workstation commands; the robot-side command watchdog then holds.

## Hardware gate

Before using `--arm` against real hardware, the local config must define reviewed
follower joint/gripper limits and an effort threshold. Start at execution horizon
1–4 and a conservative `--max-joint-speed`, verify physical e-stop and takeover,
then increase the horizon only after reviewing recorded motion and timing.
