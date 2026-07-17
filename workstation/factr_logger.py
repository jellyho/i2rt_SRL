"""FACTR 2 free-space logger — high-rate proprioception capture for NEXT training.

This is a SEPARATE tool from ``workstation.lerobot_recorder`` (that one is left
untouched). It reuses the same portal architecture (``RobotClient`` polling the YAM
robot server) but, instead of a camera-clocked LeRobot dataset, it saves the raw
per-joint signals the FACTR 2 / NEXT external-torque estimator needs:

    NEXT input  x_i = [q, q̇, Δq_d]   over a 50-step history   (Δq_d = q_d - q)
    NEXT target τ_m  = measured motor torque   (free-space, no contact)

Signal mapping on YAM (DM motors report *torque* directly over CAN, so no K·I step):

    q     -> follower ``pos``       (rad)
    q̇     -> follower ``vel``       (rad/s)
    τ_m   -> follower ``eff``       (Nm; DM feedback torque — the NEXT label)
    q_d   -> follower ``applied``   (rad; rate-limited command actually sent)
    current (derived) = eff / KT    (DM motors do NOT report raw current; this is a
                                     KT-scaled convenience column, KT=1.0 by default)

Why portal-polling and not a direct on-robot loop: the robot server owns the CAN
bus, so a second process can't open the motors. The server's control loop runs at
120 Hz and refreshes the snapshot at that rate — above the paper's 100 Hz — so
polling ``get_observation()`` and de-duplicating on the robot timestamp ``t`` yields
clean ~120 Hz data without touching the running teleop/wrapper server.

Collect FREE-SPACE only (no contact): teleoperate the follower through its whole
workspace by hand — each joint through its range, multi-joint sweeps, slow AND fast,
~10 minutes per arm — while the robot runs ``robot/yam teleop``. See the printed
guide at startup.

    python -m workstation.factr_logger --out ~/factr_data/freespace.parquet
    python -m workstation.factr_logger --duration 600            # auto-stop after 10 min
    python -m workstation.factr_logger --mock                    # no robot; synthetic data

Output: one ``.parquet`` (columns follow the recorder's ``{arm}.field.{i}`` naming)
plus a ``.meta.json`` sidecar (rate stats, dof, kt, mode). Falls back to ``.npz`` if
pyarrow is unavailable. Compute ``Δq_d = cmd - pos`` at training time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from i2rt.serving.rig_config import Resolver, load_rig

# Keep in step with the recorder's dimensions (workstation/lerobot_recorder/config.py).
ARMS = ("left", "right")
ARM_DOF = 7  # 6 arm joints + trailing gripper; NEXT should train on joints 0..5 only
NEXT_ARM_JOINTS = 6  # gripper (index 6) is excluded from external-torque estimation
FIELDS = ("pos", "vel", "eff", "cmd", "cur")  # cmd = applied (q_d), cur = eff/KT


# --------------------------------------------------------------------------- config
def _unique_out(path: str) -> str:
    """Never overwrite an existing capture: if the target (or its .npz/.meta.json
    siblings) already exists, append _01, _02, … until a free stem is found.

    Two runs with the same --out silently clobbered each other before this guard.
    """
    stem, ext = os.path.splitext(path)
    ext = ext or ".parquet"

    def _taken(base: str) -> bool:
        return any(os.path.exists(base + suffix) for suffix in (ext, ".parquet", ".npz", ".meta.json"))

    if not _taken(stem):
        return stem + ext
    for n in range(1, 1000):
        cand = f"{stem}_{n:02d}"
        if not _taken(cand):
            return cand + ext
    raise SystemExit(f"factr_logger: too many existing captures at {stem}_*; clean up or pass a new --out")


class LoggerConfig:
    """Resolved settings (CLI flag > config.yaml > default), mirroring the recorder."""

    def __init__(self, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
        rig = load_rig(args.config)
        rob = Resolver(args, parser, rig.get("robot", {}))
        self.robot_host: str = rob.get("robot_host", "host")
        self.robot_port: int = int(rob.get("robot_port", "port"))
        self.out: str = _unique_out(os.path.expanduser(args.out))
        self.duration: float = float(args.duration)  # 0 = until Ctrl-C
        self.arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) or ARMS
        self.poll_hz: float = float(args.poll_hz)
        self.min_hz: float = float(args.min_hz)
        self.contact_warn: float = float(args.contact_warn)  # |torque| Nm hinting HARD contact
        self.pipeline: int = max(1, int(args.pipeline))  # RPCs kept in flight to hide RTT
        self.kt = _parse_kt(args.kt)  # scalar or per-joint torque constant
        self.mock: bool = bool(args.mock)


def _parse_kt(text: str) -> np.ndarray:
    vals = [float(x) for x in str(text).split(",") if x.strip()]
    if not vals:
        vals = [1.0]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 1:
        arr = np.full(ARM_DOF, arr[0])
    if arr.size != ARM_DOF:  # allow 6 arm joints -> pad gripper with 1.0
        pad = np.ones(ARM_DOF)
        pad[: min(arr.size, ARM_DOF)] = arr[:ARM_DOF]
        arr = pad
    return arr


# --------------------------------------------------------------------------- sources
def _extract_frame(obs: Dict, arms: Sequence[str]) -> Optional[Dict]:
    """Pull the per-arm follower vectors out of a raw robot snapshot.

    Returns None if a required field is missing (so we don't log a half frame).
    """
    frame: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms:
        side = obs.get(arm)
        if not side:
            return None
        pos, vel, eff, cmd = side.get("pos"), side.get("vel"), side.get("eff"), side.get("applied")
        if pos is None or vel is None or eff is None or cmd is None:
            return None
        vecs = {
            "pos": np.asarray(pos, dtype=np.float64).reshape(-1),
            "vel": np.asarray(vel, dtype=np.float64).reshape(-1),
            "eff": np.asarray(eff, dtype=np.float64).reshape(-1),
            "cmd": np.asarray(cmd, dtype=np.float64).reshape(-1),
        }
        # Drop any frame whose vectors aren't the expected width (mirrors the recorder's
        # _fuse guard) so a short/partial snapshot can't IndexError the row builder and
        # crash collection mid-run — a dropped frame is logged as a rate dip instead.
        if any(v.size != ARM_DOF for v in vecs.values()):
            return None
        frame[arm] = vecs
    return frame


class RobotSource:
    """Poll the live YAM robot server over portal (read-only).

    ``get_observation()`` is a synchronous RPC, so a single blocking call per frame
    caps the rate at 1/RTT (≈72 Hz on a ~14 ms link — below NEXT's 100 Hz). We keep
    ``pipeline`` requests in flight (portal futures) so the network round-trips
    overlap and throughput approaches the server's 120 Hz snapshot rate even on a
    slow/wireless link. Duplicate snapshots (same robot ``t``) are de-duped upstream.
    """

    def __init__(self, cfg: LoggerConfig) -> None:
        import portal

        # Fast TCP preflight so we fail loudly instead of portal blocking forever.
        try:
            with socket.create_connection((cfg.robot_host, cfg.robot_port), timeout=2.0):
                pass
        except OSError as e:
            raise SystemExit(
                f"factr_logger: no robot server on {cfg.robot_host}:{cfg.robot_port} "
                f"— start it with `robot/yam teleop` ({type(e).__name__}: {e})"
            ) from e
        self.client = portal.Client(f"{cfg.robot_host}:{cfg.robot_port}")
        self.client.get_metadata().result(2.0)  # confirm the link (raises on failure)
        self.arms = cfg.arms
        self.depth = cfg.pipeline
        self._timeout = 2.0
        self._inflight: deque = deque()

    def read(self) -> Optional[Dict]:
        # Top up the in-flight queue, then block on the oldest — the others' RTTs
        # were already overlapping while we processed the previous frame.
        while len(self._inflight) < self.depth:
            self._inflight.append(self.client.get_observation())
        obs = self._inflight.popleft().result(self._timeout)
        frame = _extract_frame(obs, self.arms)
        if frame is None:
            return None
        return {
            "t": float(obs.get("t", 0.0)),
            "teleop_state": str(obs.get("teleop_state") or ""),
            "active": bool(obs.get("active", False)),
            "arms": frame,
        }

    def reset(self) -> None:
        """Drop in-flight futures after an error so a hiccup doesn't wedge the queue."""
        self._inflight.clear()


class MockSource:
    """Synthetic free-space motion so the pipeline runs with no robot."""

    def __init__(self, cfg: LoggerConfig) -> None:
        self.arms = cfg.arms
        self._t0 = time.time()

    def read(self) -> Optional[Dict]:
        t = time.time() - self._t0
        arms: Dict[str, Dict[str, np.ndarray]] = {}
        for k, arm in enumerate(self.arms):
            ph = t + k
            j = np.arange(ARM_DOF)
            pos = 0.6 * np.sin(0.7 * ph + j)
            vel = 0.6 * 0.7 * np.cos(0.7 * ph + j)
            # free-space "torque": gravity-ish (function of pos) + small viscous term
            eff = 1.5 * np.cos(pos) + 0.2 * vel
            cmd = pos + 0.01 * np.sin(5 * ph + j)  # tiny tracking error
            arms[arm] = {"pos": pos, "vel": vel, "eff": eff, "cmd": cmd}
        return {"t": t, "teleop_state": "ENGAGED", "active": True, "arms": arms}


# --------------------------------------------------------------------------- writer
def _column_names(arms: Sequence[str]) -> List[str]:
    cols = ["t_robot", "t_wall", "teleop_state", "active"]
    for arm in arms:
        for f in FIELDS:
            cols += [f"{arm}.{f}.{i}" for i in range(ARM_DOF)]
    return cols


def _save(rows: List[Dict], cfg: LoggerConfig, hz_stats: Dict) -> str:
    if not rows:
        raise SystemExit("factr_logger: nothing captured — no frames to save.")
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    cols = _column_names(cfg.arms)
    data: Dict[str, np.ndarray] = {}
    for c in cols:
        if c == "teleop_state":
            data[c] = np.asarray([r[c] for r in rows], dtype=object)
        elif c == "active":
            data[c] = np.asarray([r[c] for r in rows], dtype=bool)
        else:
            data[c] = np.asarray([r[c] for r in rows], dtype=np.float64)

    out = cfg.out
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not out.endswith(".parquet"):
            out = out + ".parquet"
        pq.write_table(pa.table(data), out)
    except Exception as e:  # pyarrow missing / write error -> npz fallback
        out = (cfg.out[:-8] if cfg.out.endswith(".parquet") else cfg.out) + ".npz"
        np.savez_compressed(out, **{k: v for k, v in data.items()})
        print(f"factr_logger: parquet unavailable ({type(e).__name__}); wrote npz instead")

    meta = {
        "tool": "factr_logger",
        "purpose": "FACTR2 / NEXT free-space external-torque estimator training data",
        "arms": list(cfg.arms),
        "arm_dof": ARM_DOF,
        "next_arm_joints": NEXT_ARM_JOINTS,
        "fields": {
            "pos": "joint position q (rad)",
            "vel": "joint velocity q_dot (rad/s)",
            "eff": "motor torque tau_m (Nm) — DM CAN feedback — NEXT TARGET",
            "cmd": "commanded/applied position q_d (rad) — for Delta_qd = cmd - pos",
            "cur": "current = eff / KT (DM gives no raw current; KT-scaled convenience)",
        },
        "kt": cfg.kt.tolist(),
        "robot_host": cfg.robot_host,
        "robot_port": cfg.robot_port,
        "mock": cfg.mock,
        "num_frames": len(rows),
        "rate_hz": hz_stats,
        "note": "Compute NEXT input Delta_qd = cmd - pos; train per arm on joints 0..5.",
    }
    meta_path = (out.rsplit(".", 1)[0]) + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return out


# --------------------------------------------------------------------------- loop
_STOP = {"flag": False}


def _install_sigint() -> None:
    def _handler(_sig, _frame) -> None:  # noqa: ANN001
        _STOP["flag"] = True
        print("\nfactr_logger: stopping, flushing to disk…")

    signal.signal(signal.SIGINT, _handler)


def _print_guide(cfg: LoggerConfig) -> None:
    print("=" * 74)
    print("FACTR 2 free-space logger — capturing NEXT training data")
    print("-" * 74)
    print(f"  robot     : {cfg.robot_host}:{cfg.robot_port}   arms={list(cfg.arms)}   mock={cfg.mock}")
    print(f"  target    : {cfg.poll_hz:.0f} Hz poll, pipeline={cfg.pipeline} (server ~120 Hz; de-duped on robot t)")
    print(f"  output    : {cfg.out}")
    print(f"  duration  : {'until Ctrl-C' if cfg.duration <= 0 else f'{cfg.duration:.0f}s then auto-stop'}")
    print("-" * 74)
    print("  TELEOPERATE THE FOLLOWER THROUGH FREE SPACE (NO CONTACT):")
    print("   • lift the leader to ENGAGE, then sweep each joint across its range + combos")
    print("   • both slow and fast motions, ~10 min total per arm")
    print("   • do NOT touch objects/table/self — contact corrupts the free-space label")
    print(f"   • gravity torque (~20 Nm) is normal; only |torque| > {cfg.contact_warn:.0f} Nm hints HARD contact")
    print("=" * 74)


def run(cfg: LoggerConfig) -> None:
    _print_guide(cfg)
    source = MockSource(cfg) if cfg.mock else RobotSource(cfg)

    rows: List[Dict] = []
    period = 1.0 / max(cfg.poll_hz, 1.0)
    last_t: Optional[float] = None
    t_start = time.time()
    last_report = t_start
    dts: List[float] = []  # inter-frame dt of *unique* frames, for Hz stats
    prev_wall: Optional[float] = None
    warned_contact = False
    n_engaged = 0  # frames where the follower was actually being teleoperated (moving)

    _install_sigint()
    while not _STOP["flag"]:
        loop_t = time.time()
        if cfg.duration > 0 and (loop_t - t_start) >= cfg.duration:
            break
        try:
            snap = source.read()
        except Exception as e:  # link hiccup: report once per change, keep going
            print(f"factr_logger: read error ({type(e).__name__}: {e})", file=sys.stderr)
            if hasattr(source, "reset"):
                source.reset()  # drop stale in-flight futures before retrying
            time.sleep(0.1)
            continue

        if snap is not None and (last_t is None or snap["t"] != last_t):
            last_t = snap["t"]
            wall = loop_t
            row: Dict = {
                "t_robot": snap["t"],
                "t_wall": wall,
                "teleop_state": snap["teleop_state"],
                "active": snap["active"],
            }
            max_tau = 0.0
            for arm in cfg.arms:
                a = snap["arms"][arm]
                cur = a["eff"] / cfg.kt
                for i in range(ARM_DOF):
                    row[f"{arm}.pos.{i}"] = a["pos"][i]
                    row[f"{arm}.vel.{i}"] = a["vel"][i]
                    row[f"{arm}.eff.{i}"] = a["eff"][i]
                    row[f"{arm}.cmd.{i}"] = a["cmd"][i]
                    row[f"{arm}.cur.{i}"] = cur[i]
                max_tau = max(max_tau, float(np.max(np.abs(a["eff"][:NEXT_ARM_JOINTS]))))
            rows.append(row)
            if snap["teleop_state"] == "ENGAGED" or snap["active"]:
                n_engaged += 1
            if prev_wall is not None:
                dts.append(wall - prev_wall)
            prev_wall = wall

            if max_tau > cfg.contact_warn and not warned_contact:
                print(f"  ⚠ CONTACT? |torque|={max_tau:.1f} Nm > {cfg.contact_warn:.0f} — keep it free-space")
                warned_contact = True
            elif max_tau <= cfg.contact_warn:
                warned_contact = False

        # live status once per second
        if loop_t - last_report >= 1.0:
            elapsed = loop_t - t_start
            eff_hz = (len(dts) / sum(dts)) if dts else 0.0
            print(
                f"  t={elapsed:6.1f}s  frames={len(rows):7d}  ~{eff_hz:5.1f} Hz  "
                f"state={rows[-1]['teleop_state'] if rows else '-':8s}",
                end="\r",
                flush=True,
            )
            last_report = loop_t

        # pace the poll a touch faster than the server so we never miss a frame
        sleep = period - (time.time() - loop_t)
        if sleep > 0:
            time.sleep(sleep)

    # ---- summarize + save
    eff_hz = (len(dts) / sum(dts)) if dts else 0.0
    engaged_pct = (100.0 * n_engaged / len(rows)) if rows else 0.0
    hz_stats = {
        "mean_hz": round(eff_hz, 2),
        "min_dt_ms": round(1e3 * min(dts), 2) if dts else None,
        "max_dt_ms": round(1e3 * max(dts), 2) if dts else None,
        "seconds": round(time.time() - t_start, 1),
        "engaged_frames": n_engaged,
        "engaged_pct": round(engaged_pct, 1),
    }
    out = _save(rows, cfg, hz_stats)
    print("\n" + "=" * 74)
    print(f"  saved {len(rows)} frames -> {out}")
    print(f"  effective rate ~{eff_hz:.1f} Hz over {hz_stats['seconds']}s (pipeline={cfg.pipeline})")
    print(f"  engaged (moving) frames: {n_engaged} = {engaged_pct:.0f}%  "
          f"(the rest were IDLE/holding — still valid gravity samples)")
    if eff_hz and eff_hz < cfg.min_hz:
        print(f"  note: {eff_hz:.1f} Hz < {cfg.min_hz:.0f} Hz. This is the robot server's real "
              f"bimanual step rate (step() > 1/rate, so the loop free-runs); the snapshot can't "
              f"go faster and client polling/--pipeline can't beat it. ~73 Hz is fine for NEXT "
              f"(gravity/friction are low-frequency; 0.5 s window ≈ 36 steps). For true 100 Hz "
              f"you'd log inside the robot process at the 250 Hz motor-chain rate.")
    if math.isfinite(eff_hz) and eff_hz >= cfg.min_hz:
        print("  rate OK for NEXT (paper uses 100 Hz).")
    if engaged_pct < 30.0:
        print("  ⚠ few moving frames — teleoperate the follower (lift the leader to ENGAGE) "
              "while logging so it actually sweeps free space.")
    print("  next: Delta_qd = cmd - pos ; train f_theta([q, q_dot, Delta_qd]) -> eff, per arm, joints 0..5")
    print("=" * 74)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="FACTR 2 free-space logger (NEXT training data)")
    p.add_argument("--config", default=None, help="config.yaml (robot host/port); auto-found in repo")
    p.add_argument("--robot-host", dest="robot_host", default="127.0.0.1", help="YAM robot server host")
    p.add_argument("--robot-port", dest="robot_port", type=int, default=11331)
    p.add_argument("--out", default="~/factr_data/freespace.parquet", help="output .parquet path")
    p.add_argument("--duration", type=float, default=0.0, help="seconds to capture (0 = until Ctrl-C)")
    p.add_argument("--arms", default="left,right", help="which follower arms to log (comma list)")
    p.add_argument("--poll-hz", dest="poll_hz", type=float, default=250.0, help="poll rate (de-duped on robot t)")
    p.add_argument("--min-hz", dest="min_hz", type=float, default=90.0, help="warn if effective rate drops below")
    p.add_argument("--pipeline", type=int, default=4,
                   help="portal RPCs kept in flight to hide network RTT (raise if rate < 100 Hz; 1 = blocking)")
    p.add_argument("--contact-warn", dest="contact_warn", type=float, default=35.0,
                   help="|torque| (Nm) hinting HARD contact (free-space gravity torque can reach ~20 Nm)")
    p.add_argument("--kt", default="1.0", help="torque constant for current=eff/KT (scalar or 7 comma values)")
    p.add_argument("--mock", action="store_true", help="synthetic free-space data (no robot)")
    args = p.parse_args(argv)
    run(LoggerConfig(args, p))


if __name__ == "__main__":
    main()
