"""Recorder data-collection (mock) + DAgger-source assembly + outcome sidecar tests."""

from __future__ import annotations

import json
import time

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import dataset_dir
from workstation.lerobot_recorder.portal_bridge import PortalBridge
from workstation.lerobot_recorder.recorder import Recorder


def test_recorder_records_episode_and_outcome(tmp_path):
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), fps=60, mock=True)
    rec = Recorder(cfg)
    rec.start()
    rec.arm()

    captured, seen = False, set()
    t0 = time.time()
    while time.time() - t0 < 10:
        st = rec.get_status()
        seen.add(st["teleop"])
        if st["pending"]:
            captured = True
            rec.keep_episode(outcome="success")  # submits to the async writer queue
            break
        time.sleep(0.05)
    rec.shutdown()  # drains the queue + finalizes

    assert captured, "gate never produced a pending episode"
    assert "ENGAGED" in seen and "IDLE" in seen
    assert rec.writer.num_episodes >= 1  # worker saved it off the queue
    final = rec.get_status()
    assert final["kept"] >= 1 and final["success"] >= 1  # live stats counted the keep
    assert final["robot_ok"] is True  # mock bridge reports connected

    # the dataset (and its outcomes sidecar) lives at <root>/<name>
    sidecar = tmp_path / "yam" / "outcomes.jsonl"
    assert sidecar.exists()
    assert dataset_dir(str(tmp_path), "test/yam") == str(tmp_path / "yam")
    entry = json.loads(sidecar.read_text().splitlines()[0])
    assert entry["outcome"] == "success"
    assert entry["episode"] == 0


def test_eval_rollout_records_from_arm_to_disarm(tmp_path):
    cfg = RecorderConfig(
        repo_id="test/eval", root=str(tmp_path), fps=60, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(1.0)  # accumulate a rollout
    frames = rec.get_status()["frames"]
    rec.disarm()  # eval: ends the rollout and submits it
    rec.shutdown()
    assert frames > 0
    assert rec.writer.num_episodes >= 1
    assert (tmp_path / "eval" / "outcomes.jsonl").exists()


def test_control_mode_in_frame():
    cfg = RecorderConfig(record_source="teleop", mock=False)
    rec = Recorder(cfg)
    snap = {
        "state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "control_mode": 2,
    }
    frame = rec._frame({"agentview": np.zeros((4, 4, 3), np.uint8)}, snap)
    assert frame["observation.control_mode"].tolist() == [2.0]
    assert frame["observation.state"].shape == (42,)
    assert frame["observation.leader"].shape == (12,)
    assert frame["observation.eef"].shape == (14,)  # zeros if the robot can't FK
    assert frame["action"].shape == (14,)
    assert "agentview" in frame["images"]


def test_dagger_source_assembly():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    human_l, human_r = np.arange(7, dtype=float), np.arange(7, 14, dtype=float)
    pose = {"pos": [0.0] * 7, "vel": [0.0] * 7, "eff": [0.0] * 7}

    intervening = {
        "intervention": True,
        "left": {**pose, "human": human_l.tolist()},
        "right": {**pose, "human": human_r.tolist()},
        "t": 1.0,
    }
    snap = bridge._assemble(intervening)
    assert snap["teleop_state"] == "ENGAGED"
    assert snap["action"] is not None and snap["action"].shape == (14,)
    assert np.allclose(snap["action"], np.concatenate([human_l, human_r]))

    idle = {"intervention": False, "left": pose, "right": pose, "t": 2.0}
    snap_idle = bridge._assemble(idle)
    assert snap_idle["teleop_state"] == "IDLE"
    assert snap_idle["action"] is None  # not intervening -> nothing to record


def test_dagger_snapshot_carries_state_and_event():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    pose = {"pos": [0.0] * 7, "vel": [0.0] * 7, "eff": [0.0] * 7}
    snap = bridge._assemble(
        {
            "intervention": False,
            "policy_running": True,
            "homing": False,
            "dagger_state": "policy",
            "last_dagger_event": {"seq": 3, "action": "keep"},
            "left": pose,
            "right": pose,
            "t": 2.0,
        }
    )
    assert snap["policy_running"] is True
    assert snap["dagger_state"] == "policy"
    assert snap["last_dagger_event"] == {"seq": 3, "action": "keep"}


def test_dagger_recorder_events_do_not_use_expert_button_map():
    cfg = RecorderConfig(record_source="dagger", mock=False, button_map={"left.0": "discard"})
    rec = Recorder(cfg)
    rec.gate.arm()
    rec.gate.update("ENGAGED")
    rec._scan_buttons({"buttons": {"left": [1]}})
    assert rec._btn_outcome is None

    rec._scan_dagger_event({"last_dagger_event": {"seq": 1, "action": "keep"}})
    assert rec._btn_outcome == "keep"
    rec._scan_dagger_event({"last_dagger_event": {"seq": 2, "action": "discard"}})
    assert rec._btn_outcome == "discard"
