"""Recorder data-collection (mock) + DAgger-source assembly + outcome sidecar tests."""

from __future__ import annotations

import json
import time

import numpy as np

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.dataset_writer import dataset_dir
from workstation.lerobot_recorder.portal_bridge import PortalBridge
from workstation.lerobot_recorder.recorder import Recorder


def test_recorder_start_failure_releases_hardware(tmp_path, monkeypatch):
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), mock=True)
    rec = Recorder(cfg)
    calls = []
    monkeypatch.setattr(rec.cameras, "start", lambda: calls.append("cameras.start"))
    monkeypatch.setattr(rec.cameras, "stop", lambda: calls.append("cameras.stop"))
    monkeypatch.setattr(rec.robot, "start", lambda: calls.append("robot.start"))
    monkeypatch.setattr(rec.robot, "stop", lambda: calls.append("robot.stop"))

    def fail_writer():
        raise RuntimeError("dataset initialization failed")

    monkeypatch.setattr(rec, "_open_writer", fail_writer)
    try:
        rec.start()
    except RuntimeError as exc:
        assert str(exc) == "dataset initialization failed"
    else:
        raise AssertionError("startup should fail")

    assert calls == ["cameras.start", "robot.start", "robot.stop", "cameras.stop"]
    assert rec.writer is None


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


def test_manual_save_finalizes_and_next_episode_reopens_writer(tmp_path):
    cfg = RecorderConfig(
        repo_id="test/manual", root=str(tmp_path), fps=60, mock=True, review_before_save=False
    )
    rec = Recorder(cfg)
    rec.start()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec.save_dataset()
        first_writer = rec.writer
        assert first_writer.finalized
        assert first_writer.num_episodes == 1

        rec._episode = [rec._sample_frame()]
        rec._submit("fail")
        assert rec.writer is not first_writer
        rec.save_dataset()
    finally:
        rec.shutdown()

    lines = (tmp_path / "manual" / "outcomes.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["episode"] for row in rows] == [0, 1]
    assert [row["outcome"] for row in rows] == ["success", "fail"]


def test_manual_save_preserves_armed_idle_state(tmp_path):
    cfg = RecorderConfig(
        repo_id="test/armed_save", root=str(tmp_path), fps=60, mock=True, review_before_save=False
    )
    rec = Recorder(cfg)
    rec.writer = rec._open_writer()
    rec.arm()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec.save_dataset()
        st = rec.get_status()
        assert rec.gate.armed is True
        assert st["armed"] is True
        assert st["recording"] is False
    finally:
        rec.shutdown()


def test_status_reports_dataset_total_across_sessions(tmp_path):
    # session 1: fresh dataset — total grows with the saves
    cfg = RecorderConfig(repo_id="test/total", root=str(tmp_path), fps=60, mock=True, review_before_save=False)
    rec = Recorder(cfg)
    rec.start()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec.save_dataset()
        assert rec.writer.total_episodes == 2
        assert rec.writer.new_episodes == 2
        assert rec.get_status()["episodes_total"] == 2
    finally:
        rec.shutdown()

    # session 2: resume — the dashboard total starts at the EXISTING count, not 0
    cfg2 = RecorderConfig(
        repo_id="test/total", root=str(tmp_path), fps=60, mock=True, review_before_save=False, resume=True
    )
    rec2 = Recorder(cfg2)
    rec2.start()
    try:
        assert rec2.writer.total_episodes == 2  # before any new episode this session
        assert rec2.writer.new_episodes == 0
        rec2._episode = [rec2._sample_frame()]
        rec2._submit("fail")
        rec2.save_dataset()
        assert rec2.writer.total_episodes == 3
        assert rec2.writer.new_episodes == 1
        assert rec2.get_status()["episodes_total"] == 3
    finally:
        rec2.shutdown()


def test_status_reports_dataset_outcome_totals_across_sessions(tmp_path):
    # session 1: one success + one fail
    cfg = RecorderConfig(repo_id="test/outcomes", root=str(tmp_path), fps=60, mock=True, review_before_save=False)
    rec = Recorder(cfg)
    rec.start()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec._episode = [rec._sample_frame()]
        rec._submit("fail")
        rec.save_dataset()
        assert rec.writer.outcome_totals == {"success": 1, "fail": 1}
        st = rec.get_status()
        assert st["success_total"] == 1 and st["fail_total"] == 1
    finally:
        rec.shutdown()

    # session 2: resume — totals seed from the dataset's outcome sidecar, then grow
    cfg2 = RecorderConfig(
        repo_id="test/outcomes", root=str(tmp_path), fps=60, mock=True, review_before_save=False, resume=True
    )
    rec2 = Recorder(cfg2)
    rec2.start()
    try:
        assert rec2.writer.outcome_totals == {"success": 1, "fail": 1}  # before recording anything
        rec2._episode = [rec2._sample_frame()]
        rec2._submit("success")
        rec2.save_dataset()
        assert rec2.writer.outcome_totals == {"success": 2, "fail": 1}
        st = rec2.get_status()
        assert st["success_total"] == 2 and st["fail_total"] == 1
    finally:
        rec2.shutdown()


def test_streaming_encoding_kwarg_flows_to_writer(tmp_path):
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter

    cfg = RecorderConfig(repo_id="t/stream", root=str(tmp_path), mock=True, streaming_encoding=True)
    w = AsyncDatasetWriter(cfg, [], {})
    assert w._create_encoding_kwargs().get("streaming_encoding") is True
    assert w._dataset_encoding_kwargs().get("streaming_encoding") is True  # resume path too

    cfg_off = RecorderConfig(repo_id="t/nostream", root=str(tmp_path), mock=True)
    w_off = AsyncDatasetWriter(cfg_off, [], {})
    # omitted when disabled so stock lerobot builds (without the kwarg) keep working
    assert "streaming_encoding" not in w_off._create_encoding_kwargs()


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
            "fine_grained": True,
            "left": pose,
            "right": pose,
            "t": 2.0,
        }
    )
    assert snap["policy_running"] is True
    assert snap["dagger_state"] == "policy"
    assert snap["last_dagger_event"] == {"seq": 3, "action": "keep"}
    assert snap["fine_grained"] is True


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
