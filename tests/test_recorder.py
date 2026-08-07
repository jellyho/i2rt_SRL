"""Recorder data-collection (mock) + DAgger-source assembly + outcome sidecar tests."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

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


def test_recenter_pauses_appends_without_closing_episode():
    cfg = RecorderConfig(record_source="teleop", mock=True)
    rec = Recorder(cfg)
    rec.writer = SimpleNamespace(
        num_episodes=0,
        total_episodes=0,
        outcome_totals={"success": 0, "fail": 0},
        queue_depth=0,
        low_disk=False,
        progress={"saving": False, "queued": 0},
    )
    rec.gate.arm()
    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}
    snap = {
        "teleop_state": "ENGAGED",
        "state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "control_mode": 0,
        "buttons": {},
        "leader_recentering": False,
    }

    rec._step(images, snap)
    assert len(rec._episode) == 1
    assert rec.gate.recording is True

    snap["leader_recentering"] = True
    rec._step(images, snap)
    rec._step(images, snap)
    assert len(rec._episode) == 1  # no frame, state, or action => no dataset time
    assert rec.gate.recording is True  # same episode remains open internally
    assert rec.get_status()["recording"] is False

    snap["leader_recentering"] = False
    rec._step(images, snap)
    assert len(rec._episode) == 2
    assert rec.get_status()["recording"] is True


def test_dagger_source_assembly():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    human_l, human_r = np.arange(7, dtype=float), np.arange(7, 14, dtype=float)
    applied_l, applied_r = human_l + 20, human_r + 20
    pose = {"pos": [0.0] * 7, "vel": [0.0] * 7, "eff": [0.0] * 7}

    intervening = {
        "intervention": True,
        "policy_running": True,
        "left": {**pose, "human": human_l.tolist(), "applied": applied_l.tolist()},
        "right": {**pose, "human": human_r.tolist(), "applied": applied_r.tolist()},
        "t": 1.0,
    }
    snap = bridge._assemble(intervening)
    assert snap["teleop_state"] == "ENGAGED"
    assert snap["action"] is not None and snap["action"].shape == (14,)
    assert np.allclose(snap["action"], np.concatenate([applied_l, applied_r]))
    assert snap["control_mode"] == 2

    policy = {
        "intervention": False,
        "policy_running": True,
        "left": {**pose, "applied": human_l.tolist()},
        "right": {**pose, "applied": human_r.tolist()},
        "t": 2.0,
    }
    snap_policy = bridge._assemble(policy)
    assert snap_policy["teleop_state"] == "ENGAGED"
    assert np.allclose(snap_policy["action"], np.concatenate([human_l, human_r]))
    assert snap_policy["control_mode"] == 1

    stopped = {"intervention": False, "policy_running": False, "left": pose, "right": pose, "t": 3.0}
    snap_stopped = bridge._assemble(stopped)
    assert snap_stopped["teleop_state"] == "IDLE"
    assert snap_stopped["action"] is None


def test_portal_bridge_queues_policy_action_for_its_client_thread():
    bridge = PortalBridge(RecorderConfig(record_source="dagger", mock=False))
    action = {"left": np.arange(7), "right": np.arange(7, 14)}

    bridge.set_policy_action(action)
    action["left"][0] = 99

    assert bridge._policy_action_seq == 1
    assert bridge._policy_action_req is not None
    assert bridge._policy_action_req["left"][0] == 0

    bridge.set_policy_running(False)
    assert bridge._policy_action_req is None


def test_portal_bridge_reissues_ui_request_after_handle_changed_robot_state():
    bridge = PortalBridge(RecorderConfig(record_source="dagger", mock=False))
    bridge._policy_running_req = True
    bridge._policy_running_sent = True

    bridge.set_policy_running(True)

    assert bridge._policy_running_req is True
    assert bridge._policy_running_sent is None


def test_finish_clears_policy_action_and_latched_rollout_request():
    bridge = PortalBridge(RecorderConfig(record_source="dagger", mock=False))
    bridge.set_policy_running(True)
    bridge.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})

    bridge.finish_dagger_run("keep")

    assert bridge._finish_req == "keep"
    assert bridge._policy_running_req is False
    assert bridge._intervention_req is False
    assert bridge._policy_action_req is None


def test_dagger_records_one_rollout_across_interventions():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    rec = Recorder(cfg)
    submitted = []
    rec.writer = SimpleNamespace(
        num_episodes=0,
        total_episodes=0,
        outcome_totals={"success": 0, "fail": 0},
        queue_depth=0,
        low_disk=False,
        progress={"saving": False, "queued": 0},
        finalized=False,
        submit=lambda frames, outcome, task: submitted.append((list(frames), outcome, task)),
    )
    rec.gate.arm()
    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}

    def snap(*, running=True, intervention=False, mode=1, event=None):
        return {
            "teleop_state": "ENGAGED" if running else "IDLE",
            "state": np.zeros(42, np.float32),
            "action": np.full(14, mode, np.float32) if running else None,
            "leader": np.zeros(12, np.float32),
            "eef": np.zeros(14, np.float32),
            "control_mode": mode,
            "buttons": {},
            "intervention": intervention,
            "leader_recentering": False,
            "last_dagger_event": event,
        }

    rec._step(images, snap())
    rec._step(images, snap(intervention=True, mode=2))
    rec._step(images, snap())
    assert rec.gate.recording is True
    assert len(rec._episode) == 3
    assert [int(f["observation.control_mode"][0]) for f in rec._episode] == [1, 2, 1]
    assert rec.get_status()["interventions"] == 1

    rec._step(images, snap(running=False, event={"seq": 1, "action": "keep"}))
    assert rec.gate.recording is False
    assert rec._pending is False
    assert rec._btn_outcome == "keep"
    assert len(submitted) == 1
    assert len(submitted[0][0]) == 3
    assert submitted[0][1] == "keep"


def test_dagger_discard_drops_the_complete_rollout():
    cfg = RecorderConfig(record_source="dagger", mock=False, review_before_save=False)
    rec = Recorder(cfg)
    rec.writer = SimpleNamespace(
        num_episodes=0,
        total_episodes=0,
        outcome_totals={"success": 0, "fail": 0},
        queue_depth=0,
        low_disk=False,
        progress={"saving": False, "queued": 0},
    )
    rec.gate.arm()
    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}
    base = {
        "state": np.zeros(42, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "buttons": {},
        "intervention": False,
        "leader_recentering": False,
    }
    rec._step(images, {**base, "teleop_state": "ENGAGED", "action": np.zeros(14), "control_mode": 1})
    rec._step(
        images,
        {
            **base,
            "teleop_state": "IDLE",
            "action": None,
            "control_mode": 1,
            "last_dagger_event": {"seq": 1, "action": "discard"},
        },
    )
    assert rec._episode == []
    assert rec.get_status()["discarded"] == 1
    assert rec.get_status()["kept"] == 0


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
            "leader_recentering": True,
            "recenter_fault": False,
            "left": pose,
            "right": pose,
            "t": 2.0,
        }
    )
    assert snap["policy_running"] is True
    assert snap["dagger_state"] == "policy"
    assert snap["last_dagger_event"] == {"seq": 3, "action": "keep"}
    assert snap["fine_grained"] is True
    assert snap["leader_recentering"] is True
    assert snap["recenter_fault"] is False


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


# --------------------------------------------------------------- deploy (no recording)
def test_deploy_source_opens_no_dataset(tmp_path):
    """`deploy` runs the policy but must not create a dataset anywhere on disk."""
    cfg = RecorderConfig(repo_id="test/deployonly", root=str(tmp_path), fps=60, mock=True,
                         record_source="deploy")
    rec = Recorder(cfg)
    rec.start()
    time.sleep(0.5)  # let the loop run — in dagger/eval this would be buffering frames
    st = rec.get_status()
    rec.shutdown()

    assert rec.writer is None  # no writer was ever opened
    assert not (tmp_path / "deployonly").exists()  # and nothing was written to disk
    assert st["frames"] == 0 and st["recording"] is False and st["armed"] is False
    assert st["robot_ok"] is True  # the robot link is live all the same


def test_deploy_source_ignores_arm_and_save(tmp_path):
    """Arming/saving are meaningless without a dataset; they must be safe no-ops, not crashes."""
    cfg = RecorderConfig(repo_id="test/deploynoop", root=str(tmp_path), fps=60, mock=True,
                         record_source="deploy")
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.4)
    armed = rec.get_status()["armed"]
    rec.save_dataset()  # would raise/AttributeError if it reached the writer
    rec.disarm()
    rec.shutdown()

    assert armed is False  # arm() did not arm anything
    assert rec.writer is None
    assert not (tmp_path / "deploynoop").exists()


def test_deploy_source_still_reports_rollout_state(tmp_path):
    """The UI needs policy/intervention state in deploy mode — that is the whole point."""
    cfg = RecorderConfig(repo_id="test/deploystate", root=str(tmp_path), fps=60, mock=True,
                         record_source="deploy")
    rec = Recorder(cfg)
    snap = {
        "teleop_state": "IDLE", "state": np.zeros(4), "action": np.zeros(4),
        "policy_running": True, "intervention": True, "dagger_state": "intervention",
        "fine_grained": False, "leader_recentering": False, "recenter_fault": False,
        "homing": False, "estop": False, "buttons": {},
    }
    rec._step({}, snap)
    st = rec.get_status()
    assert st["policy_running"] is True
    assert st["intervention"] is True
    assert st["dagger_state"] == "intervention"
    assert st["frames"] == 0  # ...but still nothing buffered


def test_deploy_source_does_not_use_expert_button_map():
    """Handle buttons drive the robot's rollout state machine here, not outcome labels."""
    cfg = RecorderConfig(record_source="deploy", mock=False, button_map={"left.0": "discard"})
    rec = Recorder(cfg)
    rec._scan_buttons({"buttons": {"left": [1]}})
    assert rec._btn_outcome is None


# ------------------------------------------------------- robot-server mode mismatch
def test_wrong_robot_mode_refuses_to_start_with_an_actionable_message(tmp_path, monkeypatch):
    """A crossed robot server otherwise fails SILENTLY (a teleop server just ignores
    policy actions), so starting must refuse and name the command to run."""
    cfg = RecorderConfig(repo_id="test/mode", root=str(tmp_path), mock=False,
                         record_source="deploy", expected_robot_mode="deploy")
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: "teleop"))

    try:
        rec.start()
    except RuntimeError as exc:
        assert "'teleop'" in str(exc) and "'deploy'" in str(exc)
        assert "robot/yam deploy" in str(exc)
    else:
        raise AssertionError("a teleop server must not satisfy a deployment session")


def test_matching_robot_mode_starts_normally(tmp_path, monkeypatch):
    cfg = RecorderConfig(repo_id="test/modeok", root=str(tmp_path), mock=False,
                         record_source="deploy", expected_robot_mode="deploy")
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: "deploy"))
    rec.start()
    rec.shutdown()
    assert rec.writer is None  # deploy source: still no dataset


def test_unknown_robot_mode_does_not_block_startup(tmp_path, monkeypatch):
    """An older server that reports no mode must not become un-startable."""
    cfg = RecorderConfig(repo_id="test/modenone", root=str(tmp_path), mock=False,
                         record_source="deploy", expected_robot_mode="deploy")
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: None))
    rec._check_robot_mode(timeout=0.1)  # warns, does not raise
    rec.shutdown()


def test_mock_sessions_skip_the_mode_check(tmp_path):
    cfg = RecorderConfig(repo_id="test/modemock", root=str(tmp_path), mock=True,
                         record_source="deploy", expected_robot_mode="deploy")
    rec = Recorder(cfg)
    rec.start()  # mock has no robot at all — the check must not fire
    rec.shutdown()


def test_older_robot_reporting_the_pre_rename_mode_is_accepted(tmp_path, monkeypatch):
    """The controller was renamed dagger -> deploy once it also served plain deployment.
    An un-updated robot server still says "dagger"; that skew must not read as a mismatch."""
    cfg = RecorderConfig(repo_id="test/modeold", root=str(tmp_path), mock=False,
                         record_source="deploy", expected_robot_mode="deploy")
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: "dagger"))
    rec.start()  # must not raise
    rec.shutdown()


# --------------------------------------------------------------- memory guard
def test_low_memory_ends_the_episode_and_saves_it(tmp_path, monkeypatch):
    """Being OOM-killed mid-write is what leaves a half-written parquet the dataset cannot
    open, so the recorder has to stop itself first -- keeping the frames recorded so far."""
    cfg = RecorderConfig(repo_id="test/ram", root=str(tmp_path), fps=60, mock=True,
                         record_source="eval", review_before_save=False, min_free_ram_gb=1000.0)
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    t0 = time.time()
    while time.time() - t0 < 10 and not rec.get_status().get("low_ram"):
        time.sleep(0.05)
    st = rec.get_status()
    rec.shutdown()

    assert st["low_ram"] is True, "the guard never fired"
    assert st["armed"] is False and st["recording"] is False
    assert rec.writer.num_episodes >= 1, "the buffered frames must be saved, not dropped"
    assert (tmp_path / "ram" / "outcomes.jsonl").exists()


def test_guard_is_off_when_the_threshold_is_zero(tmp_path):
    cfg = RecorderConfig(repo_id="test/ramoff", root=str(tmp_path), fps=60, mock=True,
                         record_source="eval", review_before_save=False, min_free_ram_gb=0.0)
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.6)
    st = rec.get_status()
    rec.disarm()
    rec.shutdown()
    assert st["low_ram"] is False and st["recording"] is True


def test_available_ram_is_read_from_proc(tmp_path):
    """A plausible number, and inf rather than a crash where /proc/meminfo is absent."""
    free = Recorder.available_ram_gb()
    assert free > 0
    if free != float("inf"):
        assert free < 10_000


def test_guard_does_not_fire_with_ample_memory(tmp_path):
    cfg = RecorderConfig(repo_id="test/ramok", root=str(tmp_path), fps=60, mock=True,
                         record_source="eval", review_before_save=False, min_free_ram_gb=0.001)
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.6)
    st = rec.get_status()
    rec.disarm()
    rec.shutdown()
    assert st["low_ram"] is False
