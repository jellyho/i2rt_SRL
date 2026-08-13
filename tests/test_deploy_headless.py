"""Headless deployment — the deploy loop without Qt (replaces the old `yam-data bridge`)."""

from __future__ import annotations

import threading
import time

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.deploy_headless import run_headless
from workstation.policy_bridge.config import BridgeConfig


def _stop_soon(delay=0.8):
    """Ctrl-C the run loop from another thread so the test terminates."""
    import os
    import signal

    threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGINT)).start()


def test_headless_runs_and_shuts_down_cleanly(tmp_path):
    cfg = RecorderConfig(repo_id="test/hl", root=str(tmp_path), fps=60, mock=True,
                         record_source="deploy")
    _stop_soon()
    rc = run_headless(cfg, BridgeConfig(), status_period=0.2)
    assert rc == 0
    assert not (tmp_path / "hl").exists()  # deploy source: nothing written


def test_headless_records_when_the_source_says_so(tmp_path):
    """The old bridge could not record at all; headless deploy inherits it from Recorder."""
    cfg = RecorderConfig(repo_id="test/hlrec", root=str(tmp_path), fps=60, mock=True,
                         record_source="eval", review_before_save=False)
    _stop_soon(1.2)
    rc = run_headless(cfg, BridgeConfig(), status_period=0.3)
    assert rc == 0
    # Ctrl-C has to close the rollout — the GUI would use "Stop collection".
    assert (tmp_path / "hlrec" / "outcomes.jsonl").exists()


def test_headless_reports_the_wrong_robot_mode_instead_of_hanging(tmp_path, monkeypatch):
    from workstation.lerobot_recorder import deploy_headless as dh

    class WrongMode:
        def __init__(self, cfg):
            self.cfg = cfg

        def start(self):
            raise RuntimeError("is running in 'teleop' mode, but this needs 'deploy'")

        def shutdown(self):
            pass

        def set_policy_running(self, flag):
            pass

    monkeypatch.setattr(dh, "Recorder", WrongMode)
    cfg = RecorderConfig(repo_id="test/hlbad", root=str(tmp_path), mock=True, record_source="deploy")
    assert run_headless(cfg, BridgeConfig()) == 1  # non-zero, not a hang


def test_autostart_can_be_disabled(tmp_path, monkeypatch):
    """Headless starts the arms on launch by default; --no-autostart must really skip it."""
    from workstation.lerobot_recorder import deploy_headless as dh

    calls = []
    real = dh.Recorder

    class Spy(real):
        def set_policy_running(self, flag):
            calls.append(bool(flag))
            super().set_policy_running(flag)

    monkeypatch.setattr(dh, "Recorder", Spy)
    cfg = RecorderConfig(repo_id="test/hlauto", root=str(tmp_path), fps=60, mock=True,
                         record_source="deploy")
    _stop_soon(0.6)
    run_headless(cfg, BridgeConfig(), autostart=False, status_period=0.2)
    assert True not in calls  # never asked the robot to start driving
    assert calls and calls[-1] is False  # but did stop it on the way out


def test_headless_disables_review_since_nobody_can_review(tmp_path):
    """review_before_save would park every rollout awaiting a Keep/Delete that can never
    come, silently dropping the whole run."""
    cfg = RecorderConfig(repo_id="test/hlrev", root=str(tmp_path), fps=60, mock=True,
                         record_source="eval", review_before_save=True)
    _stop_soon(1.2)
    assert run_headless(cfg, BridgeConfig(), status_period=0.3) == 0
    assert cfg.review_before_save is False
    assert (tmp_path / "hlrev" / "outcomes.jsonl").exists()  # saved, not held pending
