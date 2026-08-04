"""Headless policy deployment — the deploy UI's run loop without Qt.

This replaces the old ``yam-data bridge``. That was a second, independent implementation
of the same observation→action loop, and keeping two copies had already cost one bug fix
applied twice; worse, the copies decided *what the policy receives*, so drifting apart
would have made a policy behave differently depending on which command launched it.

Here the pieces are the same ones the GUI drives:

* :class:`~workstation.lerobot_recorder.recorder.Recorder` — cameras, robot link, the
  robot-mode check, and (when recording) the dataset writer.
* :class:`~workstation.policy_bridge.deploy_runner.DeploymentPolicyRunner` — builds the
  observation and streams action chunks while the robot says the policy may drive.

So headless gains everything the UI has except the screen, including **recording**, which
the old bridge could not do at all. ``mode`` and ``record`` mean exactly what they mean in
the UI (see :mod:`workstation.lerobot_recorder.deploy_gui`).

    workstation/yam-data deploy --headless
    workstation/yam-data deploy --headless --mode deploy --no-record
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Optional

from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.recorder import Recorder
from workstation.policy_bridge.config import BridgeConfig
from workstation.policy_bridge.deploy_runner import DeploymentPolicyRunner

logger = logging.getLogger(__name__)


def run_headless(
    recorder_cfg: RecorderConfig,
    bridge_cfg: BridgeConfig,
    *,
    leader_mirror: Optional[bool] = None,
    autostart: bool = True,
    status_period: float = 5.0,
) -> int:
    """Run a deployment with no GUI until Ctrl-C. Returns a process exit code.

    ``autostart`` starts the policy rollout as soon as the link is up, which is what the
    old bridge did — there is no button to press here. It is announced loudly because it
    means the arms begin moving on launch; pass ``autostart=False`` to wait for the
    handle button instead.
    """
    recording = recorder_cfg.record_source != "deploy"
    mirror = bool(leader_mirror) if leader_mirror is not None else recorder_cfg.record_source == "dagger"
    if recording and recorder_cfg.review_before_save:
        # Review holds each finished episode for a Keep/Delete nobody is here to give, so
        # it would silently drop every rollout. Auto-save instead.
        logger.info("headless: review_before_save is off (there is no one to review)")
        recorder_cfg.review_before_save = False

    recorder = Recorder(recorder_cfg)
    runner: Optional[DeploymentPolicyRunner] = None
    stop = {"now": False}

    def _sigint(_sig, _frame):
        stop["now"] = True

    previous = signal.signal(signal.SIGINT, _sigint)
    try:
        recorder.start()  # raises with an actionable message on the wrong robot mode
        runner = DeploymentPolicyRunner(bridge_cfg, recorder_cfg, recorder.get_last_images)
        runner.start()
        recorder.set_leader_mirror(mirror)
        logger.info(
            "headless deploy up: robot=%s:%d policy=%s:%d source=%s leader=%s",
            bridge_cfg.robot_host, bridge_cfg.robot_port,
            bridge_cfg.policy_host, bridge_cfg.policy_port,
            recorder_cfg.record_source, "mirroring" if mirror else "free",
        )
        if recording:
            recorder.arm()  # eval/dagger: open the gate so rollouts are captured
        if autostart:
            logger.warning("AUTOSTART: starting the policy now — the arms will begin moving")
            recorder.set_policy_running(True)
        else:
            logger.info("waiting for the handle button (left upper) to start the rollout")

        last = 0.0
        while not stop["now"]:
            time.sleep(0.2)
            now = time.monotonic()
            if now - last >= status_period:
                last = now
                st = recorder.get_status()
                rs = runner.get_status()
                logger.info(
                    "state=%s policy=%s%s robot=%s cameras=%s%s",
                    st.get("dagger_state", "?"),
                    "streaming" if rs.get("streaming") else "idle",
                    " (intervention)" if st.get("intervention") else "",
                    "up" if st.get("robot_ok") else "DOWN",
                    "ok" if st.get("cam_ok") else "FAULT",
                    f" episodes={st.get('episodes', 0)}" if recording else "",
                )
                if rs.get("last_error"):
                    logger.warning("policy: %s", rs["last_error"])
        return 0
    except Exception as e:
        logger.error("headless deploy failed: %s", e)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
        # Stop the policy before dropping the link, so the followers are not left holding
        # a stale chunk while we tear down.
        try:
            recorder.set_policy_running(False)
        except Exception:
            pass
        if runner is not None:
            runner.shutdown()
        if recording:
            # The GUI's "Stop collection" is what ends an eval rollout and hands it to the
            # writer. There is no such button here, so Ctrl-C has to do it — otherwise the
            # whole run is buffered and then thrown away on shutdown.
            try:
                recorder.disarm()
            except Exception:
                logger.exception("could not close the in-flight episode")
        try:
            recorder.shutdown()  # drains + finalizes the dataset when recording
        except Exception:
            logger.exception("error during shutdown")
        logger.info("headless deploy stopped")
