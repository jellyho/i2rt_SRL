"""Recorder orchestrator — cameras + portal bridge + episode gate + async writer.

Runs a fixed-rate loop (``cfg.fps``, 60 Hz to match the cameras). Each tick it
grabs the latest camera frames and the latest robot snapshot (polled over portal),
lets the :class:`EpisodeGate` decide start/record/stop from the source gate,
and **buffers** the frame locally. A finished episode is handed to the
:class:`AsyncDatasetWriter` queue, so LeRobot's per-trajectory encoding never
blocks the next collection.

Every frame carries ``observation.control_mode`` (teleop / policy / intervention),
plus whatever the robot reports (``observation.state``, ``observation.leader``,
``observation.eef`` when available) — so provenance is always in the dataset.

Review/delete: with ``review_before_save`` (default) each finished episode is held
unsaved in a PENDING state; the GUI plays back the buffered camera and the user
keeps (with a success/fail outcome) or deletes it. Two leader buttons can also
label+save automatically (see ``record_source`` / ``recorder.buttons``).
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder import episode_gate as eg
from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import (
    ACTION_DIM,
    CONTROL_MODE,
    EEF_DIM,
    LEADER_DIM,
    STATE_DIM,
    RecorderConfig,
)
from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter
from workstation.lerobot_recorder.episode_gate import EpisodeGate
from workstation.lerobot_recorder.portal_bridge import PortalBridge

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(
        self,
        cfg: RecorderConfig,
        on_status: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.cameras = CameraManager(cfg)
        self.robot = PortalBridge(cfg)
        self.gate = EpisodeGate()
        self.writer: Optional[AsyncDatasetWriter] = None
        self._on_status = on_status
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status = {
            "running": False,
            "armed": False,
            "recording": False,
            "pending": False,
            "teleop": "—",
            "episodes": 0,
            "episodes_total": 0,  # whole dataset incl. what a resumed one already had
            "success_total": 0,  # whole-dataset ✓ (sidecar history + this session)
            "fail_total": 0,  # whole-dataset ✗
            "frames": 0,
            "queue": 0,
            "saving": False,
            "cam_ok": True,
            "robot_ok": False,
            "disk_ok": True,
            "kept": 0,
            "success": 0,
            "fail": 0,
            "discarded": 0,
            "interventions": 0,
            "dagger_state": "stopped",
            "policy_running": False,
            "leader_mirror": True,
            "intervention": False,
            "fine_grained": False,
            "leader_recentering": False,
            "recenter_fault": False,
            "homing": False,
            "estop": False,
            "low_ram": False,
        }
        self._last_images: dict = {}
        self._pending = False
        # Frames of the in-progress / pending episode. Only ever populated on the fallback
        # path (per-frame RL signals, which need the whole episode before any frame can be
        # written); otherwise frames go straight to the writer and this stays empty --
        # see _ep_add. _n_frames is the count either way.
        self._episode: List[dict] = []
        self._n_frames = 0
        self._streaming_episode = False
        # Eval rollout logging records ONE frame per action the policy actually sends to the robot.
        # The runner calls note_action_sent() from its control-rate thread on every set_policy_action;
        # that call builds the WHOLE frame right then -- images, robot snapshot, action AND the
        # policy's per-step extras (action_samples) all captured at the SAME instant the action is
        # sent -- and appends it here. The record loop only drains this queue and hands each frame to
        # the writer (so writer/episode mutation stays single-threaded). Capturing at send time (not
        # at drain time) is what keeps every frame internally consistent and 1:1 with executions: if
        # the record loop ever falls behind, queued frames are distinct snapshots that drain later,
        # rather than several sends collapsing onto one stale drain-time snapshot (which duplicated
        # images/state/extras and made the first chunk render as a frozen fan). deque.append/popleft
        # are individually atomic in CPython, so no lock is needed across the two threads.
        self._sent_frames: "collections.deque[dict]" = collections.deque()
        self._preview: List[np.ndarray] = []  # downsampled review frames
        self._btn_prev: Dict[str, list] = {}
        self._btn_outcome: Optional[str] = None  # outcome chosen via a leader button this episode
        # "<side>.<index>" -> outcome (success/fail/discard); see RecorderConfig.button_map
        self._button_outcome: Dict[str, str] = {
            str(k).lower(): str(v).lower() for k, v in (cfg.button_map or {}).items()
        }
        if cfg.record_source in ("dagger", "deploy"):
            # Policy-driven modes: the handle buttons drive the robot's own rollout state
            # machine (start/stop, takeover, home), not an episode outcome label.
            self._button_outcome = {}
        self._last_dagger_event_seq = 0
        self._dagger_intervention_active = False
        # "eval": record a continuous rollout (policy / intervention) from arm to disarm,
        # instead of gating on the teleop engage signal.
        self._eval = cfg.record_source == "eval"
        # Set by end_rollout() and acted on by the record loop. An eval EPISODE is one ROLLOUT,
        # not one arming session: the operator arms once and runs several, and between them the
        # robot goes home -- correctly unrecorded, since nothing is sent while homing. Concatenated,
        # they became one episode in which the arm teleports ~2 rad back to home mid-trajectory
        # (measured: 7 rollouts inside one 9052-frame episode, 6 jumps of 1.7-2.1 rad).
        #
        # The signal comes from the runner, which is the authority: it decides whether to send, so
        # it knows when the rollout ended. Reading `policy_running` out of the robot snapshot would
        # be a second, laggier source of the same fact -- and the two can disagree.
        self._rollout_ended = False
        # "deploy": run the policy and watch it, but record NOTHING — no dataset is opened
        # and no frames are buffered. Cameras, robot link, live view, e-stop and human
        # takeover are unchanged, so this is the recorder minus the dataset.
        self._deploy = cfg.record_source == "deploy"
        self._last_ram_gb = 0.0
        # Per-step arrays the policy declared at handshake: {name: shape} for the schema, and
        # a callable returning this step's values. Both arrive after the policy connects, so
        # they are set before the dataset is opened rather than at construction.
        self._extra_features: Dict[str, tuple] = {}
        self._extras_fn: Optional[Callable[[], Dict[str, np.ndarray]]] = None

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """Open cameras + robot link + dataset and begin the record loop (gate stays disarmed).

        In ``deploy`` mode no dataset is opened at all — ``self.writer`` stays None, which
        every consumer below already treats as "nothing to record"."""
        try:
            self.cameras.start()
            self.robot.start()
            self._check_robot_mode()
            # Not opened here when a policy drives: the schema may include columns the policy
            # only declares at handshake, which has not happened yet. _ensure_writer_open
            # opens it on the first recorded frame, by which time it has.
            if not self._deploy and self.cfg.record_source not in ("dagger", "eval"):
                self.writer = self._open_writer()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._set(running=True)
        except Exception:
            # Startup is transactional: dataset initialization can fail after the
            # cameras have opened, and leaving them alive makes every retry fail busy.
            if self.writer is not None and not self.writer.finalized:
                try:
                    self.writer.finalize()
                except Exception:
                    logger.exception("failed to clean up dataset writer after startup error")
            self.writer = None
            self.robot.stop()
            self.cameras.stop()
            raise

    #: Which robot controller each tool needs -> the mode strings that satisfy it, and how
    #: to start the right one. "deploy" also accepts "dagger": that controller was renamed
    #: once it started serving plain deployment as well as HG-DAgger, and an older robot
    #: server still reports the old name — a version skew must not read as a mismatch.
    _ROBOT_MODES = {
        "teleop": ({"teleop"}, "robot/yam teleop"),
        "deploy": ({"deploy", "dagger"}, "robot/yam deploy"),
    }

    def _check_robot_mode(self, timeout: float = 3.0) -> None:
        """Fail loudly when the robot server is running the wrong controller.

        The robot modes are mutually exclusive and launched separately on the robot
        machine, so getting them crossed is easy — and it fails *silently*: a ``teleop``
        server simply ignores ``set_policy_action`` (the policy appears connected and the
        arms never move), and a ``dagger`` server never reports the engage gate (the
        recorder waits forever for an episode). Better to refuse to start and say which
        command to run."""
        want = str(getattr(self.cfg, "expected_robot_mode", "") or "")
        if not want or self.cfg.mock:
            return
        deadline = time.time() + timeout
        mode = self.robot.robot_mode
        while mode is None and time.time() < deadline:
            time.sleep(0.05)
            mode = self.robot.robot_mode
        if mode is None:
            logger.warning("could not read the robot server mode; skipping the %s check", want)
            return
        accepted, hint = self._ROBOT_MODES.get(want, ({want}, want))
        if mode not in accepted:
            raise RuntimeError(
                f"The robot server at {self.cfg.robot_host}:{self.cfg.robot_port} is running in "
                f"'{mode}' mode, but this needs '{want}'.\n\n"
                f"On the robot machine, restart it with:  {hint}"
            )

    def _open_writer(self) -> AsyncDatasetWriter:
        shapes = {k: self.cameras.shape_of(k) for k in self.cameras.image_keys}
        writer = AsyncDatasetWriter(self.cfg, self.cameras.image_keys, shapes, extra_features=self._extra_features)
        writer.open(self._sample_frame())
        return writer

    def _ensure_writer_open(self) -> AsyncDatasetWriter:
        """Return a writer that can accept new episodes.

        The GUI Save button finalizes the current writer so the dataset is usable on
        disk. If collection continues afterward, the next kept episode reopens the
        same dataset in resume/append mode.
        """
        if self.writer is not None and not self.writer.finalized:
            return self.writer
        if self.writer is not None and self.writer.num_episodes > 0:
            self.cfg.resume = True
        self.writer = self._open_writer()
        return self.writer

    def _sample_frame(self) -> dict:
        """The FIXED recorded schema, derived from the robot's known outputs (see config)."""
        return {
            "images": {k: np.zeros(self.cameras.shape_of(k), np.uint8) for k in self.cameras.image_keys},
            "observation.state": np.zeros(STATE_DIM, np.float32),
            "observation.leader": np.zeros(LEADER_DIM, np.float32),
            "observation.eef": np.zeros(EEF_DIM, np.float32),
            "observation.control_mode": np.zeros(1, np.float32),
            "action": np.zeros(ACTION_DIM, np.float32),
            **{name: np.zeros(int(np.prod(shape)), np.float32) for name, shape in self._extra_features.items()},
        }

    @staticmethod
    def _fit(vec: Optional[np.ndarray], dim: int) -> np.ndarray:
        """Coerce a vector to exactly ``dim`` (zeros if missing; pad/truncate otherwise)."""
        if vec is None:
            return np.zeros(dim, np.float32)
        a = np.asarray(vec, dtype=np.float32).reshape(-1)
        if a.size == dim:
            return a
        out = np.zeros(dim, np.float32)
        out[: min(dim, a.size)] = a[:dim]
        return out

    def arm(self) -> None:
        """GUI 'Start collection': begin gating (teleop/dagger) or a rollout (eval).

        No-op in ``deploy`` mode — there is no dataset to collect into, so the concept of
        arming does not apply and the GUI hides the control."""
        if self._deploy:
            return
        if not self._ram_ok_to_start():
            return
        self.gate.arm()
        self._rollout_ended = False
        if self._eval:  # eval: each rollout between arm and disarm is its own episode
            self._reset_episode()
            self._btn_outcome = None
            logger.info("collection armed (eval) — recording rollout until you stop")
        elif self.cfg.record_source == "dagger":
            logger.info("collection armed — each complete policy rollout is recorded as an episode")
        else:
            logger.info("collection armed — each teleop engage→idle is recorded as an episode")
        self._set(armed=True, recording=self._eval)

    def disarm(self) -> None:
        """GUI 'Stop collection'. In eval mode this ENDS the rollout and saves it
        (or holds it for review); otherwise it just stops the gate and drops partials.
        No-op in ``deploy`` mode (nothing was ever armed)."""
        if self._deploy:
            return
        self.gate.disarm()
        if self._eval:
            if self._btn_outcome == "discard":
                self._discard_episode(counted=True)
            elif self._ep_empty():
                self._discard_episode()
            elif self.cfg.review_before_save and self._btn_outcome is None:
                self._pending = True
            else:
                self._submit(self._btn_outcome)
            self._set(
                armed=False,
                recording=False,
                pending=self._pending,
                queue=self.writer.queue_depth if self.writer is not None else 0,
            )
            return
        if self._pending:
            self._discard_episode()
        self._set(armed=False, recording=False, pending=False)

    def keep_episode(self, outcome: Optional[str] = None) -> None:
        """Review decision: keep the pending episode (submit it), with an optional outcome label."""
        if self._pending and self.writer is not None:
            self._submit(outcome)
            self._set(
                pending=False,
                episodes=self.writer.num_episodes,
                episodes_total=self.writer.total_episodes,
                success_total=self.writer.outcome_totals["success"],
                fail_total=self.writer.outcome_totals["fail"],
                frames=0,
                queue=self.writer.queue_depth,
            )

    def delete_episode(self) -> None:
        """Review decision: discard the pending episode."""
        if self._pending:
            self._discard_episode(counted=True)
            self._set(pending=False, frames=0)

    # ---------------------------------------------------------------- episode sink
    def set_extra_features(self, features: Dict[str, tuple], values_fn) -> None:
        """Declare extra per-step columns, and where to read them each frame.

        Called once the policy handshake is in, which is why the dataset is not opened until
        the first recorded frame: the schema has to include these, and nothing knows them
        before the server says so.
        """
        self._extra_features = {str(k): tuple(v) for k, v in (features or {}).items()}
        self._extras_fn = values_fn if self._extra_features else None
        if self._extra_features:
            logger.info(
                "recording extra per-step features: %s", ", ".join(f"{k}{v}" for k, v in self._extra_features.items())
            )

    def _extra_values(self) -> Dict[str, np.ndarray]:
        """This frame's extras, zero-filled when the policy did not send one.

        A declared column has to exist on every frame -- LeRobot's schema is fixed -- so a
        missing value becomes zeros rather than a hole. That is a real loss of meaning, which
        is why _set_extras only ever drops a value it cannot use, loudly.
        """
        if self._extras_fn is None:
            return {}
        try:
            got = self._extras_fn() or {}
        except Exception as e:
            logger.debug("extra features unavailable this frame: %s", e)
            got = {}
        out = {}
        for name, shape in self._extra_features.items():
            size = int(np.prod(shape))
            value = got.get(name)
            out[name] = (
                np.zeros(size, np.float32) if value is None else np.asarray(value, np.float32).reshape(-1)[:size]
            )
        return out

    def _ep_add(self, frame: dict) -> None:
        """Take one captured frame, either streaming it out or buffering it.

        Streaming is the point: three 640x480 cameras are ~2.8 MB a frame, so holding an
        episode as a list costs ~2.5 GB for 30 s and ~7.5 GB for 90 s -- which is what pushed
        the machine into swap and got this process OOM-killed mid-episode. The writer takes
        frames one at a time over a bounded queue instead.
        """
        writer = self._ensure_writer_open()
        if not self._n_frames:
            self._streaming_episode = writer.supports_streaming()
        if self._streaming_episode:
            writer.stream_frame(frame, self.cfg.task)
        else:
            self._episode.append(frame)
        self._n_frames += 1

    def _ep_empty(self) -> bool:
        return self._n_frames == 0

    def _submit(self, outcome: Optional[str]) -> None:
        """Close the episode out to the writer and update live stats."""
        self._log_ram_after_episode(self._n_frames)
        writer = self._ensure_writer_open()
        if self._streaming_episode:
            writer.end_episode(outcome, self.cfg.task)
        else:
            writer.submit(self._episode, outcome, self.cfg.task)
        with self._lock:
            self._status["kept"] += 1
            if outcome in ("success", "fail"):
                self._status[outcome] += 1
        self._reset_episode()

    def _discard_episode(self, *, counted: bool = False) -> None:
        if counted:
            with self._lock:
                self._status["discarded"] += 1
        # A streamed episode is already inside LeRobot's buffer, so discarding it is the
        # writer's job rather than just dropping a list.
        if self._streaming_episode and self._n_frames and self.writer is not None:
            self.writer.abort_episode()
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._episode, self._preview, self._pending = [], [], False
        self._n_frames = 0
        self._streaming_episode = False
        self._sent_frames.clear()  # drop any frames not yet drained; a new episode starts empty

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.writer is not None and not self.writer.finalized:
            self.writer.finalize()
        self.cameras.stop()
        self.robot.stop()
        self._set(running=False)

    def save_dataset(self) -> None:
        """Finalize saved episodes now, without closing cameras or the robot link.
        Nothing to do in ``deploy`` mode — no dataset was ever opened."""
        if self._deploy:
            return
        if self._pending:
            raise RuntimeError("review the pending episode before saving the dataset")
        if self.gate.recording or (self._eval and self.gate.armed and not self._ep_empty()):
            raise RuntimeError("finish or stop the current recording before saving the dataset")

        was_armed = self.gate.armed
        self.gate.disarm()
        self._set(armed=False, recording=False, pending=False)
        try:
            writer = self.writer
            if writer is None:
                return
            if writer.finalized:
                logger.info("dataset already saved; no new episodes to finalize")
                return

            progress = writer.progress
            if progress.get("submitted", 0) <= 0 and writer.pending_total <= 0:
                logger.info("no new episodes to save")
                return

            logger.info("saving dataset now; waiting for queued episodes to finish")
            writer.finalize()
            if writer.num_episodes > 0:
                self.cfg.resume = True
            logger.info("manual save complete")
        finally:
            if was_armed:
                self.gate.arm()
            self._set(
                armed=self.gate.armed,
                recording=False,
                pending=False,
                episodes=self.writer.num_episodes if self.writer is not None else 0,
                episodes_total=self.writer.total_episodes if self.writer is not None else 0,
                success_total=self.writer.outcome_totals["success"] if self.writer is not None else 0,
                fail_total=self.writer.outcome_totals["fail"] if self.writer is not None else 0,
                frames=0,
                queue=0,
                saving=False,
            )

    def set_estop(self, flag: bool) -> None:
        """Forward an e-stop request to the robot (holds the followers until released)."""
        self.robot.set_estop(flag)

    def set_policy_running(self, flag: bool) -> None:
        """Forward a DAgger policy rollout start/stop request."""
        self.robot.set_policy_running(flag)

    def end_rollout(self) -> None:
        """The policy stopped driving; this rollout's episode is finished.

        Called from the runner's control thread (DeploymentPolicyRunner.on_rollout_end). It only
        raises a flag -- the record loop does the closing, keeping every episode-buffer and writer
        mutation on that one thread."""
        if self._eval and self.gate.armed:
            self._rollout_ended = True

    def _end_eval_rollout(self) -> None:
        """Hand the finished rollout over, keeping the gate armed for the next one.

        The same outcome rules disarm() uses -- a leader discard button wins, an empty rollout is
        dropped, review_before_save holds it pending, otherwise it is submitted."""
        if self._btn_outcome == "discard":
            self._discard_episode(counted=True)
        elif self._ep_empty():
            self._discard_episode()
        elif self.cfg.review_before_save and self._btn_outcome is None:
            self._pending = True
        else:
            self._submit(self._btn_outcome)
        self._btn_outcome = None
        logger.info("eval rollout ended — episode closed (still armed for the next one)")

    def note_action_sent(self, action: np.ndarray) -> None:
        """The runner calls this from its control-rate thread the instant it sends an action to the
        robot (see DeploymentPolicyRunner.on_action_sent). It captures the WHOLE frame right now --
        the latest camera images, the current robot snapshot, this action, and the policy's per-step
        extras (action_samples), all from the same instant -- and queues it; the record loop then
        writes one frame per sent action, so recorded frames == executed actions AND each frame is
        internally consistent. Capturing here rather than at drain time is deliberate: it stops a
        lagging record loop from collapsing several sends onto one stale snapshot.

        A no-op unless armed in eval mode and the sensors are ready: pre-arm/idle sends and sends
        while cameras/state are not yet healthy (or during a recentre pause) are dropped rather than
        recorded as a junk frame -- there is nothing coherent to capture then. Extras are read via
        get_extras(); this runs right after the runner's own _set_extras() on the same thread, so it
        sees exactly this step's extras."""
        if not (self._eval and self.gate.armed):
            return
        if not self.cameras.healthy:
            return
        snap = self.robot.get_snapshot()
        if snap.get("state") is None or snap.get("leader_recentering"):
            return
        # Homing is not part of the rollout. The runner already stops streaming once the robot
        # reports it (which covers the gripper release -- the controller raises `homing` for the
        # whole release/travel/close sequence), but `homing` can turn true between that check and
        # this capture, which is how a single frame of the homing pose reached the end of a
        # recording. Re-checking here closes that window: an episode ends on the last action the
        # policy drove, not on the arm going home.
        if snap.get("homing") or snap.get("teleop_state") == "HOMING":
            return
        images = self.get_last_images()
        if not images:
            return
        frame = self._frame(images, snap, action=np.asarray(action, dtype=np.float32).reshape(-1))
        self._sent_frames.append(frame)

    def set_intervention(self, flag: bool) -> None:
        """Forward a DAgger human-intervention request."""
        self.robot.set_intervention(flag)

    def set_leader_mirror(self, flag: bool) -> None:
        """Forward whether the leader tracks the follower while the policy drives."""
        self.robot.set_leader_mirror(flag)

    def finish_dagger_run(self, action: str) -> None:
        """Forward a DAgger keep/discard + home request."""
        self.robot.finish_dagger_run(action)

    def get_status(self) -> dict:
        with self._lock:
            d = dict(self._status)
        d["disk_ok"] = self.writer is None or not self.writer.low_disk
        if self.writer is not None:  # detailed writer pipeline state
            d["writer"] = self.writer.progress
            d["writer_ok"] = not bool(d["writer"].get("failed"))
            d["writer_error"] = d["writer"].get("last_error", "")
            d["saving"] = d["writer"]["saving"]
            d["queue"] = d["writer"]["queued"]
            d["saved"] = bool(d["writer"].get("finalized") and d["writer"].get("submitted", 0) > 0)
            d["saveable"] = bool(
                not d["writer"].get("finalized")
                and not d["writer"].get("failed")
                and d["writer"].get("submitted", 0) > 0
                and not d.get("recording")
                and not d.get("leader_recentering")
                and not d.get("pending")
            )
        else:
            d["saving"] = False
            d["saved"] = False
            d["saveable"] = False
            d["writer_ok"] = True
            d["writer_error"] = ""
        return d

    def get_last_images(self) -> dict:
        with self._lock:
            return dict(self._last_images)

    def get_review_frames(self) -> List[np.ndarray]:
        with self._lock:
            return list(self._preview)

    # ------------------------------------------------------------------ memory guard
    @staticmethod
    def available_ram_gb() -> float:
        """MemAvailable from /proc, i.e. what can be allocated without swapping hard.

        Deliberately not psutil: this runs inside the record loop and must not add a
        dependency or a surprise import cost. inf when the file is unreadable, so a
        platform without it simply keeps the old behaviour."""
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1e6  # kB -> GB
        except Exception:
            pass
        return float("inf")

    def _ram_ok_to_start(self) -> bool:
        """Checked once as an episode begins: is there room to buffer another one?

        The episode is held whole in RAM until it reaches the writer -- 2.8 MB a frame
        across three cameras, so several GB for a normal take. Refusing to start is better
        than stopping half way: a truncated episode is data nobody asked for, and being
        OOM-killed mid-write is what leaves a parquet the dataset cannot open.

        Only at the start, not per frame -- one read of /proc/meminfo per episode. The
        limit of that is a single unusually long take, which can still outgrow the memory
        that looked sufficient when it began; `min_free_ram_gb` carries the headroom for it.
        """
        limit = float(getattr(self.cfg, "min_free_ram_gb", 0.0) or 0.0)
        if limit <= 0:
            return True
        free = self.available_ram_gb()
        if free >= limit:
            self._last_ram_gb = free
            return True
        logger.error(
            "LOW MEMORY: %.1f GB available (< %.1f GB) — NOT starting this episode. An "
            "episode buffers ~%.1f MB per frame in RAM, and being OOM-killed mid-write "
            "leaves a dataset that will not open. Free some memory, or lower "
            "recorder.min_free_ram_gb if this is too cautious.",
            free,
            limit,
            self._frame_mb(),
        )
        with self._lock:
            self._status["low_ram"] = True
        return False

    def _frame_mb(self) -> float:
        """Bytes one buffered frame costs, so the log says why the limit is what it is."""
        try:
            shapes = [self.cameras.shape_of(k) for k in self.cameras.image_keys]
            return sum(int(np.prod(s)) for s in shapes) / 1e6
        except Exception:
            return 0.0

    def _log_ram_after_episode(self, frames: int) -> None:
        """The other half of the pair: what the episode actually cost."""
        limit = float(getattr(self.cfg, "min_free_ram_gb", 0.0) or 0.0)
        if limit <= 0 or not frames:
            return
        free = self.available_ram_gb()
        if self._streaming_episode:
            logger.info(
                "episode streamed %d frames (~%.1f GB, none of it held in RAM); %.1f GB available now",
                frames,
                frames * self._frame_mb() / 1000,
                free,
            )
        else:
            logger.info(
                "episode buffered %d frames (~%.1f GB); %.1f GB available now",
                frames,
                frames * self._frame_mb() / 1000,
                free,
            )
        if free < limit * 1.5:
            logger.warning(
                "memory is getting tight (%.1f GB free, limit %.1f GB) — the next episode may be refused",
                free,
                limit,
            )

    def _buffered_gb(self) -> float:
        """Rough size of the in-RAM episode, for the log line that explains the stop."""
        if not self._episode:
            return 0.0
        first = self._episode[0]
        per_frame = sum(getattr(v, "nbytes", 0) for v in first.get("images", {}).values())
        per_frame += sum(getattr(v, "nbytes", 0) for k, v in first.items() if k != "images")
        return len(self._episode) * per_frame / 1e9

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        # cameras.read() is now non-blocking (the CameraManager grabs on its own thread),
        # so we pace the loop ourselves at cfg.fps. Each tick samples the latest cached
        # frame + the latest robot snapshot; the robot updates faster than fps, so
        # state/action are fresh every tick (images repeat if the camera is slower).
        period = 1.0 / max(self.cfg.fps, 1.0)
        next_t = time.perf_counter()
        warned = False
        while not self._stop.is_set():
            try:
                images = self.cameras.read()
                with self._lock:
                    self._last_images = images
                snap = self.robot.get_snapshot()
                self._scan_dagger_event(snap)
                if self._pending:
                    self._set(teleop=snap["teleop_state"], **self._dagger_status(snap))
                else:
                    self._step(images, snap)
                    warned = self._warn_state(snap, warned)
            except Exception as e:  # keep the loop alive
                logger.error("step error: %s", e)
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def _frame(self, images: dict, snap: dict, action: Optional[np.ndarray] = None) -> dict:
        # The automatic HOMING return at the episode's tail is recorded but marked as
        # `homing` in observation.control_mode, so it can be filtered out at train time
        # (e.g. treat a failed episode as ending at the failure, not after the return).
        control_mode = snap.get("control_mode", 0)
        if snap.get("teleop_state") == "HOMING" or snap.get("homing"):
            control_mode = CONTROL_MODE["homing"]
        # `action` overrides the snapshot's when given (eval logging passes the exact action the
        # runner just sent, so the recorded action == the executed one with no snapshot latency).
        return {
            "images": {k: np.ascontiguousarray(v) for k, v in images.items()},
            "observation.state": self._fit(snap.get("state"), STATE_DIM),
            "observation.leader": self._fit(snap.get("leader"), LEADER_DIM),
            "observation.eef": self._fit(snap.get("eef"), EEF_DIM),
            "observation.control_mode": np.array([control_mode], dtype=np.float32),
            "action": self._fit(snap.get("action") if action is None else action, ACTION_DIM),
            **self._extra_values(),
        }

    def _step(self, images: dict, snap: dict) -> None:
        self._scan_buttons(snap)
        self._scan_dagger_event(snap)

        if self._deploy:
            # Watch-only: publish liveness + rollout state for the UI and return before
            # anything touches an episode buffer or the (absent) writer. The robot-side
            # controller still owns policy/intervention/homing exactly as in dagger mode.
            self._set(
                armed=False,
                recording=False,
                pending=False,
                teleop=snap["teleop_state"],
                frames=0,
                queue=0,
                cam_ok=self.cameras.healthy,
                robot_ok=self.robot.connected,
                **self._dagger_status(snap),
            )
            return

        # Recentring is part of the same episode, but dataset time is paused:
        # append neither sensors nor actions until physical alignment completes.
        recording_paused = bool(snap.get("leader_recentering"))

        if self._eval:  # one episode per rollout while armed; no engage gate
            # Close the episode where the runner says the rollout ended (see end_rollout), so the
            # next one is its own rather than spliced onto this across an unrecorded homing.
            # Drained here, on the record loop, because that is the only thread that touches the
            # episode buffer and the writer.
            if self._rollout_ended:
                self._rollout_ended = False
                if not self._ep_empty():
                    self._end_eval_rollout()

            # Drain the frames note_action_sent already captured (one per action the runner sent).
            # They are complete, self-consistent snapshots taken at send time -- held "waiting for
            # inference" ticks and paused ticks send nothing, so they contribute nothing. Writing
            # here keeps all episode-buffer/writer mutation on this single thread (the runner only
            # appends to the deque). If the gate is closed or paused, drop whatever is queued rather
            # than fold it into a stale episode.
            if self.gate.armed and not recording_paused:
                while self._sent_frames:
                    frame = self._sent_frames.popleft()
                    self._ep_add(frame)
                    self._buffer_preview(frame["images"])
            else:
                self._sent_frames.clear()
            self._set(
                armed=self.gate.armed,
                recording=self.gate.armed and not recording_paused,
                pending=self._pending,
                teleop=snap["teleop_state"],
                # self.writer is opened lazily on the first appended frame (see
                # _ensure_writer_open); it is still None on the ticks before that,
                # e.g. while cameras/state are not yet healthy right after arming.
                episodes=self.writer.num_episodes if self.writer is not None else 0,
                episodes_total=self.writer.total_episodes if self.writer is not None else 0,
                success_total=self.writer.outcome_totals["success"] if self.writer is not None else 0,
                fail_total=self.writer.outcome_totals["fail"] if self.writer is not None else 0,
                frames=self._n_frames,
                queue=self.writer.queue_depth if self.writer is not None else 0,
                cam_ok=self.cameras.healthy,
                robot_ok=self.robot.connected,
                **self._dagger_status(snap),
            )
            return

        event = self.gate.update(snap["teleop_state"])

        if self.cfg.record_source == "dagger":
            # Count actual takeover segments, not rollouts.  Releasing and later
            # re-entering human control within one rollout counts twice.
            intervening = bool(snap.get("intervention"))
            if self.gate.recording:
                if intervening and not self._dagger_intervention_active:
                    with self._lock:
                        self._status["interventions"] += 1
                self._dagger_intervention_active = intervening
            else:
                self._dagger_intervention_active = False

        if event in (eg.EV_START, eg.EV_RECORD, eg.EV_STOP):
            if event == eg.EV_START:
                if not self._ram_ok_to_start():
                    # Refuse the episode rather than buffer one there is no room for; the
                    # gate is disarmed so the operator sees it stop instead of silently
                    # recording nothing.
                    self.gate.disarm()
                    self._set(armed=False, recording=False)
                    return
                self._reset_episode()
                self._btn_outcome = None
                if self.cfg.record_source == "dagger":
                    logger.info("● recording DAgger rollout (policy started)")
                else:
                    logger.info("● recording episode (teleop engaged)")
            if (
                not recording_paused
                and snap["state"] is not None
                and snap["action"] is not None
                and self.cameras.healthy
            ):
                self._ep_add(self._frame(images, snap))
                self._buffer_preview(images)

        if event == eg.EV_STOP:
            if self._ep_empty():
                logger.warning(
                    "episode ended with 0 frames — nothing saved (cameras healthy? robot state/action present?)"
                )
            elif self._btn_outcome == "discard":
                self._discard_episode(counted=True)  # leader discard button: end without saving
            elif self._btn_outcome is not None:
                self._submit(self._btn_outcome)  # button auto-label+save
            elif self.cfg.review_before_save:
                self._pending = True  # hold for Keep/Delete
                logger.info("episode ready for review (%d frames) — keep [S/F] or delete [D]", self._n_frames)
            else:
                self._submit(None)

        self._set(
            armed=self.gate.armed,
            recording=self.gate.recording and not recording_paused,
            pending=self._pending,
            teleop=snap["teleop_state"],
            # self.writer is opened lazily on the first appended frame (see
            # _ensure_writer_open); it is still None before that, e.g. before the
            # first EV_START or before cameras/state are healthy.
            episodes=self.writer.num_episodes if self.writer is not None else 0,
            episodes_total=self.writer.total_episodes if self.writer is not None else 0,
            success_total=self.writer.outcome_totals["success"] if self.writer is not None else 0,
            fail_total=self.writer.outcome_totals["fail"] if self.writer is not None else 0,
            frames=self._n_frames,
            queue=self.writer.queue_depth if self.writer is not None else 0,
            cam_ok=self.cameras.healthy,
            robot_ok=self.robot.connected,
            **self._dagger_status(snap),
        )

    @staticmethod
    def _dagger_status(snap: dict) -> dict:
        return {
            "dagger_state": snap.get("dagger_state", "stopped"),
            "policy_running": bool(snap.get("policy_running")),
            "leader_mirror": bool(snap.get("leader_mirror", True)),
            "intervention": bool(snap.get("intervention")),
            "fine_grained": bool(snap.get("fine_grained")),
            "leader_recentering": bool(snap.get("leader_recentering")),
            "recenter_fault": bool(snap.get("recenter_fault")),
            "homing": bool(snap.get("homing")),
            "estop": bool(snap.get("estop")),
        }

    def _scan_dagger_event(self, snap: dict) -> None:
        """Apply robot-side DAgger keep/discard events exactly once."""
        if self.cfg.record_source != "dagger":
            return
        event = snap.get("last_dagger_event")
        if not isinstance(event, dict):
            return
        try:
            seq = int(event.get("seq", 0))
        except Exception:
            return
        if seq <= self._last_dagger_event_seq:
            return
        self._last_dagger_event_seq = seq
        action = str(event.get("action", "")).lower()
        if action == "keep":
            if self._pending:
                self.keep_episode(outcome="keep")
            else:
                self._btn_outcome = "keep"
        elif action == "discard":
            if self._pending:
                self.delete_episode()
            else:
                self._btn_outcome = "discard"

    def _scan_buttons(self, snap: dict) -> None:
        """Latch a success/fail/discard outcome on a rising edge of a leader label button,
        using the per-(side, index) ``button_map`` so each arm's buttons can differ."""
        if not (self.gate.recording or (self._eval and self.gate.armed)):
            return
        for side, btns in (snap.get("buttons") or {}).items():
            prev = self._btn_prev.get(side, [])
            for idx in range(len(btns)):
                outcome = self._button_outcome.get(f"{side}.{idx}")
                if outcome is None:
                    continue
                pressed = bool(btns[idx])
                was = idx < len(prev) and bool(prev[idx])
                if pressed and not was:
                    self._btn_outcome = outcome
                    logger.info("leader button %s.%d -> %s", side, idx, outcome)
            self._btn_prev[side] = list(btns)

    def _warn_state(self, snap: dict, warned: bool) -> bool:
        if (
            self.gate.recording
            and not snap.get("leader_recentering")
            and (snap["state"] is None or snap["action"] is None)
        ):
            if not warned:
                if snap["state"] is None:
                    logger.warning("recording but robot state is missing — check the robot link")
                else:
                    logger.warning("recording is waiting for the first applied policy action")
            return True
        return False

    # ------------------------------------------------------------------ review buffer
    def _buffer_preview(self, images: dict) -> None:
        key = self.cfg.review_cam if self.cfg.review_cam in images else next(iter(images), None)
        if key is None:
            return
        s = max(self.cfg.review_downscale, 1)
        small = np.ascontiguousarray(images[key][::s, ::s])
        with self._lock:
            self._preview.append(small)
            if len(self._preview) > 1200:  # cap memory (~20 s @ 60 fps downsampled)
                self._preview.pop(0)

    # ------------------------------------------------------------------ status
    def _set(self, **kw: object) -> None:
        with self._lock:
            self._status.update(kw)
        if self._on_status is not None:
            self._on_status(self.get_status())
