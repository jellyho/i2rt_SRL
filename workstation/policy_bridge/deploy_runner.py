"""Controllable policy streamer used by the DAgger deployment UI.

Unlike the headless bridge, this runner does not own cameras and does not decide
whether policy rollout is active. The robot-side DAgger controller is the source
of truth; this runner only sends policy actions while the robot snapshot reports
``policy_running`` and not intervention/homing/e-stop.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Callable, Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import (
    ARM_DOF,
    ARMS,
    CONTROL_MODE,
    EEF_DIM,
    LEADER_DIM,
    RecorderConfig,
)
from workstation.policy_bridge.config import BridgeConfig

logger = logging.getLogger(__name__)


#: Frameworks a server may name itself as, mapped to how it is shown.
_FRAMEWORK_LABELS = {
    "openpi": "openpi",
    "acrft": "ACRFT",
    "lerobot": "LeRobot",
    "yam-policy": "yam-policy",
    "dataset-replay": "replay",
}


def _describe_policy(meta: Dict) -> str:
    """A short label for what is actually answering on the policy port.

    Three different stacks can serve this wire -- openpi, its ACRFT fork, and a LeRobot
    checkpoint behind our adapter -- and the observations they take are not interchangeable. When
    the wrong one is up, every symptom appears at the robot: an action chunk arrives, of the right
    shape, computed from an observation the checkpoint never trained on. Naming the server at the
    handshake is what turns that into something visible before the rollout starts.

    A server that names itself is taken at its word. One that does not is described from what it
    does advertise, and said to be a guess -- rather than shown a confident wrong name.
    """
    if not isinstance(meta, dict) or not meta:
        return "unidentified server"

    framework = str(meta.get("framework") or "").strip().lower()
    name = next(
        (str(meta[k]) for k in ("policy_name", "policy_type", "train_config", "policy") if meta.get(k)),
        "",
    )
    name = name.rsplit(":", 1)[-1] if name else ""

    if framework:
        label = _FRAMEWORK_LABELS.get(framework, framework)
        return f"{label} · {name}" if name else label

    # Undeclared. `supports_multi_sample` is ACRFT's marker for the several-chunks-per-observation
    # request, which upstream openpi has no notion of -- but a fork could add one, so this stays
    # flagged as inferred rather than asserted.
    guess = "ACRFT?" if meta.get("supports_multi_sample") else "openpi-compatible?"
    return f"{guess} · {name}" if name else guess


#: Per-step timing/provenance columns the runner adds to every eval recording, on top of whatever
#: the policy declares. They answer "where did this action come from?": which chunk and which step
#: within it, how long that chunk's inference took, how many control ticks passed between the
#: observation it was computed from and this action being sent (the RTC inference delay), and when
#: the send happened. Prefixed so they never collide with a policy's own extras.
#:
#: `elapsed_s` is seconds since the runner started, NOT a unix timestamp: the columns are recorded
#: as float32, whose spacing at 1.8e9 is 128 SECONDS, so a wall-clock stamp quantized every frame
#: of a rollout onto three distinct values and destroyed the cadence it was added to measure. Time
#: since start stays under a few thousand, where float32 still resolves a quarter of a millisecond.
PROVENANCE_PREFIX = "policy."
PROVENANCE_FIELDS = ("chunk_index", "step_in_chunk", "infer_ms", "delay_ticks", "elapsed_s")

#: How many recent chunk lengths to keep for the live plot (~2 minutes of replans at a 1 s chunk).
CHUNK_HISTORY = 120


def _extra_features_from_meta(meta: Dict) -> Dict[str, tuple]:
    """What extra per-step arrays this policy sends, from its handshake metadata.

    The client hard-codes none of it. A policy that wants its critic values, its candidate
    chunks, or anything else recorded declares them once:

        {"extra_features": {"critic_q": [1], "action_samples": [8, 14]}}

    The shape given is ONE STEP's shape. The chunk axis is deliberately left out: the chunk
    length is adaptive -- whatever a reply happens to carry -- so it is not something either
    side can declare, while the per-step shape is fixed and is exactly what a dataset column
    needs. A critic scoring N candidates declares ``[N]`` and sends ``(X, N)``; the candidate
    chunks declare ``[N, action_dim]`` and arrive as ``(X, N, action_dim)``.

    Anything malformed is dropped with a warning rather than taken on faith: a wrong shape here
    becomes a wrong column in every episode of the dataset.
    """
    declared = meta.get("extra_features") or {}
    if not isinstance(declared, dict):
        logger.warning("ignoring extra_features: expected a mapping, got %s", type(declared).__name__)
        return {}
    out: Dict[str, tuple] = {}
    for name, shape in declared.items():
        try:
            dims = tuple(int(d) for d in (shape if isinstance(shape, (list, tuple)) else [shape]))
        except (TypeError, ValueError):
            logger.warning("ignoring extra feature %r: shape %r is not a list of ints", name, shape)
            continue
        if not dims or any(d <= 0 for d in dims):
            logger.warning("ignoring extra feature %r: shape %r has no positive dimensions", name, shape)
            continue
        out[str(name)] = dims
    return out


class DeploymentPolicyRunner:
    def __init__(
        self, cfg: BridgeConfig, recorder_cfg: RecorderConfig, images_fn: Callable[[], Dict[str, np.ndarray]]
    ):
        self.cfg = cfg
        self.recorder_cfg = recorder_cfg
        self.images_fn = images_fn
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._robot = None
        self._policy = None
        self._policy_client = None
        self._image_shape = (cfg.image_size, cfg.image_size)
        # Declared by the server at handshake; see _extra_features_from_meta.
        self._extra_features: Dict[str, tuple] = {}
        self._extras: Dict[str, "np.ndarray"] = {}
        self._extra_warned: set = set()
        #: This step's provenance (see PROVENANCE_FIELDS / _note_timing).
        self._timing: Dict[str, float] = {name: 0.0 for name in PROVENANCE_FIELDS}
        self._t0: Optional[float] = None  # first send; elapsed_s is measured from it
        #: Length of each chunk the server has answered with, newest last. The chunk is adaptive --
        #: the broker takes the horizon from every reply rather than from a setting -- so a policy
        #: may answer with a different number of steps each replan, and this is the only record of
        #: what it actually sent. Bounded: it feeds a live plot, not an archive.
        self._chunk_lengths: "deque[int]" = deque(maxlen=CHUNK_HISTORY)
        self._last_chunk_index = -1
        #: Called once the handshake is in, so the recorder can declare its columns.
        self.on_connected = None
        #: Called with the sent action vector every time an action is pushed to the robot, so the
        #: recorder can log exactly one eval frame per executed action (see Recorder.note_action_sent).
        self.on_action_sent: Callable[[np.ndarray], None] | None = None
        #: Called when the policy stops driving -- the operator stopping it, an intervention, or
        #: the arm going home. This runner is the authority on that: it is what decides to send or
        #: not, and it already resets the chunk here. A recorder uses it to end the episode, so one
        #: eval episode is one ROLLOUT rather than everything between arming and disarming.
        self.on_rollout_end: Callable[[], None] | None = None
        self._was_streaming = False
        # Replay pause: PURELY a send-gate on this (workstation) side. When paused the loop stops
        # calling infer()/set_policy_action, so the DatasetPolicy cursor freezes and the robot just
        # holds its last command (the deploy controller's command_timeout hold) -- no policy_running
        # toggle, so the robot side runs none of its start/stop logic (no gripper close). Resume
        # continues from the same frame. Only meaningful in replay_mode.
        self._replay_paused = False
        self._lock = threading.Lock()
        # Seconds between idle reachability probes -- see _probe_policy.
        self._PROBE_PERIOD_S = 2.0
        self._last_probe = -1e9
        self._last_probe_error = ""
        self._status = {
            "robot_connected": False,
            "policy_connected": False,
            "streaming": False,
            # Not "" -- an empty error next to a red dot reads as "no problem", when what it
            # really means is that nothing has been tried yet.
            "last_error": "connecting…",
            # Which stack is answering (openpi / ACRFT / LeRobot), read off the handshake.
            "policy_name": "",
            "policy_framework": "",
            # Set only by a dataset-replay server; see _connect_policy.
            "replay_dataset": "",
            "replay_episode": -1,
            "replay_fps": 0.0,
            "action_horizon": 0,  # filled in from the first chunk the policy returns
            "image_size": cfg.image_size,
            "image_shape": self._image_shape,
            # Inference timing (async broker only): how long the policy takes, how many control
            # ticks that costs, and how often it was too slow to cover the chunk (each underrun is
            # a tick the robot had nothing new to execute).
            "infer_ms": 0.0,
            "delay_ticks": 0,
            "underruns": 0,
        }

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._policy_client is not None:
            try:
                self._policy_client.close()
            except Exception:
                pass

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _set(self, **kw: object) -> None:
        """Update the status, announcing the link transitions on the way through.

        Every client -- the GUI, ``--headless``, the sim harness -- watches this same dict, so
        saying it here says it once for all of them. A periodic status line cannot: it prints
        "policy idle", which means connected-but-not-running and not-connected-at-all equally
        well. What an operator needs is the moment it changes.
        """
        with self._lock:
            transitions = [
                (key, was, kw[key])
                for key, was in ((k, self._status.get(k)) for k in ("robot_connected", "policy_connected"))
                if key in kw and bool(kw[key]) is not bool(was)
            ]
            self._status.update(kw)
            error = str(kw.get("last_error", self._status.get("last_error", "")))

        for key, _was, now in transitions:
            link = "policy" if key == "policy_connected" else "robot"
            host, port = (
                (self.cfg.policy_host, self.cfg.policy_port)
                if link == "policy"
                else (self.cfg.robot_host, self.cfg.robot_port)
            )
            if now:
                logger.info("%s CONNECTED at %s:%s", link, host, port)
            else:
                logger.warning("%s DISCONNECTED from %s:%s%s", link, host, port, f" ({error})" if error else "")

    def _connect_robot(self) -> None:
        if self.recorder_cfg.mock or self._robot is not None:
            self._set(robot_connected=True)
            return
        from i2rt.serving.robot_client import RobotClient

        self._robot = RobotClient(host=self.cfg.robot_host, port=self.cfg.robot_port, timeout=2.0)
        self._set(robot_connected=True)

    def _policy_port_open(self) -> bool:
        """Is the policy server up? Asked over HTTP, because the alternative is noisy.

        This check has to exist at all because `WebsocketClientPolicy` retries a refused
        connection forever, five seconds at a time -- calling it against a server that is not
        up would wedge the control loop instead of reporting the state.

        It used to open a bare TCP connection and close it. That works, but the server is a
        websocket server: a socket that connects and says nothing is a failed opening
        handshake, and it logs one with a full traceback --

            ERROR:websockets.server:opening handshake failed
            EOFError: stream ends after 0 bytes, before end of line

        -- immediately before the real connection succeeds. An operator watching the server
        cannot tell that apart from a genuine fault. openpi's server answers `/healthz` for
        exactly this, so ask it properly.

        Any HTTP answer means something is listening and speaking HTTP, so a non-200 (an older
        server without the route) still counts as up; only a connection-level failure is down.
        """
        url = f"http://{self.cfg.policy_host}:{self.cfg.policy_port}/healthz"
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return True
        except urllib.error.HTTPError:
            return True
        except OSError:
            return False

    def _build_replay_policy(self) -> tuple:
        """An in-process DatasetPolicy wrapped as the policy, for a dataset replay source.

        Replay differs from deployment in exactly ONE thing -- where the actions come from -- so it
        reuses the whole deploy stack (smoother, e-stop, takeover, overlay) by building the same
        ``ActionChunkBroker`` over a local ``DatasetPolicy`` instead of a websocket client. No
        server, no port, no subprocess: DatasetPolicy reads the episode's actions from the parquet
        directly. ``build_metadata`` gives the exact handshake meta the websocket path would have
        received (framework=dataset-replay, replay_dataset/episode/fps, extra_features), so the meta
        processing below is shared."""
        from yam_policy import ActionChunkBroker
        from yam_policy.policies.dataset_policy import DatasetPolicy
        from yam_policy.serve import build_metadata

        from workstation.lerobot_recorder.dataset_writer import dataset_dir

        root = dataset_dir(self.recorder_cfg.root, self.cfg.replay_dataset)
        policy = DatasetPolicy(
            root=root,
            episode=int(self.cfg.replay_episode),
            speed=float(self.cfg.replay_speed),
            loop=bool(self.cfg.replay_loop),
        )
        meta = build_metadata(policy, "yam_policy.policies.dataset_policy:DatasetPolicy", {})
        return ActionChunkBroker(policy), meta

    def set_replay_source(self, dataset: str, episode: int) -> None:
        """Point the in-process replay at a dataset+episode (chosen on the run page's reference
        panel). Rebuilds the policy on the next probe, so switching episodes just re-points here --
        no server to restart. A no-op change is ignored so re-selecting the same row does nothing."""
        dataset, episode = str(dataset or ""), int(episode)
        if (dataset, episode) == (self.cfg.replay_dataset, self.cfg.replay_episode) and self._policy is not None:
            return
        self.cfg.replay_dataset = dataset
        self.cfg.replay_episode = episode
        # Drop the current policy so the idle probe rebuilds DatasetPolicy for the new episode.
        self._policy = None
        self._policy_client = None
        self._last_probe = 0.0  # rebuild promptly, don't wait out the probe throttle
        self._replay_paused = False  # a freshly picked episode plays from the start
        self._set(policy_connected=False)

    def set_replay_paused(self, paused: bool) -> None:
        """Pause/resume a dataset replay -- a send-gate only (see _replay_paused). Pausing freezes
        the episode cursor and lets the robot hold; resuming continues from the same frame. No
        effect on a live policy."""
        self._replay_paused = bool(paused)

    @property
    def replay_paused(self) -> bool:
        return self._replay_paused

    def _connect_policy(self) -> None:
        if self.recorder_cfg.mock or self._policy is not None:
            self._set(policy_connected=True)
            return

        if self.cfg.replay_mode:
            # In-process dataset replay -- no server to reach, so no port check / websocket client.
            # The dataset+episode are chosen on the run page (set_replay_source); until one is,
            # there is nothing to build, which surfaces as an ordinary "not connected" state.
            if not self.cfg.replay_dataset:
                raise ConnectionError("select an episode to replay")
            policy, meta = self._build_replay_policy()
        else:
            if not self._policy_port_open():
                raise ConnectionError(f"policy server offline at {self.cfg.policy_host}:{self.cfg.policy_port}")
            from yam_policy import ActionChunkBroker, AsyncChunkBroker, WebsocketClientPolicy

            client = WebsocketClientPolicy(host=self.cfg.policy_host, port=self.cfg.policy_port)
            meta = client.get_server_metadata() or {}
            self._policy_client = client
            if self.cfg.async_inference:
                # Infer the next chunk while this one is still executing; the reply is spliced in
                # at the inference delay, so the robot keeps moving through the chunk boundary.
                policy = AsyncChunkBroker(
                    client,
                    rate_hz=self.cfg.rate_hz,
                    margin_ticks=self.cfg.prefetch_margin_ticks,
                    prefetch_ticks=self.cfg.prefetch_ticks,
                )
            else:
                policy = ActionChunkBroker(client)

        self._image_shape = self._image_shape_from_meta(meta)
        image_keys = meta.get("image_keys", self.cfg.image_keys)
        self._extra_features = _extra_features_from_meta(meta)
        if self._extra_features:
            logger.info(
                "policy also returns per-step %s", ", ".join(f"{k}{tuple(v)}" for k, v in self._extra_features.items())
            )
        # No action_horizon here on purpose: the broker reads the chunk size off each
        # response, so a checkpoint's horizon can never disagree with a client setting.
        self._policy = policy
        self.cfg.image_keys = image_keys
        self._set(
            policy_connected=True,
            policy_name=_describe_policy(meta),
            # Kept separate from the label so a UI can act on it -- replay drives the overlay
            # from this rather than parsing the display string.
            policy_framework=str(meta.get("framework") or "").strip().lower(),
            # Only a replay server sets these; a UI uses them to line its past-demonstration
            # overlay up with the episode being replayed instead of asking the operator to
            # find it again in a list.
            replay_dataset=str(meta.get("replay_dataset") or ""),
            replay_episode=int(meta["replay_episode"]) if meta.get("replay_episode") is not None else -1,
            replay_fps=float(meta.get("replay_fps") or 0.0),
            image_size=self._image_shape[0],
            image_shape=self._image_shape,
        )
        logger.info("deploy policy: %s | metadata: %s", _describe_policy(meta), meta)
        # The extra-feature declaration only exists now, and the dataset schema is fixed on the
        # first recorded frame -- so whoever owns the recorder is told here, not at construction.
        if self.on_connected is not None:
            try:
                self.on_connected()
            except Exception as e:
                logger.error("on_connected failed: %s", e)

    def _set_extras(self, result: Dict) -> None:
        """Keep this step's declared extras, so the recorder can write them.

        Only the declared keys, and only at the declared shape. A policy that changes what it
        sends mid-run would otherwise change the dataset's columns mid-episode, which no
        reader can make sense of; a mismatch is dropped and said once.
        """
        if not self._extra_features:
            return
        values = {}
        for name, shape in self._extra_features.items():
            value = result.get(name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            if array.size != int(np.prod(shape)):
                if name not in self._extra_warned:
                    self._extra_warned.add(name)
                    logger.warning(
                        "extra feature %r: expected %s values, got %d — not recorded",
                        name,
                        int(np.prod(shape)),
                        array.size,
                    )
                continue
            values[name] = array
        with self._lock:
            self._extras = values

    def get_extras(self) -> Dict[str, "np.ndarray"]:
        """The declared per-step extras from the most recent step, plus this step's provenance.

        Read on the control thread immediately after the action is sent (see
        Recorder.note_action_sent), so the values belong to the action being recorded.
        """
        with self._lock:
            out = {k: v.copy() for k, v in self._extras.items()}
            for name, value in self._timing.items():
                out[PROVENANCE_PREFIX + name] = np.asarray([value], dtype=np.float32)
        return out

    def extra_features(self) -> Dict[str, tuple]:
        """``{name: per-step shape}``: what the policy declared at handshake, plus provenance."""
        features = dict(self._extra_features)
        features.update({PROVENANCE_PREFIX + name: (1,) for name in PROVENANCE_FIELDS})
        return features

    def chunk_lengths(self) -> list:
        """Length of every chunk the server has answered with recently, oldest first."""
        with self._lock:
            return list(self._chunk_lengths)

    def _note_chunk(self) -> None:
        """Record this reply's length, once per chunk.

        The horizon is read off each reply rather than configured, so it can differ every replan;
        `chunk_index` is what says a NEW reply arrived (the same length twice in a row is still two
        entries). Works with either broker -- both expose the same two attributes.
        """
        index = getattr(self._policy, "chunk_index", None)
        horizon = int(getattr(self._policy, "action_horizon", 0) or 0)
        if index is None or index == self._last_chunk_index or horizon <= 0:
            return
        self._last_chunk_index = index
        with self._lock:
            self._chunk_lengths.append(horizon)

    def _note_timing(self, now: float) -> None:
        """Record where the action just sent came from, so the dataset can be audited.

        Without this a rollout says what was executed but not when it was decided: an action is
        the k-th step of a chunk inferred from an observation `delay_ticks` earlier, and only
        `chunk_index`/`step_in_chunk` distinguish "the policy reacted" from "the policy was still
        executing a plan made a second ago". `wall_time` is the real send instant, so the true
        cadence survives the dataset's uniform frame timestamps.
        """
        stats = getattr(self._policy, "stats", None)
        got = stats() if callable(stats) else {}
        timing = {name: float(got.get(name, 0.0)) for name in PROVENANCE_FIELDS if name != "elapsed_s"}
        if self._t0 is None:
            self._t0 = now
        timing["elapsed_s"] = float(now - self._t0)
        with self._lock:
            self._timing = timing
        if got:
            self._set(
                infer_ms=round(float(got.get("infer_ms", 0.0)), 1),
                delay_ticks=int(got.get("delay_ticks", 0)),
                underruns=int(got.get("underruns", 0)),
            )

    def _image_shape_from_meta(self, meta: Dict) -> tuple[int, int]:
        image_shape = meta.get("image_shape")
        if isinstance(image_shape, (list, tuple)) and len(image_shape) == 2:
            return (int(image_shape[0]), int(image_shape[1]))
        image_size = int(meta.get("image_size", self.cfg.image_size))
        return (image_size, image_size)

    def _loop(self) -> None:
        period = 1.0 / max(self.cfg.rate_hz, 1.0)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self.recorder_cfg.mock:
                    # No robot or policy server: simulate a live policy driving at the control rate,
                    # so the record path runs end-to-end (headless/GUI wire on_action_sent ->
                    # Recorder.note_action_sent). One synthetic send per tick == one eval frame,
                    # exactly as a real rollout would produce -- a no-op unless armed in eval mode.
                    self._set(robot_connected=True, policy_connected=True, streaming=True, last_error="")
                    if self.on_action_sent is not None:
                        self.on_action_sent(np.zeros(1, dtype=float))
                else:
                    self._connect_robot()
                    obs = self._robot.get_observation()
                    should_stream = bool(obs.get("policy_running")) and not (
                        obs.get("intervention") or obs.get("homing") or obs.get("estop")
                    )
                    if should_stream and self.cfg.replay_mode and self._replay_paused:
                        # Paused replay: DON'T infer or send. The cursor stays put and the robot
                        # holds its last command -- policy_running is untouched, so the robot side
                        # runs none of its start/stop logic. `_was_streaming` is left as-is so
                        # resuming does not count as a fresh rollout (no reset).
                        self._connect_policy()
                        self._set(streaming=False, last_error="")
                    elif should_stream:
                        self._connect_policy()
                        if not self._was_streaming:
                            self._reset_policy_chunk()
                        policy_obs = self._build_obs(obs, self.images_fn())
                        if policy_obs:
                            result = self._policy.infer(policy_obs)
                            action = result["actions"]
                            self._set_extras(result)
                            self._note_chunk()  # a new reply? record how many steps it carried
                            action_vec = np.asarray(action, dtype=float).reshape(-1)
                            self._robot.set_policy_action(self._split(action_vec))
                            # Stamp where this action came from BEFORE the recorder captures the
                            # frame -- note_action_sent reads get_extras() synchronously, so the
                            # provenance written with the frame is this action's, not the previous.
                            self._note_timing(time.time())
                            # Log one eval frame for the action we just executed (1 frame == 1 send).
                            if self.on_action_sent is not None:
                                self.on_action_sent(action_vec)
                            self._set(
                                streaming=True,
                                last_error="",
                                action_horizon=getattr(self._policy, "action_horizon", 0),
                            )
                            self._was_streaming = True
                        else:
                            if self._was_streaming and self.on_rollout_end is not None:
                                self.on_rollout_end()
                            self._set(streaming=False)
                            self._was_streaming = False
                    else:
                        if self._was_streaming:
                            self._reset_policy_chunk()
                            if self.on_rollout_end is not None:
                                self.on_rollout_end()
                        self._set(streaming=False)
                        self._was_streaming = False
                        self._probe_policy()
            except Exception as e:
                self._set(streaming=False, last_error=f"{type(e).__name__}: {e}")
                self._was_streaming = False
                msg = str(e).lower()
                if "policy" in msg:
                    self._policy = None
                    self._policy_client = None
                    # Drop the name too: a stale one next to a red dot reads as still-identified.
                    self._set(policy_connected=False, policy_name="", policy_framework="")
                else:
                    self._robot = None
                    self._set(robot_connected=False)
                logger.warning("deploy policy tick failed: %s", e)
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _probe_policy(self) -> None:
        """Connect to the policy while idle, so the UI can say whether it is there.

        The connection used to be made only inside the streaming branch, which needs the robot
        to already be running a rollout. Before that, `policy_connected` sat at its initial
        False with `last_error` at its initial "" -- a red dot next to "policy idle" and no
        reason given. Worse, the GUI refuses to start a rollout unless the policy is connected,
        so nothing could ever connect it: the only path to connected ran through a rollout that
        could not be started. Probing while idle is what breaks that circle.

        Throttled, because the probe is a real websocket round-trip (`get_server_metadata`) and
        the loop runs at the control rate -- retrying every tick would hammer a server that is
        still loading its checkpoint.
        """
        if self._policy is not None:
            return
        now = time.monotonic()
        if now - self._last_probe < self._PROBE_PERIOD_S:
            return
        self._last_probe = now
        try:
            self._connect_policy()
            self._set(last_error="")
            self._last_probe_error = ""
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            # _set only announces a *transition*, and a policy that was never up does not
            # transition -- so starting with no server at all would say nothing whatsoever.
            # Report each distinct reason once: enough to see why, without a line every probe.
            # Stay quiet when _set is about to announce the drop itself, or losing a live
            # policy would report the same event twice.
            was_connected = bool(self.get_status().get("policy_connected"))
            if reason != self._last_probe_error and not was_connected:
                self._last_probe_error = reason
                logger.warning(
                    "policy NOT CONNECTED at %s:%s (%s) — retrying every %.0fs",
                    self.cfg.policy_host,
                    self.cfg.policy_port,
                    reason,
                    self._PROBE_PERIOD_S,
                )
            self._set(policy_connected=False, policy_name="", policy_framework="", last_error=reason)

    def _reset_policy_chunk(self) -> None:
        if self._policy is None:
            return
        try:
            self._policy.reset()
        except Exception as e:
            logger.warning("policy reset failed: %s", e)

    @staticmethod
    def _fuse(sides: list[Dict], fields: tuple, per_arm: int | None = None) -> np.ndarray | None:
        parts = []
        for side in sides:
            if not side or any(side.get(field) is None for field in fields):
                return None
            vec = np.concatenate([np.asarray(side[field], dtype=np.float32).reshape(-1) for field in fields])
            if per_arm is not None and vec.size != per_arm:
                return None
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _fit(value: np.ndarray | None, dim: int) -> np.ndarray:
        """Exactly ``dim`` values: zeros when the robot cannot report this at all (an arm with
        no FK reports no eef), padded or truncated otherwise, so the column keeps its width."""
        out = np.zeros(dim, dtype=np.float32)
        if value is None:
            return out
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        n = min(dim, arr.size)
        if n:
            out[:n] = arr[:n]
        return out

    def _build_obs(self, robot_obs: Dict, images: Dict[str, np.ndarray]) -> Dict:
        from yam_policy import image_tools

        sides = [robot_obs.get(arm) for arm in ARMS]
        # `observation/state` must be the SAME vector the policy was trained on. Training
        # repacks the dataset's `observation.state` (pos+vel+eff per arm, 42) into this
        # key, so sending only the 14 joint positions would keep the key valid, pass every
        # check, and quietly normalize against the wrong statistics. Send all 42 and let
        # the policy's input transform take the slice it wants.
        state = self._fuse(sides, ("pos", "vel", "eff"), ARM_DOF * 3)
        if state is None:
            return {}

        obs = {"observation/state": state, "prompt": self.cfg.prompt}
        # The rest of the dataset's non-image columns, under the names the dataset uses. openpi
        # reads `observation/state` and ignores these; a LeRobot policy trained on data from this
        # recorder may declare any of them as an input, because LeRobot's trainer takes every
        # column it finds. Sending them costs 27 floats and is what makes such a checkpoint
        # deployable without retraining.
        obs["observation.leader"] = self._fit(self._fuse(sides, ("leader_pos",)), LEADER_DIM)
        obs["observation.eef"] = self._fit(self._fuse(sides, ("eef",)), EEF_DIM)
        # The policy is what is driving when this observation is used, so that is what is
        # reported -- the recorder labels those frames the same way.
        obs["observation.control_mode"] = np.array([CONTROL_MODE["policy"]], dtype=np.float32)
        # Nothing here asks the policy HOW to answer -- not how many candidates to draw, not
        # whether to run a critic. That is the server's configuration, and the reply already
        # carries everything this side needs: the chunk to execute, its length, and whatever
        # per-step arrays the handshake declared. A knob that lives on both sides is a knob that
        # can disagree, which is exactly how `num_samples` used to drop the action_samples column
        # from every frame of a rollout.

        height, width = self._image_shape
        for role, key in self.cfg.image_keys.items():
            if role in images:
                img = image_tools.resize_with_pad(images[role], height, width)
                obs[key] = image_tools.convert_to_uint8(img)
        return obs

    @staticmethod
    def _split(action: np.ndarray) -> Dict[str, np.ndarray]:
        return {arm: np.asarray(action[i * ARM_DOF : (i + 1) * ARM_DOF], dtype=float) for i, arm in enumerate(ARMS)}
