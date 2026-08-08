"""Drawing sampled action chunks onto a wrist-camera frame."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")
pytest.importorskip("PIL")

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from workstation.policy_bridge.chunk_overlay import (
    _visible_runs,
    draw_chunk_paths,
    overlay_samples,
    project_samples,
)
from workstation.policy_bridge.wrist_view import CameraIntrinsics, WristCameraGeometry


@pytest.fixture(scope="module")
def geometry():
    return WristCameraGeometry(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))


def _intrinsics():
    return CameraIntrinsics(fx=430, fy=430, cx=320, cy=240, width=640, height=480)


def _samples(n=4, horizon=30, amplitude=0.2):
    """n chunks that sweep one joint by differing amounts — a spread, as a real sample set is."""
    chunks = np.zeros((n, horizon, 7))
    for i in range(n):
        chunks[i, :, 1] = np.linspace(0.0, amplitude * (i + 1) / n, horizon)
    return chunks


def test_a_hidden_middle_point_breaks_the_line_instead_of_shortcutting_it():
    """The failure this guards against draws a segment the gripper never travels.

    Filtering hidden points out joins the two either side of the gap, and a fabricated
    straight line is worse than a missing one: it is indistinguishable from a real prediction.
    """
    path = np.array([
        [10.0, 10.0, 1.0],
        [20.0, 20.0, 1.0],
        [30.0, 30.0, 0.0],   # behind the lens
        [40.0, 40.0, 1.0],
        [50.0, 50.0, 1.0],
    ])
    runs = _visible_runs(path)
    assert [len(r) for r in runs] == [2, 2]
    assert runs[0] == [(10.0, 10.0), (20.0, 20.0)]
    assert runs[1] == [(40.0, 40.0), (50.0, 50.0)]


def test_a_fully_visible_path_is_one_run():
    path = np.column_stack([np.arange(5.0), np.arange(5.0), np.ones(5)])
    assert len(_visible_runs(path)) == 1


def test_a_fully_hidden_path_draws_nothing():
    path = np.column_stack([np.arange(5.0), np.arange(5.0), np.zeros(5)])
    assert _visible_runs(path) == []
    frame = np.full((64, 64, 3), 40, np.uint8)
    assert np.array_equal(draw_chunk_paths(frame, [path]), frame)


def test_paths_without_a_visibility_column_are_all_visible():
    path = np.column_stack([np.arange(5.0), np.arange(5.0)])
    assert len(_visible_runs(path)[0]) == 5


def test_projection_turns_chunks_into_one_path_each(geometry):
    samples = _samples(n=4)
    paths = project_samples(geometry, samples, np.zeros(6), _intrinsics())
    assert len(paths) == 4
    assert all(p.shape == (30, 3) for p in paths)
    # Different chunks must project differently, or the picture says "confident" when it isn't.
    endpoints = {tuple(np.round(p[-1, :2], 3)) for p in paths}
    assert len(endpoints) == 4


def test_the_overlay_actually_marks_the_frame_and_leaves_the_original_alone(geometry):
    frame = np.full((480, 640, 3), 40, np.uint8)
    out = overlay_samples(frame, geometry, _samples(), np.zeros(6), _intrinsics())
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert (out != frame).any(), "nothing was drawn"
    assert (frame == 40).all(), "the caller's frame was modified"


def test_intrinsics_are_rescaled_to_the_frame_actually_passed_in(geometry):
    """Frames get resized on the way to a viewer. Projecting with the sensor's numbers onto a
    resized image pushes every point away from the centre by the resize factor — which looks
    like a plausible-but-wrong prediction, not like a mistake.
    """
    samples = _samples()
    q = np.zeros(6)
    full = _intrinsics()

    small = np.full((240, 320, 3), 40, np.uint8)
    out = overlay_samples(small, geometry, samples, q, full)
    assert (out != small).any(), "nothing drawn on the resized frame"

    # The same scene projected with correctly-scaled intrinsics puts points at half the
    # coordinates; using the full-size ones would place them outside a 320x240 frame.
    at_full = project_samples(geometry, samples, q, full)[0]
    at_half = project_samples(geometry, samples, q, full.scaled_to(320, 240))[0]
    assert np.allclose(at_half[:, :2], at_full[:, :2] / 2.0, atol=1e-6)


def test_the_executed_chunk_is_drawn_differently(geometry):
    """It has to be findable among the candidates: it is the one that will actually run."""
    frame = np.full((480, 640, 3), 40, np.uint8)
    samples = _samples(n=4)
    q = np.zeros(6)
    intrinsics = _intrinsics()

    first = overlay_samples(frame, geometry, samples, q, intrinsics, executed_index=0)
    third = overlay_samples(frame, geometry, samples, q, intrinsics, executed_index=2)
    assert not np.array_equal(first, third)


def test_a_bimanual_action_can_be_sliced_to_one_arm(geometry):
    """The policy emits both arms in one 14-d vector; each wrist camera sees only its own."""
    both = np.zeros((3, 30, 14))
    both[:, :, 1] = np.linspace(0.0, 0.2, 30)          # left arm joint 2
    both[:, :, 8] = np.linspace(0.0, -0.9, 30)         # right arm, must not leak in
    left = project_samples(geometry, both, np.zeros(6), _intrinsics(), arm_slice=slice(0, 7))
    left_only = project_samples(geometry, both[:, :, :7], np.zeros(6), _intrinsics())
    assert np.allclose(left[0], left_only[0])


def test_sample_shape_is_checked(geometry):
    with pytest.raises(ValueError, match="samples"):
        project_samples(geometry, np.zeros((30, 7)), np.zeros(6), _intrinsics())


def test_a_chunk_set_survives_the_broker_intact():
    """The broker hands out one STEP of a chunk at a time. `action_samples` describes the whole
    chunk, so indexing it by step would return sample number `step` — not corrupted-looking
    data, but a different chunk presented as the current action.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_serving"))
    from yam_policy.action_chunk_broker import ActionChunkBroker

    samples = np.stack([np.full((30, 14), i, float) for i in range(5)])

    class _Policy:
        def infer(self, obs):
            return {"actions": np.arange(30 * 14, dtype=float).reshape(30, 14),
                    "action_samples": samples.copy()}

        def reset(self):
            pass

    broker = ActionChunkBroker(_Policy())
    for _ in range(3):
        result = broker.infer({})
        assert result["actions"].shape == (14,), "the executed action is still per-step"
        assert np.array_equal(result["action_samples"], samples), "the chunk set was sliced"


# --------------------------------------------------------------------------------------- #
# The renderer both surfaces share
# --------------------------------------------------------------------------------------- #
def _wrist_frames():
    return {k: np.full((480, 640, 3), 40, np.uint8)
            for k in ("agentview", "wrist_left", "wrist_right")}


def _bimanual_samples(n=4, horizon=30):
    samples = np.zeros((n, horizon, 14))
    for i in range(n):
        samples[i, :, 1] = np.linspace(0.0, 0.05 * (i + 1), horizon)     # left
        samples[i, :, 8] = np.linspace(0.0, -0.05 * (i + 1), horizon)    # right
    return samples


def test_the_renderer_decorates_both_wrists_and_leaves_the_agentview_alone():
    """The agentview is not on a wrist, so nothing here knows where it is pointing."""
    from workstation.policy_bridge.chunk_overlay import WristOverlayRenderer

    renderer = WristOverlayRenderer()
    assert renderer.available, renderer.error

    images = _wrist_frames()
    out = renderer.draw(images, _bimanual_samples(), np.zeros(42), lambda _k: _intrinsics())

    assert np.array_equal(out["agentview"], images["agentview"])
    assert (out["wrist_left"] != images["wrist_left"]).any()
    assert (out["wrist_right"] != images["wrist_right"]).any()


def test_each_wrist_is_drawn_from_its_own_arm():
    """Left and right chunks differ, so the two views must not come out identical -- which is
    what happens if the action or state slice is taken from the wrong arm."""
    from workstation.policy_bridge.chunk_overlay import WristOverlayRenderer

    renderer = WristOverlayRenderer()
    out = renderer.draw(_wrist_frames(), _bimanual_samples(), np.zeros(42), lambda _k: _intrinsics())
    assert not np.array_equal(out["wrist_left"], out["wrist_right"])


@pytest.mark.parametrize(
    ("samples", "state", "intrinsics_for"),
    [
        (None, np.zeros(42), lambda _k: _intrinsics()),                    # nothing to draw
        (_bimanual_samples(), None, lambda _k: _intrinsics()),             # no pose to project from
        (_bimanual_samples(), np.zeros(42), lambda _k: None),              # camera never reported
        (np.zeros((30, 14)), np.zeros(42), lambda _k: _intrinsics()),      # wrong rank
        (_bimanual_samples(), np.zeros(6), lambda _k: _intrinsics()),      # truncated state
    ],
)
def test_missing_pieces_leave_the_view_alone_rather_than_breaking_it(samples, state, intrinsics_for):
    """A frame that cannot be decorated is still a frame worth showing: the overlay is a
    diagnostic, and taking the live view down with it would be a worse failure than not
    drawing."""
    from workstation.policy_bridge.chunk_overlay import WristOverlayRenderer

    renderer = WristOverlayRenderer()
    images = _wrist_frames()
    out = renderer.draw(images, samples, state, intrinsics_for)
    for key, frame in images.items():
        assert np.array_equal(out[key], frame), key


def test_an_unusable_model_disables_the_overlay_instead_of_raising():
    from workstation.policy_bridge.chunk_overlay import WristOverlayRenderer

    renderer = WristOverlayRenderer(xml_path="/nonexistent/model.xml")
    assert not renderer.available
    assert renderer.error
    images = _wrist_frames()
    assert renderer.draw(images, _bimanual_samples(), np.zeros(42), lambda _k: _intrinsics()) == images
