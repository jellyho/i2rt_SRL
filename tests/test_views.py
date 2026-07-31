"""Operator view composition (pure numpy) tests."""

from __future__ import annotations

import numpy as np

from workstation.lerobot_recorder.views import compose_agentview, compose_camera_strip, overlay, overlay_camera_views


def test_compose_camera_strip_orders_left_agent_right():
    h, w = 120, 160
    images = {
        "agentview": np.full((h, w, 3), 20, np.uint8),
        "wrist_left": np.full((h, w, 3), 100, np.uint8),
        "wrist_right": np.full((h, w, 3), 200, np.uint8),
    }
    out = compose_camera_strip(images, agent_key="agentview")
    assert out.shape == (h, w * 3, 3)
    assert np.all(out[:, :w] == 100)
    assert np.all(out[:, w : 2 * w] == 20)
    assert np.all(out[:, 2 * w :] == 200)


def test_compose_agentview_insets():
    h, w = 120, 160
    images = {
        "agentview": np.zeros((h, w, 3), np.uint8),
        "wrist_left": np.full((48, 64, 3), 200, np.uint8),
        "wrist_right": np.full((48, 64, 3), 100, np.uint8),
    }
    out = compose_agentview(images, agent_key="agentview", inset_frac=0.33)
    assert out.shape == (h, w, 3)
    # bottom-left and bottom-right corners should now contain inset content (non-zero)
    assert out[h - 1, 0:5].max() > 0
    assert out[h - 1, w - 5 : w].max() > 0
    # top stays the (black) agentview
    assert out[0:5, w // 2].max() == 0


def test_overlay_blend():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((20, 20, 3), 200, np.uint8)  # different size -> resized to a
    blended = overlay(a, b, alpha=0.5)
    assert blended.shape == (10, 10, 3)
    assert np.all(blended == 100)  # 0*0.5 + 200*0.5
    assert overlay(None, b) is b  # degenerate inputs pass through


def test_overlay_camera_views_matches_keys_without_mutating_live_frames():
    live = {
        "wrist_left": np.full((4, 5, 3), 200, np.uint8),
        "agentview": np.full((4, 5, 3), 100, np.uint8),
    }
    reference = {
        "wrist_left": np.zeros((8, 10, 3), np.uint8),
        "wrist_right": np.full((4, 5, 3), 50, np.uint8),
    }

    blended = overlay_camera_views(live, reference, live_alpha=0.25)

    assert np.all(blended["wrist_left"] == 50)
    assert blended["agentview"] is live["agentview"]
    assert np.all(live["wrist_left"] == 200)


def test_overlay_camera_views_opacity_endpoints():
    live = {"agentview": np.full((2, 2, 3), 220, np.uint8)}
    reference = {"agentview": np.full((2, 2, 3), 20, np.uint8)}

    assert np.all(overlay_camera_views(live, reference, live_alpha=0)["agentview"] == 20)
    assert np.all(overlay_camera_views(live, reference, live_alpha=1)["agentview"] == 220)
