import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_dataset import (
    _video_stats_from_raw,
    action_idle_mask,
    ffmpeg_select_expression,
    sustained_mask,
    true_runs,
)


class FilterDatasetTests(unittest.TestCase):
    def test_arm_idle_modes(self) -> None:
        actions = np.zeros((4, 14), dtype=np.float32)
        actions[1, 0] = 0.1
        actions[2, 7] = 0.1
        actions[3, :7] = actions[2, :7]
        actions[3, 7:] = actions[2, 7:]

        np.testing.assert_array_equal(action_idle_mask(actions, 1e-3, "left"), [False, False, False, True])
        np.testing.assert_array_equal(action_idle_mask(actions, 1e-3, "right"), [False, True, False, True])
        np.testing.assert_array_equal(action_idle_mask(actions, 1e-3, "either"), [False, True, False, True])
        np.testing.assert_array_equal(action_idle_mask(actions, 1e-3, "both"), [False, False, False, True])

    def test_sustained_mask_only_removes_long_runs(self) -> None:
        candidate = np.asarray([False, True, True, False, True, True, True, False], dtype=bool)
        np.testing.assert_array_equal(
            sustained_mask(candidate, 3),
            [False, False, False, False, True, True, True, False],
        )

    def test_true_runs_and_ffmpeg_expression(self) -> None:
        mask = np.asarray([True, True, False, True, False], dtype=bool)
        self.assertEqual(true_runs(mask), [(0, 1), (3, 3)])
        self.assertEqual(ffmpeg_select_expression(mask), "between(n\\,0\\,1)+eq(n\\,3)")

    def test_raw_video_statistics_validate_byte_count(self) -> None:
        raw = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).tobytes()
        stats = _video_stats_from_raw(raw, frame_count=2, height=2, width=3)
        self.assertEqual(stats["mean"].shape, (3, 1, 1))
        np.testing.assert_array_equal(stats["count"], [2])
        with self.assertRaisesRegex(RuntimeError, "expected"):
            _video_stats_from_raw(raw[:-1], frame_count=2, height=2, width=3)


if __name__ == "__main__":
    unittest.main()
