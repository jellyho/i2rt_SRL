"""Recording the action chunks a policy sampled, beside the dataset."""

from __future__ import annotations

import numpy as np
import pytest

import os

from yam_policy.viz.sample_log import EpisodeSampleLog, episode_path, load, row_at, samples_at


def _samples(value, n=4, horizon=30, dim=14):
    return np.full((n, horizon, dim), float(value), np.float32)


def test_a_chunk_is_stored_once_not_once_per_frame():
    """The broker infers when its chunk runs out, so the thirty frames in between share one
    identical set. Storing per frame is thirty times the bytes for the same information."""
    log = EpisodeSampleLog()
    for frame in range(90):
        log.add(frame, _samples(frame // 30))     # a new set every 30 frames
    assert log.rows == 3


def test_the_row_records_the_frame_the_set_became_current_at(tmp_path):
    log = EpisodeSampleLog()
    for frame in range(90):
        log.add(frame, _samples(frame // 30))
    log.save(str(tmp_path), 0)

    stored = load(str(tmp_path), 0)
    assert list(stored["frame_index"]) == [0, 30, 60]


def test_the_set_in_force_is_found_for_every_frame(tmp_path):
    """Rows mark where a set BECAME current, so an exact-match lookup finds nothing for 29
    frames out of 30."""
    log = EpisodeSampleLog()
    for frame in range(90):
        log.add(frame, _samples(frame // 30))
    log.save(str(tmp_path), 0)
    stored = load(str(tmp_path), 0)

    for frame in range(90):
        found = samples_at(stored, frame)
        assert found is not None, frame
        assert found[0, 0, 0] == pytest.approx(frame // 30), frame


def test_a_frame_before_the_first_set_has_none(tmp_path):
    log = EpisodeSampleLog()
    log.add(10, _samples(1))
    log.save(str(tmp_path), 0)
    assert samples_at(load(str(tmp_path), 0), 3) is None


def test_stored_as_float16(tmp_path):
    """Half the file for a diagnostic nobody trains on, whose interesting property — how far
    apart the chunks are — survives three decimal places comfortably."""
    log = EpisodeSampleLog()
    log.add(0, _samples(1.0))
    log.save(str(tmp_path), 7)

    with np.load(episode_path(str(tmp_path), 7)) as data:
        assert data["samples"].dtype == np.float16


def test_nothing_recorded_writes_nothing(tmp_path):
    assert EpisodeSampleLog().save(str(tmp_path), 0) is None
    assert load(str(tmp_path), 0) is None


def test_absent_samples_are_skipped_not_stored_as_zeros():
    log = EpisodeSampleLog()
    for frame in range(10):
        log.add(frame, None)
    assert log.rows == 0


def test_a_shape_change_mid_episode_keeps_one_shape(tmp_path):
    """Changing the sample count mid-episode would otherwise produce an array nothing can
    stack. Keep the first shape and say so, rather than write a file no reader can open."""
    log = EpisodeSampleLog()
    log.add(0, _samples(1, n=4))
    log.add(30, _samples(2, n=8))
    log.add(60, _samples(3, n=4))
    log.save(str(tmp_path), 0)

    stored = load(str(tmp_path), 0)
    assert stored["samples"].shape[0] == 2
    assert list(stored["frame_index"]) == [0, 60]


def test_reset_starts_a_fresh_episode():
    log = EpisodeSampleLog()
    log.add(0, _samples(1))
    log.reset()
    assert log.rows == 0
    log.add(0, _samples(1))       # the same values again: a new episode, so it must be kept
    assert log.rows == 1


def test_a_wrongly_shaped_sample_set_is_ignored():
    log = EpisodeSampleLog()
    log.add(0, np.zeros((30, 14)))       # a single chunk, not a set of them
    assert log.rows == 0


def test_it_compresses_well_because_of_the_repeats(tmp_path):
    """Sanity on the size claim: an episode of a minute is megabytes, not hundreds."""
    log = EpisodeSampleLog()
    rng = np.random.default_rng(0)
    for chunk in range(60):                    # 60 chunks ~= one minute at 30 fps / horizon 30
        log.add(chunk * 30, rng.standard_normal((8, 30, 14)).astype(np.float32))
    path = log.save(str(tmp_path), 0)

    import os

    size_mb = os.path.getsize(path) / 1e6
    assert size_mb < 5.0, f"{size_mb:.1f} MB for one minute"


def test_the_executed_candidate_is_recorded_not_assumed(tmp_path):
    """Plain multi-sampling puts the executed chunk first; a critic picking best-of-N does not.

    Assuming index 0 would draw a candidate the robot never ran as "what it did" — a picture
    that looks correct and is not, which is the worst kind of wrong for a diagnostic.
    """
    log = EpisodeSampleLog()
    log.add(0, _samples(1), chosen=3, scores=np.array([0.1, 0.2, 0.3, 0.9]))
    log.save(str(tmp_path), 0)

    row = row_at(load(str(tmp_path), 0), 5)
    assert row["chosen"] == 3
    assert row["scores"][3] == pytest.approx(0.9, abs=1e-2)


def test_scores_are_absent_rather_than_zero_when_nobody_scored(tmp_path):
    """An array of zeros would read as "every candidate is worthless" instead of "no critic"."""
    log = EpisodeSampleLog()
    log.add(0, _samples(1))
    log.save(str(tmp_path), 0)

    stored = load(str(tmp_path), 0)
    assert stored["scores"] is None
    assert row_at(stored, 0)["scores"] is None
    assert row_at(stored, 0)["chosen"] == 0


def test_a_log_without_a_chosen_column_still_reads(tmp_path):
    """Logs written before the critic existed must not need a reader that knows their vintage."""
    import numpy as _np

    path = episode_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _np.savez_compressed(path, frame_index=_np.array([0], _np.int32),
                         samples=_samples(1).astype(_np.float16)[None])
    row = row_at(load(str(tmp_path), 0), 0)
    assert row["chosen"] == 0 and row["scores"] is None
