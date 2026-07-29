#!/usr/bin/env python3
"""Rebuild an i2rt LeRobot dataset while removing sustained idle transitions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from lerobot.datasets.compute_stats import (
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


def write_status(path: Path, **updates: object) -> None:
    current: dict[str, object] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except Exception:
            pass
    current.update(updates, updatedAt=time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2))
    os.replace(temporary, path)


def load_outcomes(root: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    path = root / "outcomes.jsonl"
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row["episode"])] = row
    return result


def action_idle_mask(actions: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] < 14:
        raise ValueError(f"Expected action shape (frames, >=14), got {actions.shape}")
    changes = np.abs(actions[1:, :14] - actions[:-1, :14])
    left = np.max(changes[:, :7], axis=1) < threshold
    right = np.max(changes[:, 7:14], axis=1) < threshold
    if mode == "left":
        idle = left
    elif mode == "right":
        idle = right
    elif mode == "either":
        idle = left | right
    elif mode == "both":
        idle = left & right
    else:
        raise ValueError(f"Unknown idle mode: {mode}")
    return np.concatenate(([False], idle))


def sustained_mask(candidate: np.ndarray, minimum_run: int) -> np.ndarray:
    padded = np.concatenate(([False], candidate, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    remove = np.zeros(candidate.shape, dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        if end - start >= minimum_run:
            remove[start:end] = True
    return remove


def to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def recompute_rl_features(
    features: dict, frames: int, outcome: str | None, rl_config: dict
) -> dict[str, np.ndarray] | None:
    keys = {"success", "reward", "mc_return"} & set(features)
    if not keys:
        return None
    success = outcome == "success"
    discount = float(rl_config.get("discount_factor", 0.99))
    mode = str(rl_config.get("reward_mode", "sparse"))
    if mode != "sparse":
        raise ValueError(f"Filtering RL datasets currently supports reward_mode='sparse', got {mode!r}")
    values = {
        "success": np.full((frames, 1), float(success), dtype=np.float32),
        "reward": np.zeros((frames, 1), dtype=np.float32),
        "mc_return": np.zeros((frames, 1), dtype=np.float32),
    }
    if success:
        values["reward"][-1, 0] = 1.0
        values["mc_return"][:, 0] = discount ** np.arange(frames - 1, -1, -1)
    return {key: values[key] for key in keys}


def should_filter(outcome: str | None, scope: str) -> bool:
    if scope == "both":
        return True
    if scope == "success":
        return outcome == "success"
    if scope == "failure":
        return outcome == "fail"
    raise ValueError(f"Unknown outcome scope: {scope}")


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive runs of true values."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def ffmpeg_select_expression(mask: np.ndarray) -> str:
    """Build a compact FFmpeg select expression from a boolean frame mask."""
    terms = []
    for start, end in true_runs(mask):
        if start == end:
            terms.append(f"eq(n\\,{start})")
        else:
            terms.append(f"between(n\\,{start}\\,{end})")
    if not terms:
        raise ValueError("Cannot encode an empty frame mask")
    return "+".join(terms)


def video_source_path(source: LeRobotDataset, metadata: dict, video_key: str) -> Path:
    chunk = int(metadata[f"videos/{video_key}/chunk_index"])
    file = int(metadata[f"videos/{video_key}/file_index"])
    return source.root / source.meta.video_path.format(video_key=video_key, chunk_index=chunk, file_index=file)


def _video_stats_from_raw(raw: bytes, frame_count: int, height: int, width: int) -> dict:
    expected = frame_count * height * width * 3
    if len(raw) != expected:
        raise RuntimeError(
            f"FFmpeg returned {len(raw)} statistics bytes; expected {expected} "
            f"for {frame_count} RGB frames at {width}x{height}"
        )
    images = np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, height, width, 3)
    images = np.moveaxis(images, -1, 1)
    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    return {key: value if key == "count" else np.squeeze(value / 255.0, axis=0) for key, value in stats.items()}


def encode_filtered_video(
    source_path: Path,
    output_path: Path,
    from_timestamp: float,
    source_frames: int,
    keep: np.ndarray,
    fps: int,
    frame_shape: tuple[int, int],
    encoder_threads: int,
) -> tuple[Path, dict]:
    """Select frames and encode an episode in one sequential FFmpeg pass.

    The second filter output samples frames before re-encoding for LeRobot's
    per-episode image statistics. It is piped through memory, so no PNG staging
    or random MP4 seeks are involved.
    """
    kept = int(keep.sum())
    sample_output_indices = np.asarray(sample_indices(kept), dtype=np.int64)
    sample_mask = np.zeros(kept, dtype=bool)
    sample_mask[sample_output_indices] = True
    keep_expr = ffmpeg_select_expression(keep)
    sample_expr = ffmpeg_select_expression(sample_mask)

    source_height, source_width = frame_shape
    # Match LeRobot's spatial downsampling rule. FFmpeg's nearest-neighbour
    # downsampling avoids material interpolation of the statistics samples.
    probe = np.empty((3, source_height, source_width), dtype=np.uint8)
    downsampled = auto_downsample_height_width(probe)
    stats_height, stats_width = int(downsampled.shape[1]), int(downsampled.shape[2])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporary video: {output_path}")
    filter_graph = (
        f"[0:v]select='{keep_expr}',split=2[video][statsin];"
        f"[video]setpts=N/({fps}*TB)[videoout];"
        f"[statsin]select='{sample_expr}',"
        f"scale={stats_width}:{stats_height}:flags=neighbor,format=rgb24[statsout]"
    )
    duration = source_frames / fps
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{from_timestamp:.9f}",
        "-t",
        f"{duration:.9f}",
        "-i",
        str(source_path),
        "-filter_complex",
        filter_graph,
        "-map",
        "[videoout]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "30",
        "-g",
        "2",
        "-threads",
        str(encoder_threads),
        "-fps_mode",
        "cfr",
        "-frames:v",
        str(kept),
        str(output_path),
        "-map",
        "[statsout]",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-frames:v",
        str(len(sample_output_indices)),
        "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        shutil.rmtree(output_path.parent, ignore_errors=True)
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed for {source_path.name}: {error}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg produced no video at {output_path}")
    probe_result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(output_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    try:
        packet_count = int(probe_result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not validate encoded video {output_path}: {probe_result.stderr.strip()}") from exc
    if probe_result.returncode or packet_count != kept:
        raise RuntimeError(
            f"Encoded video frame count mismatch for {output_path.name}: expected {kept}, found {packet_count}"
        )

    stats = _video_stats_from_raw(result.stdout, len(sample_output_indices), stats_height, stats_width)
    return output_path, stats


def task_for_index(source: LeRobotDataset, task_index: int) -> str:
    matches = source.meta.tasks[source.meta.tasks["task_index"] == task_index]
    if len(matches) != 1:
        raise ValueError(f"Could not resolve task_index={task_index}")
    return str(matches.index[0])


def build_episode_buffer(
    target: LeRobotDataset,
    source: LeRobotDataset,
    source_batch: dict,
    kept_indices: np.ndarray,
    features: dict,
    rl_values: dict[str, np.ndarray] | None,
) -> dict:
    """Build a LeRobot episode buffer without decoding any video fields."""
    buffer = target.create_episode_buffer()
    kept = len(kept_indices)
    buffer["size"] = kept
    buffer["frame_index"] = list(range(kept))
    buffer["timestamp"] = [index / target.fps for index in range(kept)]
    source_task_indices = [int(to_numpy(source_batch["task_index"][int(index)])) for index in kept_indices]
    task_map = {task_index: task_for_index(source, task_index) for task_index in set(source_task_indices)}
    buffer["task"] = [task_map[task_index] for task_index in source_task_indices]

    for key, feature in features.items():
        if key in DEFAULT_FEATURES:
            continue
        if feature["dtype"] == "video":
            buffer[key] = [None] * kept
            continue
        values = []
        for output_index, source_index in enumerate(kept_indices):
            if key in ("success", "reward", "mc_return") and rl_values is not None:
                value = rl_values[key][output_index]
            else:
                value = to_numpy(source_batch[key][int(source_index)])
            shape = tuple(feature.get("shape") or ())
            if shape:
                value = np.asarray(value).reshape(shape)
            values.append(value)
        buffer[key] = values
    return buffer


def save_filtered_episode(
    target: LeRobotDataset,
    source: LeRobotDataset,
    source_metadata: dict,
    episode_buffer: dict,
    keep: np.ndarray,
    features: dict,
    encoder_threads: int,
) -> None:
    """Save tabular data and native-selected videos using LeRobot metadata APIs."""
    episode_length = int(episode_buffer.pop("size"))
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = target.meta.total_episodes
    episode_buffer["index"] = np.arange(target.meta.total_frames, target.meta.total_frames + episode_length)
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
    target.meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.asarray([target.meta.get_task_index(task) for task in tasks], dtype=np.int64)

    for key, feature in features.items():
        if key in ("index", "episode_index", "task_index") or feature["dtype"] in (
            "image",
            "video",
        ):
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    non_video_buffer = {
        key: value
        for key, value in episode_buffer.items()
        if features.get(key, {}).get("dtype") not in ("image", "video")
    }
    non_video_features = {
        key: value for key, value in target.meta.features.items() if value["dtype"] not in ("image", "video")
    }
    episode_stats = compute_episode_stats(non_video_buffer, non_video_features)
    episode_metadata = target._save_episode_data(episode_buffer)

    video_jobs: dict[concurrent.futures.Future, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(source.meta.video_keys))) as executor:
        for video_key in source.meta.video_keys:
            feature = source.meta.features[video_key]
            height, width = map(int, feature["shape"][:2])
            temp_dir = Path(tempfile.mkdtemp(dir=target.root))
            output_path = temp_dir / f"{video_key.replace('/', '_')}_{episode_index:06d}.mp4"
            future = executor.submit(
                encode_filtered_video,
                video_source_path(source, source_metadata, video_key),
                output_path,
                float(source_metadata[f"videos/{video_key}/from_timestamp"]),
                len(keep),
                keep,
                target.fps,
                (height, width),
                encoder_threads,
            )
            video_jobs[future] = video_key

        encoded: dict[str, tuple[Path, dict]] = {}
        for future in concurrent.futures.as_completed(video_jobs):
            video_key = video_jobs[future]
            encoded[video_key] = future.result()

    for video_key in source.meta.video_keys:
        temp_path, stats = encoded[video_key]
        episode_stats[video_key] = stats
        episode_metadata.update(target._save_episode_video(video_key, episode_index, temp_path=temp_path))
    target.meta.save_episode(episode_index, episode_length, episode_tasks, episode_stats, episode_metadata)
    target.episode_buffer = target.create_episode_buffer()


def main(config_path: Path, status_path: Path) -> None:
    config = json.loads(config_path.read_text())
    data_root = Path(config["dataRoot"]).expanduser().resolve()
    source_name = config["source"]
    output_name = config["output"]
    source_root = (data_root / source_name).resolve()
    output_root = (data_root / output_name).resolve()
    temporary_root = data_root / f".{output_name}.filter-{config['jobId']}.tmp"

    if source_root.parent != data_root or output_root.parent != data_root:
        raise ValueError("Datasets must be direct children of the configured data root")
    if output_root.exists() or temporary_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    source_bytes = sum(path.stat().st_size for path in source_root.rglob("*") if path.is_file())
    free_bytes = shutil.disk_usage(data_root).free
    required_bytes = int(source_bytes * 1.2) + 1_000_000_000
    if free_bytes < required_bytes:
        raise OSError(
            f"Not enough free disk space: need approximately {required_bytes / 1e9:.1f} GB, "
            f"have {free_bytes / 1e9:.1f} GB"
        )

    write_status(status_path, state="running", phase="loading", message="Loading source dataset")
    source = LeRobotDataset(source_name, root=source_root)
    if source.meta.image_keys:
        raise ValueError("The fast idle exporter currently supports i2rt video datasets, not embedded image features")
    depth_keys = set(getattr(source.meta, "depth_keys", []))
    if depth_keys:
        raise ValueError(
            "The fast idle exporter currently supports RGB video only; depth video keys found: "
            + ", ".join(sorted(depth_keys))
        )
    for video_key in source.meta.video_keys:
        shape = tuple(source.meta.features[video_key].get("shape") or ())
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Expected RGB HWC video for {video_key}, got shape {shape}")
    outcomes = load_outcomes(source_root)
    rl_config_path = source_root / "rl_config.json"
    rl_config = json.loads(rl_config_path.read_text()) if rl_config_path.exists() else {}
    features = {key: value for key, value in source.meta.features.items() if key not in DEFAULT_FEATURES}

    video_codec = "h264"
    for key in source.meta.video_keys:
        video_codec = source.meta.features[key].get("info", {}).get("video.codec", video_codec)
        break

    target = LeRobotDataset.create(
        repo_id=output_name,
        root=temporary_root,
        fps=source.meta.fps,
        features=features,
        robot_type=source.meta.robot_type,
        use_videos=bool(source.meta.video_keys),
        vcodec=video_codec,
        batch_encoding_size=1,
    )
    # Store one video file per episode. This avoids repeatedly remuxing an
    # ever-growing 200 MB aggregate after every episode, while remaining a
    # standard LeRobot v3 layout (up to chunks_size files per chunk).
    target.meta.info["video_files_size_in_mb"] = 0
    default_encoder_threads = min(4, (os.cpu_count() or 4) // max(1, len(source.meta.video_keys)))
    encoder_threads = max(1, int(config.get("encoderThreads", default_encoder_threads)))

    total_episodes = source.meta.total_episodes
    total_source_frames = source.meta.total_frames
    processed_frames = kept_total = removed_total = 0
    output_outcomes: list[dict] = []
    write_status(
        status_path,
        state="running",
        phase="encoding",
        message="Source loaded; starting episode 0",
        currentEpisode=0,
        totalEpisodes=total_episodes,
        currentFrame=0,
        currentEpisodeFrames=int(source.meta.episodes[0]["length"]),
        processedFrames=0,
        totalSourceFrames=total_source_frames,
        keptFrames=0,
        removedFrames=0,
    )

    for episode_index in range(total_episodes):
        metadata = source.meta.episodes[episode_index]
        start = int(metadata["dataset_from_index"])
        end = int(metadata["dataset_to_index"])
        actions = np.asarray(source.hf_dataset[start:end]["action"], dtype=np.float32)
        outcome_row = outcomes.get(episode_index)
        outcome = outcome_row.get("outcome") if outcome_row else None

        remove = np.zeros(end - start, dtype=bool)
        if should_filter(outcome, config["outcomeScope"]):
            candidate = action_idle_mask(actions, float(config["threshold"]), config["idleMode"])
            remove = sustained_mask(candidate, int(config["minimumRun"]))

        kept_indices = np.flatnonzero(~remove)
        if kept_indices.size < 2:
            raise ValueError(
                f"Episode {episode_index} would retain only {kept_indices.size} frames; "
                "use a less aggressive idle rule"
            )
        removed_fraction = float(remove.mean())
        if removed_fraction > float(config.get("maxEpisodeRemovalFraction", 0.9)):
            raise ValueError(
                f"Episode {episode_index} would lose {removed_fraction:.1%} of its frames; "
                "this exceeds the safety limit"
            )

        rl_values = recompute_rl_features(features, len(kept_indices), outcome, rl_config)
        source_batch = source.hf_dataset[start:end]
        episode_buffer = build_episode_buffer(target, source, source_batch, kept_indices, features, rl_values)
        write_status(
            status_path,
            state="running",
            phase="encoding",
            message=(f"Episode {episode_index}: encoding {len(source.meta.video_keys)} video streams"),
            currentEpisode=episode_index,
            totalEpisodes=total_episodes,
            currentFrame=0,
            currentEpisodeFrames=end - start,
            processedFrames=processed_frames,
            totalSourceFrames=total_source_frames,
            keptFrames=kept_total,
            removedFrames=removed_total,
        )
        save_filtered_episode(
            target,
            source,
            metadata,
            episode_buffer,
            ~remove,
            target.meta.features,
            encoder_threads,
        )
        kept = int(kept_indices.size)
        removed = int(remove.sum())
        processed_frames += end - start
        kept_total += kept
        removed_total += removed

        if outcome_row:
            new_row = dict(outcome_row)
            new_row["episode"] = episode_index
            new_row["frames"] = kept
            output_outcomes.append(new_row)

        write_status(
            status_path,
            state="running",
            phase="encoding",
            message=f"Finished episode {episode_index}",
            currentEpisode=episode_index + 1,
            totalEpisodes=total_episodes,
            currentFrame=end - start,
            currentEpisodeFrames=end - start,
            processedFrames=processed_frames,
            totalSourceFrames=total_source_frames,
            keptFrames=kept_total,
            removedFrames=removed_total,
        )

    write_status(status_path, state="running", phase="finalizing", message="Finalizing dataset metadata")
    target.finalize()
    if output_outcomes:
        (temporary_root / "outcomes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in output_outcomes))
    if rl_config_path.exists():
        shutil.copy2(rl_config_path, temporary_root / "rl_config.json")
    check = LeRobotDataset(output_name, root=temporary_root)
    if check.meta.total_episodes != total_episodes or check.meta.total_frames != kept_total:
        raise RuntimeError("Output validation failed after finalization")
    for episode_index in range(total_episodes):
        episode = check.meta.episodes[episode_index]
        expected_duration = int(episode["length"]) / check.meta.fps
        for video_key in check.meta.video_keys:
            actual_duration = float(episode[f"videos/{video_key}/to_timestamp"]) - float(
                episode[f"videos/{video_key}/from_timestamp"]
            )
            if abs(actual_duration - expected_duration) > 1 / check.meta.fps:
                raise RuntimeError(
                    f"Output validation failed for episode {episode_index}, {video_key}: "
                    f"video duration {actual_duration:.6f}s != {expected_duration:.6f}s"
                )
    os.rename(temporary_root, output_root)
    write_status(
        status_path,
        state="complete",
        phase="complete",
        message="Filtered dataset is ready",
        currentEpisode=total_episodes,
        totalEpisodes=total_episodes,
        processedFrames=processed_frames,
        totalSourceFrames=total_source_frames,
        keptFrames=kept_total,
        removedFrames=removed_total,
        outputPath=str(output_root),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.config, args.status)
    except BaseException as exc:
        write_status(
            args.status,
            state="failed",
            phase="failed",
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
        # A failed job never exposes a partial dataset. Remove only this
        # job's uniquely named staging tree; source and completed outputs are
        # outside it. Running jobs created by an older worker are untouched.
        try:
            failed_config = json.loads(args.config.read_text())
            failed_root = Path(failed_config["dataRoot"]).expanduser().resolve()
            failed_temp = failed_root / (f".{failed_config['output']}.filter-{failed_config['jobId']}.tmp")
            if failed_temp.parent == failed_root and failed_temp.exists():
                shutil.rmtree(failed_temp)
        except Exception:
            pass
        sys.exit(1)
