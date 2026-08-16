# Dataset Episode Visualizer — Design

Date: 2026-07-09
Status: **historical.** This shipped as `workstation/lerobot_dataset_visualizer/`, not as the
`visualize_dataset.py` sketched below. Kept for the reasoning, not as instructions.

## Goal

Produce an MP4 that concatenates the three camera views (**wrist_left | agentview |
wrist_right**, in that fixed order) for a recorded episode, at a chosen fast-forward
ratio. Input is a dataset name, a speed ratio, and one or more episode indices.

## Context

Datasets are recorded by `yam-data record` (`workstation/lerobot_recorder`) into
**LeRobot v3.0** format under `~/lerobot_data/<dataset>/`:

- `meta/info.json` — `fps`, `video_path` template
  (`videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`), feature list.
- `meta/episodes/**/*.parquet` — one row per episode. All episodes of a camera are
  **concatenated into a single MP4**; per-episode boundaries live in columns
  `videos/<key>/chunk_index`, `videos/<key>/file_index`,
  `videos/<key>/from_timestamp`, `videos/<key>/to_timestamp`.
- Camera image keys: `observation.images.wrist_left`, `observation.images.agentview`,
  `observation.images.wrist_right`. Each frame is 480×640×3.

The repo's `.venv` has `pyarrow` (to read parquet); base conda does not. The script
must be run with `.venv/bin/python`. `ffmpeg`/`ffprobe` are on PATH with `drawtext`,
`hstack`, `libx264`, and the DejaVu Sans Mono font at
`/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf`.

## Interface

Standalone CLI: `workstation/lerobot_recorder/visualize_dataset.py`

```
.venv/bin/python workstation/lerobot_recorder/visualize_dataset.py \
    --dataset yam_lego_yellow_taxi --speed 4 --episode 0 1 2
```

Arguments:

- `--dataset` (required) — dataset name (dir under root).
- `--speed` — fast-forward ratio, float ≥ 1. Default `1`.
- `--episode` — one or more indices, or the literal `all`. Default `all`.
- `--root` — dataset root. Default: `recorder.root` from `config.yaml`
  (`~/lerobot_data`), expanded.
- `--out` — output directory. Default `<root>/viz/` — deliberately **outside** the
  dataset folder so renders never get uploaded along with the dataset.
- `--fps-cap` — max output fps after speed-up. Default `60`.

Output file per episode: `<out>/<dataset>_ep<N>_<speed>x.mp4`.

## How it works (ffmpeg-direct, no torch decode)

1. Resolve `root` and dataset dir; read `meta/info.json` for `fps` and `video_path`.
2. Read `meta/episodes/**/*.parquet` (concat all parquet files) into one table keyed
   by `episode_index`.
3. For the requested episode, for each of the three cameras, read `chunk_index`,
   `file_index`, `from_timestamp`, `to_timestamp`; format the source MP4 path from the
   `video_path` template.
4. Build a single ffmpeg command:
   - Three inputs, each trimmed with `-ss <from> -to <to> -i <mp4>` (input-level seek).
   - Per-panel `drawtext` static label (camera name) bottom-left.
   - `hstack=inputs=3` → 1920×480.
   - Header `drawtext` on the stacked frame:
     `<dataset>  ep<N>  t=%{pts\:hms}  frame %{n}` (sped-up/output timeline; see Decisions).
   - `setpts=PTS/<speed>` for fast-forward.
   - `-an` drop audio; `-r <fps-cap>` cap output fps; encode `libx264 -pix_fmt yuv420p`.
5. Run via `subprocess`, write the MP4, print the output path.

Filtergraph sketch (labelled per input, then stacked):

```
[0:v]drawtext=...:text='wrist_left'[l];
[1:v]drawtext=...:text='agentview'[a];
[2:v]drawtext=...:text='wrist_right'[r];
[l][a][r]hstack=inputs=3,drawtext=...header...,setpts=PTS/SPEED[out]
```

## Structure

Single self-contained file. Functions:

- `find_root(cli_root)` — CLI value, else parse `config.yaml` `recorder.root`, expanduser.
- `load_episode_windows(dataset_dir)` → dict `episode_index -> {cam: (mp4_path, from, to)}`
  plus `fps`. Reads info.json + episodes parquet with pyarrow.
- `build_ffmpeg_cmd(windows, dataset, ep, speed, out_path, fps_cap)` → list[str].
- `main()` — argparse, resolve episodes (`all` → every key), loop, run ffmpeg.

Dependencies: `pyarrow`, `json`, `subprocess`, `argparse`, `pathlib`, `re`/`yaml` for
config. No `lerobot`/`torch` import.

## Decisions

- **Counter timeline**: header shows the **output (sped-up)** time and frame — i.e.
  wall-clock of the produced file (`%{pts}` / `%{n}` evaluated after `setpts`). This is
  the natural "how long does the fast-forwarded clip run" reading.
- **Panel order** fixed: wrist_left, agentview, wrist_right. All 480 tall → clean hstack,
  no resize.
- **Fast-forward** via `setpts=PTS/speed` (keeps all frames, smooth), with `-r` cap so
  high ratios produce a valid fps.

## Edge cases

- Episode index out of range → error listing the valid range, skip it.
- Missing camera key for the dataset → warn and drop that panel (hstack the remaining).
- `--out` created if absent.
- Non-integer / `<1` speed → argparse validation error.

## Testing

- Run on `yam_lego_yellow_taxi` (3 episodes) at speed 1 and 4, episodes `0` and `all`.
- Verify each output MP4 opens (`ffprobe`), has expected width (≈1920, or fewer if a
  camera is missing), and duration ≈ episode_length / fps / speed.
