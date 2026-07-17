---
title: Visualize Dataset (v2.0+ latest dataset format)
emoji: 💻
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
hf_oauth: true
hf_oauth_scopes:
  - read-repos
hf_oauth_expiration_minutes: 480
---

# LeRobot Dataset Visualizer

## i2rt offline mode

This vendored copy is configured to discover and serve local LeRobot datasets
written by i2rt. By default it scans `~/lerobot_data`; choose a dataset on the
landing page and then choose any episode from the original LeRobot sidebar.

```bash
npm install
./serve-local.sh
```

The server binds to `127.0.0.1:3000` by default. From another computer:

```bash
ssh -L 3000:127.0.0.1:3000 USER@ROBOT_COMPUTER
```

Then open `http://127.0.0.1:3000`. Both the UI and dataset files travel through
the SSH tunnel; no separate file server or Hub upload is needed.

Configuration:

```bash
PORT=3100 ./serve-local.sh
LEROBOT_DATA_ROOT=/data/lerobot PORT=3100 ./serve-local.sh
```

The Episodes view includes a full-resolution idle timeline. It marks left,
right, and combined idle transitions using the maximum per-arm action change,
with adjustable threshold and minimum-run controls. The **OpenPI preset** uses
`threshold=0.001` and `minimum run=7` frames. Click any band or interval to seek
the synchronized videos.

Each dataset card also has a **Filter copy** action. It creates a new dataset
while keeping the source untouched. The export can match left-idle, right-idle,
either-arm-idle, or both-arms-idle runs and can apply the cuts to successful
episodes, failed episodes, or both. A background job reports per-episode
progress and rebuilds all camera videos, Parquet data, timestamps, statistics,
episode metadata, `outcomes.jsonl`, and RL sidecars consistently. Exports refuse
to overwrite existing datasets and abort if a rule would remove more than 90%
of any episode.

The exporter bulk-copies retained Parquet rows and runs the camera encodes in
parallel through FFmpeg frame-selection filters. Video statistics are sampled
from a second in-memory FFmpeg branch. It therefore does not perform a random
MP4 seek per retained frame and does not stage decoded frames as PNG files.
Output is built under a job-specific temporary directory, validated as a
LeRobot dataset, and renamed into place only after the complete export succeeds.

LeRobot Dataset Tool and Visualizer is a web application for interactive exploration and visualization of robotics datasets, particularly those in the LeRobot format. It enables users to browse, view, and analyze episodes from large-scale robotics datasets, combining synchronized video playback with rich, interactive data graphs.

## Project Overview

This tool is designed to help robotics researchers and practitioners quickly inspect and understand large, complex datasets. It fetches dataset metadata and episode data (including video and sensor/telemetry data), and provides a unified interface for:

- Navigating between organizations, datasets, and episodes
- Watching episode videos
- Exploring synchronized time-series data with interactive charts
- Analyzing action quality and identifying problematic episodes
- Visualizing robot poses in 3D using URDF models
- Paginating through large datasets efficiently

## Key Features

- **Dataset & Episode Navigation:** Quickly jump between organizations, datasets, and episodes using a sidebar and navigation controls.
- **Synchronized Video & Data:** Video playback is synchronized with interactive data graphs for detailed inspection of sensor and control signals.
- **Overview Panel:** At-a-glance summary of dataset metadata, camera info, and episode details.
- **Statistics Panel:** Dataset-level statistics including episode count, total recording time, frames-per-second, and an episode-length histogram.
- **Action Insights Panel:** Data-driven analysis tools to guide training configuration — includes autocorrelation, state-action alignment, speed distribution, and cross-episode variance heatmap.
- **Filtering Panel:** Identify and flag problematic episodes (low movement, jerky motion, outlier length) for removal. Exports flagged episode IDs as a ready-to-run LeRobot CLI command.
- **3D URDF Viewer:** Visualize robot joint poses frame-by-frame in an interactive 3D scene, with end-effector trail rendering. Supports SO-100, SO-101, and OpenArm bimanual robots.
- **Annotations Panel:** Hand-edit the v3.1 language schema (`language_persistent` + `language_events`) — subtask, plan, memory, interjection + paired speech, and VQA atoms with bounding-box / keypoint / count / attribute / spatial answers. VQA bboxes and keypoints render as overlays on the video player; drag or click on a camera to draw new ones. Backed by an optional FastAPI service (in `backend/`) for parquet rewrites and HF Hub push.
- **Efficient Data Loading:** Uses parquet and JSON loading for large dataset support, with pagination, chunking, and lazy-loaded panels for fast initial load.
- **Responsive UI:** Built with React, Next.js, and Tailwind CSS for a fast, modern user experience.

## Technologies Used

- **Next.js** (App Router)
- **React**
- **Recharts** (for data visualization)
- **Three.js** + **@react-three/fiber** + **@react-three/drei** (for 3D URDF visualization)
- **urdf-loader** (for parsing URDF robot models)
- **hyparquet** (for reading Parquet files)
- **Tailwind CSS** (styling)

## Getting Started

### Prerequisites

This project uses [Bun](https://bun.sh) as its package manager. If you don't have it installed:

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash
```

### Installation

Install dependencies:

```bash
bun install
```

### Development

Run the development server:

```bash
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `src/app/page.tsx` or other files in the `src/` directory. The app supports hot-reloading for rapid development.

### Other Commands

```bash
# Build for production
bun run build

# Start production server
bun start

# Run linter
bun run lint

# Format code
bun run format
```

### Environment Variables

- `DATASET_URL`: (optional) Base URL for dataset hosting (defaults to HuggingFace Datasets).
- `NEXT_PUBLIC_ANNOTATE_BACKEND_URL`: (optional) URL of the FastAPI annotation
  backend (`backend/app.py`). When set, the Annotations tab can save edits and
  rewrite parquet shards / push to the Hub. When unset the tab is read/edit
  only with sessionStorage persistence.

## Annotations backend (optional)

The Annotations tab edits LeRobot v3.1 language atoms — `language_persistent`
(broadcast subtask/plan/memory) and `language_events` (per-frame
interjection / vqa / speech) — and renders existing bbox/keypoint atoms over
the video player. Edits live in `sessionStorage` by default; to write the
new columns into `data/chunk-*/file-*.parquet` (matching the writer in
[lerobot#3471](https://github.com/huggingface/lerobot/pull/3471)) and push the
result to the Hub, run the bundled FastAPI service:

```bash
# 1. install + start the backend (port 7861 by default)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 7861 --reload

# 2. start the visualizer with the backend URL configured
cd ..
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run dev
```

The backend exposes:

- `POST /api/dataset/load` — load a dataset by `repo_id` or `local_path`
- `GET  /api/episodes/{ep}/atoms` — list atoms for an episode
- `POST /api/episodes/{ep}/atoms` — replace atoms (event timestamps are
  snapped to exact source-frame timestamps before persisting)
- `GET  /api/episodes/{ep}/frame_timestamps` — used client-side for snapping
- `POST /api/export` — rewrite parquet with the new language columns plus
  the dataset-level `tools` column (drops legacy `subtask_index`)
- `POST /api/push_to_hub` — export and push to a target repo

## Docker Deployment

This application can be deployed using Docker with bun for optimal performance and self-contained builds.

### Build the Docker image

```bash
docker build -t lerobot-visualizer .
```

### Run the container

```bash
docker run -p 7860:7860 lerobot-visualizer
```

The application will be available at [http://localhost:7860](http://localhost:7860).

### Run with custom environment variables

```bash
docker run -p 7860:7860 -e DATASET_URL=your-url lerobot-visualizer
```

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

### Acknowledgement

The app was orignally created by [@Mishig25](https://github.com/mishig25) and taken from this PR [#1055](https://github.com/huggingface/lerobot/pull/1055)
