"""Dataset doctor — quick health/stats for a collected dataset.

Summarizes the episode verdicts the dataset carries in its own schema (``next.success`` /
``next.done``, read from ``meta/episodes``: counts, success rate, per-task breakdown) and —
if ``lerobot`` is installed and a ``--repo-id`` is given — validates the LeRobot dataset
(features, episode/frame counts).

    python -m workstation.lerobot_recorder.doctor --root ~/lerobot_data/yam_pick
    python -m workstation.lerobot_recorder.doctor --root ~/lerobot_data/yam_pick --repo-id user/yam_pick

A dataset recorded before the verdict moved into the schema is reported as such, with the
one command that migrates it.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict

from workstation.lerobot_recorder import outcomes as _outcomes


def summarize_outcomes(root: str) -> Dict:
    """Counts + success rate + per-task stats from the dataset's own metadata."""
    ds_dir = os.path.expanduser(root)
    if not os.path.isfile(os.path.join(ds_dir, "meta", "info.json")):
        return {"exists": False, "path": ds_dir, "episodes": 0, "outcomes": {}, "by_task": {}, "success_rate": None}
    if _outcomes.predates_outcome_schema(ds_dir):
        return {
            "exists": True,
            "legacy": True,
            "path": ds_dir,
            "episodes": 0,
            "outcomes": {},
            "by_task": {},
            "success_rate": None,
        }

    episodes = _outcomes.episode_table(ds_dir)
    outcomes = Counter(e["outcome"] for e in episodes.values())
    succ, fail = outcomes.get(_outcomes.SUCCESS, 0), outcomes.get(_outcomes.FAIL, 0)
    by_task: Dict[str, Counter] = defaultdict(Counter)
    for e in episodes.values():
        by_task[e["task"]][e["outcome"]] += 1
    return {
        "exists": True,
        "legacy": False,
        "path": ds_dir,
        "episodes": len(episodes),
        "frames_total": sum(int(e["length"]) for e in episodes.values()),
        "outcomes": dict(outcomes),
        "success_rate": (succ / (succ + fail)) if (succ + fail) else None,
        "by_task": {k: dict(v) for k, v in by_task.items()},
    }


def outcomes_by_episode(root: str) -> Dict[int, str]:
    """Map episode_index -> ``success`` / ``fail`` / ``unknown`` (for annotating episode lists).

    Empty for a directory that is not a dataset or predates the outcome features.
    """
    ds_dir = os.path.expanduser(root)
    if not os.path.isfile(os.path.join(ds_dir, "meta", "info.json")):
        return {}
    return _outcomes.episode_outcomes(ds_dir, strict=False)


def _print_summary(s: Dict) -> None:
    if not s["exists"]:
        print(f"[doctor] no LeRobot dataset at {s['path']} (nothing recorded yet?)")
        return
    if s.get("legacy"):
        print(f"[doctor] {s['path']} predates the outcome features; migrate it with")
        print(f"    workstation/yam-data migrate-outcomes {s['path']}")
        return
    rate = "n/a" if s["success_rate"] is None else f"{100 * s['success_rate']:.0f}%"
    print(f"[doctor] {s['path']}")
    print(f"  episodes: {s['episodes']}   frames: {s['frames_total']}   success rate: {rate}")
    print(f"  outcomes: {s['outcomes']}")
    for task, counts in s["by_task"].items():
        print(f"    task {task!r}: {counts}")


def _validate_lerobot(repo_id: str, root: str) -> None:
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception as e:
        print(f"[doctor] lerobot not available, skipping dataset validation ({e})")
        return
    try:
        ds = LeRobotDataset(repo_id, root=os.path.expanduser(root))
        feats = sorted(getattr(ds, "features", {}))
        n_ep = getattr(ds, "num_episodes", getattr(getattr(ds, "meta", None), "total_episodes", "?"))
        n_fr = getattr(ds, "num_frames", getattr(getattr(ds, "meta", None), "total_frames", "?"))
        print(f"[doctor] LeRobot dataset OK: {n_ep} episodes, {n_fr} frames")
        print(f"  features: {feats}")
    except Exception as e:
        print(f"[doctor] FAILED to open LeRobot dataset: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Dataset health / stats")
    p.add_argument("--root", required=True, help="the dataset directory (the one with meta/ and data/)")
    p.add_argument("--repo-id", default=None, help="also validate the LeRobot dataset (needs lerobot)")
    args = p.parse_args()
    _print_summary(summarize_outcomes(args.root))
    if args.repo_id:
        _validate_lerobot(args.repo_id, args.root)


if __name__ == "__main__":
    main()
