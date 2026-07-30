"""Auto-annotate the homing tail of a recorded dataset (non-destructive).

Every episode ends with the arms returning home and the gripper closing. This detects
that homing return from the gripper signal and marks those frames as
``observation.control_mode = homing`` so training can drop them (filter
``control_mode == homing``) — e.g. treat a failed episode as ending at the failure,
not after the return. It only relabels the scalar ``control_mode`` column: no frames
are removed and no video is re-encoded, and it is reversible.

    # preview what would be marked (no writes)
    workstation/yam-data mark-homing --dataset yam_lego_taxi --dry-run

    # apply it
    workstation/yam-data mark-homing --dataset yam_lego_taxi

Newly recorded datasets are already tagged live by the recorder (teleop_state HOMING →
control_mode=homing); this tool is for datasets collected before that, or to re-derive
the annotation. Root/repo default to config.yaml's recorder section.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from i2rt.serving.rig_config import Resolver, load_rig
from workstation.lerobot_recorder.dataset_editor import DatasetEditor
from workstation.lerobot_recorder.dataset_writer import dataset_dir, list_datasets


def _process(name: str, root: str, dry_run: bool, episodes: Optional[List[int]]) -> int:
    ed = DatasetEditor(name, root)
    ds_dir = dataset_dir(root, name)
    starts = ed.episode_homing_starts(episodes)
    if not starts:
        print(f"[{name}] no homing detected (no episode clearly ends with the gripper closing) — nothing to do")
        return 0
    # frames marked per episode = episode length - start
    lengths = ed.frames_by_episode()
    total = 0
    print(f"[{name}] detected homing in {len(starts)} episode(s):")
    for ep in sorted(starts):
        start = starts[ep]
        length = lengths.get(ep)
        tail = (length - start) if length else None
        total += tail or 0
        print(f"    ep {ep:>3}: homing from frame {start}" + (f"  ({tail} frames)" if tail else ""))
    if dry_run:
        print(f"[{name}] --dry-run: nothing written ({total} frames would be marked). Re-run without --dry-run to apply.")
        return 0
    ed.auto_mark_homing(episodes)
    print(f"[{name}] marked {total} frames as homing across {len(starts)} episode(s) in {ds_dir}")
    print(f"[{name}] undo with the editor's Clear, or filter control_mode == homing (4) at train time")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config.yaml (recorder root/repo defaults)")
    p.add_argument("--dataset", default=None, help="dataset folder under --root (default: config repo_id)")
    p.add_argument("--root", default=None, help="parent dir holding datasets (default: config recorder.root)")
    p.add_argument("--all", action="store_true", help="process every dataset folder under --root")
    p.add_argument("--episodes", type=int, nargs="+", default=None, help="limit to these episode indices")
    p.add_argument("--dry-run", action="store_true", help="print what would be marked, write nothing")
    args = p.parse_args(argv)

    rig = load_rig(args.config)
    rec = Resolver(args, p, rig.get("recorder", {}))
    root = args.root or rec.get("root")
    if args.all:
        names = list_datasets(root)
        if not names:
            print(f"no datasets found under {root}", file=sys.stderr)
            return 1
    else:
        repo = args.dataset or rec.get("repo_id")
        names = [repo.strip("/").split("/")[-1]]

    rc = 0
    for name in names:
        rc |= _process(name, root, args.dry_run, args.episodes)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
