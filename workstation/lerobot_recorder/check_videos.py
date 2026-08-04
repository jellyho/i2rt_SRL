"""Check (and repair) a recorded dataset's videos against its episode metadata.

LeRobot v3 packs many episodes into one ``.mp4`` per camera. If the encoder or the
stream-copy concat that appends an episode drops a frame, the file ends up shorter than
``meta/episodes`` believes — every episode after the drop is silently misaligned, and the
last one's window runs off the end, so training later dies with::

    IndexError: Invalid frame index=6498 for streamIndex=0; must be less than 6498

No metadata-only check can see this (every window still spans exactly ``length`` frames),
so this tool reads the real frame count out of each mp4.

    # report only (no writes)
    workstation/yam-data check-videos --dataset yam_lego_taxi

    # repair
    workstation/yam-data check-videos --dataset yam_lego_taxi --fix

    # sweep every dataset under the root
    workstation/yam-data check-videos --all

``--fix`` prefers the repair that touches least: frames lost at an interior join are fixed
in **metadata** (the affected episodes are moved back onto the frames that are really theirs,
which also undoes their silent misalignment); only a loss off the *end* of a file appends a
duplicate frame, since a truncated final episode has no later episode to realign against.
Video bytes are otherwise untouched, and ``meta/`` is backed up before any write.

Newly recorded datasets run this check automatically when the recorder finalizes; this tool
is for datasets collected before that, or to re-verify one. Root/repo default to config.yaml's
recorder section.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import List, Optional

from i2rt.serving.rig_config import Resolver, load_rig
from workstation.lerobot_recorder.dataset_writer import dataset_dir, list_datasets
from workstation.lerobot_recorder.video_integrity import (
    diagnose_shortfall,
    repair_short_videos,
    video_file_shortfalls,
)


def _backup_meta(ds_dir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(ds_dir, f"meta.backup-check-videos.{stamp}")
    shutil.copytree(os.path.join(ds_dir, "meta"), dst)
    return dst


def _process(name: str, root: str, fix: bool) -> int:
    ds_dir = dataset_dir(root, name)
    if not os.path.isdir(os.path.join(ds_dir, "meta")):
        print(f"[{name}] not a LeRobot dataset (no meta/) — skipped")
        return 0

    short = video_file_shortfalls(ds_dir)
    if not short:
        print(f"[{name}] OK — every video file has as many frames as the metadata claims")
        return 0

    total = sum(s["missing"] for s in short)
    bad_eps = sorted({e for s in short for e in s["overrun_episodes"]})
    print(f"[{name}] {len(short)} truncated video file(s), {total} frame(s) missing:")
    for s in short:
        diag = diagnose_shortfall(ds_dir, s)
        shifted = sorted(e for e, off in diag["offsets"].items() if off)
        print(
            f"    {s['video_key'].split('.')[-1]:12s} chunk-{s['chunk']:03d}/file-{s['file']:03d}: "
            f"{s['actual']} frames, metadata expects {s['claimed']} (missing {s['missing']})"
        )
        if shifted:
            print(f"        {len(shifted)} episode(s) misaligned: {shifted}")
        if diag["eof_missing"]:
            print(f"        {diag['eof_missing']} frame(s) lost off the end of the file")
    print(f"    episodes that FAIL to load: {bad_eps}")

    if not fix:
        print(f"[{name}] read-only: nothing written. Re-run with --fix to repair.")
        return 1

    backup = _backup_meta(ds_dir)
    print(f"[{name}] backed up metadata to {os.path.basename(backup)}")
    failed = False
    for r in repair_short_videos(ds_dir):
        tag = f"{r['video_key'].split('.')[-1]} file-{r['file']:03d}"
        if r["status"] == "repaired":
            print(f"    repaired {tag}: realigned {len(r['shifted_episodes'])} episode(s), "
                  f"appended {r['appended_frames']} frame(s)")
        else:
            failed = True
            print(f"    FAILED {tag}: {r.get('error', r['status'])}", file=sys.stderr)

    left = video_file_shortfalls(ds_dir)
    if left or failed:
        print(f"[{name}] {len(left)} file(s) still short — metadata backup kept at {backup}", file=sys.stderr)
        return 1
    print(f"[{name}] repaired — all {len(bad_eps)} previously-broken episode(s) load again")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config.yaml (recorder root/repo defaults)")
    p.add_argument("--dataset", default=None, help="dataset folder under --root (default: config repo_id)")
    p.add_argument("--root", default=None, help="parent dir holding datasets (default: config recorder.root)")
    p.add_argument("--all", action="store_true", help="check every dataset folder under --root")
    p.add_argument("--fix", action="store_true", help="repair what is found (default: report only)")
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
        rc |= _process(name, root, args.fix)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
