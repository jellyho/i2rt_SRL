"""Recover a dataset left broken by a recorder crash.

An episode is committed in stages — staged PNGs, per-camera mp4s, the shared camera file,
a data parquet, and finally ``meta/episodes``. A crash in the middle leaves files the
metadata never learned about, and a **half-written data parquet has no footer**. LeRobot
globs ``data/`` instead of reading the file list from the metadata, so that single file
stops the whole dataset opening — every good episode included:

    DatasetGenerationError: An error occurred while generating the dataset

The committed episodes are almost always fine: the metadata is written last, so anything it
references finished before the crash. Recovery is removing what the metadata does not claim,
not repairing data.

    # look first -- nothing is written
    workstation/yam-data recover --dataset yam_lego_taxi

    # move the leftovers out and check the dataset opens
    workstation/yam-data recover --dataset yam_lego_taxi --fix

    # sweep everything under the root
    workstation/yam-data recover --all --fix

Leftovers are **moved, never deleted**, into a sibling ``<dataset>.crash-orphans.<stamp>/``,
so this is reversible and a partial episode worth salvaging is still there. A file the
metadata *does* reference is never touched — if one of those is unreadable that is real data
loss and it is reported instead, because no cleanup can fix it.

It also repairs the crash that leaves *no* leftover files: LeRobot rewrites ``meta/info.json``
every episode but flushes the episode parquets in batches, so Ctrl-C leaves the counters
claiming episodes that only ever existed in memory, and the next open fails with

    FileNotFoundError: Cached dataset doesn't contain all requested episodes
    ... RepositoryNotFoundError: 404 ... /api/datasets/<repo>/refs

Those episodes are unrecoverable; ``--fix`` rolls the counters back onto what is on disk so
the surviving episodes open and recording can continue.

The recorder runs the same recovery by itself when it resumes a dataset, so this command is
for datasets you are not about to record into.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from i2rt.serving.rig_config import Resolver, load_rig
from workstation.lerobot_recorder.crash_recovery import (
    describe,
    find_crash_leftovers,
    find_uncommitted_metadata,
    recover,
    verify,
)
from workstation.lerobot_recorder.dataset_writer import dataset_dir, list_datasets


def _process(name: str, root: str, fix: bool) -> int:
    ds_dir = dataset_dir(root, name)
    found = find_crash_leftovers(ds_dir)
    drift = find_uncommitted_metadata(ds_dir)

    if not found and drift is None:
        state = verify(ds_dir)
        if state["ok"]:
            print(f"[{name}] clean — opens fine ({state['episodes']} episodes, {state['frames']} frames)")
            return 0
        # Nothing to clean up, yet it will not open: say so plainly rather than implying
        # this tool can fix it.
        print(f"[{name}] no crash leftovers, but the dataset does not open:\n    {state['error']}", file=sys.stderr)
        return 1

    total = sum(len(v) for v in found.values()) + (1 if drift else 0)
    print(f"[{name}] {total} leftover item(s) from an interrupted recording:")
    print(describe(ds_dir))
    if drift:
        print(
            f"[{name}] --fix rolls the counters back to {drift['committed_episodes']} episodes; "
            f"the {drift['lost_episodes']} buffered episode(s) never reached disk and cannot be recovered."
        )

    if not fix:
        state = verify(ds_dir)
        if not state["ok"]:
            print(f"[{name}] the dataset currently does NOT open: {state['error']}")
        print(f"[{name}] read-only: nothing moved. Re-run with --fix.")
        return 1

    result = recover(ds_dir)
    print(f"[{name}] moved {len(result['moved'])} item(s) to {result['quarantine']}")
    if result["uncommitted"]:
        u = result["uncommitted"]
        print(
            f"[{name}] rolled meta/info.json back {u['info_episodes']} -> {u['committed_episodes']} episodes "
            f"({u['info_frames']} -> {u['committed_frames']} frames)"
        )

    state = verify(ds_dir)
    if state["ok"]:
        print(f"[{name}] recovered — opens fine ({state['episodes']} episodes, {state['frames']} frames)")
        return 0
    print(f"[{name}] still does not open after recovery: {state['error']}", file=sys.stderr)
    print(f"[{name}] the quarantine at {result['quarantine']} has everything that was moved", file=sys.stderr)
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config.yaml (recorder root/repo defaults)")
    p.add_argument("--dataset", default=None, help="dataset folder under --root (default: config repo_id)")
    p.add_argument("--root", default=None, help="parent dir holding datasets (default: config recorder.root)")
    p.add_argument("--all", action="store_true", help="check every dataset folder under --root")
    p.add_argument("--fix", action="store_true", help="move the leftovers aside (default: report only)")
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
