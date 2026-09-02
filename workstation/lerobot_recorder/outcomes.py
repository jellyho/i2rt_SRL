"""Episode outcomes (success / fail / no verdict) inside the LeRobot schema itself.

They used to live in ``outcomes.jsonl``, a sidecar the recorder appended one row to per
episode. That file was invisible to everything LeRobot offers for selecting episodes
(``episodes=[...]``, and ``episode_filter=`` in newer releases), it fell out of step with the
data on every delete / re-index / split / merge, and every consumer -- the trainer, the
critic caches, the visualizer, the doctor -- re-parsed it by hand with its own idea of what
``"keep"`` or ``"unknown"`` meant.

Now the verdict is two **per-frame features**, under the names LeRobot's own sim and
recording datasets use:

    next.success   bool (1,)   True on the LAST frame of an episode  iff  it succeeded
    next.done      bool (1,)   True on the LAST frame of an episode  iff  it was judged at all

and nothing else. Because they are ordinary features, LeRobot computes per-episode
statistics for them on every ``save_episode`` and writes them to ``meta/episodes`` as
``stats/next.success/max`` etc. -- so the episode-level verdict is recoverable from the
metadata alone, without touching a data file:

    success      stats/next.success/max > 0
    fail         stats/next.done/max > 0   and not success
    no verdict   stats/next.done/max == 0      (an eval rollout kept without judging it, or an
                                                episode recovered from a crash)

``next.done`` exists for that third state. With ``next.success`` alone, "not judged" and
"judged a failure" would collapse into one value, and a success-only training filter would
then quietly treat every unjudged rollout as a failure.

Terminal-frame semantics (not "True on every frame") are deliberate: it is what
``next.success`` means in the LeRobot ecosystem, it is what a sparse reward derives from, and
the episode-level reading above does not care either way.

Everything that needs a verdict goes through :func:`episode_outcomes`; nothing reads the
sidecar any more except :func:`migrate`, which turns an old dataset into a new one.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

SUCCESS_KEY = "next.success"
DONE_KEY = "next.done"
OUTCOME_FEATURES: Dict[str, dict] = {
    SUCCESS_KEY: {"dtype": "bool", "shape": (1,), "names": None},
    DONE_KEY: {"dtype": "bool", "shape": (1,), "names": None},
}

#: The three states an episode can be in, as strings, so callers compare against names and
#: not against a bool that means two different things.
SUCCESS, FAIL, UNKNOWN = "success", "fail", "unknown"
#: Sidecar values that meant "judged a success" or "judged a failure". ``keep`` is a legacy
#: name the handle used before it distinguished the two; the recorder already maps it to
#: success at the button, and the migration does the same for old rows.
_SIDECAR_SUCCESS = {"success", "keep"}
_SIDECAR_FAIL = {"fail", "failure"}

LEGACY_SIDECAR = "outcomes.jsonl"


class OutcomeColumnsMissing(RuntimeError):
    """The dataset predates the outcome features. Says how to fix it, because the fix is one
    command and the alternative -- silently treating every episode as unjudged -- is exactly
    the failure the columns exist to prevent."""

    def __init__(self, ds_dir: str):
        super().__init__(
            f"{ds_dir} has no '{SUCCESS_KEY}' / '{DONE_KEY}' features: it was recorded before the "
            "outcome moved into the LeRobot schema. Migrate it in place with\n"
            f"    workstation/yam-data migrate-outcomes {ds_dir}\n"
            "(reads the old outcomes.jsonl once, adds the two columns, and rebuilds the stats)."
        )


# ------------------------------------------------------------------ writing


def terminal_flags(n_frames: int, outcome: Optional[str]) -> Dict[str, np.ndarray]:
    """Per-frame ``{next.success, next.done}`` for one episode of ``n_frames``.

    Both are False everywhere except the last frame, which carries the verdict. ``None`` (or
    anything that is not a success or a failure) is "no verdict" -- both stay False -- which
    is what an eval rollout kept without judging it, or a crash-recovered episode, records.
    """
    n = int(n_frames)
    success = np.zeros((n, 1), dtype=bool)
    done = np.zeros((n, 1), dtype=bool)
    state = normalize(outcome)
    if n > 0 and state in (SUCCESS, FAIL):
        done[-1, 0] = True
        success[-1, 0] = state == SUCCESS
    return {SUCCESS_KEY: success, DONE_KEY: done}


def frame_value(flag: bool) -> np.ndarray:
    """The value one frame carries for either feature, in the dtype/shape the feature declares."""
    return np.array([bool(flag)], dtype=bool)


def normalize(outcome: Optional[str]) -> str:
    """Any spelling a sidecar, a button or a GUI ever used -> one of SUCCESS / FAIL / UNKNOWN."""
    if outcome is None:
        return UNKNOWN
    o = str(outcome).strip().lower()
    if o in _SIDECAR_SUCCESS:
        return SUCCESS
    if o in _SIDECAR_FAIL:
        return FAIL
    return UNKNOWN


# ------------------------------------------------------------------ reading


def has_outcome_features(ds_dir: str) -> bool:
    info = os.path.join(ds_dir, "meta", "info.json")
    try:
        with open(info) as fh:
            feats = json.load(fh).get("features", {})
    except (OSError, ValueError):
        return False
    return SUCCESS_KEY in feats and DONE_KEY in feats


def predates_outcome_schema(ds_dir: str) -> bool:
    """True for a real dataset recorded before the outcome features existed: ``meta/info.json``
    declares features and ours are not among them. A directory with no feature declaration at
    all (a fresh recording, a stub) is not a legacy dataset and returns False."""
    info = os.path.join(ds_dir, "meta", "info.json")
    try:
        with open(info) as fh:
            feats = json.load(fh).get("features")
    except (OSError, ValueError):
        return False
    if not isinstance(feats, dict) or not feats:
        return False
    return SUCCESS_KEY not in feats or DONE_KEY not in feats


def _episode_files(ds_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(ds_dir, "meta", "episodes", "**", "*.parquet"), recursive=True))


def episode_outcomes(ds_dir: str, *, strict: bool = True) -> Dict[int, str]:
    """``{episode_index: "success" | "fail" | "unknown"}`` from the metadata alone.

    Reads ``meta/episodes`` directly rather than through ``LeRobotDatasetMetadata``: the
    installed LeRobot drops every ``stats/*`` column when it loads that table (they are large
    for images), and the two that carry the verdict go with them.

    ``strict=False`` returns ``{}`` instead of raising on a pre-migration dataset, for callers
    that are only decorating a listing.
    """
    import pandas as pd

    s_col, d_col = f"stats/{SUCCESS_KEY}/max", f"stats/{DONE_KEY}/max"
    out: Dict[int, str] = {}
    files = _episode_files(ds_dir)
    if not files:
        return out
    for f in files:
        df = pd.read_parquet(f, columns=None)
        if s_col not in df.columns or d_col not in df.columns:
            if strict:
                raise OutcomeColumnsMissing(ds_dir)
            return {}
        for _, row in df[["episode_index", s_col, d_col]].iterrows():
            ep = int(row["episode_index"])
            succ = bool(np.asarray(row[s_col]).reshape(-1)[0])
            done = bool(np.asarray(row[d_col]).reshape(-1)[0])
            out[ep] = SUCCESS if succ else (FAIL if done else UNKNOWN)
    return dict(sorted(out.items()))


def outcome_totals(ds_dir: str) -> Dict[str, int]:
    """Whole-dataset success / fail counts, for the recorder's status line."""
    totals = {SUCCESS: 0, FAIL: 0}
    try:
        for state in episode_outcomes(ds_dir, strict=False).values():
            if state in totals:
                totals[state] += 1
    except Exception as e:  # a status counter must never break recording
        logger.warning("could not read outcome totals from %s: %s", ds_dir, e)
    return totals


def success_episodes(ds_dir: str) -> List[int]:
    return [e for e, s in episode_outcomes(ds_dir).items() if s == SUCCESS]


# ------------------------------------------------------------------ legacy sidecar (migration input only)


def episode_table(ds_dir: str) -> Dict[int, dict]:
    """``{episode_index: {"outcome", "task", "length"}}`` -- the verdict plus the two
    columns a listing wants next to it, from ``meta/episodes`` alone. Raises
    :class:`OutcomeColumnsMissing` on a pre-migration dataset."""
    import pandas as pd

    verdicts = episode_outcomes(ds_dir, strict=True)
    out: Dict[int, dict] = {}
    for f in _episode_files(ds_dir):
        df = pd.read_parquet(f, columns=["episode_index", "tasks", "length"])
        for ep, tasks, n in zip(
            df["episode_index"].tolist(), df["tasks"].tolist(), df["length"].tolist(), strict=True
        ):
            tasks = list(tasks) if tasks is not None else []
            out[int(ep)] = {
                "outcome": verdicts.get(int(ep), UNKNOWN),
                "task": str(tasks[0]) if tasks else "?",
                "length": int(n),
            }
    return out


def read_legacy_sidecar(ds_dir: str) -> Dict[int, dict]:
    """The old ``outcomes.jsonl``, one dict per episode. Only the migration should need this."""
    path = os.path.join(ds_dir, LEGACY_SIDECAR)
    rows: Dict[int, dict] = {}
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows[int(row["episode"])] = row
            except (ValueError, KeyError, TypeError):
                logger.warning("skipping malformed row in %s", path)
    return rows


# ------------------------------------------------------------------ migration


def _episode_lengths(ds_dir: str) -> Dict[int, int]:
    import pandas as pd

    out: Dict[int, int] = {}
    for f in _episode_files(ds_dir):
        df = pd.read_parquet(f, columns=["episode_index", "length"])
        for ep, n in zip(df["episode_index"].tolist(), df["length"].tolist(), strict=True):
            out[int(ep)] = int(n)
    return out


def _feature_stats(values: np.ndarray) -> Dict[str, np.ndarray]:
    """The stats LeRobot itself would have written for a ``(n, 1)`` bool feature, in the same
    shapes AND dtypes, so a migrated episode row is indistinguishable from a recorded one and
    ``aggregate_stats`` / ``crash_recovery.reaggregate_stats`` treat both alike.

    Mirrors ``compute_episode_stats``: a 2-D ``(n, 1)`` column is reduced over axis 0 with
    ``keepdims=False`` -> every stat has shape ``(1,)``; min/max of a bool column stay bool.
    (The first version of this used keepdims=True and wrote ``[[0.0]]`` where LeRobot writes
    ``[False]`` -- same numbers, different shape, and the shape is what a later aggregation
    trips over.)"""
    import warnings

    from lerobot.datasets.compute_stats import get_feature_stats

    with warnings.catch_warnings():
        # the quantile estimator histograms a bool column and warns about its bin edges
        warnings.simplefilter("ignore", RuntimeWarning)
        return get_feature_stats(np.asarray(values, dtype=bool), axis=0, keepdims=False)


def migrate(ds_dir: str, *, outcomes: Optional[Dict[int, str]] = None, force: bool = False) -> Dict[str, int]:
    """Add the two outcome features to an existing dataset, in place.

    Verdicts come from ``outcomes`` (``{episode: outcome}``) or, by default, from the legacy
    sidecar. Episodes with no row get "no verdict". Touches:

      data/**/*.parquet          two new bool columns per row
      meta/info.json             the two features declared (a parquet column that info.json
                                 does not declare makes LeRobot refuse to open the dataset)
      meta/episodes/**/*.parquet stats/next.success/* and stats/next.done/* per episode
      meta/stats.json            re-aggregated from the per-episode stats

    and nothing under ``videos/``. Every file is written to a sibling temp path and swapped in
    with ``os.replace``, so a crash mid-way leaves either the old file or the new one.

    Returns ``{"episodes", "success", "fail", "unknown"}``. Idempotent: a dataset that already
    has the features is left alone unless ``force``.
    """
    import pandas as pd

    ds_dir = os.path.expanduser(ds_dir)
    info_path = os.path.join(ds_dir, "meta", "info.json")
    with open(info_path) as fh:
        info = json.load(fh)
    if not force and SUCCESS_KEY in info.get("features", {}) and DONE_KEY in info.get("features", {}):
        logger.info("%s already carries the outcome features; nothing to do", ds_dir)
        current = episode_outcomes(ds_dir)
        return _tally(current)

    if outcomes is None:
        outcomes = {ep: row.get("outcome") for ep, row in read_legacy_sidecar(ds_dir).items()}
    verdict = {int(e): normalize(o) for e, o in outcomes.items()}
    lengths = _episode_lengths(ds_dir)
    missing = sorted(set(lengths) - set(verdict))
    if missing:
        logger.warning("%d episode(s) have no verdict and are migrated as 'unknown': %s", len(missing), missing[:20])

    # ---- data parquets: the per-frame columns
    data_files = sorted(glob.glob(os.path.join(ds_dir, "data", "**", "*.parquet"), recursive=True))
    per_episode_stats: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for f in data_files:
        df = pd.read_parquet(f)
        succ = np.zeros(len(df), dtype=bool)
        done = np.zeros(len(df), dtype=bool)
        ep_col = df["episode_index"].to_numpy()
        fr_col = df["frame_index"].to_numpy()
        for ep in np.unique(ep_col):
            ep = int(ep)
            n = lengths.get(ep, int(fr_col[ep_col == ep].max()) + 1)
            flags = terminal_flags(n, verdict.get(ep))
            rows = np.flatnonzero(ep_col == ep)
            # frame_index is the position inside the episode, so it indexes the flags directly
            succ[rows] = flags[SUCCESS_KEY][fr_col[rows], 0]
            done[rows] = flags[DONE_KEY][fr_col[rows], 0]
            per_episode_stats[ep] = {
                SUCCESS_KEY: _feature_stats(flags[SUCCESS_KEY]),
                DONE_KEY: _feature_stats(flags[DONE_KEY]),
            }
        df[SUCCESS_KEY] = succ
        df[DONE_KEY] = done
        _atomic_parquet(df[_recorded_order(list(df.columns))], f)

    # ---- info.json: declare them, so the columns are legal
    info["features"] = _declare(info.get("features", {}))
    _atomic_json(info, info_path)

    # ---- meta/episodes: the per-episode stats LeRobot would have written
    for f in _episode_files(ds_dir):
        df = pd.read_parquet(f)
        for key in OUTCOME_FEATURES:
            for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                col = f"stats/{key}/{stat}"
                df[col] = [
                    per_episode_stats.get(int(ep), _empty_stats())[key][stat].tolist()
                    for ep in df["episode_index"].tolist()
                ]
        _atomic_parquet(df, f)

    # ---- stats.json: add the two keys; everything already there is left exactly as it was
    _update_outcome_stats(ds_dir)

    result = _tally(verdict, lengths)
    logger.info("migrated %s: %s", ds_dir, result)
    return result


def episode_lengths(ds_dir: str) -> Dict[int, int]:
    """``{episode_index: frame count}`` from ``meta/episodes``."""
    return _episode_lengths(ds_dir)


def relabel(ds_dir: str, episode: int, outcome: Optional[str]) -> str:
    """Change one episode's verdict in place. Rewrites the two bool columns for that episode's
    rows (terminal-frame semantics, like a fresh recording), replaces its ``stats/next.*``
    row in ``meta/episodes`` and re-aggregates ``meta/stats.json``. Every other column, and
    every video, is untouched. Returns the normalized verdict written.

    Raises :class:`OutcomeColumnsMissing` on a pre-migration dataset and ``KeyError`` for an
    episode the dataset does not have.
    """
    import pandas as pd

    ds_dir = os.path.expanduser(ds_dir)
    if not has_outcome_features(ds_dir):
        raise OutcomeColumnsMissing(ds_dir)
    episode = int(episode)
    lengths = _episode_lengths(ds_dir)
    if episode not in lengths:
        raise KeyError(f"episode {episode} is not in {ds_dir} (has {len(lengths)} episodes)")
    state = normalize(outcome)
    flags = terminal_flags(lengths[episode], state)

    touched = 0
    for f in sorted(glob.glob(os.path.join(ds_dir, "data", "**", "*.parquet"), recursive=True)):
        ep_col = pd.read_parquet(f, columns=["episode_index"])["episode_index"].to_numpy()
        rows = np.flatnonzero(ep_col == episode)
        if rows.size == 0:
            continue
        df = pd.read_parquet(f)
        fr = df["frame_index"].to_numpy()[rows]
        succ = df[SUCCESS_KEY].to_numpy().copy()
        done = df[DONE_KEY].to_numpy().copy()
        succ[rows] = flags[SUCCESS_KEY][fr, 0]
        done[rows] = flags[DONE_KEY][fr, 0]
        df[SUCCESS_KEY] = succ.astype(bool)
        df[DONE_KEY] = done.astype(bool)
        _atomic_parquet(df, f)
        touched += rows.size
    if touched != lengths[episode]:
        logger.warning("episode %d: rewrote %d rows but meta says %d frames", episode, touched, lengths[episode])

    stats = {SUCCESS_KEY: _feature_stats(flags[SUCCESS_KEY]), DONE_KEY: _feature_stats(flags[DONE_KEY])}
    for f in _episode_files(ds_dir):
        df = pd.read_parquet(f)
        hit = df.index[df["episode_index"].to_numpy() == episode]
        if len(hit) == 0:
            continue
        for key in OUTCOME_FEATURES:
            for stat, value in stats[key].items():
                col = f"stats/{key}/{stat}"
                if col not in df.columns:
                    raise OutcomeColumnsMissing(ds_dir)
                series = df[col].astype(object)
                for i in hit:
                    series.at[i] = value.tolist()
                df[col] = series
        _atomic_parquet(df, f)

    _update_outcome_stats(ds_dir)
    return state


#: LeRobot appends these to every dataset after the user's features; the recorder declares
#: the outcome features before them, so a migrated dataset puts them in the same place.
_LEROBOT_DEFAULT_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _recorded_order(columns: List[str]) -> List[str]:
    """``columns`` with the two outcome columns moved to where the recorder writes them: right
    before LeRobot's own bookkeeping columns. Column order is cosmetic to LeRobot (it maps
    by name) but a migrated file that matches a recorded one byte-for-byte in layout is one
    less thing to explain."""
    ours = [c for c in (SUCCESS_KEY, DONE_KEY) if c in columns]
    rest = [c for c in columns if c not in ours]
    cut = next((i for i, c in enumerate(rest) if c in _LEROBOT_DEFAULT_KEYS), len(rest))
    return rest[:cut] + ours + rest[cut:]


def _declare(features: Dict[str, dict]) -> Dict[str, dict]:
    """``features`` with the outcome features (re)declared, in the recorder's position."""
    keep = {k: v for k, v in features.items() if k not in OUTCOME_FEATURES}
    ours = {k: {**spec, "shape": list(spec["shape"])} for k, spec in OUTCOME_FEATURES.items()}
    order = _recorded_order(list(keep) + list(ours))
    return {k: (ours[k] if k in ours else keep[k]) for k in order}


def _update_outcome_stats(ds_dir: str) -> None:
    """Write the whole-dataset aggregate for the two outcome features into ``meta/stats.json``,
    touching nothing else in that file.

    Not a full re-aggregation on purpose: on a dataset that has been edited in place
    (``mark-homing`` rewrites ``observation.control_mode``, ``set-task`` rewrites
    ``task_index``) the per-episode stats and ``stats.json`` can already disagree, and which
    of the two is current differs per feature. A migration that owns two keys should
    rewrite two keys.
    """
    import pandas as pd
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import serialize_dict

    stats_names = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
    per_episode: List[Dict[str, Dict[str, np.ndarray]]] = []
    cols = ["episode_index"] + [f"stats/{k}/{st}" for k in OUTCOME_FEATURES for st in stats_names]
    for f in _episode_files(ds_dir):
        df = pd.read_parquet(f, columns=cols)
        for _, row in df.iterrows():
            ep: Dict[str, Dict[str, np.ndarray]] = {}
            for k in OUTCOME_FEATURES:
                ep[k] = {}
                for st in stats_names:
                    v = row[f"stats/{k}/{st}"]
                    v = v.tolist() if hasattr(v, "tolist") else v
                    dtype = np.int64 if st == "count" else (np.bool_ if st in ("min", "max") else np.float64)
                    ep[k][st] = np.asarray(v, dtype=dtype).reshape(-1)
            per_episode.append(ep)
    if not per_episode:
        return
    agg = serialize_dict(aggregate_stats(per_episode))
    stats_path = os.path.join(ds_dir, "meta", "stats.json")
    try:
        with open(stats_path) as fh:
            stats = json.load(fh)
    except (OSError, ValueError):
        stats = {}
    for k in OUTCOME_FEATURES:
        stats[k] = agg[k]
    _atomic_json(stats, stats_path)


def _empty_stats() -> Dict[str, Dict[str, np.ndarray]]:
    """Stats for an episode the data files did not contain (should not happen; keeps the
    parquet rectangular rather than failing halfway)."""
    per = {s: np.zeros((1,), np.float32) for s in ("mean", "std", "q01", "q10", "q50", "q90", "q99")}
    per["min"] = per["max"] = np.zeros((1,), bool)
    per["count"] = np.array([0])
    return {SUCCESS_KEY: per, DONE_KEY: per}


def _tally(verdict: Dict[int, str], lengths: Optional[Dict[int, int]] = None) -> Dict[str, int]:
    eps = set(verdict) | set(lengths or {})
    counts = {SUCCESS: 0, FAIL: 0, UNKNOWN: 0}
    for e in eps:
        counts[verdict.get(e, UNKNOWN)] += 1
    return {"episodes": len(eps), **counts}


def _atomic_parquet(df: "pd.DataFrame", path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=4)
    os.replace(tmp, path)


# ------------------------------------------------------------------ verification


def verify_against_sidecar(ds_dir: str) -> List[str]:
    """Every disagreement between the migrated columns and the legacy sidecar, as messages.
    Empty means they agree episode for episode -- the check run before a migrated dataset is
    uploaded anywhere."""
    columns = episode_outcomes(ds_dir)
    legacy = {ep: normalize(row.get("outcome")) for ep, row in read_legacy_sidecar(ds_dir).items()}
    problems = []
    for ep in sorted(set(columns) | set(legacy)):
        a, b = columns.get(ep, "<absent>"), legacy.get(ep, "<absent>")
        if a != b:
            problems.append(f"episode {ep}: columns say {a}, sidecar says {b}")
    return problems


# ------------------------------------------------------------------ CLI


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Move episode outcomes from outcomes.jsonl into the LeRobot schema.")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="add next.success / next.done to a dataset in place")
    m.add_argument("ds_dir", help="the dataset folder (the one holding meta/ and data/)")
    m.add_argument("--force", action="store_true", help="rewrite even if the features already exist")
    v = sub.add_parser("verify", help="compare the columns with the legacy sidecar")
    v.add_argument("ds_dir")
    s = sub.add_parser("show", help="print each episode's verdict from the columns")
    s.add_argument("ds_dir")
    r = sub.add_parser("relabel", help="change one episode's verdict (success / fail / unknown)")
    r.add_argument("ds_dir")
    r.add_argument("episode", type=int)
    r.add_argument("outcome", choices=[SUCCESS, FAIL, UNKNOWN])
    a = p.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd == "migrate":
        print(migrate(a.ds_dir, force=a.force))
        problems = verify_against_sidecar(a.ds_dir)
        if problems:
            print(f"WARNING: {len(problems)} disagreement(s) with the legacy sidecar:")
            for line in problems[:20]:
                print("  " + line)
            return 1
        print("columns agree with the legacy sidecar for every episode")
        return 0
    if a.cmd == "verify":
        problems = verify_against_sidecar(a.ds_dir)
        for line in problems:
            print(line)
        print("OK" if not problems else f"{len(problems)} disagreement(s)")
        return 0 if not problems else 1
    if a.cmd == "relabel":
        print(f"episode {a.episode} -> {relabel(a.ds_dir, a.episode, a.outcome)}")
        return 0
    outcomes = episode_outcomes(os.path.expanduser(a.ds_dir))
    for ep, state in outcomes.items():
        print(f"{ep:5d}  {state}")
    print(_tally(outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
