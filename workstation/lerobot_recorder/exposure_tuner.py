"""Interactive exposure tuner — match camera brightness by eye and by number.

Live feed from every configured camera side by side, sliders for the sensor controls
that matter (auto-exposure, exposure, gain, white balance), and a mean-luma readout
per camera plus the delta against a chosen reference. Drive the delta to ~0 and the
cameras match; then write the values straight into ``config.yaml``.

    workstation/yam-data tune

Why brightness needs matching at all: the wrists are D405s and the agentview is a
D455, and they do NOT share an exposure scale. Color exposure on a D455's ``RGB
Camera`` counts in 100 us steps (range 1..10000); a D405 has no RGB sensor, so its
color stream is driven by the ``Stereo Module``, which counts in 1 us steps (range
1..165000). The same number means a 100x different exposure time across the two, so
values are never portable between models — tune each camera against the luma readout
instead of copying numbers.

Clipping is shown next to each luma value ("clip 66%" = that share of pixels is at
or near pure white). A camera whose highlights are clipped has thrown detail away
that no downstream training can recover, so prefer matching cameras DOWN to the
darkest well-exposed one rather than up into saturation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Controls offered per camera, in application order (auto toggles first — writing
# `exposure` while auto-exposure is on is silently ignored by RealSense).
TUNABLE = ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance")

# Options that are 0/1 toggles rather than continuous sliders.
TOGGLES = ("enable_auto_exposure", "enable_auto_white_balance")


def mean_luma(frame: np.ndarray) -> float:
    """Rec.601 luma averaged over an RGB uint8 frame (0..255)."""
    if frame is None or frame.size == 0:
        return 0.0
    f = frame.astype(np.float32)
    return float((0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).mean())


def clipped_fraction(frame: np.ndarray, threshold: int = 250) -> float:
    """Share of pixels at/above `threshold` luma — i.e. blown-out highlights (0..1)."""
    if frame is None or frame.size == 0:
        return 0.0
    f = frame.astype(np.float32)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return float((luma >= threshold).mean())


def brightness_report(frames: Dict[str, np.ndarray], reference: Optional[str] = None) -> Dict[str, dict]:
    """Per-camera ``{luma, clipped, delta}``; ``delta`` is luma minus the reference's.

    With no reference (or one that isn't present) deltas are ``None`` — the caller
    just shows absolute values.
    """
    out = {key: {"luma": mean_luma(f), "clipped": clipped_fraction(f)} for key, f in frames.items()}
    ref_luma = out.get(reference, {}).get("luma") if reference else None
    for key, stats in out.items():
        stats["delta"] = None if ref_luma is None else stats["luma"] - ref_luma
    return out


def suggest_exposure(current: float, luma: float, target: float, rng: Tuple[float, float]) -> float:
    """Exposure that should move `luma` toward `target`, clamped to the sensor range.

    Sensor response to exposure time is close enough to linear in the mid-range for a
    ratio step to converge in a few iterations. Guarded against a black frame (luma
    ~0 would ask for an infinite step) and capped at a 4x change per step so a wildly
    off starting point can't slam into the rail and oscillate.
    """
    lo, hi = rng
    if luma <= 1.0:  # essentially black: nudge up rather than divide by ~zero
        return min(hi, max(lo, current * 2.0))
    ratio = max(0.25, min(4.0, target / luma))
    return min(hi, max(lo, current * ratio))


def format_options_yaml(key: str, options: Dict[str, float], indent: int = 2) -> str:
    """The ``config.yaml`` block for one camera, ready to paste.

    Integral values print without a trailing ``.0`` so the file stays readable.
    """
    pad = " " * indent
    lines = [f"{pad}{key}:", f"{pad}  serial: \"{options.get('serial', '')}\"" if options.get("serial") else None]
    lines = [ln for ln in lines if ln]
    if any(k != "serial" for k in options):
        lines.append(f"{pad}  options:")
        for name in TUNABLE:
            if name not in options:
                continue
            value = options[name]
            text = str(int(value)) if float(value).is_integer() else f"{value:g}"
            lines.append(f"{pad}    {name}: {text}")
    return "\n".join(lines)


def splice_camera_options(text: str, key: str, options: Dict[str, float], serial: str = "") -> str:
    """Replace (or insert) one camera's entry under ``cameras:`` in a config.yaml.

    Deliberately a line-range splice rather than a YAML load/dump round-trip: this
    file is heavily commented and a round-trip would strip every comment and reflow
    the whole document. Only the target camera's lines are touched; everything else
    in the file — including comments — is preserved byte for byte.

    Raises ``ValueError`` if there is no ``cameras:`` section or no entry for `key`,
    rather than guessing where to put it.
    """
    lines = text.splitlines()

    # locate the top-level `cameras:` mapping
    start = next(
        (i for i, ln in enumerate(lines) if ln.rstrip().startswith("cameras:") and not ln.startswith((" ", "\t"))),
        None,
    )
    if start is None:
        raise ValueError("no top-level 'cameras:' section in config.yaml")

    # its body runs until the next top-level (column-0, non-comment, non-blank) line
    end = len(lines)
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln.strip() and not ln.startswith((" ", "\t")) and not ln.lstrip().startswith("#"):
            end = i
            break

    # find `  <key>:` within the body and the extent of its (more-indented) block
    body = range(start + 1, end)
    key_line = next((i for i in body if lines[i].strip().startswith(f"{key}:")), None)
    if key_line is None:
        raise ValueError(f"no '{key}:' entry under 'cameras:' in config.yaml")
    key_indent = len(lines[key_line]) - len(lines[key_line].lstrip())

    block_end = key_line + 1
    for i in range(key_line + 1, end):
        ln = lines[i]
        if not ln.strip():  # blank lines inside the block are absorbed below
            block_end = i + 1
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent <= key_indent:
            break
        block_end = i + 1

    # keep any trailing comment on the key line (e.g. "# D455") so labels survive
    comment = ""
    if "#" in lines[key_line]:
        after = lines[key_line].split("#", 1)[1]
        comment = f"  # {after.strip()}"

    merged = dict(options)
    merged.pop("serial", None)
    replacement = [f"{' ' * key_indent}{key}:{comment}"]
    if serial:
        replacement.append(f"{' ' * key_indent}  serial: \"{serial}\"")
    if merged:
        replacement.append(f"{' ' * key_indent}  options:")
        for name in TUNABLE:
            if name not in merged:
                continue
            value = merged[name]
            shown = str(int(value)) if float(value).is_integer() else f"{value:g}"
            replacement.append(f"{' ' * key_indent}    {name}: {shown}")

    out = lines[:key_line] + replacement + lines[block_end:]
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def same_model_groups(models: Dict[str, str]) -> Dict[str, List[str]]:
    """``{model: [camera keys]}`` for models with more than one camera.

    Only same-model cameras may share settings: exposure units and ranges are per
    model (a D455's RGB sensor counts in 100 us steps, a D405's stereo sensor in
    1 us), so ganging a D405 to a D455 would be meaningless. Two D405s, by contrast,
    take identical numbers and should generally be configured identically.
    """
    groups: Dict[str, List[str]] = {}
    for key, model in models.items():
        if model:
            groups.setdefault(model, []).append(key)
    return {model: keys for model, keys in groups.items() if len(keys) > 1}


# --------------------------------------------------------------------------- device
def sensor_controls(device) -> Tuple[object, Dict[str, Tuple[float, float, float]]]:
    """``(sensor, {option_name: (min, max, current)})`` for the tunable controls.

    The sensor is the one that actually owns color exposure for this model (the
    ``Stereo Module`` on a D405, ``RGB Camera`` on a D455) — see
    :func:`~workstation.lerobot_recorder.cameras.color_control_sensor`.
    """
    import pyrealsense2 as rs

    from workstation.lerobot_recorder.cameras import color_control_sensor

    sensor = color_control_sensor(device)
    controls: Dict[str, Tuple[float, float, float]] = {}
    for name in TUNABLE:
        option = getattr(rs.option, name, None)
        if option is None:
            continue
        try:
            if not sensor.supports(option):
                continue
            rng = sensor.get_option_range(option)
            controls[name] = (rng.min, rng.max, sensor.get_option(option))
        except Exception:  # option present but unreadable on this firmware
            continue
    return sensor, controls


def set_control(sensor, name: str, value: float) -> bool:
    """Set one option, clamped to its range. False if it could not be applied."""
    import pyrealsense2 as rs

    option = getattr(rs.option, name, None)
    if option is None:
        return False
    try:
        if not sensor.supports(option):
            return False
        rng = sensor.get_option_range(option)
        sensor.set_option(option, min(rng.max, max(rng.min, value)))
        return True
    except Exception as e:
        logger.warning("could not set %s=%g: %s", name, value, e)
        return False
