"""Wire compatibility with openpi — checked against openpi's OWN code, not a restatement.

openpi is the standard for the policy link: `~/jellyho/ACRFT` is an openpi fork whose
client-side files (`msgpack_numpy.py`, `image_tools.py`) are untouched since its `openpi
init` commit. These tests import those files directly when the fork is present, so the
assertions fail if either side drifts rather than passing against our own idea of the
format. They skip when it is not checked out.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_OPENPI = pathlib.Path.home() / "jellyho" / "ACRFT" / "packages" / "openpi-client" / "src" / "openpi_client"
# Load OUR modules by path, not `import yam_policy`. The env's editable install of
# policy_serving can point at a different worktree of this repo, in which case a plain
# import would test that checkout's copy instead of the one being changed here.
_OURS = pathlib.Path(__file__).resolve().parent.parent / "policy_serving" / "yam_policy"


def _load_from(directory: pathlib.Path, name: str, *, prefix: str):
    path = directory / f"{name}.py"
    if not path.exists():
        pytest.skip(f"reference not available at {path}")
    spec = importlib.util.spec_from_file_location(f"{prefix}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(name: str):
    return _load_from(_OPENPI, name, prefix="_openpi")


ours_msgpack = _load_from(_OURS, "msgpack_numpy", prefix="_ours")
ours_image_tools = _load_from(_OURS, "image_tools", prefix="_ours")


@pytest.fixture(scope="module")
def openpi_msgpack():
    return _load("msgpack_numpy")


@pytest.fixture(scope="module")
def openpi_image_tools():
    return _load("image_tools")


# --------------------------------------------------------------------------- msgpack
def _sample_obs():
    return {
        "observation/state": np.arange(42, dtype=np.float32),
        "observation/image": np.zeros((4, 4, 3), np.uint8),
        "prompt": "pick up the cube",
    }


def test_our_packing_is_readable_by_openpi(openpi_msgpack):
    """What the deploy client sends must arrive at an openpi server as real arrays."""
    got = openpi_msgpack.unpackb(ours_msgpack.Packer().pack(_sample_obs()))
    assert isinstance(got["observation/state"], np.ndarray)
    assert got["observation/state"].dtype == np.float32
    np.testing.assert_array_equal(got["observation/state"], np.arange(42, dtype=np.float32))
    assert got["observation/image"].shape == (4, 4, 3)
    assert got["prompt"] == "pick up the cube"


def test_openpi_packing_is_readable_by_us(openpi_msgpack):
    """And the action chunk coming back must arrive here as a real array."""
    chunk = np.arange(30 * 14, dtype=np.float32).reshape(30, 14)
    got = ours_msgpack.unpackb(openpi_msgpack.Packer().pack({"actions": chunk}))
    assert isinstance(got["actions"], np.ndarray)
    np.testing.assert_array_equal(got["actions"], chunk)


def test_byte_identical_encoding(openpi_msgpack):
    """Not just mutually decodable — the same bytes, so neither can drift unnoticed."""
    obs = _sample_obs()
    assert ours_msgpack.Packer().pack(obs) == openpi_msgpack.Packer().pack(obs)


def test_scalars_survive(openpi_msgpack):
    got = openpi_msgpack.unpackb(ours_msgpack.Packer().pack({"x": np.float32(1.5)}))
    assert got["x"] == np.float32(1.5)


# ----------------------------------------------------------------------- image tools
@pytest.mark.parametrize("shape", [(480, 640), (480, 848), (240, 320), (376, 672), (100, 133)])
@pytest.mark.parametrize("target", [224, 256])
def test_resize_with_pad_is_pixel_identical(openpi_image_tools, shape, target):
    """A policy is trained on openpi's resize; a different one is a silent input shift.

    848x480 is the case that used to differ (we rounded where openpi truncates), and it is
    reachable through the per-camera width/height in config.yaml.
    """
    img = np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8)
    ours = ours_image_tools.resize_with_pad(img, target, target)
    theirs = np.asarray(openpi_image_tools.resize_with_pad(img, target, target))
    assert ours.shape == theirs.shape
    np.testing.assert_array_equal(ours, theirs)


def test_convert_to_uint8_matches(openpi_image_tools):
    f = np.linspace(0, 1, 12, dtype=np.float32).reshape(2, 2, 3)
    np.testing.assert_array_equal(
        ours_image_tools.convert_to_uint8(f), openpi_image_tools.convert_to_uint8(f)
    )
