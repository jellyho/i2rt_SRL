"""CameraManager sensor options — applied on open, ordered, failure-tolerant.

pyrealsense2 is absent on CI/dev boxes, so a minimal fake `rs` module stands in for
the SDK: enough of `rs.option`, the sensor, and the pipeline profile for
``_apply_options`` to exercise its real logic.
"""

from __future__ import annotations

import sys
import types

import pytest

from workstation.lerobot_recorder.cameras import CameraManager
from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig


class FakeSensor:
    def __init__(self, supported=None, ranges=None, fail=()):
        # supported=None -> every known option supported
        self._supported = supported
        self._ranges = ranges or {}
        self._fail = set(fail)
        self.calls = []  # (option_name, value) in application order

    def supports(self, option):
        return self._supported is None or option in self._supported

    def get_option_range(self, option):
        lo, hi = self._ranges.get(option, (0.0, 10000.0))
        return types.SimpleNamespace(min=lo, max=hi)

    def set_option(self, option, value):
        if option in self._fail:
            raise RuntimeError("device busy")
        self.calls.append((option, value))


def _fake_rs(sensor, *, options=("enable_auto_exposure", "exposure", "gain", "white_balance")):
    """A stand-in pyrealsense2 whose rs.option members are plain strings."""
    rs = types.ModuleType("pyrealsense2")
    rs.option = types.SimpleNamespace(**{name: name for name in options})
    return rs


def _profile(sensor, *, raise_on_get=False):
    class Device:
        def first_color_sensor(self):
            if raise_on_get:
                raise RuntimeError("no color sensor")
            return sensor

    return types.SimpleNamespace(get_device=lambda: Device())


@pytest.fixture
def install_rs(monkeypatch):
    def _install(sensor, **kw):
        rs = _fake_rs(sensor, **kw)
        monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
        return rs

    return _install


def _manager(options):
    spec = CameraSpec("agentview", serial="AAA", options=options)
    return CameraManager(RecorderConfig(cameras=[spec])), spec


def test_options_applied_with_auto_exposure_first(install_rs):
    sensor = FakeSensor()
    install_rs(sensor)
    mgr, spec = _manager({"exposure": 300.0, "gain": 64.0, "enable_auto_exposure": 0.0})

    mgr._apply_options(spec, _profile(sensor))

    # auto-exposure must land before exposure/gain, else the manual values are no-ops
    assert sensor.calls[0] == ("enable_auto_exposure", 0.0)
    assert dict(sensor.calls) == {"enable_auto_exposure": 0.0, "exposure": 300.0, "gain": 64.0}


def test_no_options_is_a_noop(install_rs):
    sensor = FakeSensor()
    install_rs(sensor)
    mgr, spec = _manager({})

    mgr._apply_options(spec, _profile(sensor))

    assert sensor.calls == []


def test_unknown_unsupported_and_out_of_range_options_are_skipped(install_rs):
    sensor = FakeSensor(supported={"exposure", "enable_auto_exposure"}, ranges={"exposure": (1.0, 200.0)})
    install_rs(sensor)
    mgr, spec = _manager(
        {
            "enable_auto_exposure": 0.0,
            "not_a_real_option": 1.0,  # unknown -> skipped
            "gain": 64.0,  # unsupported by this model -> skipped
            "exposure": 5000.0,  # out of range -> skipped
        }
    )

    mgr._apply_options(spec, _profile(sensor))

    assert sensor.calls == [("enable_auto_exposure", 0.0)]


def test_one_failing_option_does_not_abort_the_rest(install_rs):
    sensor = FakeSensor(fail={"exposure"})
    install_rs(sensor)
    mgr, spec = _manager({"enable_auto_exposure": 0.0, "exposure": 300.0, "gain": 64.0})

    mgr._apply_options(spec, _profile(sensor))

    assert dict(sensor.calls) == {"enable_auto_exposure": 0.0, "gain": 64.0}


def test_missing_color_sensor_is_survivable(install_rs):
    sensor = FakeSensor()
    install_rs(sensor)
    mgr, spec = _manager({"enable_auto_exposure": 0.0})

    mgr._apply_options(spec, _profile(sensor, raise_on_get=True))  # must not raise

    assert sensor.calls == []
