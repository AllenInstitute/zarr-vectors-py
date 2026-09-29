"""``zarr_vectors._gds``: GDS detection and read-path choice, across kvikio APIs.

kvikio is replaced by small fakes shaped like its 25.x and 26.x APIs, so
these run without kvikio, CUDA or a GPU.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from zarr_vectors import _gds
from zarr_vectors.exceptions import ArrayError

AUTO, ON = 2, 1


class _Settings:
    def __init__(self, compat_mode: int = AUTO, num_threads: int = 1) -> None:
        self.values = {"compat_mode": compat_mode, "num_threads": num_threads}

    @contextlib.contextmanager
    def scoped(self, name: str, value: int):
        old = self.values[name]
        self.values[name] = value
        try:
            yield
        finally:
            self.values[name] = old


def _install(monkeypatch, *, api: int, gds: bool | Exception, **settings) -> _Settings:
    """Put a fake kvikio of ``api`` (25 or 26) in ``sys.modules``."""
    s = _Settings(**settings)
    kv = types.ModuleType("kvikio")
    kv.__version__ = f"{api}.0.0"
    defaults = types.ModuleType("kvikio.defaults")
    driver = types.ModuleType("kvikio.cufile_driver")

    class Props:
        @property
        def is_gds_available(self) -> bool:
            if isinstance(gds, Exception):
                raise gds
            return gds

    if api == 26:
        defaults.get = lambda name: s.values[name]
        defaults.set = lambda name, value: s.scoped(name, value)
        driver.properties = Props()
    else:
        defaults.compat_mode = lambda: s.values["compat_mode"]
        defaults.get_num_threads = lambda: s.values["num_threads"]
        defaults.set_num_threads = lambda n: s.scoped("num_threads", n)
        driver.DriverProperties = Props
    kv.defaults, kv.cufile_driver = defaults, driver
    monkeypatch.setitem(sys.modules, "kvikio", kv)
    monkeypatch.setitem(sys.modules, "kvikio.defaults", defaults)
    monkeypatch.setitem(sys.modules, "kvikio.cufile_driver", driver)
    return s


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    _gds._cufile_has_gds.cache_clear()
    monkeypatch.delenv(_gds.ENV, raising=False)
    monkeypatch.delenv("KVIKIO_NTHREADS", raising=False)
    yield
    _gds._cufile_has_gds.cache_clear()


def test_without_kvikio_everything_is_host(monkeypatch):
    monkeypatch.setitem(sys.modules, "kvikio", None)
    status = _gds.gds_status()
    assert status == _gds.GdsStatus(None, False, "kvikio is not installed")
    assert _gds.choose_io("auto") == "host"
    assert _gds.choose_io("host") == "host"
    with pytest.raises(ImportError, match="needs kvikio"):
        _gds.choose_io("kvikio")


@pytest.mark.parametrize("api", [25, 26])
def test_auto_follows_cufile(monkeypatch, api):
    _install(monkeypatch, api=api, gds=True)
    status = _gds.gds_status()
    assert status.available and status.why is None and status.kvikio == f"{api}.0.0"
    assert _gds.choose_io("auto") == "kvikio"

    _gds._cufile_has_gds.cache_clear()
    _install(monkeypatch, api=api, gds=False)
    status = _gds.gds_status()
    assert not status.available and "nvidia-fs" in status.why
    assert _gds.choose_io("auto") == "host"
    # Forcing kvikio still works without GDS: it reads in compatibility mode.
    assert _gds.choose_io("kvikio") == "kvikio"


@pytest.mark.parametrize("api", [25, 26])
def test_compat_mode_on_means_host(monkeypatch, api):
    _install(monkeypatch, api=api, gds=True, compat_mode=ON)
    status = _gds.gds_status()
    assert not status.available and "compatibility mode" in status.why
    assert _gds.choose_io("auto") == "host"


@pytest.mark.parametrize("api", [25, 26])
def test_a_failing_driver_answers_host(monkeypatch, api):
    _install(monkeypatch, api=api, gds=RuntimeError("cuFileDriverOpen failed"))
    status = _gds.gds_status()
    assert not status.available and "could not be asked" in status.why
    assert _gds.choose_io("auto") == "host"


def test_missing_properties_answer_host(monkeypatch):
    _install(monkeypatch, api=26, gds=True)
    del sys.modules["kvikio.cufile_driver"].properties
    assert _gds.choose_io("auto") == "host"


def test_precedence_argument_then_environment_then_auto(monkeypatch):
    _install(monkeypatch, api=26, gds=True)
    monkeypatch.setenv(_gds.ENV, "host")
    assert _gds.choose_io("auto") == "host"
    assert _gds.choose_io("kvikio") == "kvikio"

    _gds._cufile_has_gds.cache_clear()
    _install(monkeypatch, api=26, gds=False)
    monkeypatch.setenv(_gds.ENV, "kvikio")
    assert _gds.choose_io("auto") == "kvikio"
    assert _gds.choose_io("host") == "host"
    # Anything else in the environment is auto.
    monkeypatch.setenv(_gds.ENV, "nvme-please")
    assert _gds.choose_io("auto") == "host"


def test_unknown_io_is_refused():
    with pytest.raises(ArrayError, match="io='gds'"):
        _gds.choose_io("gds")


@pytest.mark.parametrize("api", [25, 26])
def test_threads_raised_only_while_reading(monkeypatch, api):
    s = _install(monkeypatch, api=api, gds=False)
    monkeypatch.setattr(_gds.os, "cpu_count", lambda: 64)
    with _gds.kvikio_threads():
        assert s.values["num_threads"] == _gds._THREADS
    assert s.values["num_threads"] == 1


@pytest.mark.parametrize("api", [25, 26])
def test_threads_someone_chose_are_left_alone(monkeypatch, api):
    s = _install(monkeypatch, api=api, gds=False, num_threads=4)
    with _gds.kvikio_threads():
        assert s.values["num_threads"] == 4

    s = _install(monkeypatch, api=api, gds=False)
    monkeypatch.setenv("KVIKIO_NTHREADS", "1")
    with _gds.kvikio_threads():
        assert s.values["num_threads"] == 1
