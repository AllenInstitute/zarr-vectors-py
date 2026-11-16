"""Whether reads can use GPUDirect Storage, and which way a device read goes.

Local device reads go through ``kvikio`` (straight to device memory) or
``host`` (pinned host reads, one copy up); kvikio is only worth it with GDS.
"With GDS" is cuFile's ``is_gds_available`` verdict: kvikio's compat-mode
setting cannot tell. Settings are read both the kvikio 25.x and 26.x way,
answering ``host`` rather than raising. No cupy import, so
:func:`zarr_vectors.runtime_capabilities` and CPU tests can use it.
"""

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from zarr_vectors.exceptions import ArrayError

IO = Literal["auto", "kvikio", "host"]
_CHOICES = ("auto", "kvikio", "host")

#: Environment variable that picks the read path when a call leaves it
#: at ``"auto"``.
ENV = "ZARR_VECTORS_GPU_IO"

#: kvikio threads for a read, when nobody has chosen a count. kvikio's own
#: default is 1, which runs every cell's read one after another.
_THREADS = 16

_COMPAT_ON = 1  # kvikio.CompatMode.ON


@dataclass(frozen=True)
class GdsStatus:
    """What this process can do with GPUDirect Storage."""

    kvikio: str | None   # kvikio's version; None when it does not import
    available: bool      # kvikio reads here would use GDS
    why: str | None      # why they would not; None when they would


def _kvikio() -> Any | None:
    try:
        import kvikio  # type: ignore[import-not-found]
        import kvikio.defaults  # noqa: F401
    except Exception:
        return None
    return kvikio


@functools.cache
def _cufile_has_gds() -> tuple[bool, str | None]:
    """cuFile's verdict, asked once per process (it opens the driver)."""
    try:
        import kvikio.cufile_driver as driver  # type: ignore[import-not-found]

        props = getattr(driver, "properties", None)
        if props is None:  # kvikio 25.x
            props = driver.DriverProperties()
        ok = bool(props.is_gds_available)
    except Exception as exc:  # noqa: BLE001 - a probe answers, never raises
        return False, f"cuFile could not be asked ({type(exc).__name__}: {exc})"
    if not ok:
        return False, (
            "cuFile reports GPUDirect Storage unavailable (no nvidia-fs "
            "driver, or an unsupported filesystem)"
        )
    return True, None


def _setting(kv: Any, name: str) -> Any:
    """A kvikio default, from 26.x's ``get`` or 25.x's per-name getter."""
    d = kv.defaults
    if hasattr(d, "get"):
        return d.get(name)
    return getattr(d, {"compat_mode": "compat_mode", "num_threads": "get_num_threads"}[name])()


def gds_status() -> GdsStatus:
    """Whether kvikio reads in this process would use GPUDirect Storage.

    Opens the cuFile driver the first time, which initialises CUDA; a
    process that will fork should ask in its children.
    """
    kv = _kvikio()
    if kv is None:
        return GdsStatus(None, False, "kvikio is not installed")
    version = str(getattr(kv, "__version__", "")) or None
    try:
        forced = int(_setting(kv, "compat_mode")) == _COMPAT_ON
    except Exception:  # noqa: BLE001
        forced = False
    if forced:
        return GdsStatus(version, False, "kvikio compatibility mode is switched on")
    ok, why = _cufile_has_gds()
    return GdsStatus(version, ok, why)


def check_io(io: str) -> None:
    if io not in _CHOICES:
        raise ArrayError(f"io={io!r}; expected 'auto', 'kvikio' or 'host'")


def choose_io(io: str = "auto") -> Literal["kvikio", "host"]:
    """The read path for local files: ``"kvikio"`` or ``"host"``.

    ``"auto"`` defers to ``$ZARR_VECTORS_GPU_IO`` (``kvikio`` or ``host``),
    else is ``kvikio`` only when :func:`gds_status` reports GDS available.
    Explicit ``kvikio`` without GDS runs in compatibility mode.
    """
    check_io(io)
    if io == "auto":
        env = os.environ.get(ENV, "auto")
        io = env if env in ("kvikio", "host") else "auto"
    if io == "host":
        return "host"
    if io == "kvikio":
        if _kvikio() is None:
            raise ImportError(
                "io='kvikio' needs kvikio. Install it with "
                "pip install 'zarr-vectors[gpu-io]', or in a conda environment "
                "from the rapidsai channel."
            )
        return "kvikio"
    return "kvikio" if gds_status().available else "host"


@contextlib.contextmanager
def kvikio_threads() -> Iterator[None]:
    """Run kvikio's reads on several threads, for the duration.

    Only when ``$KVIKIO_NTHREADS`` is unset and the setting is still at
    kvikio's default of 1; restored afterwards.
    """
    kv = _kvikio()
    if kv is None:
        yield
        return
    try:
        current = int(_setting(kv, "num_threads"))
    except Exception:  # noqa: BLE001
        current = None
    if "KVIKIO_NTHREADS" in os.environ or current != 1:
        yield
        return
    n = min(_THREADS, os.cpu_count() or 1)
    d = kv.defaults
    scope = d.set("num_threads", n) if hasattr(d, "get") else d.set_num_threads(n)
    with scope:
        yield
