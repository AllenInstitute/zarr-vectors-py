"""Whether reads can use GPUDirect Storage, and which way a device read goes.

A device read of local files goes one of two ways (see
:mod:`zarr_vectors.gpu._fetch`): ``kvikio``, straight into device memory,
or ``host``, plain reads into pinned host memory and one copy up. kvikio
is only worth it with GPUDirect Storage (GDS); without it cuFile reads
through a bounce buffer, one copy per cell, which the host path does in
one copy for all of them.

"With GDS" is cuFile's own verdict, ``is_gds_available``: it needs the
nvidia-fs driver and a supported filesystem. kvikio's compat-mode
setting cannot tell: its default, ``AUTO``, is the same whether or not
GDS works, and ``is_compat_mode_preferred()`` is False whenever libcufile
loads, even when cuFile then falls back to POSIX reads.

kvikio moved its settings between releases (``defaults.compat_mode()``
in 25.x, ``defaults.get("compat_mode")`` in 26.x; cuFile's properties
likewise), so everything here asks both ways and answers ``host`` rather
than raise. This module does not import cupy, so
:func:`zarr_vectors.runtime_capabilities` and the CPU tests can use it.
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

    An explicit ``io`` wins; ``"auto"`` defers to ``$ZARR_VECTORS_GPU_IO``
    (``kvikio`` or ``host``; anything else is ``auto``), and ``auto``
    itself is ``kvikio`` only when :func:`gds_status` says GDS is
    available. Asking for ``kvikio`` without GDS still works: kvikio then
    reads in compatibility mode.
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

    Only when nobody has chosen a count: ``$KVIKIO_NTHREADS`` unset and
    the setting still at kvikio's default of 1. The setting is restored
    afterwards, so a process that sets its own count keeps it.
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
