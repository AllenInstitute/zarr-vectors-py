"""Which device an array lives on, and moving arrays between host and device.

The core works in numpy and never imports cupy. A caller may still hand
it device arrays (cupy, or anything exposing ``__cuda_array_interface__``)
and ask for device arrays back; this module is the one place that knows
how. Everything device-specific is behind :func:`_gpu`, which imports the
optional :mod:`zarr_vectors.gpu` extension only when a device is actually
involved.

The contract every array-form entry point keeps:

- a writer copies each device argument to the host **once**, at entry
  (:func:`to_host`), and encodes there, so the bytes it writes are the
  ones a numpy caller would get;
- a reader decodes on the host and copies each array it returns to the
  device **once**, at exit (:func:`to_device`);
- ``device="cuda"`` without the extension is an error, never a host
  array returned in its place.

Device arrays are recognised by ``__cuda_array_interface__`` rather than
the array-API ``__array_namespace__``, which cupy 13 does not define.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from zarr_vectors.exceptions import ZVError

Device = Literal["cpu", "cuda"]
_DEVICES: tuple[str, ...] = ("cpu", "cuda")

_INSTALL_HINT = (
    "Install the optional GPU extension: pip install 'zarr-vectors[gpu]', "
    "or in a conda environment, conda install -c conda-forge cupy."
)


def is_device_array(a: Any) -> bool:
    """Whether ``a`` lives in device memory.

    Checked on the type, so a property that would itself touch the device
    is never evaluated.
    """
    return hasattr(type(a), "__cuda_array_interface__")


def resolve_device(device: str | None, *like: Any) -> Device:
    """The device a result should land on.

    ``None`` means "where the inputs are": ``"cuda"`` if any of ``like``
    is a device array, else ``"cpu"``. A reader with no array input
    therefore returns host arrays unless asked otherwise.
    """
    if device is None:
        return "cuda" if any(is_device_array(a) for a in like) else "cpu"
    if device not in _DEVICES:
        raise ZVError(f"device={device!r}; expected one of {_DEVICES} or None")
    return device  # type: ignore[return-value]


@dataclass
class TransferStats:
    """Host-device copies made inside :func:`count_transfers`."""

    d2h_calls: int = 0
    d2h_bytes: int = 0
    h2d_calls: int = 0
    h2d_bytes: int = 0


_local = threading.local()


@contextmanager
def count_transfers() -> Iterator[TransferStats]:
    """Count the copies :func:`to_host` and :func:`to_device` make.

    For tests: "one copy per array" is the promise, and counting is how
    it is kept. Per thread, and nests.
    """
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    stats = TransferStats()
    stack.append(stats)
    try:
        yield stats
    finally:
        stack.remove(stats)


def _record(kind: str, nbytes: int) -> None:
    for stats in getattr(_local, "stack", ()):
        setattr(stats, f"{kind}_calls", getattr(stats, f"{kind}_calls") + 1)
        setattr(stats, f"{kind}_bytes", getattr(stats, f"{kind}_bytes") + int(nbytes))


def to_host(a: Any, *, dtype: Any = None) -> np.ndarray:
    """``a`` as a numpy array, copying it off the device if it is there.

    A device array is copied exactly once; a host array is not copied
    unless ``dtype`` asks for a cast.
    """
    if is_device_array(a):
        get = getattr(a, "get", None)
        host = get() if callable(get) else _gpu().to_host(a)
        _record("d2h", host.nbytes)
    else:
        host = np.asarray(a)
    if dtype is not None:
        host = host.astype(dtype, copy=False)
    return host


def to_device(a: Any, device: str | None) -> Any:
    """``a`` on ``device``; ``None`` leaves it where it is."""
    if device is None:
        return a
    if device == "cpu":
        return to_host(a)
    if device == "cuda":
        if is_device_array(a):
            return a
        host = np.asarray(a)
        out = _gpu().upload(host)
        _record("h2d", host.nbytes)
        return out
    raise ZVError(f"device={device!r}; expected one of {_DEVICES} or None")


def namespace(*arrays: Any) -> Any:
    """The array module for ``arrays``: cupy if any is on the device."""
    if any(is_device_array(a) for a in arrays):
        return _gpu().xp
    return np


def encode_on_device(*arrays: Any) -> bool:
    """Whether an encoder handed ``arrays`` should run on the device.

    True when any of them is a device array that cupy can take as it is
    (a cupy array, or one exporting DLPack, such as a torch tensor) and
    the GPU extension imports, unless ``ZARR_VECTORS_GPU_ENCODE=0`` asks
    for the host encoders -- a way to rule the device encoders out when
    chasing a difference; the bytes are the same either way. Any other
    device array is copied to the host once and encoded there.
    """
    import os

    device = [a for a in arrays if is_device_array(a)]
    if not device or os.environ.get("ZARR_VECTORS_GPU_ENCODE", "1") == "0":
        return False
    if not all(
        type(a).__module__.partition(".")[0] == "cupy" or hasattr(type(a), "__dlpack__")
        for a in device
    ):
        return False
    from zarr_vectors._runtime import _gpu_extension

    return _gpu_extension()


def _gpu() -> Any:
    """The optional GPU extension, imported on first use."""
    try:
        import zarr_vectors.gpu as gpu
    except ImportError as exc:
        raise ZVError(f"A device array was requested. {_INSTALL_HINT}") from exc
    return gpu
