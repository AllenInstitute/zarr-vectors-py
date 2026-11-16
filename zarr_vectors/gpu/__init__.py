"""Optional GPU extension: device arrays for zarr-vectors reads and writes.

Install with ``pip install "zarr-vectors[gpu]"`` (or cupy from conda-forge).
Loaded lazily, through ``device="cuda"`` or a device array passed to a writer:
device reads (:mod:`._read`, :mod:`._fetch`, :mod:`._kernels`, :mod:`._codecs`)
and device encoders (:mod:`._encode`). Compression and writes stay on the
host, so stored bytes do not depend on where arrays came from. Reach it
through ``device=`` and :func:`zarr_vectors.runtime_capabilities`; the one
public import is :func:`decode_zstd` (``FEATURES`` entry ``"decode-zstd"``).
See ``docs/how_to/gpu.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    import cupy  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - exercised without cupy
    raise ImportError(
        "zarr_vectors.gpu needs cupy. Install it with "
        "pip install 'zarr-vectors[gpu]', or in a conda environment "
        "conda install -c conda-forge cupy."
    ) from exc

#: The device array module.
xp = cupy


def upload(a: Any) -> Any:
    """A host array as a device array (one host-to-device copy)."""
    return cupy.asarray(np.asarray(a))


def to_host(a: Any) -> np.ndarray:
    """Any object exposing ``__cuda_array_interface__`` as a numpy array."""
    return cupy.asarray(a).get()


def decode_zstd(
    frames: Sequence[Any],
    expected_nbytes: int | Sequence[int] | None = None,
    *,
    out: Sequence[Any] | None = None,
    stream: Any = None,
) -> tuple[list[Any], dict[int, str]]:
    """Decompress zstd frames in device memory, checking each one first.

    Catches truncation and broken frame structure, not damage inside a
    block; decode only trusted data (see ``docs/how_to/gpu.md``).

    Args:
        frames: C-contiguous device byte buffers, one zstd frame each, each
            declaring its content size.
        expected_nbytes: Required decoded size, for all frames or per frame.
        out: Optional device arrays to decode into, sized as declared.
        stream: cupy stream (default: current) on which the frames are
            ready; synchronised on return.

    Returns:
        ``(outputs, errors)``: decoded ``uint8`` arrays or None, with each
        None's reason in ``errors``. Errors are reported, never raised.
    """
    from zarr_vectors.gpu import _codecs

    if _codecs.nvcomp() is None:
        raise ImportError(
            "decode_zstd needs nvCOMP. Install it with "
            "pip install 'zarr-vectors[gpu-codecs]'."
        )
    frames = list(frames)
    n = len(frames)
    for i, f in enumerate(frames):
        if not _is_contiguous(f):
            raise ValueError(f"frame {i} is not C-contiguous")
    ptrs = np.array([_codecs.device_pointer(f) for f in frames], dtype=np.uint64)
    sizes = np.array([_codecs.device_nbytes(f) for f in frames], dtype=np.int64)
    expected = None
    if expected_nbytes is not None:
        expected = np.broadcast_to(np.asarray(expected_nbytes, dtype=np.int64), (n,))
    if out is not None:
        out = list(out)
        if len(out) != n:
            raise ValueError(f"{len(out)} outputs for {n} frames")
        for i, o in enumerate(out):
            if not _is_contiguous(o):
                raise ValueError(f"output {i} is not C-contiguous")
    outputs, errors = _codecs.decode_frames(
        ptrs, sizes, expected=expected, out=out, stream=stream,
    )
    return [
        None if o is None else cupy.asarray(o).reshape(-1).view(cupy.uint8)
        for o in outputs
    ], errors


def _is_contiguous(a: Any) -> bool:
    iface = a.__cuda_array_interface__
    strides = iface.get("strides")
    if strides is None:
        return True
    step = np.dtype(iface["typestr"]).itemsize
    for extent, stride in zip(reversed(iface["shape"]), reversed(strides)):
        if extent > 1 and stride != step:
            return False
        step *= extent
    return True


def device_count() -> int:
    """How many CUDA devices are visible; 0 if the runtime cannot say.

    Initialises the CUDA driver, so a process that will fork should call
    it in the children.
    """
    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except cupy.cuda.runtime.CUDARuntimeError:
        return 0


__all__ = ["decode_zstd", "device_count", "to_host", "upload", "xp"]
