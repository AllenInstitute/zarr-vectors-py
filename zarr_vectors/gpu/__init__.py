"""Optional GPU extension: device arrays for zarr-vectors reads and writes.

Installed with ``pip install "zarr-vectors[gpu]"``, or in a conda
environment by installing cupy from conda-forge (the extra would add a
second, pip-built cupy). Nothing in the core imports this module at import
time; it is loaded the first time a device array is involved, through
``device="cuda"`` on a reader or a device array passed to a writer.

What it does:

- reads (:func:`zarr_vectors.core.cells.read_cells` with
  ``device="cuda"``) fetch cells' stored bytes into device memory and
  decode them there (:mod:`._read`, :mod:`._fetch`, :mod:`._kernels`),
  with zstd through nvCOMP when asked (:mod:`._codecs`);
- writers handed device arrays run the fragment, manifest and link
  partition encoders on the device and download the encoded form
  (:mod:`._encode`);
- everything else moves arrays between host and device once per array.

Compression and the write itself stay on the host, so the bytes on disk
are the same whether the arrays came from numpy or from the device.
:func:`zarr_vectors.runtime_capabilities` reports what this install can
do (``device_decode``, ``gpu_encode``, ``gpu_codecs``, ``gpu_io``).

Callers should reach this through ``device=`` and ``runtime_capabilities``
rather than import it: its own surface is not yet promised. The one
function meant to be imported is :func:`decode_zstd`, for callers that
fetch zstd-compressed bytes into device memory themselves (a dense image
reader, say) and want them checked before nvCOMP sees them; its presence
is advertised in ``zarr_vectors.FEATURES`` as ``"decode-zstd"``.
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

    nvCOMP does not validate its input: a truncated frame can hang its
    kernel, and a damaged one can hit an illegal memory access, which
    kills the process's CUDA context. So each frame's structure (header,
    every block header, the exact end) is walked on the device before
    nvCOMP sees it, and an output is kept only when its length is the
    content size the frame declares. That catches truncation and broken
    structure. It cannot catch damage inside a compressed block, which
    only a content checksum would; decode only data you trust.

    Args:
        frames: One zstd frame per entry: C-contiguous device arrays (or
            anything exposing ``__cuda_array_interface__``), read as bytes.
            Frames must declare their content size, as zarr's and
            numcodecs' zstd do.
        expected_nbytes: The size every frame (an int) or each frame (a
            sequence) must decode to; a frame declaring anything else is
            reported, not decoded. ``None`` accepts what frames declare.
        out: Optional device arrays to decode into, one per frame, each
            exactly the size its frame declares.
        stream: A cupy stream to run on; the current stream by default.
            It is synchronised before returning. The frames must be ready
            on it: work that produced them on another stream (a copy to
            the device on the default stream, say) has to be ordered
            before the call, as for any CUDA stream.

    Returns:
        ``(outputs, errors)``: ``outputs[i]`` is frame ``i`` decoded, as a
        cupy ``uint8`` array (a view of ``out[i]`` when ``out`` is given),
        or None, with the reason in ``errors[i]``. Errors are reported,
        never raised.
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
