"""Decompressing cells on the device.

zstd is the one byte codec decoded here, because it is the only one a
zarr-vectors store writes besides none: the store default is
``vlen-bytes`` alone, and ``compressor="zstd"`` (BRIDGE's default, level
5) adds a single zstd frame per cell. Decompression goes through nvCOMP's
batched zstd decoder, one call for every cell of a read.

nvCOMP does not validate what it is given: a buffer that is not zstd at
all decodes, without an error, to bytes of some length, and a truncated
frame can hang its kernel. So a frame is only handed to it after
:func:`~zarr_vectors.gpu._kernels.zstd_walk` has checked its structure
(header, every block header, exact end), and its output is only kept when
its length is the content size the header declares. A frame that fails
either check is reported per cell, the way a host decode error is.

What the walk cannot check is the entropy-coded inside of a block, and
there nvCOMP is not safe. Measured on 40 frames each corrupted by one
byte inside a block (structure intact, so the walk passes them): libzstd
raised for 32 and returned wrong bytes for 8; nvCOMP returned wrong
bytes for 9, hung its kernel for 10, and hit an illegal memory access --
which poisons the process's CUDA context -- for 21. Neither decoder
catches every corruption without a content checksum; only nvCOMP turns
one into a hang or a dead process. That is why :func:`~zarr_vectors.core.cells.read_cells`
decodes zstd arrays on the device only when asked to outright
(``decode="device"``), and on the host under ``decode="auto"``: a store
whose bytes may be damaged should not be read through here.

Compression never happens here. A store written from device arrays is
compressed on the host by libzstd, which is what keeps it byte-identical
to one written from numpy.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np


@functools.cache
def nvcomp() -> Any | None:
    """nvCOMP's Python module, or None when it is not installed."""
    try:
        from nvidia import nvcomp as mod  # type: ignore[import-not-found]
    except Exception:
        return None
    return mod


def zstd_decode(
    frames: list[Any], *, out: list[Any] | None = None, stream: Any = None,
) -> list[Any]:
    """Decompress device byte arrays, one zstd frame each, in one batch.

    Returns device arrays in the same order (``out`` itself when given);
    the caller checks their lengths against the declared content sizes.
    Runs on ``stream`` (default: the current stream) and synchronises it
    before returning, since the caller reads the outputs by pointer.
    """
    mod = nvcomp()
    if mod is None:  # pragma: no cover - callers check first
        raise RuntimeError("nvCOMP is not installed")
    if not frames:
        return []
    import cupy as cp

    stream = stream if stream is not None else cp.cuda.get_current_stream()
    codec = _codec(mod, int(stream.ptr))
    srcs = [mod.as_array(f) for f in frames]
    if out is None:
        decoded = list(codec.decode(srcs))
        stream.synchronize()
        return decoded
    try:
        got = list(codec.decode(srcs, out=[mod.as_array(o) for o in out]))
    except TypeError:
        # nvCOMP before 5 has no out=: decode, then copy into place.
        got = list(codec.decode(srcs))
        for o, d in zip(out, got):
            cp.cuda.runtime.memcpyAsync(
                device_pointer(o), device_pointer(d),
                min(device_nbytes(o), device_nbytes(d)),
                cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr,
            )
    stream.synchronize()
    # Each output as long as what was decoded into it, so the caller's
    # length check sees a short frame.
    return [
        cp.asarray(o).reshape(-1).view(cp.uint8)[:min(device_nbytes(o), device_nbytes(d))]
        for o, d in zip(out, got)
    ]


def _codec(mod: Any, stream_ptr: int = 0) -> Any:
    # A codec is bound to its stream, so only the default stream's is
    # kept: one kept for a caller's stream would outlive it, and its
    # teardown at exit fails once the CUDA context is gone.
    if not stream_ptr:
        return _default_codec(mod)
    return mod.Codec(
        algorithm="Zstd", bitstream_kind=mod.BitstreamKind.RAW, cuda_stream=stream_ptr,
    )


@functools.cache
def _default_codec(mod: Any) -> Any:
    return mod.Codec(algorithm="Zstd", bitstream_kind=mod.BitstreamKind.RAW)


class _Frame:
    """A device byte range, as nvCOMP takes it (``__cuda_array_interface__``)."""

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, size: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (size,), "typestr": "|u1", "data": (ptr, True), "version": 3,
            "strides": None,
        }


def decode_frames(
    ptrs: np.ndarray,
    sizes: np.ndarray,
    *,
    expected: np.ndarray | None = None,
    out: list[Any] | None = None,
    stream: Any = None,
) -> tuple[list[Any], dict[int, str]]:
    """Check, then decompress, zstd frames at device addresses.

    Frame ``j`` is ``sizes[j]`` bytes at ``ptrs[j]``. Its structure is
    walked on the device first (:func:`~zarr_vectors.gpu._kernels.zstd_walk`);
    it must declare ``expected[j]`` bytes when ``expected`` is given, and
    fit ``out[j]`` exactly when ``out`` is. Only frames that pass go to
    nvCOMP, in one batch, and an output is kept only when its length is
    what the frame declares. Returns one device array or None per frame,
    and the reason for each None.
    """
    import cupy as cp

    from zarr_vectors.gpu import _kernels

    n = len(ptrs)
    outputs: list[Any] = [None] * n
    errors: dict[int, str] = {}
    if n == 0:
        return outputs, errors
    stream = stream if stream is not None else cp.cuda.get_current_stream()
    with stream:
        declared, walked = _kernels.zstd_walk(
            cp.asarray(np.asarray(ptrs, dtype=np.uint64)),
            cp.asarray(np.asarray(sizes, dtype=np.int64)),
        )
    for j in np.flatnonzero(walked).tolist():
        errors[j] = _kernels.UNFRAME_ERRORS[int(walked[j])]
    for j in range(n):
        if j in errors:
            continue
        if expected is not None and int(declared[j]) != int(expected[j]):
            errors[j] = (
                f"zstd frame declares {int(declared[j])} bytes; "
                f"{int(expected[j])} expected"
            )
        elif out is not None and device_nbytes(out[j]) != int(declared[j]):
            errors[j] = (
                f"zstd frame declares {int(declared[j])} bytes; its output "
                f"holds {device_nbytes(out[j])}"
            )
    good = [j for j in range(n) if j not in errors]
    decoded = zstd_decode(
        [_Frame(int(ptrs[j]), int(sizes[j])) for j in good],
        out=None if out is None else [out[j] for j in good],
        stream=stream,
    )
    for j, d in zip(good, decoded):
        got = device_nbytes(d)
        if got != int(declared[j]):
            errors[j] = f"zstd frame decoded to {got} bytes; its header declares {int(declared[j])}"
            continue
        outputs[j] = d
    return outputs, errors


def device_pointer(a: Any) -> int:
    """The device address of an array exposing ``__cuda_array_interface__``."""
    return int(a.__cuda_array_interface__["data"][0])


def device_nbytes(a: Any) -> int:
    iface = a.__cuda_array_interface__
    count = int(np.prod(iface["shape"])) if iface["shape"] else 1
    return count * np.dtype(iface["typestr"]).itemsize
