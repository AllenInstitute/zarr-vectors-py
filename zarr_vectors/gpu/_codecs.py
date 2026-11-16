"""zstd decompression of cells on the device, through nvCOMP's batched decoder.

nvCOMP does not validate its input, so a frame reaches it only after
:func:`~zarr_vectors.gpu._kernels.zstd_walk` has checked its structure, and
an output is kept only when its length is the declared content size;
failures are reported per cell. Damage inside a block can still hang nvCOMP
or kill the CUDA context, which is why ``read_cells`` decodes on the device
only under ``decode="device"`` (see ``docs/how_to/gpu.md``). Compression is
never done here: device-written stores are compressed on the host.
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

    Returns device arrays in order (views of ``out`` when given), unchecked
    against declared sizes. Synchronises ``stream`` (default: current).
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
    # Only the default stream's codec is cached: one bound to a caller's
    # stream would outlive it and fail teardown at exit.
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

    Frame ``j`` is ``sizes[j]`` bytes at ``ptrs[j]`` and must declare
    ``expected[j]`` bytes and fit ``out[j]`` exactly, when given. Returns
    one device array or None per frame, and the reason for each None.
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
