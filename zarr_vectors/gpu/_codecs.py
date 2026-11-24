"""zstd decompression of cells on the device, through nvCOMP's batched decoder.

nvCOMP does not validate its input, so a frame reaches it only after
:func:`~zarr_vectors.gpu._kernels.zstd_walk` has checked its structure, and
an output is kept only when its length is the declared content size;
failures are reported per cell. A frame written with a content checksum
(zstd ``checksum=True``) is verified after decoding, as the host decoder
does, so wrong bytes are reported rather than returned. Damage inside a
block can still hang nvCOMP or kill the CUDA context before any check
runs, which is why ``read_cells`` decodes on the device only under
``decode="device"`` (see ``docs/how_to/gpu.md``). Compression is
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
    if out is None:
        decoded = list(_decode_batch(mod, codec, frames))
        stream.synchronize()
        return decoded
    try:
        got = list(_decode_batch(mod, codec, frames, out=out))
    except TypeError:
        # nvCOMP before 5 has no out=: decode, then copy into place.
        got = list(_decode_batch(mod, codec, frames))
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


def _batch(mod: Any, buffers: list[Any]) -> Any:
    """``buffers`` as ``Codec.decode`` takes a batch: a list of ``Array``
    (nvCOMP 4 and 5), or a ``BatchArray`` (nvCOMP 6, which rejects lists)."""
    if _takes_batch_array(mod):
        return mod.as_arrays(list(buffers))
    return [mod.as_array(b) for b in buffers]


@functools.cache
def _takes_batch_array(mod: Any) -> bool:
    # Read off decode's (pybind11) signature: the batch overload names
    # ``BatchArray`` from nvCOMP 6 on, ``list[Array]`` before.
    doc = getattr(mod.Codec.decode, "__doc__", None) or ""
    return hasattr(mod, "as_arrays") and "BatchArray" in doc and "list[" not in doc.lower()


def _decode_batch(mod: Any, codec: Any, srcs: list[Any], **kwargs: Any) -> Any:
    """``codec.decode`` on a batch, in whichever form this nvCOMP takes."""
    return codec.decode(_batch(mod, srcs), **{k: _batch(mod, v) for k, v in kwargs.items()})


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
        declared, walked, has_checksum = _kernels.zstd_walk(
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
    # A frame of empty content decodes to nothing, and nvCOMP refuses the
    # null buffer such an output gets, failing the whole batch: it is
    # produced here, after the walk has checked its structure, and its
    # content checksum (if any) is still verified below.
    for j in range(n):
        if j not in errors and int(declared[j]) == 0:
            outputs[j] = out[j] if out is not None else cp.empty(0, dtype=cp.uint8)
    good = [j for j in range(n) if j not in errors and int(declared[j]) != 0]
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
    # Frames written with a content checksum (zstd checksum=True) are
    # checked against it, as the host decoder does: wrong bytes out of a
    # damaged block become an error rather than data.
    summed = [j for j in range(n) if outputs[j] is not None and has_checksum[j]]
    if summed:
        with stream:
            ok = _kernels.zstd_checksums_match(
                cp.asarray(np.array([device_pointer(outputs[j]) for j in summed], np.uint64)),
                cp.asarray(np.array([device_nbytes(outputs[j]) for j in summed], np.int64)),
                cp.asarray(np.array(
                    [int(ptrs[j]) + int(sizes[j]) for j in summed], dtype=np.uint64,
                )),
            )
        for j, good_sum in zip(summed, ok.tolist()):
            if not good_sum:
                errors[j] = _kernels.UNFRAME_ERRORS[18]
                outputs[j] = None
    return outputs, errors


def device_pointer(a: Any) -> int:
    """The device address of an array exposing ``__cuda_array_interface__``."""
    return int(a.__cuda_array_interface__["data"][0])


def device_nbytes(a: Any) -> int:
    iface = a.__cuda_array_interface__
    count = int(np.prod(iface["shape"])) if iface["shape"] else 1
    return count * np.dtype(iface["typestr"]).itemsize
