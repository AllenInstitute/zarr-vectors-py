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


def zstd_decode(frames: list[Any]) -> list[Any]:
    """Decompress device byte arrays, one zstd frame each, in one batch.

    Returns nvCOMP arrays in the same order; the caller checks their
    lengths against the declared content sizes. Synchronises the device
    before returning, since the caller reads the outputs by pointer.
    """
    mod = nvcomp()
    if mod is None:  # pragma: no cover - callers check first
        raise RuntimeError("nvCOMP is not installed")
    if not frames:
        return []
    import cupy as cp

    codec = _codec(mod)
    out = codec.decode([mod.as_array(f) for f in frames])
    cp.cuda.Device().synchronize()
    return list(out)


@functools.cache
def _codec(mod: Any) -> Any:
    return mod.Codec(algorithm="Zstd", bitstream_kind=mod.BitstreamKind.RAW)


def device_pointer(a: Any) -> int:
    """The device address of an array exposing ``__cuda_array_interface__``."""
    return int(a.__cuda_array_interface__["data"][0])


def device_nbytes(a: Any) -> int:
    iface = a.__cuda_array_interface__
    count = int(np.prod(iface["shape"])) if iface["shape"] else 1
    return count * np.dtype(iface["typestr"]).itemsize
