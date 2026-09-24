"""Device-side decode for :func:`zarr_vectors.core.cells.read_cells`.

For every array :func:`~zarr_vectors.gpu._fetch.cell_source` recognises,
a device read of ``C`` cells is:

1. one fetch of the cells' stored bytes into device memory
   (:mod:`~zarr_vectors.gpu._fetch`);
2. for a zstd array, one kernel walking every frame's structure (nvCOMP
   trusts its input, and a truncated frame can hang it), then one batched
   nvCOMP decode (:mod:`~zarr_vectors.gpu._codecs`);
3. one kernel reading every vlen frame (and a ragged link cell's group
   header), and one download of the payload lengths;
4. one kernel copying every payload into the output buffer, which is then
   viewed as the array's dtype -- no per-row work anywhere, and no host
   copy of the data.

The result is the same :class:`~zarr_vectors.core.cells.CellColumn` the
host path builds, element for element; the host path stays the reference
and the tests compare the two. A cell that fails any check is reported
the way the host path reports it, and reads as empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np

from zarr_vectors.gpu import _codecs, _fetch, _kernels

if TYPE_CHECKING:  # pragma: no cover
    from zarr_vectors.core.cells import _Layout
    from zarr_vectors.core.group import Group


@dataclass
class Payloads:
    """One array's decoded cell payloads, by unique chunk key."""

    index: dict[str, int]          # chunk key -> row in the arrays below
    ptrs: np.ndarray               # (U,) uint64 device addresses, on the host
    lengths: np.ndarray            # (U,) int64 payload bytes
    errors: dict[str, str]         # chunk key -> reason
    keepalive: list[Any] = field(default_factory=list)

    def length(self, key: str) -> int:
        i = self.index.get(key)
        return 0 if i is None else int(self.lengths[i])


def supports(
    level_group: Group, name: str, lay: _Layout | None, *, zstd: bool,
) -> _fetch.CellSource | None:
    """The array's cell source, if its cells can be decoded on the device.

    ``zstd`` admits zstd-compressed arrays, which go through nvCOMP. It
    is off unless the caller asked for device decode outright: nvCOMP
    does not survive corrupt input the way libzstd does (see
    :mod:`zarr_vectors.gpu._codecs`).
    """
    if lay is not None and not lay.dtype.isnative:
        return None
    node = level_group._sharded_chunk_array(name)
    src = _fetch.cell_source(name, node)
    if src is None or (src.zstd and (not zstd or _codecs.nvcomp() is None)):
        return None
    return src


def fetch_payloads(
    items: list[tuple[_fetch.CellSource, list[str], np.ndarray, bool]],
    *,
    io: _fetch.IO = "auto",
) -> list[Payloads]:
    """Fetch, decompress and unframe many arrays' cells as one job.

    ``items`` is ``[(source, chunk_keys, chunk_coords, ragged), ...]``;
    ``ragged`` marks a ragged links array. Every array's files are read
    in one pooled pass, every zstd frame is checked in one kernel and
    decoded in one nvCOMP batch, and every frame is unframed in one
    kernel, however many arrays there are.
    """
    requests = []
    for src, keys, coords, _ in items:
        cells = np.asarray(coords, dtype=np.int64).reshape(len(keys), -1)
        if src.origin is not None:
            cells = cells - np.asarray(src.origin, dtype=np.int64)
        requests.append((src, cells))
    raws = _fetch.fetch_many(requests, io=io)

    # One flat table of every cell of every array.
    counts = [len(keys) for _, keys, _, _ in items]
    bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(bounds[-1])
    ptrs = np.zeros(total, dtype=np.uint64)
    sizes = np.zeros(total, dtype=np.int64)
    zstd = np.zeros(total, dtype=bool)
    ragged = np.zeros(total, dtype=np.uint8)
    errors: dict[int, str] = {}
    keepalive: list[Any] = []
    for r, ((src, _, _, is_ragged), raw) in enumerate(zip(items, raws)):
        lo, hi = int(bounds[r]), int(bounds[r + 1])
        keepalive.append(raw.data)
        base = int(raw.data.data.ptr) if raw.data.size else 0
        size = raw.sizes.copy()
        for i, message in raw.errors.items():
            errors[lo + i] = message
            size[i] = 0
        sizes[lo:hi] = size
        ptrs[lo:hi] = np.where(size > 0, base + raw.starts, 0).astype(np.uint64)
        zstd[lo:hi] = src.zstd
        ragged[lo:hi] = is_ragged

    live = np.flatnonzero(zstd & (sizes > 0))
    if live.size:
        declared, walked = _kernels.zstd_walk(cp.asarray(ptrs[live]), cp.asarray(sizes[live]))
        bad = walked != 0
        for j in np.flatnonzero(bad).tolist():
            errors[int(live[j])] = _kernels.UNFRAME_ERRORS[int(walked[j])]
        good = live[~bad]
        want = declared[~bad]
        frames = [_Frame(int(ptrs[i]), int(sizes[i])) for i in good.tolist()]
        decoded = _codecs.zstd_decode(frames)
        keepalive.append(decoded)
        for i, need, out in zip(good.tolist(), want.tolist(), decoded):
            got = _codecs.device_nbytes(out)
            if got != need:
                errors[i] = f"zstd frame decoded to {got} bytes; its header declares {need}"
                continue
            ptrs[i] = _codecs.device_pointer(out)
            sizes[i] = got
    for i in errors:
        ptrs[i], sizes[i] = 0, 0

    payload, lengths, status = _kernels.unframe(cp.asarray(ptrs), cp.asarray(sizes), ragged)
    for i in np.flatnonzero(status).tolist():
        errors[i] = _kernels.UNFRAME_ERRORS[int(status[i])]
    lengths = np.where(status == 0, lengths, 0)
    for i in errors:
        lengths[i] = 0

    out = []
    for r, (_, keys, _, _) in enumerate(items):
        lo, hi = int(bounds[r]), int(bounds[r + 1])
        out.append(Payloads(
            index={k: i for i, k in enumerate(keys)},
            ptrs=payload[lo:hi],
            lengths=lengths[lo:hi],
            errors={keys[i - lo]: m for i, m in errors.items() if lo <= i < hi},
            keepalive=keepalive,
        ))
    return out


class _Frame:
    """A device byte range, as nvCOMP takes it (``__cuda_array_interface__``)."""

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, size: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (size,), "typestr": "|u1", "data": (ptr, True), "version": 3,
            "strides": None,
        }


def column(
    name: str,
    payloads: Payloads,
    keys: list[str],
    lay: _Layout,
    vertex_rows: dict[str, int],
    on_error: str,
    errors: list[Any],
) -> Any:
    """The array's :class:`CellColumn` over ``keys``, in request order."""
    from zarr_vectors.core.cells import CellColumn, CellReadError
    from zarr_vectors.exceptions import ArrayError

    itemsize = lay.dtype.itemsize
    n = len(keys)
    lengths = np.array([payloads.length(k) for k in keys], dtype=np.int64)
    bad = np.zeros(n, dtype=bool)

    def fail(i: int, message: str) -> None:
        if on_error == "raise":
            raise ArrayError(f"{lay.kind} cell {keys[i]}: {message}")
        errors.append(CellReadError(name, keys[i], f"ArrayError: {message}"))
        bad[i] = True

    if lay.kind == "links":
        tail: tuple[int, ...] = (lay.ncols,)
    elif lay.row_shape is None:
        # Width is whatever divides each cell against its vertex rows.
        widths = set()
        for i, key in enumerate(keys):
            elems = lengths[i] // itemsize
            if lengths[i] % itemsize:
                fail(i, f"{lengths[i]} bytes are not whole {lay.dtype} elements")
                continue
            if elems == 0:
                continue
            rows = vertex_rows.get(key, 0)
            if not rows or elems % rows:
                fail(i, f"{elems} attribute elements do not divide into this "
                        f"cell's {rows} vertex rows")
                continue
            widths.add(int(elems // rows))
        if len(widths) > 1:
            raise ArrayError(f"cells disagree on attribute width: {sorted(widths)}")
        tail = (widths.pop() if widths else 1,)
    else:
        tail = lay.row_shape
    row_bytes = itemsize * int(np.prod(tail, dtype=np.int64))
    uneven = (lengths % row_bytes != 0) & ~bad
    for i in np.flatnonzero(uneven).tolist():
        fail(i, f"{lengths[i]} bytes do not divide into rows of {row_bytes} bytes")
    lengths = np.where(bad, 0, lengths)

    counts = lengths // row_bytes
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    dest = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    ptrs = np.zeros(n, dtype=np.uint64)
    for i, key in enumerate(keys):
        j = payloads.index.get(key)
        if j is not None and lengths[i]:
            ptrs[i] = payloads.ptrs[j]
    out = _kernels.gather(cp.asarray(ptrs), lengths, dest, int(lengths.sum()))
    data = out.view(lay.dtype).reshape(-1, *tail)
    if lay.kind == "links":
        data = data.astype(cp.int64, copy=False)
    return CellColumn(data=data, offsets=cp.asarray(offsets), _host_offsets=offsets)
