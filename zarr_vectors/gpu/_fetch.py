"""Fetching cells' stored (undecoded) bytes into device memory.

Cell locations come from :mod:`zarr_vectors.core._cells_on_disk`. Local
arrays are read with kvikio (``io="kvikio"``) or into pinned host memory and
copied up once (``io="host"``; see :mod:`zarr_vectors._gds` for ``auto``);
any other zarr store through its async ``get`` with byte ranges. A layout
:func:`cell_source` does not positively recognise returns None and is read
on the host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cupy as cp
import numpy as np

from zarr_vectors import _gds
from zarr_vectors._gds import IO
from zarr_vectors.core._cells_on_disk import (  # noqa: F401  (re-exported)
    CellSource,
    _batched,
    _batched_groups,
    _file,
    _index_nbytes,
    _key,
    _locate,
    _parse_index,
    _Range,
    _read_ranges,
    cell_source,
    locate_local,
)


@dataclass
class RawCells:
    """Stored bytes of many cells in one device buffer.

    Cell ``i`` is ``data[starts[i]:starts[i] + sizes[i]]``; size 0 reads as
    an empty cell. ``path`` is how the bytes were read: ``gds``,
    ``kvikio-compat``, ``pinned-host`` or ``store``.
    """

    data: Any
    starts: np.ndarray
    sizes: np.ndarray
    errors: dict[int, str]
    path: str = ""


# --------------------------------------------------------------------
# Locating cells


# --------------------------------------------------------------------
# Reading


def fetch_many(
    requests: list[tuple[CellSource, np.ndarray]], *, io: IO = "auto",
) -> list[RawCells]:
    """Read the stored bytes of many arrays' cells as one job.

    ``requests`` pairs each array with ``(C, K)`` cell indices into it.
    Returns one :class:`RawCells` per request, in order; they may share a
    device buffer.
    """
    how = _gds.choose_io(io)
    out: list[RawCells | None] = [None] * len(requests)
    local = [i for i, (src, _) in enumerate(requests) if src.local_root is not None]
    remote = [i for i, (src, _) in enumerate(requests) if src.local_root is None]
    if local:
        got = _fetch_local([requests[i] for i in local], kvikio=how == "kvikio")
        if how == "host":
            path = "pinned-host"
        else:
            path = "gds" if _gds.gds_status().available else "kvikio-compat"
        for i, raw in zip(local, got):
            raw.path = path
            out[i] = raw
    for i in remote:
        out[i] = _fetch_store(*requests[i])
        out[i].path = "store"  # type: ignore[union-attr]
    return out  # type: ignore[return-value]


def fetch(src: CellSource, cells: np.ndarray, *, io: IO = "auto") -> RawCells:
    """Read the stored bytes of one array's ``cells``."""
    return fetch_many([(src, cells)], io=io)[0]


def _fetch_local(
    requests: list[tuple[CellSource, np.ndarray]], *, kvikio: bool,
) -> list[RawCells]:
    flat, errors_of = locate_local(requests)
    sizes_of = [
        np.zeros(np.asarray(cells).reshape(-1, len(src.shape)).shape[0], dtype=np.int64)
        for src, cells in requests
    ]

    if kvikio:
        return _fetch_kvikio(requests, flat, sizes_of, errors_of)

    blobs = _batched_groups(_read_ranges, [(p, off, size) for _, _, p, off, size in flat])
    kept: list[tuple[int, int, bytes]] = []
    for (r, i, *_), blob in zip(flat, blobs):
        if isinstance(blob, Exception):
            errors_of[r][i] = f"{type(blob).__name__}: {blob}"
        elif blob:
            sizes_of[r][i] = len(blob)
            kept.append((r, i, blob))
    total = sum(len(b) for _, _, b in kept)
    host = cp.cuda.alloc_pinned_memory(max(total, 1))
    view = np.frombuffer(host, dtype=np.uint8, count=total)
    starts_of = [np.zeros(len(s), dtype=np.int64) for s in sizes_of]
    pos = 0
    for r, i, blob in kept:
        view[pos:pos + len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        starts_of[r][i] = pos
        pos += len(blob)
    data = cp.empty(total, dtype=cp.uint8)
    if total:
        data.set(view)
    return [
        RawCells(data, starts, sizes, errors)
        for starts, sizes, errors in zip(starts_of, sizes_of, errors_of)
    ]


def _fetch_kvikio(
    requests: list[tuple[CellSource, np.ndarray]],
    flat: list[tuple[int, int, str, int, int]],
    sizes_of: list[np.ndarray],
    errors_of: list[dict[int, str]],
) -> list[RawCells]:
    import kvikio as kv

    def stat(path: str) -> int:
        try:
            return os.stat(path).st_size
        except FileNotFoundError:
            return 0

    whole = [j for j, f in enumerate(flat) if f[4] < 0]
    whole_sizes = dict(zip(whole, _batched(stat, [flat[j][2] for j in whole])))
    starts_of = [np.zeros(len(s), dtype=np.int64) for s in sizes_of]
    pos = 0
    jobs = []
    for j, (r, i, path, offset, size) in enumerate(flat):
        n = whole_sizes[j] if size < 0 else size
        if n <= 0:
            continue
        sizes_of[r][i] = n
        starts_of[r][i] = pos
        jobs.append((r, i, path, offset, n, pos))
        pos += n
    data = cp.empty(pos, dtype=cp.uint8)
    handles: dict[str, Any] = {}
    futures = []
    try:
        with _gds.kvikio_threads():
            for r, i, path, offset, n, at in jobs:
                fh = handles.get(path)
                if fh is None:
                    fh = handles[path] = kv.CuFile(path, "r")
                futures.append((r, i, n, fh.pread(data[at:at + n], n, offset)))
            for r, i, n, fut in futures:
                if fut.get() != n:
                    errors_of[r][i] = "short read"
                    sizes_of[r][i] = 0
    finally:
        for fh in handles.values():
            fh.close()
    return [
        RawCells(data, starts, sizes, errors)
        for starts, sizes, errors in zip(starts_of, sizes_of, errors_of)
    ]


def _fetch_store(src: CellSource, cells: np.ndarray) -> RawCells:
    """Any zarr store: async byte-range gets, one upload."""
    import asyncio

    from zarr.abc.store import RangeByteRequest, SuffixByteRequest
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    cells = np.asarray(cells, dtype=np.int64).reshape(-1, len(src.shape))
    proto = default_buffer_prototype()

    async def get(key: str, byte_range: Any = None) -> bytes | None:
        buf = await src.store.get(key, prototype=proto, byte_range=byte_range)
        return None if buf is None else buf.to_bytes()

    def read_indexes(keys: list[str]) -> dict[str, Any]:
        n = _index_nbytes(src)
        rng = SuffixByteRequest(n) if src.index_at_end else RangeByteRequest(0, n)

        async def one(key: str) -> Any:
            try:
                return _parse_index(src, await get(key, rng))
            except Exception as exc:  # noqa: BLE001 - reported per cell
                return exc

        async def run() -> list[Any]:
            return await asyncio.gather(*(one(k) for k in keys))

        return dict(zip(keys, sync(run())))

    ranges, errors = _locate(src, cells, read_indexes)

    async def one_cell(r: _Range | None) -> Any:
        if r is None:
            return None
        try:
            if r.size < 0:
                return await get(r.key)
            return await get(r.key, RangeByteRequest(r.offset, r.offset + r.size))
        except Exception as exc:  # noqa: BLE001 - reported per cell
            return exc

    async def run_cells() -> list[Any]:
        return await asyncio.gather(*(one_cell(r) for r in ranges))

    blobs = sync(run_cells())
    parts: list[bytes] = []
    sizes = np.zeros(len(ranges), dtype=np.int64)
    for i, blob in enumerate(blobs):
        if isinstance(blob, Exception):
            errors[i] = f"{type(blob).__name__}: {blob}"
        elif blob:
            parts.append(blob)
            sizes[i] = len(blob)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    joined = b"".join(parts)
    data = cp.asarray(np.frombuffer(joined, dtype=np.uint8)) if joined else cp.empty(0, cp.uint8)
    return RawCells(data, starts, sizes, errors)
