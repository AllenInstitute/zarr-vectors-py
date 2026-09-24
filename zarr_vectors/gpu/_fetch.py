"""Fetching cells' stored bytes into device memory, undecoded.

A per-chunk array is a zarr v3 vlen-bytes array with one cell per zarr
chunk, optionally sharded. The host reader asks zarr for each cell and
gets it decoded; the device reader needs the *stored* bytes instead, so
it can decompress and unframe them on the GPU. This module works out
where each cell's bytes live -- a file, or a byte range of a shard given
by the shard's index -- and reads them all into one device buffer.

Three ways to read, chosen per call:

- ``kvikio``: straight into device memory with kvikio, which uses
  GPUDirect Storage when the system has it. Local files only.
- ``host``: into pinned host memory with plain reads on a thread pool,
  then one host-to-device copy. Local files only; the default, because
  without GPUDirect Storage kvikio does the same thing in smaller pieces.
- any other zarr store (fsspec, obstore, icechunk, memory) is read
  through the store's own async ``get`` with byte ranges, then copied up
  once.

:func:`cell_source` returns None for a layout it does not positively
recognise, and the caller then reads that array on the host. It never
guesses: an unknown codec, chunk shape, key encoding or shard index
codec all mean "not here".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import cupy as cp
import numpy as np

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

IO = Literal["auto", "kvikio", "host"]


@dataclass
class RawCells:
    """Stored bytes of many cells in one device buffer.

    Cell ``i`` is ``data[starts[i]:starts[i] + sizes[i]]``; a size of 0
    means nothing is stored there, which reads as an empty cell.
    """

    data: Any
    starts: np.ndarray
    sizes: np.ndarray
    errors: dict[int, str]


# --------------------------------------------------------------------
# Locating cells


# --------------------------------------------------------------------
# Reading


def _choose_io(io: IO) -> str:
    if io == "auto":
        io = os.environ.get("ZARR_VECTORS_GPU_IO", "auto")  # type: ignore[assignment]
    if io in ("kvikio", "host"):
        return io
    kvikio = _kvikio()
    if kvikio is None:
        return "host"
    import kvikio.defaults

    # Only when GPUDirect Storage has been asked for: in compatibility
    # mode kvikio reads through a bounce buffer, which the pinned host
    # path does in one copy instead of one per cell.
    return "kvikio" if int(kvikio.defaults.compat_mode()) == 0 else "host"


def _kvikio() -> Any | None:
    try:
        import kvikio  # type: ignore[import-not-found]
    except Exception:
        return None
    return kvikio


def fetch_many(
    requests: list[tuple[CellSource, np.ndarray]], *, io: IO = "auto",
) -> list[RawCells]:
    """Read the stored bytes of many arrays' cells as one job.

    ``requests`` pairs each array with ``(C, K)`` cell indices into it.
    Local arrays are read together -- one pooled pass over every file,
    one host-to-device copy (or kvikio reads) -- and so are the others,
    through their stores. The result is one :class:`RawCells` per request,
    in order; they may share a device buffer.
    """
    how = _choose_io(io)
    out: list[RawCells | None] = [None] * len(requests)
    local = [i for i, (src, _) in enumerate(requests) if src.local_root is not None]
    remote = [i for i, (src, _) in enumerate(requests) if src.local_root is None]
    if local:
        got = _fetch_local([requests[i] for i in local], kvikio=how == "kvikio")
        for i, raw in zip(local, got):
            out[i] = raw
    for i in remote:
        out[i] = _fetch_store(*requests[i])
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
