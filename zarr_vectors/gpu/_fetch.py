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

_EMPTY = np.uint64(2**64 - 1)

IO = Literal["auto", "kvikio", "host"]


@dataclass(frozen=True)
class CellSource:
    """Where one array's cells are stored, and how they are encoded."""

    name: str
    store: Any
    path: str                              # the array's key prefix in the store
    shape: tuple[int, ...]                 # the array's shape, in cells
    origin: tuple[int, ...] | None         # chunk coords of cell 0
    shard_shape: tuple[int, ...] | None    # cells per shard; None: unsharded
    index_at_end: bool
    index_crc: bool
    zstd: bool
    metadata: Any                          # for encode_chunk_key
    local_root: str | None                 # filesystem root of a LocalStore


def cell_source(name: str, node: Any) -> CellSource | None:
    """How to fetch ``node``'s cells undecoded, or None to leave it to the host."""
    import zarr

    if not isinstance(node, zarr.Array):
        return None
    meta = node.metadata
    if getattr(meta, "zarr_format", None) != 3 or getattr(meta, "storage_transformers", ()):
        return None
    if meta.chunk_key_encoding.to_dict().get("name") not in ("default", "v2"):
        return None
    grid = getattr(meta.chunk_grid, "chunk_shape", None)
    if grid is None:
        return None
    codecs = tuple(meta.codecs)
    shard_shape: tuple[int, ...] | None = None
    index_at_end, index_crc = True, False
    if len(codecs) == 1 and type(codecs[0]).__name__ == "ShardingCodec":
        shard = codecs[0]
        if any(int(c) != 1 for c in shard.chunk_shape):
            return None
        shard_shape = tuple(int(c) for c in grid)
        names = [type(c).__name__ for c in shard.index_codecs]
        if names not in (["BytesCodec"], ["BytesCodec", "Crc32cCodec"]):
            return None
        endian = getattr(shard.index_codecs[0], "endian", None)
        if endian is not None and getattr(endian, "value", endian) != "little":
            return None
        index_crc = len(names) == 2
        index_at_end = getattr(shard.index_location, "value", shard.index_location) == "end"
        codecs = tuple(shard.codecs)
    elif any(int(c) != 1 for c in grid):
        return None
    if not codecs or type(codecs[0]).__name__ != "VLenBytesCodec":
        return None
    rest = [type(c).__name__ for c in codecs[1:]]
    if rest not in ([], ["ZstdCodec"]):
        return None
    store = node.store_path.store
    origin = node.attrs.get("chunk_grid_origin")
    return CellSource(
        name=name,
        store=store,
        path=node.store_path.path,
        shape=tuple(int(s) for s in meta.shape),
        origin=tuple(int(o) for o in origin) if origin else None,
        shard_shape=shard_shape,
        index_at_end=index_at_end,
        index_crc=index_crc,
        zstd=bool(rest),
        metadata=meta,
        local_root=str(store.root) if type(store).__name__ == "LocalStore" else None,
    )


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


def _key(src: CellSource, coords: tuple[int, ...]) -> str:
    key = src.metadata.encode_chunk_key(coords)
    return f"{src.path}/{key}" if src.path else key


@dataclass
class _Range:
    key: str
    offset: int
    size: int      # -1: the whole object (unsharded)


def _index_nbytes(src: CellSource) -> int:
    n = int(np.prod(src.shard_shape))
    return 16 * n + (4 if src.index_crc else 0)


def _parse_index(src: CellSource, raw: bytes | None) -> np.ndarray | None:
    """``(n_inner, 2)`` uint64 (offset, nbytes) rows, or None if absent."""
    if raw is None:
        return None
    n = int(np.prod(src.shard_shape))
    body = raw[: 16 * n]
    if len(body) != 16 * n:
        raise ValueError("shard index is truncated")
    if src.index_crc:
        import google_crc32c

        want = int.from_bytes(raw[16 * n: 16 * n + 4], "little")
        if google_crc32c.value(body) != want:
            raise ValueError("shard index fails its crc32c check")
    return np.frombuffer(body, dtype="<u8").reshape(n, 2)


def _locate(
    src: CellSource, cells: np.ndarray, read_indexes: Any,
) -> tuple[list[_Range | None], dict[int, str]]:
    """The byte range holding each cell (None: nothing stored)."""
    errors: dict[int, str] = {}
    if src.shard_shape is None:
        return [_Range(_key(src, tuple(c)), 0, -1) for c in cells.tolist()], errors
    shard_shape = np.asarray(src.shard_shape, dtype=np.int64)
    shards = cells // shard_shape
    inner = cells % shard_shape
    flat_inner = np.ravel_multi_index(inner.T, src.shard_shape) if len(cells) else inner[:, 0]
    keys = [_key(src, tuple(s)) for s in shards.tolist()]
    indexes = read_indexes(sorted(set(keys)))
    out: list[_Range | None] = []
    for i, (key, j) in enumerate(zip(keys, flat_inner.tolist())):
        index = indexes.get(key)
        if isinstance(index, Exception):
            errors[i] = f"{type(index).__name__}: {index}"
            out.append(None)
            continue
        if index is None:
            out.append(None)
            continue
        offset, nbytes = index[j]
        if offset == _EMPTY and nbytes == _EMPTY:
            out.append(None)
        else:
            out.append(_Range(key, int(offset), int(nbytes)))
    return out, errors


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


#: Files read per pool task. One task per file spends more on futures
#: than on ``open`` for a warm cache; this many keeps every worker busy
#: without that overhead.
_BATCH = 64


def _batched(fn: Any, items: list[Any]) -> list[Any]:
    """``[fn(x) for x in items]``, spread over the shared read pool."""
    from zarr_vectors.core._batch_reader import _PARALLEL_READ_MIN, _read_pool

    if len(items) < _PARALLEL_READ_MIN:
        return [fn(x) for x in items]
    batches = [items[i:i + _BATCH] for i in range(0, len(items), _BATCH)]
    out: list[Any] = []
    for part in _read_pool().map(lambda b: [fn(x) for x in b], batches):
        out.extend(part)
    return out


def _batched_groups(fn: Any, items: list[Any]) -> list[Any]:
    """``fn(batch)`` over consecutive batches of ``items``, pooled, flattened."""
    from zarr_vectors.core._batch_reader import _PARALLEL_READ_MIN, _read_pool

    if len(items) < _PARALLEL_READ_MIN:
        return fn(items)
    batches = [items[i:i + _BATCH] for i in range(0, len(items), _BATCH)]
    out: list[Any] = []
    for part in _read_pool().map(fn, batches):
        out.extend(part)
    return out


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


def _file(src: CellSource, key: str) -> str:
    return os.path.join(src.local_root or "", *key.split("/"))


def _read_ranges(items: list[tuple[str, int, int]]) -> list[bytes | None | Exception]:
    """``size`` bytes of ``path`` at ``offset`` for each item (``size < 0``:
    the whole file), opening each file once however many ranges it holds."""
    out: list[bytes | None | Exception] = []
    fds: dict[str, int | None] = {}
    try:
        for path, offset, size in items:
            if path not in fds:
                try:
                    fds[path] = os.open(path, os.O_RDONLY)
                except FileNotFoundError:
                    fds[path] = None
                except Exception as exc:  # noqa: BLE001 - reported per cell
                    out.append(exc)
                    fds[path] = None
                    continue
            fd = fds[path]
            if fd is None:
                out.append(None)
                continue
            try:
                if size < 0:
                    size = os.fstat(fd).st_size
                data = os.pread(fd, size, offset)
            except Exception as exc:  # noqa: BLE001 - reported per cell
                out.append(exc)
                continue
            out.append(data if len(data) == size else ValueError("short read"))
    finally:
        for fd in fds.values():
            if fd is not None:
                os.close(fd)
    return out


def _read_index(item: tuple[CellSource, str]) -> Any:
    src, path = item
    try:
        with open(path, "rb") as fh:
            n = _index_nbytes(src)
            if src.index_at_end:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size < n:
                    raise ValueError("shard is shorter than its index")
                fh.seek(size - n)
            return _parse_index(src, fh.read(n))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 - reported per cell
        return exc


def _fetch_local(
    requests: list[tuple[CellSource, np.ndarray]], *, kvikio: bool,
) -> list[RawCells]:
    # Every shard index any request needs, read in one pooled pass.
    wanted: dict[tuple[int, str], None] = {}
    plans = []
    for r, (src, cells) in enumerate(requests):
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, len(src.shape))
        plans.append(cells)
        if src.shard_shape is not None and len(cells):
            shards = cells // np.asarray(src.shard_shape, dtype=np.int64)
            for s in shards.tolist():
                wanted[(r, _key(src, tuple(s)))] = None
    index_keys = list(wanted)
    indexes = dict(zip(index_keys, _batched(
        _read_index, [(requests[r][0], _file(requests[r][0], k)) for r, k in index_keys],
    )))

    by_request: list[dict[str, Any]] = [{} for _ in requests]
    for (r, key), index in indexes.items():
        by_request[r][key] = index
    per_request: list[tuple[list[_Range | None], dict[int, str]]] = [
        _locate(src, cells, lambda _keys, mine=mine: mine)
        for (src, _), cells, mine in zip(requests, plans, by_request)
    ]

    # One flat list of every range, across requests.
    flat: list[tuple[int, int, str, int, int]] = []   # request, cell, path, offset, size
    for r, ((src, _), (ranges, _)) in enumerate(zip(requests, per_request)):
        for i, rng in enumerate(ranges):
            if rng is not None:
                flat.append((r, i, _file(src, rng.key), rng.offset, rng.size))

    sizes_of = [np.zeros(len(ranges), dtype=np.int64) for ranges, _ in per_request]
    errors_of = [dict(errs) for _, errs in per_request]

    if kvikio:
        return _fetch_kvikio(requests, flat, sizes_of, errors_of)

    # Sorted by file, so a batch opens each shard once for all its ranges.
    flat.sort(key=lambda f: (f[2], f[3]))
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
