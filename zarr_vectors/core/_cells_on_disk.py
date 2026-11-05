"""Where a per-chunk array's cells are stored, and reading them raw.

A per-chunk array is a zarr v3 vlen-bytes array with one cell per zarr
chunk, optionally sharded. zarr reads a sharded cell through its
sharding codec one cell at a time -- the shard index, then the cell's
range, per cell -- which on a local store is almost all overhead. This
module works out where each cell's *stored* bytes are (a file, or a
byte range of a shard given by that shard's index, read once) and reads
many of them in one pooled pass, opening each file once.

Both readers use it: the host prefetch (:mod:`zarr_vectors.core._batch_reader`),
which decodes the bytes with the array's own codecs, and the device
reader (:mod:`zarr_vectors.gpu._fetch`), which decodes them on the GPU.

:func:`cell_source` returns None for a layout it does not positively
recognise, and the caller then reads that array through zarr. It never
guesses: an unknown codec, chunk shape, key encoding or shard index
codec all mean "not here".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

_EMPTY = np.uint64(2**64 - 1)


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


def cell_source(name: str, node: Any, *, any_codecs: bool = False) -> CellSource | None:
    """How to fetch ``node``'s cells undecoded, or None to leave them to zarr.

    ``any_codecs`` accepts any byte codecs a host decode can run (those
    with a synchronous decoder), rather than only none or zstd, which is
    what the device can decode.
    """
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
    if any_codecs:
        if not all(hasattr(c, "_decode_sync") for c in codecs[1:]):
            return None
    elif rest not in ([], ["ZstdCodec"]):
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
        zstd=rest == ["ZstdCodec"],
        metadata=meta,
        local_root=str(store.root) if type(store).__name__ == "LocalStore" else None,
    )


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


def locate_local(
    requests: list[tuple[CellSource, np.ndarray]],
) -> tuple[list[tuple[int, int, str, int, int]], list[dict[int, str]]]:
    """Where every requested cell of every local array is stored.

    ``requests`` pairs each array with ``(C, K)`` cell indices into it.
    Every shard index the cells need is read once, in one pooled pass.
    Returns the stored ranges as ``(request, cell, path, offset, size)``
    rows (``size < 0``: the whole file), sorted by file so a batch opens
    each shard once, and per-request errors keyed by cell. A cell with
    nothing stored has no row.
    """
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
    flat: list[tuple[int, int, str, int, int]] = []
    errors_of: list[dict[int, str]] = []
    for r, ((src, _), cells, mine) in enumerate(zip(requests, plans, by_request)):
        ranges, errors = _locate(src, cells, lambda _keys, mine=mine: mine)
        errors_of.append(dict(errors))
        for i, rng in enumerate(ranges):
            if rng is not None:
                flat.append((r, i, _file(src, rng.key), rng.offset, rng.size))
    flat.sort(key=lambda f: (f[2], f[3]))
    return flat, errors_of


def read_local(
    requests: list[tuple[CellSource, np.ndarray]],
) -> list[list[bytes | None | Exception]]:
    """The stored bytes of every requested cell, per request, in cell order.

    ``None`` is a cell with nothing stored; an exception is a cell whose
    location or bytes could not be read.
    """
    flat, errors_of = locate_local(requests)
    out: list[list[Any]] = [
        [None] * int(np.asarray(cells).reshape(-1, len(src.shape)).shape[0])
        for src, cells in requests
    ]
    for r, errors in enumerate(errors_of):
        for i, message in errors.items():
            out[r][i] = ValueError(message)
    blobs = _batched_groups(_read_ranges, [(p, off, size) for _, _, p, off, size in flat])
    for (r, i, *_), blob in zip(flat, blobs):
        out[r][i] = blob
    return out


# --------------------------------------------------------------------
# Presence from shard indexes


#: A stored inner cell this size or smaller is read and decoded before it
#: counts as present: an empty payload can be stored (a writer with
#: ``write_empty_chunks`` on), and encodes in tens of bytes under every
#: codec chain zarr-vectors writes. Anything larger holds a payload.
SMALL_CELL_BYTES = 128


def _store_get_many(store: Any, items: list[tuple[str, Any]]) -> list[bytes | None | Exception]:
    """``store.get(key, byte_range=...)`` for each item, gathered."""
    import asyncio

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    prototype = default_buffer_prototype()

    async def one(key: str, request: Any) -> bytes | None | Exception:
        try:
            buf = await store.get(key, prototype=prototype, byte_range=request)
        except Exception as exc:  # noqa: BLE001 - reported per object
            return exc
        return None if buf is None else buf.to_bytes()

    async def run() -> list[Any]:
        return list(await asyncio.gather(*(one(k, r) for k, r in items)))

    return sync(run())


def read_indexes(src: CellSource, keys: list[str]) -> list[np.ndarray | None | Exception]:
    """Each shard's index alone, by one byte-range read of the object.

    The last (or first) ``16 * cells + 4`` bytes, never the cells: from
    the file on a local store (pooled), else through the store's own
    ranged ``get`` -- a suffix request -- gathered. ``None`` for an object
    that is not there; an exception for one too short for its index, or
    failing its checksum.
    """
    if src.local_root is not None:
        return _batched(_read_index, [(src, _file(src, k)) for k in keys])
    from zarr.abc.store import RangeByteRequest, SuffixByteRequest

    n = _index_nbytes(src)
    request = SuffixByteRequest(n) if src.index_at_end else RangeByteRequest(0, n)
    out: list[np.ndarray | None | Exception] = []
    for raw in _store_get_many(src.store, [(k, request) for k in keys]):
        if raw is None or isinstance(raw, Exception):
            out.append(raw)
            continue
        try:
            out.append(_parse_index(src, raw))
        except Exception as exc:  # noqa: BLE001 - reported per object
            out.append(exc)
    return out


def _read_cell_ranges(src: CellSource, items: list[tuple[str, int, int]]) -> list[Any]:
    """The stored bytes at ``(key, offset, size)`` for each item."""
    if src.local_root is not None:
        return _batched_groups(_read_ranges, [(_file(src, k), off, size) for k, off, size in items])
    from zarr.abc.store import RangeByteRequest

    return _store_get_many(src.store, [
        (k, RangeByteRequest(off, off + size)) for k, off, size in items
    ])


def shard_cells(
    node: Any, shard_indices: list[tuple[int, ...]],
) -> dict[tuple[int, ...], np.ndarray | None] | None:
    """The cells each shard holds with a payload, from the shard indexes.

    For a sharded per-chunk array: ``{shard_index: (C, K) cell indices}``
    -- array indices, not absolute coordinates -- read from each shard's
    index rather than its cells, so a shard costs ``16 * cells + 4``
    bytes however much it holds. A cell the index says is stored counts
    when it is larger than :data:`SMALL_CELL_BYTES`; a smaller one is
    read and decoded, so an empty payload stored as such does not.

    A shard maps to ``None`` when its index could not be read (truncated,
    a failed checksum): the caller reads that shard whole. Returns
    ``None`` for an array whose layout this does not recognise.
    """
    from zarr_vectors.core._batch_reader import _decode_direct, _is_icechunk_store

    src = cell_source("", node, any_codecs=True)
    if src is None or src.shard_shape is None or _is_icechunk_store(src.store):
        return None
    try:
        decoder = _inner_decoder(node)
    except Exception:  # noqa: BLE001 - not a layout this can decode
        return None
    ndim = len(src.shape)
    shard_shape = np.asarray(src.shard_shape, dtype=np.int64)
    shape = np.asarray(src.shape, dtype=np.int64)
    keys = [_key(src, s) for s in shard_indices]
    out: dict[tuple[int, ...], np.ndarray | None] = {}
    large: dict[tuple[int, ...], np.ndarray] = {}
    small: list[tuple[tuple[int, ...], list[int], str, int, int]] = []
    for s, key, index in zip(shard_indices, keys, read_indexes(src, keys)):
        if isinstance(index, Exception):
            out[s] = None
            continue
        if index is None:  # gone since it was listed: nothing in it
            out[s] = np.empty((0, ndim), dtype=np.int64)
            continue
        flat = np.nonzero(~((index[:, 0] == _EMPTY) & (index[:, 1] == _EMPTY)))[0]
        cells = np.asarray(s, dtype=np.int64) * shard_shape + np.stack(
            np.unravel_index(flat, src.shard_shape), axis=1,
        ).reshape(-1, ndim)
        inside = np.all(cells < shape, axis=1)
        flat, cells = flat[inside], cells[inside]
        is_small = index[flat, 1].astype(np.int64) <= SMALL_CELL_BYTES
        large[s] = cells[~is_small]
        for cell, j in zip(cells[is_small].tolist(), flat[is_small].tolist()):
            small.append((s, cell, key, int(index[j, 0]), int(index[j, 1])))
    raws = _read_cell_ranges(src, [(k, off, size) for *_c, k, off, size in small]) if small else []
    found: dict[tuple[int, ...], list[list[int]]] = {}
    failed: set[tuple[int, ...]] = set()
    for (s, cell, *_rest), raw in zip(small, raws):
        try:
            if isinstance(raw, Exception):
                raise raw
            data = _decode_direct(decoder, raw)
        except Exception:  # noqa: BLE001 - that shard is read whole instead
            failed.add(s)
            continue
        if data:
            found.setdefault(s, []).append(cell)
    for s, cells in large.items():
        if s in failed:
            out[s] = None
        elif s in found:
            out[s] = np.concatenate([cells, np.asarray(found[s], dtype=np.int64)])
        else:
            out[s] = cells
    return out


def _inner_decoder(node: Any) -> Any:
    """What :func:`~zarr_vectors.core._batch_reader._decode_direct` needs
    to decode one inner cell of ``node``."""
    from zarr.core.array_spec import ArraySpec

    from zarr_vectors.core._batch_reader import _BUFFER_PROTOTYPE, _ShardedSpec

    meta = node.metadata
    return _ShardedSpec(
        source=None,
        codecs=tuple(meta.codecs[0].codecs[1:]),
        spec=ArraySpec(
            shape=(1,) * len(meta.shape),
            dtype=meta.data_type,
            fill_value=meta.fill_value,
            config=node._async_array.config,
            prototype=_BUFFER_PROTOTYPE,
        ),
    )
