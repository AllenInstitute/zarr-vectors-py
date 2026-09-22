"""Async-batched chunk + array-metadata writer.

The default ZV write paths
(:meth:`zarr_vectors.core.group.Group.write_bytes` and
:meth:`zarr_vectors.core.group.Group.write_array_meta`) each issue
synchronous zarr operations that bottom out at one obstore PUT per
operation, serialised through zarr's sync→async bridge.  Against a
high-latency object store (GCS/S3/Azure) every PUT round-trip is paid
serially — for 200k points + 3 attributes that's ~1000 chunk PUTs +
~6 per-array metadata PUTs at ~15 ms each ≈ 15 s of wall time spent
waiting on the network.

This module provides a deferred-batch path that:

1. Collects ``(array_name, chunk_key, bytes, record_presence)`` cell
   writes and ``{group_name: meta_dict}`` group metadata while the
   caller iterates (no round-trips during collection).
2. On flush, writes each array's queued cells in one concurrent
   ``set_coordinate_selection``, so N cells cost roughly one round-trip
   rather than N.
3. PUTs each queued group ``zarr.json`` directly, in the same
   :func:`asyncio.gather`.

The on-disk layout matches what ``write_bytes`` produces.  Chunk-array
codecs come from the array's own creation (see
:meth:`zarr_vectors.core.group.Group.create_sharded_chunk_array`), which
honours the session compressor — so unlike the pre-0.9 per-cell layout
there is no codec decision left to make at flush time.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import threading
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.errors import UnstableSpecificationWarning

from zarr_vectors.exceptions import StoreError

# Upper bound on concurrent per-array flushes.  The work is I/O-bound
# (each worker blocks on zarr's shared event loop), so this is about how
# many array writes may be in flight, not about CPU parallelism.
_FLUSH_MAX_WORKERS = 32

# Pyodide/WASM has no thread support: ``threading.Thread.start()`` raises
# ``RuntimeError: can't start new thread``, so constructing a pool at all
# is fatal there — clamping ``_FLUSH_MAX_WORKERS`` to 1 would not help,
# since ThreadPoolExecutor still starts a worker thread.  Flush serially
# instead.  The arrays are independent, so the only cost is wall-clock.
_NO_THREADS = sys.platform == "emscripten" or sys.platform == "wasi"


def _is_icechunk_store(store: Any) -> bool:
    """Return True when ``store`` is an icechunk-backed :class:`zarr.abc.store.Store`.

    Detection is by class name to avoid importing icechunk at module
    load time (it's an optional dep).  Icechunk tracks groups as
    first-class entities; a raw ``store.set("…/zarr.json", …)`` PUT
    doesn't register them, so those stores need the sync
    ``require_group`` path the icechunk session knows how to commit.
    """
    cls = type(store)
    return cls.__name__ == "IcechunkStore" or cls.__module__.startswith("icechunk")


# ---------------------------------------------------------------------------
# Async gather
# ---------------------------------------------------------------------------


async def _async_put_chunks(
    store: Any,
    items: list[tuple[str, bytes]],
) -> None:
    """``await asyncio.gather`` every ``store.set(key, buffer)`` in ``items``."""
    proto = default_buffer_prototype()
    tasks = [
        store.set(key, proto.buffer.from_bytes(data))
        for key, data in items
    ]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


_GROUP_ZARR_JSON_PREFIX = '{"zarr_format":3,"node_type":"group","attributes":'


def _group_zarr_json_bytes(attributes: dict[str, Any]) -> bytes:
    """Build a v3 group ``zarr.json`` with the given attributes."""
    return (
        _GROUP_ZARR_JSON_PREFIX + json.dumps(attributes) + "}"
    ).encode("utf-8")


def _presence_after(
    arr: zarr.Array, cells: dict[str, tuple[bytes, bool]],
) -> list[str] | None:
    """The ``nonempty_chunks`` manifest ``cells`` leaves behind, or
    ``None`` when no cell asked to be recorded.

    Only over the cells whose ``record_presence`` is True: an array
    written entirely with ``record_presence=False`` keeps its manifest
    untouched, because that attribute is shared by every cell and
    stamping it would reintroduce, at batch granularity, the very
    cross-worker race the flag exists to avoid.  See
    :meth:`zarr_vectors.core.group.Group.write_bytes`.
    """
    from zarr_vectors.core.group import _NONEMPTY_CHUNKS_ATTR

    current = arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
    if current is None:
        # No manifest: presence is derived from the store, and a list of
        # this batch's keys would hide every cell written before it.
        return None
    present = set(current)
    stamp = False
    for chunk_key, (data, record_presence) in cells.items():
        if not record_presence:
            continue
        stamp = True
        if data:
            present.add(chunk_key)
        else:
            present.discard(chunk_key)
    return sorted(present) if stamp else None


def _emit_batch_presence(
    array_name: str, cells: dict[str, tuple[bytes, bool]],
) -> None:
    """Report this array's landed stamps to ``observe_presence_writes``.

    Only the cells that asked to be recorded, matching what
    ``_presence_after`` folded into the manifest.  The batch paths write
    the whole list at once rather than through
    ``_record_nonempty_chunk``, so without this the instrument would be
    silent on exactly the writes that bypass it.
    """
    from zarr_vectors.core.group import _emit_presence, _presence_observers

    if not _presence_observers:
        return
    for chunk_key, (data, record_presence) in cells.items():
        if record_presence:
            _emit_presence(array_name, chunk_key, bool(data))


def _flush_one_array(
    zarr_group: zarr.Group,
    array_name: str,
    cells: dict[str, tuple[bytes, bool]],
    *,
    arr: zarr.Array | None = None,
) -> None:
    """Write one array's queued cells and stamp its manifest.

    Self-contained per array — it resolves its own node and touches no
    shared state — which is what lets :func:`_flush_native_cells` run
    these concurrently.  This is the general path, through zarr's own
    ``set_coordinate_selection``; a local unsharded array takes the
    direct path below instead.
    """
    from zarr_vectors.core.group import (
        _CHUNK_GRID_ORIGIN_ATTR,
        _NONEMPTY_CHUNKS_ATTR,
    )

    if arr is None:
        arr = zarr_group[array_name]
    ndim = arr.ndim
    origin_raw = arr.attrs.get(_CHUNK_GRID_ORIGIN_ATTR)
    origin = (
        tuple(int(o) for o in origin_raw) if origin_raw else None
    )
    axis_coords: list[list[int]] = [[] for _ in range(ndim)]
    values: list[bytes] = []
    for chunk_key, (data, _record_presence) in cells.items():
        coords = tuple(int(p) for p in chunk_key.split("."))
        # Cell index = coord - origin (grid anchored at min coord).
        # A rank disagreement leaves the coords alone rather than letting
        # ``zip`` truncate them into a plausible-looking wrong cell; the
        # arity check below then rejects it. Same reasoning as
        # ``group._coord_to_index``.
        index = (
            coords if origin is None or len(coords) != len(origin)
            else tuple(c - o for c, o in zip(coords, origin))
        )
        if len(index) != ndim:
            raise StoreError(
                f"Cannot write to array {array_name!r}: chunk_key "
                f"{chunk_key!r} has rank {len(index)} but the array's "
                f"grid has rank {ndim}"
            )
        for ax in range(ndim):
            axis_coords[ax].append(index[ax])
        values.append(bytes(data))

    obj = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        obj[i] = v
    selection = tuple(np.asarray(a, dtype=np.intp) for a in axis_coords)
    arr.set_coordinate_selection(selection, obj)
    present = _presence_after(arr, cells)
    if present is not None:
        arr.attrs[_NONEMPTY_CHUNKS_ATTR] = present
        # On a worker thread when several arrays flush at once, which is
        # what ``observe_presence_writes`` promises: the calling thread
        # is whichever one performed the write.
        _emit_batch_presence(array_name, cells)


# --------------------------------------------------------------------
# Local-filesystem fast path
# --------------------------------------------------------------------
#
# The write-side twin of the direct reader in ``_batch_reader``.  Against
# a LocalStore, ``set_coordinate_selection`` costs far more than the
# bytes it writes.  A coordinate selection is never a "complete chunk"
# as far as zarr's pipeline can tell, so for every cell it first reads
# the object back (a file that does not exist yet -- an open that fails,
# an exception, a thread hop), encodes on the event loop, and writes
# through pathlib: mkdir, a uuid-named temporary, open, replace, each
# hop bounced through ``sync()``.  Measured on 512 compressed cells:
# 625 ms through zarr against 116 ms encoding and writing the same files
# from a thread pool -- and a 1M-point write spent over half its wall
# time in that flush.
#
# The encode is zarr's own, not a reimplementation: the numcodecs
# VLenBytes instance ``VLenBytesCodec`` delegates to, then each
# BytesBytes codec's ``_encode_sync`` -- so a cell written here decodes
# byte-for-byte through ``getitem``, and through the direct reader.  The
# file lands the way zarr lands it, too: written to a temporary and
# renamed over, so a crash leaves no half-written cell.  Anything this
# path does not positively recognise (a shard, another store, a codec
# with no sync entry point) goes through ``_flush_one_array``, which
# stays the definition.


class _DirectWriteSpec(NamedTuple):
    """Everything needed to write one array's cells onto the filesystem."""

    root: str                       # directory holding the chunk tree
    separator: str                  # chunk-key separator ("/" or ".")
    origin: tuple[int, ...] | None  # chunk_grid_origin, or None for zero
    shape: tuple[int, ...]          # cell-grid shape, for bounds checks
    codecs: tuple[Any, ...]         # BytesBytes codecs, encode order
    spec: Any                       # ArraySpec the codecs encode against
    write_empty_chunks: bool        # zarr's array.write_empty_chunks


def _direct_write_spec(
    zarr_group: zarr.Group, array_name: str, arr: zarr.Array,
) -> _DirectWriteSpec | None:
    """A :class:`_DirectWriteSpec` for ``arr``, or None to use zarr.

    Built on the reader's :func:`_direct_spec`, so the two paths agree
    on exactly which arrays are direct-addressable, then narrowed to
    codecs that can also encode synchronously.
    """
    from zarr_vectors.core._batch_reader import _direct_spec

    spec = _direct_spec(zarr_group, array_name, arr)
    if spec is None:
        return None
    if not all(hasattr(c, "_encode_sync") for c in spec.codecs):
        return None
    try:
        write_empty = bool(arr._async_array.config.write_empty_chunks)
    except Exception:
        return None
    return _DirectWriteSpec(
        root=spec.root,
        separator=spec.separator,
        origin=spec.origin,
        shape=spec.shape,
        codecs=spec.codecs,
        spec=spec.spec,
        write_empty_chunks=write_empty,
    )


def _encode_direct(spec: _DirectWriteSpec, data: bytes) -> bytes:
    """One cell's payload as the bytes zarr would store for it.

    Two routes to the same bytes.  When every BytesBytes codec is zstd
    -- the default pipeline, and the only one a ``Layout`` resolves to
    besides none -- the vlen frame is packed by hand (numcodecs'
    VLenBytes layout for one item: item count, byte length, payload) and
    handed straight to the numcodecs Zstd instance the zarr codec itself
    delegates to.  That is byte-identical to the general route and about
    twenty times cheaper per cell, which matters because a links family
    is a hundred thousand cells of a few rows each and the encode was
    half the flush.  Anything else takes the general route: a numpy
    object cell through zarr's own ``_encode_sync`` chain.
    """
    codecs = spec.codecs
    if all(type(codec).__name__ == "ZstdCodec" for codec in codecs):
        framed: Any = struct.pack("<II", 1, len(data)) + data
        for codec in codecs:
            framed = codec._zstd_codec.encode(framed)
        return bytes(framed)

    from zarr_vectors.core._batch_reader import _VLEN_BYTES

    obj = np.empty(1, dtype=object)
    obj[0] = data
    buffer = spec.spec.prototype.buffer.from_bytes(_VLEN_BYTES.encode(obj))
    for codec in codecs:
        buffer = codec._encode_sync(buffer, spec.spec)
    return bytes(buffer.to_bytes())


_MADE_DIRS_LOCK = threading.Lock()


def _direct_write_one(
    spec: _DirectWriteSpec, path: str, data: bytes, made: set[str],
) -> None:
    """Encode and land one cell at ``path``.

    An empty payload is what zarr treats as the fill value and, with
    ``write_empty_chunks`` off (its default), *deletes* rather than
    writes -- so this does the same, and a cell that reads back ``b""``
    is one with no object behind it either way.
    """
    if not data and not spec.write_empty_chunks:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    encoded = _encode_direct(spec, data)
    directory = os.path.dirname(path)
    if directory not in made:
        os.makedirs(directory, exist_ok=True)
        with _MADE_DIRS_LOCK:
            made.add(directory)
    tmp = f"{path}.{uuid.uuid4().hex}.partial"
    try:
        with open(tmp, "wb") as fh:
            fh.write(encoded)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


#: Cells above which the direct path writes from a pool.  Encoding
#: (numcodecs releases the GIL) and the file I/O both overlap, and a
#: pool costs a few milliseconds to stand up, so below this it is not
#: repaid.
_PARALLEL_WRITE_MIN = 32
_PARALLEL_WRITE_WORKERS = min(16, (os.cpu_count() or 4) * 2)

_WRITE_POOL: Any = None
_WRITE_POOL_LOCK = threading.Lock()


def _write_pool() -> Any:
    """The shared writer pool, created on first use."""
    global _WRITE_POOL
    if _WRITE_POOL is None:
        with _WRITE_POOL_LOCK:
            if _WRITE_POOL is None:
                _WRITE_POOL = ThreadPoolExecutor(
                    max_workers=_PARALLEL_WRITE_WORKERS,
                    thread_name_prefix="zv-write",
                )
    return _WRITE_POOL


def _direct_write_many(
    jobs: list[tuple[_DirectWriteSpec, str, bytes]],
) -> None:
    """Write every ``(spec, path, data)`` job, in parallel when there are
    enough of them to be worth it.  Raises the first failure."""
    if not jobs:
        return
    made: set[str] = set()
    if len(jobs) < _PARALLEL_WRITE_MIN or _NO_THREADS:
        for spec, path, data in jobs:
            _direct_write_one(spec, path, data, made)
        return
    futures = [
        _write_pool().submit(_direct_write_one, spec, path, data, made)
        for spec, path, data in jobs
    ]
    for future in futures:
        future.result()


def _stamp_presence(
    stamps: list[tuple[str, zarr.Array, list[str], dict[str, tuple[bytes, bool]]]],
) -> None:
    """Rewrite each array's ``nonempty_chunks`` in one gather.

    One ``zarr.json`` rewrite per array, issued together: a links family
    with hundreds of offsets arrays would otherwise pay a serial
    ``sync()`` per array for a metadata write that is independent of
    every other.

    Each entry carries its array name and the cell batch it came from,
    used only to report the landed stamps to
    :func:`~zarr_vectors.core.group.observe_presence_writes`.
    """
    from zarr_vectors.core.group import _NONEMPTY_CHUNKS_ATTR

    if not stamps:
        return
    if len(stamps) == 1:
        array_name, arr, present, cells = stamps[0]
        arr.attrs[_NONEMPTY_CHUNKS_ATTR] = present
        _emit_batch_presence(array_name, cells)
        return

    async def _all() -> None:
        await asyncio.gather(*(
            arr._async_array.update_attributes({_NONEMPTY_CHUNKS_ATTR: present})
            for _name, arr, present, _cells in stamps
        ))

    sync(_all())
    # The sync handles carry a copy of the metadata; the async twins were
    # replaced by ``update_attributes``, so anything still holding the
    # old handle would read a stale manifest.
    for array_name, arr, present, cells in stamps:
        try:
            arr.metadata.attributes[_NONEMPTY_CHUNKS_ATTR] = present
        except Exception:
            pass
        _emit_batch_presence(array_name, cells)


def _flush_native_cells(
    zarr_group: zarr.Group,
    cells_in: list[tuple[str, str, bytes, bool]],
) -> None:
    """Flush queued cell writes into their vlen-bytes chunk arrays.

    Each ``array_name`` is a single multidim vlen-bytes Zarr array whose
    cells are spatial chunks.

    Arrays on a local filesystem take the direct path above: every cell
    of every such array is encoded and written from one pool, then each
    array's ``nonempty_chunks`` is stamped in one gather.  That is the
    layout a bulk local build produces -- one file per cell, hundreds of
    arrays when a links family fans out by offsets -- and it is exactly
    the case where zarr's per-cell pipeline was the cost.

    Everything else goes through :func:`_flush_one_array`, one
    ``set_coordinate_selection`` per array, and those are issued
    **concurrently**: each one is a blocking ``sync()`` hop onto zarr's
    shared event loop, so driving them from one thread would serialise
    the arrays.  Workers share nothing -- each resolves its own node and
    writes disjoint keys.

    ``nonempty_chunks`` is stamped once per array rather than once per
    cell, and only over the cells whose ``record_presence`` is True; see
    :func:`_presence_after`.
    """
    by_array: dict[str, dict[str, tuple[bytes, bool]]] = defaultdict(dict)
    for array_name, chunk_key, data, record_presence in cells_in:
        # Last write wins for a repeated key within the batch.
        by_array[array_name][chunk_key] = (data, record_presence)

    if not by_array:
        return

    from zarr_vectors.core._batch_reader import _direct_path
    from zarr_vectors.core.aio import _resolve_nodes

    # Every array's node in one gather, rather than one blocking lookup
    # per array from each worker.
    resolved = sync(_resolve_nodes(zarr_group._async_group, set(by_array)))

    direct_jobs: list[tuple[_DirectWriteSpec, str, bytes]] = []
    stamps: list[
        tuple[str, zarr.Array, list[str], dict[str, tuple[bytes, bool]]]
    ] = []
    general: dict[str, tuple[zarr.Array | None, dict[str, tuple[bytes, bool]]]] = {}
    for array_name, cells in by_array.items():
        node = resolved.get(array_name)
        arr = node if isinstance(node, zarr.Array) else None
        if arr is None:
            # Unresolved here; the general path resolves it itself.
            general[array_name] = (None, cells)
            continue
        spec = _direct_write_spec(zarr_group, array_name, arr)
        if spec is None:
            general[array_name] = (arr, cells)
            continue
        for chunk_key, (data, _record_presence) in cells.items():
            path = _direct_path(spec, chunk_key)
            if path is None:
                raise StoreError(
                    f"Cannot write to array {array_name!r}: chunk_key "
                    f"{chunk_key!r} is off its grid {spec.shape}"
                )
            direct_jobs.append((spec, path, bytes(data)))
        present = _presence_after(arr, cells)
        if present is not None:
            stamps.append((array_name, arr, present, cells))

    # Suppressed out here, once: ``catch_warnings`` swaps global state and
    # is not safe to enter from the workers.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        _direct_write_many(direct_jobs)
        _stamp_presence(stamps)

        if not general:
            return
        if len(general) == 1 or _NO_THREADS:
            for array_name, (arr, cells) in general.items():
                _flush_one_array(zarr_group, array_name, cells, arr=arr)
            return
        with ThreadPoolExecutor(
            max_workers=min(_FLUSH_MAX_WORKERS, len(general)),
        ) as pool:
            futures = [
                pool.submit(
                    _flush_one_array, zarr_group, array_name, cells, arr=arr,
                )
                for array_name, (arr, cells) in general.items()
            ]
            for future in futures:
                # Re-raise the first failure, after every worker settles.
                future.result()


def flush_batch(
    zarr_group: zarr.Group,
    cells: Iterable[tuple[str, str, bytes, bool]],
    *,
    array_metas: dict[str, dict[str, Any]] | None = None,
    codecs: list[dict[str, Any]] | None = None,
) -> None:
    """Flush a batch of cell writes + group-metadata writes.

    ``cells`` is the chunk batch: ``(array_name, chunk_key, data,
    record_presence)``.  Every ``array_name`` is a single multidim vlen
    array — the only per-spatial-chunk layout — so the batch flushes via
    :func:`_flush_native_cells`, one concurrent
    ``set_coordinate_selection`` per array.

    ``array_metas`` is ``{group_name: attributes_dict}`` for the **group**
    nodes in the layout — the ``links/<delta>`` family groups, attribute
    namespaces, ``object_index``.  Chunk arrays are excluded by
    construction: :meth:`zarr_vectors.core.group.Group.write_array_meta`
    writes an existing array's metadata straight to its ``attrs``, so
    only group paths ever reach the queue.  The PUT is what materializes
    the group, so a queued meta also stands in for ``require_group``.

    ``codecs`` is accepted for call-compatibility and unused: a chunk
    array's codec pipeline is fixed when the array is created.

    All PUTs go through one :func:`asyncio.gather`, then the function
    blocks until they complete (or the first error propagates).
    Idempotent on empty inputs.
    """
    del codecs
    cells = list(cells)
    array_metas = dict(array_metas or {})

    if not cells and not array_metas:
        return

    if cells:
        _flush_native_cells(zarr_group, cells)

    if array_metas:
        _flush_group_metas(zarr_group, array_metas)


def _flush_group_metas(
    zarr_group: zarr.Group,
    array_metas: dict[str, dict[str, Any]],
) -> None:
    """PUT each queued group ``zarr.json`` in one gather.

    Each PUT both creates the group node and sets its attributes, which
    is why :func:`zarr_vectors.core.arrays._ensure_array_dir` skips its
    ``require_group`` while a batch is open.
    """
    if not array_metas:
        return

    # Icechunk tracks groups as first-class entities and doesn't pick up
    # one added via a raw ``store.set`` of its ``zarr.json``.
    if _is_icechunk_store(zarr_group.store):
        from zarr_vectors.core.group import _merge_attributes

        for group_name, meta in sorted(array_metas.items()):
            _merge_attributes(zarr_group.require_group(group_name), meta)
        return

    base_path = zarr_group.path
    base_prefix = f"{base_path.rstrip('/')}/" if base_path else ""
    puts = [
        (f"{base_prefix}{group_name}/zarr.json", _group_zarr_json_bytes(meta))
        for group_name, meta in sorted(array_metas.items())
    ]
    sync(_async_put_chunks(zarr_group.store, puts))
