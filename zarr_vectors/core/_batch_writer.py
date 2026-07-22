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
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.errors import UnstableSpecificationWarning

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


def _flush_one_array(
    zarr_group: zarr.Group,
    array_name: str,
    cells: dict[str, tuple[bytes, bool]],
) -> None:
    """Write one array's queued cells and stamp its manifest.

    Self-contained per array — it resolves its own node and touches no
    shared state — which is what lets :func:`_flush_native_cells` run
    these concurrently.
    """
    from zarr_vectors.core.group import (
        _CHUNK_GRID_ORIGIN_ATTR,
        _NONEMPTY_CHUNKS_ATTR,
    )

    arr = zarr_group[array_name]
    ndim = arr.ndim
    present = set(arr.attrs.get(_NONEMPTY_CHUNKS_ATTR) or [])
    stamp = False
    origin_raw = arr.attrs.get(_CHUNK_GRID_ORIGIN_ATTR)
    origin = (
        tuple(int(o) for o in origin_raw) if origin_raw else None
    )
    axis_coords: list[list[int]] = [[] for _ in range(ndim)]
    values: list[bytes] = []
    for chunk_key, (data, record_presence) in cells.items():
        coords = tuple(int(p) for p in chunk_key.split("."))
        # Cell index = coord - origin (grid anchored at min coord).
        index = (
            coords if origin is None
            else tuple(c - o for c, o in zip(coords, origin))
        )
        for ax in range(ndim):
            axis_coords[ax].append(index[ax])
        values.append(bytes(data))
        if not record_presence:
            continue
        stamp = True
        if data:
            present.add(chunk_key)
        else:
            present.discard(chunk_key)

    obj = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        obj[i] = v
    selection = tuple(np.asarray(a, dtype=np.intp) for a in axis_coords)
    arr.set_coordinate_selection(selection, obj)
    if stamp:
        arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(present)


def _flush_native_cells(
    zarr_group: zarr.Group,
    cells_in: list[tuple[str, str, bytes, bool]],
) -> None:
    """Flush queued cell writes into their vlen-bytes chunk arrays.

    Each ``array_name`` is a single multidim vlen-bytes Zarr array whose
    cells are spatial chunks.  For every array we write all of its queued
    cells in one :meth:`zarr.Array.set_coordinate_selection` — zarr fans
    the per-cell chunk writes out across its async pipeline, so N cells
    cost roughly one round-trip rather than N.

    Those per-array writes are then issued **concurrently**.  Each one is
    a blocking ``sync()`` hop onto zarr's shared event loop, so driving
    them from one thread serialises the arrays: only the cells *within* an
    array overlap, and a batch spread thinly over many arrays (the links
    family allocates one array per offsets segment, so a store with N
    distinct offsets has N of them) would pay a full round-trip per array.
    Submitting from a pool lets the loop interleave every array's cells,
    which is the same fan-out a single wide array already got for free.
    Workers share nothing: each resolves its own node and writes disjoint
    keys.

    ``nonempty_chunks`` is then stamped once per array rather than once
    per cell — but only over the cells whose ``record_presence`` is True.
    An array written entirely with ``record_presence=False`` keeps its
    manifest untouched, because that attribute is shared by every cell
    and stamping it would reintroduce, at batch granularity, the very
    cross-worker race the flag exists to avoid.  See
    :meth:`zarr_vectors.core.group.Group.write_bytes`.
    """
    by_array: dict[str, dict[str, tuple[bytes, bool]]] = defaultdict(dict)
    for array_name, chunk_key, data, record_presence in cells_in:
        # Last write wins for a repeated key within the batch.
        by_array[array_name][chunk_key] = (data, record_presence)

    if not by_array:
        return

    # Suppressed out here, once: ``catch_warnings`` swaps global state and
    # is not safe to enter from the workers.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        if len(by_array) == 1 or _NO_THREADS:
            for array_name, cells in by_array.items():
                _flush_one_array(zarr_group, array_name, cells)
            return
        with ThreadPoolExecutor(
            max_workers=min(_FLUSH_MAX_WORKERS, len(by_array)),
        ) as pool:
            futures = [
                pool.submit(_flush_one_array, zarr_group, array_name, cells)
                for array_name, cells in by_array.items()
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
        for group_name, meta in sorted(array_metas.items()):
            zarr_group.require_group(group_name).attrs.update(meta)
        return

    base_path = zarr_group.path
    base_prefix = f"{base_path.rstrip('/')}/" if base_path else ""
    puts = [
        (f"{base_prefix}{group_name}/zarr.json", _group_zarr_json_bytes(meta))
        for group_name, meta in sorted(array_metas.items())
    ]
    sync(_async_put_chunks(zarr_group.store, puts))
