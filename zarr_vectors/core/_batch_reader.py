"""Async-batched chunk reader.

Symmetric to :mod:`zarr_vectors.core._batch_writer`.  Where the writer
turns ``N`` serial ``store.set`` PUTs into one :func:`asyncio.gather`,
this module does the same for ``store.get`` reads — collapsing the
round-trip cost of an ``N``-chunk read-all from ``O(N)`` to ``O(1)``
against high-latency object stores.

The single entry point is :func:`flush_prefetch`, driven by the
:meth:`zarr_vectors.core.group.Group.batched_reads` context manager.
The caller supplies a ``plan`` of ``(array_name, [chunk_keys, ...])``
pairs; this module loads every requested chunk in one async gather and
returns a ``{(array_name, chunk_key): bytes}`` cache that the Group
serves :meth:`read_bytes` calls from while the context is active.

Every ``<array_name>`` is a multidim vlen-bytes Zarr Array whose cell
``arr[coords]`` holds one chunk's payload; the ``sharding_indexed``
codec transparently performs the byte-range reads needed to fetch a
single inner-chunk from its outer shard.  A plan entry naming anything
else is skipped, which degrades to the sync :meth:`read_bytes` path —
and that path must agree, so this module must not learn to serve a
layout ``read_bytes`` would reject.

Icechunk-backed stores fall back to the synchronous :meth:`read_bytes`
path because icechunk tracks arrays as session-managed entities and the
async gather pattern bypasses that contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import zarr
from zarr.core.sync import sync


def _is_icechunk_store(store: Any) -> bool:
    """Return True when ``store`` is an icechunk-backed Store.

    Detection is by class name to avoid importing icechunk at module
    load time.  Mirrors the check in
    :mod:`zarr_vectors.core._batch_writer`.
    """
    cls = type(store)
    return cls.__name__ == "IcechunkStore" or cls.__module__.startswith("icechunk")


def _parse_coords(chunk_key: str) -> tuple[int, ...] | None:
    """Parse ``"0.1.2"`` → ``(0, 1, 2)``; ``None`` if any segment is
    not an integer.  Local copy of the helper in
    :mod:`zarr_vectors.core.group` to avoid an import cycle.
    """
    try:
        return tuple(int(p) for p in chunk_key.split("."))
    except ValueError:
        return None


def _apply_origin(
    coords: tuple[int, ...], attributes: dict[str, Any],
) -> tuple[int, ...]:
    """Translate an absolute chunk coord to a 0-based cell index using
    the array's stored ``chunk_grid_origin`` (absent → zero origin).
    """
    origin = attributes.get("chunk_grid_origin")
    if not origin:
        return coords
    return tuple(c - int(o) for c, o in zip(coords, origin))


async def _async_get_sharded_cell(
    async_array: Any,
    chunk_key: str,
) -> bytes | None:
    """Read one cell from a vlen-bytes chunk ``AsyncArray``.

    Builds the ``(slice(c, c+1),) * ndim`` region so zarr's
    ``sharding_indexed`` codec fetches exactly the inner chunk's byte
    range (plus the shard index tail) — typically two range reads per
    cell regardless of shard size.

    Mirrors the sync ``_vlen_get_cell`` helper in
    :mod:`zarr_vectors.core.group`.
    """
    coords = _parse_coords(chunk_key)
    if coords is None:
        return None
    if len(coords) != len(async_array.shape):
        return None
    coords = _apply_origin(coords, async_array.metadata.attributes)
    if any(c < 0 or c >= s for c, s in zip(coords, async_array.shape)):
        return None
    region = tuple(slice(c, c + 1) for c in coords)
    arr = np.asarray(await async_array.getitem(region))
    if arr.size == 0:
        return b""
    val = arr.flat[0]
    return b"" if val is None else bytes(val)


async def _gather_plan(
    async_group: Any,
    plan: list[tuple[str, list[str]]],
) -> dict[tuple[str, str], bytes]:
    """Resolve each ``array_name`` once, then fan out per-chunk reads
    via :func:`asyncio.gather`.

    Returns a flat ``{(array_name, chunk_key): bytes}`` cache; entries
    whose underlying node is missing or is not a chunk array, or whose
    read returned ``None``, are omitted — the caller's sync
    :meth:`Group.read_bytes` then raises :class:`StoreError` on the
    cache miss, exactly as it would have without the prefetch.
    """
    # Resolve each array_name once.
    nodes: dict[str, Any] = {}
    for array_name, _ in plan:
        if array_name in nodes:
            continue
        try:
            nodes[array_name] = await async_group.getitem(array_name)
        except KeyError:
            nodes[array_name] = None

    # Build the per-chunk task list.  Each task returns ``bytes | None``
    # so the post-gather assembly is uniform.
    flat: list[tuple[str, str]] = []
    tasks: list[Any] = []
    for array_name, chunk_keys in plan:
        node = nodes.get(array_name)
        if not isinstance(node, zarr.AsyncArray):
            continue
        for chunk_key in chunk_keys:
            flat.append((array_name, chunk_key))
            tasks.append(_async_get_sharded_cell(node, chunk_key))

    if not tasks:
        return {}
    results = await asyncio.gather(*tasks)

    cache: dict[tuple[str, str], bytes] = {}
    for (array_name, chunk_key), data in zip(flat, results):
        if data is not None:
            cache[(array_name, chunk_key)] = data
    return cache


def flush_prefetch(
    zarr_group: zarr.Group,
    plan: list[tuple[str, list[str]]],
) -> dict[tuple[str, str], bytes]:
    """Prefetch every chunk in ``plan`` and return a flat cache.

    ``plan`` is a list of ``(array_name, [chunk_keys, ...])`` tuples.
    Each ``chunk_key`` is the standard chunk_key string (e.g.
    ``"0.1.2"``) used by :meth:`Group.read_bytes`.  Returns a dict keyed
    by ``(array_name, chunk_key)`` whose values are the decoded chunk
    bytes; missing chunks are omitted (the caller falls through to the
    sync ``read_bytes`` path on a cache miss).

    For icechunk-backed stores, falls back to serial sync reads via
    :func:`_sync_fallback` — the async-gather pattern bypasses
    icechunk's session-tracking contract.
    """
    if not plan:
        return {}

    if _is_icechunk_store(zarr_group.store):
        return _sync_fallback(zarr_group, plan)

    return sync(_gather_plan(zarr_group._async_group, plan))


def _sync_fallback(
    zarr_group: zarr.Group,
    plan: list[tuple[str, list[str]]],
) -> dict[tuple[str, str], bytes]:
    """Serial-read fallback for icechunk and other stores that don't
    play well with the async-gather pattern.

    Walks the plan one entry at a time using sync zarr access, returning
    the same dict shape :func:`flush_prefetch` does.
    """
    cache: dict[tuple[str, str], bytes] = {}
    for array_name, chunk_keys in plan:
        try:
            node = zarr_group[array_name]
        except KeyError:
            continue
        if not isinstance(node, zarr.Array):
            continue
        shape = node.shape
        attrs = dict(node.attrs)
        for chunk_key in chunk_keys:
            coords = _parse_coords(chunk_key)
            if coords is None or len(coords) != len(shape):
                continue
            coords = _apply_origin(coords, attrs)
            if any(c < 0 or c >= s for c, s in zip(coords, shape)):
                continue
            region = tuple(slice(c, c + 1) for c in coords)
            cell = np.asarray(node[region])
            if cell.size == 0:
                cache[(array_name, chunk_key)] = b""
                continue
            val = cell.flat[0]
            cache[(array_name, chunk_key)] = (
                b"" if val is None else bytes(val)
            )
    return cache
