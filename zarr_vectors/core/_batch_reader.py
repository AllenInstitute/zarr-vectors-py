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
import os
from typing import Any, NamedTuple

import numpy as np
import zarr
from numcodecs.vlen import VLenBytes
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

# Leaf module (no other zarr_vectors imports), so importing it here does
# not reintroduce the import cycle this module otherwise avoids.
from zarr_vectors.core._vlen import (
    cell_region as _vlen_cell_region,
    region_to_bytes as _vlen_region_to_bytes,
)

# The same instance ``zarr.codecs.vlen_utf8.VLenBytesCodec`` decodes
# through, so a cell read by the direct path below is byte-identical to
# one read through ``AsyncArray.getitem``.  Stateless, hence module-level.
_VLEN_BYTES = VLenBytes()

# Resolved once for the same reason: it is a module-level singleton in
# zarr, and looking it up per cell cost 1.2 us a call on a path that runs
# twice per read.  Same bet ``_VLEN_BYTES`` above already takes -- a
# process that reconfigures zarr's buffer prototype mid-run would need a
# restart, which nothing in ZV does.
_BUFFER_PROTOTYPE = default_buffer_prototype()


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
    region = _vlen_cell_region(coords)
    return _vlen_region_to_bytes(await async_array.getitem(region))


def _as_async_array(node: Any) -> Any:
    """The ``AsyncArray`` behind a resolved node, or the node itself.

    A caller's node cache holds sync :class:`zarr.Array` handles; the
    gather needs their async twin.  Anything else passes through so the
    ``isinstance`` check downstream still rejects it.
    """
    return node._async_array if isinstance(node, zarr.Array) else node


async def _gather_plan(
    async_group: Any,
    plan: list[tuple[str, list[str]]],
    resolved: dict[str, Any] | None = None,
) -> dict[tuple[str, str], bytes]:
    """Resolve each ``array_name`` once, then fan out per-chunk reads
    via :func:`asyncio.gather`.

    ``resolved`` supplies nodes the caller has already looked up —
    :meth:`Group.batched_reads` passes the ones its node cache is holding
    — so a prefetch does not re-read a ``zarr.json`` the caller has in
    hand.  A name absent from it is resolved here as before.

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
        hit = (resolved or {}).get(array_name)
        if hit is not None:
            nodes[array_name] = _as_async_array(hit)
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


# --------------------------------------------------------------------
# Local-filesystem fast path
# --------------------------------------------------------------------
#
# Against a LocalStore the gather above is almost all overhead.  Zarr
# runs each ``store.get`` on the thread pool and hands the result back
# through the event loop, so a cell costs a thread hop, a Future and a
# loop callback -- ~0.15 ms of scheduling to read a file the OS already
# has in page cache.  Measured on a 10^6-point store (125 chunks, zstd):
# 250 cells cost 40 ms through the async store layer and 3.3 ms read
# serially with ``open()``.  Add the per-cell ``AsyncArray.getitem``
# above it -- indexers, an ArraySpec, a one-element codec pipeline pass
# -- and a read-everything spent ~80% of its wall time scheduling I/O it
# had already finished.
#
# That trade is the right one for GCS/S3, where a round-trip dwarfs the
# scheduling and concurrency is the whole game.  It is the wrong one for
# a local directory, where there is no round-trip to hide.
#
# So: read the storage objects with plain ``open()`` and decode them
# through the array's own codecs.  The decode is zarr's, not a
# reimplementation of it -- each BytesBytes codec's ``_decode_sync``,
# then the same ``numcodecs`` VLenBytes instance ``VLenBytesCodec``
# itself delegates to -- so a cell decoded here is byte-identical to one
# decoded by ``getitem``.  Anything this path does not positively
# recognise (a shard, an unexpected serializer, a codec with no sync
# entry point, a non-local store) returns None from :func:`_direct_spec`
# and falls through to the gather, which stays the general path.


class _DirectSpec(NamedTuple):
    """Everything needed to read one array's cells off the filesystem."""

    root: str                       # directory holding the chunk tree
    separator: str                  # chunk-key separator ("/" or ".")
    origin: tuple[int, ...] | None  # chunk_grid_origin, or None for zero
    shape: tuple[int, ...]          # cell-grid shape, for bounds checks
    codecs: tuple[Any, ...]         # BytesBytes codecs, decode order
    spec: Any                       # ArraySpec the codecs decode against


def _direct_spec(
    zarr_group: zarr.Group,
    array_name: str,
    resolved: Any = None,
) -> _DirectSpec | None:
    """Return a :class:`_DirectSpec` for ``array_name``, or None.

    None means "not recognised, use the gather" — never "absent".  Every
    condition checked here is one the direct reader would otherwise have
    to guess at, so a new layout degrades to the general path rather
    than to wrong bytes.

    ``resolved`` is the node if the caller already has it, saving the
    ``zarr.json`` read this would otherwise do per array per prefetch.
    """
    node = resolved
    if node is None:
        try:
            node = zarr_group[array_name]
        except KeyError:
            return None
    if not isinstance(node, zarr.Array):
        return None

    store = node.store_path.store
    if type(store).__name__ != "LocalStore":
        return None

    meta = node.metadata
    if getattr(meta, "zarr_format", None) != 3:
        return None
    if getattr(meta, "storage_transformers", ()):    # sharding via transformer
        return None
    # One storage object per cell.  A chunk_shape with any dimension > 1
    # (or a sharding codec, caught below) means a file holds several
    # cells and the byte ranges are the shard index's business.
    grid = getattr(meta, "chunk_grid", None)
    chunk_shape = getattr(grid, "chunk_shape", None)
    if chunk_shape is None or any(c != 1 for c in chunk_shape):
        return None

    codecs = tuple(meta.codecs)
    if not codecs or type(codecs[0]).__name__ != "VLenBytesCodec":
        return None
    bytes_codecs = codecs[1:]
    if not all(hasattr(c, "_decode_sync") for c in bytes_codecs):
        return None

    cke = meta.chunk_key_encoding.to_dict()
    if cke.get("name") != "default":
        return None
    separator = (cke.get("configuration") or {}).get("separator", "/")

    origin = node.attrs.get("chunk_grid_origin")
    prototype = _BUFFER_PROTOTYPE
    try:
        spec = ArraySpec(
            shape=tuple(chunk_shape),
            dtype=meta.data_type,
            fill_value=meta.fill_value,
            config=node._async_array.config,
            prototype=prototype,
        )
    except Exception:
        return None

    return _DirectSpec(
        root=os.path.join(str(store.root), node.store_path.path, "c"),
        separator=separator,
        origin=tuple(int(o) for o in origin) if origin else None,
        shape=tuple(meta.shape),
        codecs=bytes_codecs,
        spec=spec,
    )


def _direct_read(spec: _DirectSpec, chunk_key: str) -> bytes | None:
    """Read and decode one cell.

    Returns ``b""`` for a cell with no stored object — that is the fill
    value, and it is what the gather path caches for one too, so a plan
    naming an unwritten cell costs the same either way.  Returns None
    only for a key this array cannot hold (unparseable, wrong arity, off
    the grid), which the caller omits from the cache exactly as the
    gather does.
    """
    coords = _parse_coords(chunk_key)
    if coords is None or len(coords) != len(spec.shape):
        return None
    if spec.origin is not None:
        coords = tuple(c - o for c, o in zip(coords, spec.origin))
    if any(c < 0 or c >= s for c, s in zip(coords, spec.shape)):
        return None
    path = spec.root + spec.separator + spec.separator.join(
        str(c) for c in coords
    )
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return b""              # no stored object: the fill value
    buffer = spec.spec.prototype.buffer.from_bytes(raw)
    for codec in reversed(spec.codecs):
        buffer = codec._decode_sync(buffer, spec.spec)
    decoded = _VLEN_BYTES.decode(buffer.as_numpy_array())
    if decoded.size != 1:
        return None
    value = decoded.flat[0]
    return b"" if value is None else bytes(value)


def flush_prefetch(
    zarr_group: zarr.Group,
    plan: list[tuple[str, list[str]]],
    nodes: dict[str, Any] | None = None,
    specs: dict[str, Any] | None = None,
) -> dict[tuple[str, str], bytes]:
    """Prefetch every chunk in ``plan`` and return a flat cache.

    ``plan`` is a list of ``(array_name, [chunk_keys, ...])`` tuples.
    Each ``chunk_key`` is the standard chunk_key string (e.g.
    ``"0.1.2"``) used by :meth:`Group.read_bytes`.  Returns a dict keyed
    by ``(array_name, chunk_key)`` whose values are the decoded chunk
    bytes; missing chunks are omitted (the caller falls through to the
    sync ``read_bytes`` path on a cache miss).

    ``nodes`` lets the caller hand over array handles it has already
    resolved, and ``specs`` the direct-read specs derived from them
    (``None`` in that mapping meaning "not direct-readable", which is an
    answer worth carrying).  Inside a :meth:`Group.cached_nodes` block a
    prefetch therefore costs no node lookups and no spec derivation at
    all.

    Arrays on a local filesystem take the direct path described above —
    plain ``open()`` plus the array's own codecs, no event loop — and
    everything else the async gather.  A plan naming both splits between
    them and the results merge, so a caller never has to know which it
    got.

    For icechunk-backed stores, falls back to serial sync reads via
    :func:`_sync_fallback` — the async-gather pattern bypasses
    icechunk's session-tracking contract.
    """
    if not plan:
        return {}

    if _is_icechunk_store(zarr_group.store):
        return _sync_fallback(zarr_group, plan)

    # Resolve each array_name once, as ``_gather_plan`` does, then split
    # the plan on whether the direct reader recognised it.
    direct: dict[str, _DirectSpec] = {}
    seen: set[str] = set()
    for array_name, _ in plan:
        if array_name in seen:
            continue
        seen.add(array_name)
        if specs is not None and array_name in specs:
            spec = specs[array_name]
        else:
            spec = _direct_spec(
                zarr_group, array_name, (nodes or {}).get(array_name),
            )
        if spec is not None:
            direct[array_name] = spec
    gathered = [(name, keys) for name, keys in plan if name not in direct]

    cache: dict[tuple[str, str], bytes] = {}
    if gathered:
        cache.update(
            sync(_gather_plan(zarr_group._async_group, gathered, nodes))
        )

    for array_name, chunk_keys in plan:
        spec = direct.get(array_name)
        if spec is None:
            continue
        for chunk_key in chunk_keys:
            data = _direct_read(spec, chunk_key)
            if data is not None:
                cache[(array_name, chunk_key)] = data
    return cache


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
            region = _vlen_cell_region(coords)
            cache[(array_name, chunk_key)] = _vlen_region_to_bytes(
                node[region]
            )
    return cache
