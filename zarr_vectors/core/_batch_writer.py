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

1. Collects ``(array_name, chunk_key, bytes)`` chunk triples and
   ``{array_name: meta_dict}`` array metadata while the caller iterates
   (no round-trips during collection).
2. On flush, ensures each unique parent group either gets its
   ``zarr.json`` PUT directly (when the caller queued metadata for it)
   or via a single sync ``require_group`` (when only chunk PUTs are
   queued, no metadata to merge).
3. Builds inner-array metadata + data buffers in pure Python.
4. Issues every store ``set()`` in a single :func:`asyncio.gather`, so
   the obstore-backed pipe runs at full async parallelism — the latency
   of the slowest PUT, not the sum of all PUTs.

The on-disk layout matches what ``write_bytes`` produces.  Inner-array
codecs are configurable per-batch via the ``codecs`` kwarg of
:func:`flush_batch` (which the calling
:meth:`zarr_vectors.core.group.Group.batched_writes` derives from its
``compressor=`` argument).  ``compressor=None`` (the default) resolves
to ``BytesCodec``-only — no compression, fastest cloud-write path.
Users who want compression can pass ``compressor='zstd'``,
``'blosc'``, or a full codec-spec list; that opts in to the sync
fallback below so zarr's encoder can run the codec pipeline before
each PUT.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.errors import UnstableSpecificationWarning


def _is_icechunk_store(store: Any) -> bool:
    """Return True when ``store`` is an icechunk-backed :class:`zarr.abc.store.Store`.

    Detection is by class name to avoid importing icechunk at module
    load time (it's an optional dep).  Icechunk tracks arrays and
    groups as first-class entities; raw ``store.set("…/zarr.json", …)``
    PUTs don't register them, so the batched direct-PUT path can't be
    used and we have to fall back to the synchronous
    ``zarr.Array.create_array`` + ``array[:] = …`` path that the
    icechunk session knows how to commit.
    """
    cls = type(store)
    return cls.__name__ == "IcechunkStore" or cls.__module__.startswith("icechunk")


# ---------------------------------------------------------------------------
# Inner-array metadata template
# ---------------------------------------------------------------------------

# zarr v3 metadata for a 1D uint8 array with shape=(N,), chunks=(N,).
# Verified round-trip against zarr 3.2.x.  The ``__CODECS__`` token is
# replaced per call with a compact-JSON-encoded codecs list; ``__SIZE__``
# is replaced with the per-chunk byte length.  We use str.replace instead
# of json.dumps for the per-chunk loop because every chunk differs only
# by length and reconstructing the dict + serialising adds meaningful CPU
# when the batch is large.
_INNER_ARRAY_META_PARAM_TEMPLATE = (
    '{"shape":[__SIZE__],"data_type":"uint8",'
    '"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[__SIZE__]}},'
    '"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},'
    '"fill_value":0,"codecs":__CODECS__,"attributes":{},'
    '"zarr_format":3,"node_type":"array","storage_transformers":[]}'
)

# Empty-chunk variant: shape=(0,) with chunks=(1,) so zarr accepts it.
# Matches the legacy "n == 0" branch in ``FsGroup.write_bytes``.
_EMPTY_INNER_ARRAY_META_TEMPLATE = (
    '{"shape":[0],"data_type":"uint8",'
    '"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1]}},'
    '"chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},'
    '"fill_value":0,"codecs":__CODECS__,"attributes":{},'
    '"zarr_format":3,"node_type":"array","storage_transformers":[]}'
)

# Fallback codecs JSON used when ``flush_batch`` is called without an
# explicit list (legacy callers).  Matches the project default of
# BytesCodec-only — keeps the fast async PUT path active.  Callers who
# want compression must pass ``compressor='zstd'`` / ``'blosc'`` / a
# codec list at the ``batched_writes`` boundary.
_DEFAULT_CODECS_JSON = '[{"name":"bytes"}]'


def _codecs_json(codecs: list[dict[str, Any]] | None) -> str:
    """Serialise a codec list to the compact JSON used in inner metadata."""
    if codecs is None:
        return _DEFAULT_CODECS_JSON
    return json.dumps(codecs, separators=(",", ":"))


def _inner_array_meta_bytes(
    n: int,
    codecs_json: str = _DEFAULT_CODECS_JSON,
) -> bytes:
    """Return the inner-array ``zarr.json`` bytes for an N-byte chunk."""
    if n == 0:
        return (
            _EMPTY_INNER_ARRAY_META_TEMPLATE
            .replace("__CODECS__", codecs_json)
            .encode("utf-8")
        )
    return (
        _INNER_ARRAY_META_PARAM_TEMPLATE
        .replace("__CODECS__", codecs_json)
        .replace("__SIZE__", str(n))
        .encode("utf-8")
    )


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


def _is_bytes_only_codecs(codecs: list[dict[str, Any]] | None) -> bool:
    """Return True iff ``codecs`` is the BytesCodec-only pipeline.

    The fast async PUT path can only emit raw chunk bytes; any pipeline
    that includes a compressor (``zstd``, ``blosc``, …) has to go through
    zarr's encoder, which only happens on the sync fallback.
    """
    if codecs is None:
        return False
    return len(codecs) == 1 and codecs[0].get("name") == "bytes"


def _flush_batch_sync(
    zarr_group: zarr.Group,
    triples: list[tuple[str, str, bytes]],
    array_metas: dict[str, dict[str, Any]],
    codecs: list[dict[str, Any]] | None = None,
) -> None:
    """Synchronous fallback for stores that can't accept raw ``set()``
    PUTs against array paths (icechunk) **and** for batches that need a
    real compressor in the codec pipeline (the fast async path can only
    PUT raw chunk bytes; with a compressor in the pipeline we have to go
    through zarr's encoder).

    Replays each queued operation through the zarr ``Array.create_array``
    + ``array[:] = data`` path that ``Group.write_bytes`` would have
    used in unbatched mode, so the codec pipeline runs on every chunk.
    """
    # Strip the BytesCodec serializer; ``create_array`` adds it back via
    # the ``serializer=`` default and only accepts BytesBytes codecs in
    # ``compressors=``.
    from zarr_vectors.encoding.compression import codecs_for_create_array
    extra_kwargs: dict[str, Any] = {}
    if codecs is not None:
        extra_kwargs["compressors"] = codecs_for_create_array(codecs)

    # Array metadata first so the parent group exists with the right
    # attributes before any per-chunk array creation.
    for array_name, meta in array_metas.items():
        arr_group = zarr_group.require_group(array_name)
        arr_group.attrs.update(meta)

    for array_name, chunk_key, data in triples:
        arr_group = zarr_group.require_group(array_name)
        if chunk_key in arr_group:
            del arr_group[chunk_key]
        n = len(data)
        if n == 0:
            arr_group.create_array(
                chunk_key, shape=(0,), chunks=(1,), dtype="uint8",
                **extra_kwargs,
            )
            continue
        a = arr_group.create_array(
            chunk_key, shape=(n,), chunks=(n,), dtype="uint8",
            **extra_kwargs,
        )
        a[:] = np.frombuffer(data, dtype="uint8")


def _flush_native_cells(
    zarr_group: zarr.Group,
    triples: list[tuple[str, str, bytes]],
) -> None:
    """Flush queued cell writes into single vlen-bytes chunk arrays.

    Each ``array_name`` in ``triples`` is a single multidim vlen-bytes
    Zarr array (the default single-array layout) whose cells are spatial
    chunks.  For every array we write all of its queued cells in one
    :meth:`zarr.Array.set_coordinate_selection` — zarr fans the per-cell
    chunk writes out across its async pipeline, so N cells cost roughly
    one round-trip rather than N.  The ``nonempty_chunks`` presence
    manifest is then stamped once (instead of once per cell).
    """
    from zarr_vectors.core.group import (
        _CHUNK_GRID_ORIGIN_ATTR,
        _NONEMPTY_CHUNKS_ATTR,
    )

    by_array: dict[str, dict[str, bytes]] = defaultdict(dict)
    for array_name, chunk_key, data in triples:
        # Last write wins for a repeated key within the batch.
        by_array[array_name][chunk_key] = data

    for array_name, cells in by_array.items():
        arr = zarr_group[array_name]
        ndim = arr.ndim
        present = set(arr.attrs.get(_NONEMPTY_CHUNKS_ATTR) or [])
        origin_raw = arr.attrs.get(_CHUNK_GRID_ORIGIN_ATTR)
        origin = (
            tuple(int(o) for o in origin_raw) if origin_raw else None
        )
        axis_coords: list[list[int]] = [[] for _ in range(ndim)]
        values: list[bytes] = []
        for chunk_key, data in cells.items():
            coords = tuple(int(p) for p in chunk_key.split("."))
            # Cell index = coord - origin (grid anchored at min coord).
            index = (
                coords if origin is None
                else tuple(c - o for c, o in zip(coords, origin))
            )
            for ax in range(ndim):
                axis_coords[ax].append(index[ax])
            values.append(bytes(data))
            if data:
                present.add(chunk_key)
            else:
                present.discard(chunk_key)

        obj = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            obj[i] = v
        selection = tuple(np.asarray(a, dtype=np.intp) for a in axis_coords)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            arr.set_coordinate_selection(selection, obj)
            arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(present)


def flush_batch(
    zarr_group: zarr.Group,
    triples: Iterable[tuple[str, str, bytes]],
    *,
    array_metas: dict[str, dict[str, Any]] | None = None,
    codecs: list[dict[str, Any]] | None = None,
) -> None:
    """Flush a batch of chunk writes + array-metadata writes.

    ``triples`` is the chunk batch: ``(array_name, chunk_key, data)``.
    Two physical destinations are handled and dispatched per array by
    inspecting the node already at ``array_name``:

    * **Single vlen array** (``zarr.Array`` node) — the default
      single-array layout for every per-spatial-chunk array.  Cells
      flush via :func:`_flush_native_cells` (one concurrent
      ``set_coordinate_selection`` per array).
    * **Per-cell group** (``zarr.Group`` node, or absent) — retained for
      ``cross_chunk_links`` / ``…_attributes``, whose keys are endpoint
      tuples.  Each cell becomes a single-chunk ``uint8`` inner array,
      flushed by :func:`_flush_legacy`.

    ``array_metas`` is the per-array metadata batch for the per-cell
    groups: ``{array_name: attributes_dict}``.  (Single vlen arrays get
    their metadata written directly to the array's ``attrs`` at create
    time, so they never appear here.)

    ``codecs`` is the per-batch codec list applied to the per-cell inner
    arrays.  When BytesCodec-only the fast async PUT path runs; any other
    pipeline (or an icechunk store) forces the sync fallback.

    All PUTs go through one :func:`asyncio.gather`, then the function
    blocks until they complete (or the first error propagates).
    Idempotent on empty inputs.
    """
    triples = list(triples)
    array_metas = dict(array_metas or {})

    if not triples and not array_metas:
        return

    # Dispatch each triple by the node already at its ``array_name``:
    # a ``zarr.Array`` is a single vlen array (default layout); a group
    # (or absent) is the per-cell primitive used by cross_chunk_links.
    native_triples: list[tuple[str, str, bytes]] = []
    legacy_triples: list[tuple[str, str, bytes]] = []
    is_native: dict[str, bool] = {}
    for item in triples:
        array_name = item[0]
        native = is_native.get(array_name)
        if native is None:
            node = zarr_group[array_name] if array_name in zarr_group else None
            native = isinstance(node, zarr.Array)
            is_native[array_name] = native
        (native_triples if native else legacy_triples).append(item)

    if native_triples:
        _flush_native_cells(zarr_group, native_triples)

    if legacy_triples or array_metas:
        _flush_legacy(zarr_group, legacy_triples, array_metas, codecs=codecs)


def _flush_legacy(
    zarr_group: zarr.Group,
    triples: list[tuple[str, str, bytes]],
    array_metas: dict[str, dict[str, Any]],
    *,
    codecs: list[dict[str, Any]] | None = None,
) -> None:
    """Flush the per-cell layout (cross_chunk_links): one single-chunk
    ``uint8`` inner array per cell key, plus per-group ``zarr.json``.
    """
    if not triples and not array_metas:
        return

    # Icechunk doesn't pick up arrays added via raw ``store.set`` of
    # their ``zarr.json`` — it tracks arrays as first-class entities and
    # expects them through ``zarr.Array.create_array``.  Same path is
    # required when the codec pipeline includes a compressor: the fast
    # async PUT below can only emit raw bytes, so a compressor in the
    # pipeline must go through zarr's encoder.
    if _is_icechunk_store(zarr_group.store) or not _is_bytes_only_codecs(codecs):
        _flush_batch_sync(zarr_group, triples, array_metas, codecs=codecs)
        return

    # Resolve the addressing prefix once.
    base_path = zarr_group.path
    base_prefix = f"{base_path.rstrip('/')}/" if base_path else ""

    # Chunk-write array_names that have no queued metadata still need a
    # parent zarr.json.  Create those via require_group (sync, amortised
    # — typically 2-6 names per batch).  Array_names with queued meta
    # skip require_group entirely; the meta PUT below creates the group
    # directly.
    chunk_array_names = {array_name for array_name, _, _ in triples}
    needs_require_group = chunk_array_names - array_metas.keys()
    for array_name in sorted(needs_require_group):
        zarr_group.require_group(array_name)

    puts: list[tuple[str, bytes]] = []
    codecs_json = _codecs_json(codecs)

    # Group-level zarr.json PUTs (one per queued array meta).
    for array_name, meta in sorted(array_metas.items()):
        puts.append(
            (f"{base_prefix}{array_name}/zarr.json", _group_zarr_json_bytes(meta))
        )

    # Per-chunk inner-array PUTs.
    for array_name, chunk_key, data in triples:
        inner_prefix = f"{base_prefix}{array_name}/{chunk_key}/"
        n = len(data)
        puts.append(
            (f"{inner_prefix}zarr.json", _inner_array_meta_bytes(n, codecs_json))
        )
        if n > 0:
            puts.append((f"{inner_prefix}c/0", data))

    sync(_async_put_chunks(zarr_group.store, puts))
