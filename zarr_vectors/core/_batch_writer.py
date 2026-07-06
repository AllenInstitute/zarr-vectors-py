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
from typing import Any, Iterable

import numpy as np
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync


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


def _normalize_triple(
    item: tuple,
) -> tuple[str, str, bytes, list[dict[str, Any]] | None]:
    """Accept 3-tuples (legacy) or 4-tuples (with a per-write codec override).

    The 4th element, when present, is a full Zarr V3 codecs list overriding
    the batch-level ``codecs`` for that single write (used to write one
    array uncompressed while siblings stay compressed).
    """
    array_name, chunk_key, data, *rest = item
    override = rest[0] if rest else None
    return array_name, chunk_key, data, override


def _flush_batch_sync(
    zarr_group: zarr.Group,
    triples: list[tuple],
    array_metas: dict[str, dict[str, Any]],
    codecs: list[dict[str, Any]] | None = None,
) -> None:
    """Synchronous fallback for stores that can't accept raw ``set()``
    PUTs against array paths (icechunk) **and** for batches that need a
    real compressor in the codec pipeline (the fast async path can only
    PUT raw chunk bytes; with a compressor in the pipeline we have to go
    through zarr's encoder) **and** for batches that mix per-write codec
    overrides.

    Replays each queued operation through the zarr ``Array.create_array``
    + ``array[:] = data`` path that ``Group.write_bytes`` would have
    used in unbatched mode, so the codec pipeline runs on every chunk.
    """
    # Strip the BytesCodec serializer; ``create_array`` adds it back via
    # the ``serializer=`` default and only accepts BytesBytes codecs in
    # ``compressors=``.
    from zarr_vectors.encoding.compression import codecs_for_create_array

    def _extra_kwargs(effective: list[dict[str, Any]] | None) -> dict[str, Any]:
        return (
            {"compressors": codecs_for_create_array(effective)}
            if effective is not None
            else {}
        )

    batch_kwargs = _extra_kwargs(codecs)

    # Array metadata first so the parent group exists with the right
    # attributes before any per-chunk array creation.
    for array_name, meta in array_metas.items():
        arr_group = zarr_group.require_group(array_name)
        arr_group.attrs.update(meta)

    for item in triples:
        array_name, chunk_key, data, override = _normalize_triple(item)
        extra_kwargs = _extra_kwargs(override) if override is not None else batch_kwargs
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


def flush_batch(
    zarr_group: zarr.Group,
    triples: Iterable[tuple[str, str, bytes]],
    *,
    array_metas: dict[str, dict[str, Any]] | None = None,
    codecs: list[dict[str, Any]] | None = None,
) -> None:
    """Flush a batch of chunk writes + array-metadata writes.

    ``triples`` is the chunk batch: ``(array_name, chunk_key, data)``.
    Each item produces an inner-array ``zarr.json`` and (for non-empty
    data) a ``c/0`` PUT.

    ``array_metas`` is the per-array metadata batch:
    ``{array_name: attributes_dict}``.  Each entry produces one PUT for
    ``{array_name}/zarr.json`` with the attributes inlined into a fresh
    v3 group node.  This collapses the legacy ``require_group(name)`` +
    ``attrs.update(meta)`` sync pair (2-3 round-trips) into a single
    PUT that joins the chunk gather.

    ``codecs`` is the per-batch codec list (full Zarr V3 ``codecs`` JSON
    shape — BytesCodec serializer plus any compressors).  When the list
    is BytesCodec-only the fast async path runs; any other pipeline
    forces the sync fallback so zarr's encoder can compress the chunk
    bytes before they are written.

    For array_names that appear only in ``triples`` (chunk writes
    without queued metadata), the parent group is created with a single
    sync ``require_group`` call — fast and amortised over the batch.

    All PUTs go through one :func:`asyncio.gather`, then the function
    blocks until they complete (or the first error propagates).
    Idempotent on empty inputs.
    """
    triples = [_normalize_triple(t) for t in triples]
    array_metas = dict(array_metas or {})

    if not triples and not array_metas:
        return

    # Any per-write codec override forces the sync encoder path (the fast
    # async PUT below can only emit raw bytes under one uniform codec).
    has_overrides = any(override is not None for _, _, _, override in triples)

    # Icechunk doesn't pick up arrays added via raw ``store.set`` of
    # their ``zarr.json`` — it tracks arrays as first-class entities and
    # expects them through ``zarr.Array.create_array``.  Same path is
    # required when the codec pipeline includes a compressor: the fast
    # async PUT below can only emit raw bytes, so a compressor in the
    # pipeline must go through zarr's encoder.
    if (
        _is_icechunk_store(zarr_group.store)
        or not _is_bytes_only_codecs(codecs)
        or has_overrides
    ):
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
    chunk_array_names = {array_name for array_name, _, _, _ in triples}
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

    # Per-chunk inner-array PUTs.  (This path only runs when no per-write
    # override is present, so the uniform batch ``codecs_json`` applies.)
    for array_name, chunk_key, data, _override in triples:
        inner_prefix = f"{base_prefix}{array_name}/{chunk_key}/"
        n = len(data)
        puts.append(
            (f"{inner_prefix}zarr.json", _inner_array_meta_bytes(n, codecs_json))
        )
        if n > 0:
            puts.append((f"{inner_prefix}c/0", data))

    sync(_async_put_chunks(zarr_group.store, puts))
