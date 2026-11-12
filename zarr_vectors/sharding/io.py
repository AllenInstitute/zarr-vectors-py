"""Sharding for ZV stores using Zarr v3's native ``sharding_indexed`` codec.

Sharding packs many per-chunk byte blobs into a single storage object
to reduce object count on cloud stores (S3/GCS) and inode pressure on
local filesystems.  Every per-chunk array is a multidim vlen-bytes Zarr
array whose shape is the level's chunk grid; this module owns the one
thing that distinguishes its two packings:

* **Unsharded** — one storage object per Zarr Vectors chunk, at
  ``<array>/c/i/j/k``.
* **Sharded** — Zarr v3's built-in ``sharding_indexed`` codec packs the
  chunk-grid cells into outer-chunk shards.  One storage object per
  shard (default 512 Zarr Vectors chunks per shard).

The logical layout is identical either way — only the codec pipeline in
each array's ``zarr.json`` changes — so both are read transparently by
:class:`zarr_vectors.core.group.Group` with no dispatch at all.  Both
link families are ordinary rank-D grids (cell = the record's source
chunk), so they shard exactly like ``vertices``.

Public API
----------

``shard_store(path, *, shard_shape=8)``
    Repack every per-chunk array in the store with the
    ``sharding_indexed`` codec.  ``shard_shape`` is either an int (broadcast to
    every axis) or an explicit per-axis tuple.  Idempotent: arrays
    already sharded with the requested shape are skipped.

``unshard_store(path)``
    Reverse direction: every sharded array is unpacked back to one
    storage object per chunk.  The array itself stays put.

``reshard(path, shard_shape)``
    Convenience wrapper: ``shard_shape=None`` → unshard, otherwise
    re-shard with the requested shape (round-trips through unsharded
    when the current shape differs).

``set_octree(path, octree=True)``
    Mark the store octree (zarr-octree's ``zarr_octree_sharding`` codec)
    or unmark it, and repack its shards to match; each array through a
    scratch copy, so an interrupted run loses nothing and a re-run
    finishes it.

``is_sharded(path) -> bool`` / ``get_shard_info(path) -> dict``
    Status queries — checks whether any array in the store uses the
    ``sharding_indexed`` codec.

There is no Morton / Hilbert curve indirection: the native sharding
codec already clusters spatially-adjacent inner chunks
into the same shard via its C-order outer grid, giving the same
read-locality benefit without a custom mapping.

Post-hoc, not the only way
--------------------------

These verbs repack a store that already exists.  A store can also be
born sharded -- ``create_store(shard_shape=)`` records the shape and
every array allocated afterwards honours it, including the ones created
lazily in workers.  That is cheaper than writing flat and repacking, and
it is what a pipeline that knows its own object budget should do.

Sharding need not be the last coordinator pass:
:meth:`~zarr_vectors.core.group.Group.derive_nonempty_chunks` reads a
sharded array's cells out of its shard objects.  One ordering does
hold: ``shard_store`` selects cells through the presence manifest, so a store whose workers
wrote with ``record_presence=False`` still needs its rebuild before it
is repacked -- otherwise there is nothing to migrate and the pass
silently does nothing.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zarr_vectors.core.group import _parse_chunk_coords
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group


# ===================================================================
# Walking per-chunk arrays
# ===================================================================


def _is_native_sharded(zarr_node) -> bool:
    """True iff ``zarr_node`` is a Zarr Array using a sharding codec:
    ``sharding_indexed``, or one built on zarr's ``ShardingCodec`` (such as
    zarr-octree's ``zarr_octree_sharding``), whose shards zarr reports the same way."""
    import zarr
    from zarr.codecs.sharding import ShardingCodec

    if not isinstance(zarr_node, zarr.Array):
        return False
    return any(isinstance(c, ShardingCodec) or _codec_name(c) == "sharding_indexed"
               for c in zarr_node.metadata.codecs)


def _codec_name(codec: Any) -> str | None:
    """Return the on-wire name of a Zarr codec (``sharding_indexed``,
    ``vlen-bytes``, ``zstd``, ...).

    Zarr 3.2's ``Codec`` instances expose the registered name via
    ``to_dict()["name"]`` rather than a Python ``.name`` attribute.
    """
    name = getattr(codec, "name", None)
    if name:
        return name
    to_dict = getattr(codec, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict().get("name")
        except Exception:
            return None
    return None


def _repacking(level: Group) -> Any:
    """Mark ``level`` as rewritten without changing its cells, for the
    duration of the block, so a level with octree chunks lets the write
    through (see :mod:`zarr_vectors.octree_chunks.guard`)."""
    from zarr_vectors.octree_chunks.guard import converting

    return converting(level)


def _refuse_unfinished_conversions(root: Group, what: str) -> None:
    from zarr_vectors.octree_chunks.guard import refuse_unfinished_conversions

    refuse_unfinished_conversions(root, what)


def _octree_chunks_marker(zarr_node: Any) -> list[dict[str, Any]] | None:
    """``[marker]`` when ``zarr_node``'s cell codecs hold the octree-chunks
    marker (:mod:`zarr_vectors.encoding.octree_marker`), else ``None``.

    The repackers below rebuild an array without its cell compressors;
    the marker is kept, since without it a reader that does not know
    octree chunks would read the level's leaves as uniform cells.
    """
    from zarr_vectors.encoding.octree_marker import OCTREE_MARKER_CODEC, is_marker

    codecs = list(zarr_node.metadata.codecs)
    if codecs and _codec_name(codecs[0]) in ("sharding_indexed", "zarr_octree_sharding"):
        codecs = list(getattr(codecs[0], "codecs", ()))
    return [{"name": OCTREE_MARKER_CODEC}] if any(is_marker(c) for c in codecs) else None


def _list_array_names(level_group, requested: list[str] | None) -> list[str]:
    """Enumerate the per-spatial-chunk array paths under a level.

    Every such array is a single vlen-bytes Zarr array (``vertices``,
    ``vertex_fragments``, ``links/<delta>/<offsets>``,
    ``vertex_attributes/<name>``,
    ``link_attributes/<name>/<delta>/<offsets>``, …).  Recursively walks
    the level's group hierarchy — the link families nest two levels
    deeper than the rest — and returns each array path for which
    :func:`zarr_vectors.core.arrays._is_per_chunk_array` holds.  That
    predicate is what excludes ``object_index``, ``object_attributes``
    and ``groups``, whose arrays have no spatial chunk grid; both link
    families ARE rank-D grids and shard like any other array.

    ``requested`` short-circuits the walk: callers pass an explicit list
    to limit the migration to specific arrays.
    """
    import zarr

    from zarr_vectors.core.arrays import _is_per_chunk_array

    if requested is not None:
        return requested

    names: list[str] = []

    def _walk(prefix: str, group: zarr.Group) -> None:
        for name in group.array_keys():
            path = f"{prefix}{name}"
            if _is_per_chunk_array(path):
                names.append(path)
        for name in group.group_keys():
            _walk(f"{prefix}{name}/", group[name])

    _walk("", level_group.zarr_group)
    # The store lists keys in whatever order it likes (directory order on
    # a local store), so sort: get_shard_info reports, and the migrations
    # process, the arrays in one stable order.
    return sorted(names)


# ===================================================================
# Shape inference
# ===================================================================


def _normalise_shard_shape(
    shard_shape: int | Sequence[int] | None, ndim: int,
) -> tuple[int, ...] | None:
    if shard_shape is None:
        return None
    if isinstance(shard_shape, int):
        if shard_shape < 1:
            raise ValueError(
                f"shard_shape must be >= 1, got {shard_shape}"
            )
        return (shard_shape,) * ndim
    shape = tuple(int(s) for s in shard_shape)
    if len(shape) != ndim:
        raise ValueError(
            f"shard_shape {shape} has rank {len(shape)} but chunks "
            f"have rank {ndim}"
        )
    if any(s < 1 for s in shape):
        raise ValueError(f"shard_shape components must be >= 1, got {shape}")
    return shape


# ===================================================================
# shard_store / unshard_store / reshard
# ===================================================================


def shard_store(
    store_path: str | Path | Group,
    *,
    shard_shape: int | Sequence[int] = 8,
    arrays: list[str] | None = None,
    object_shard_rows: int | None = None,
) -> dict[str, Any]:
    """Repack every per-chunk array in the store with the
    ``sharding_indexed`` codec.

    The sharded layout uses Zarr v3's built-in ``sharding_indexed`` codec
    — readable by any standards-compliant Zarr v3 implementation
    (zarrs, tensorstore, neuroglancer-precomputed/zarr3, ...).  No
    ZV-specific metadata is needed; the sharding configuration lives
    in each array's ``zarr.json``.

    Args:
        store_path: Path or URL to the ZV store.
        shard_shape: Outer-chunk shape in *inner-chunk* units (one
            inner chunk == one Zarr Vectors spatial chunk).  An ``int`` is
            broadcast to every axis (e.g. ``8`` → ``(8,8,8)`` for a
            3-D store).  A tuple sets the per-axis shard shape
            explicitly.  Default ``8`` ≈ 512 inner chunks per shard,
            a reasonable cloud default per
            :doc:`/spec/chunking/sharding`.
        arrays: Optional list of logical array names to migrate.  When
            omitted, every per-array container under every resolution
            level is converted.
        object_shard_rows: Also repack every level's object layer into
            shards of this many rows (a multiple of 65,536); see
            :func:`shard_object_layer`.  ``None`` leaves it alone.

    Returns:
        Stats dict with ``arrays_sharded``, ``chunks_packed``,
        ``shard_shape``, and with ``object_shard_rows`` the
        ``object_arrays_repacked`` count.
    """

    root = open_store(store_path, mode="r+")
    _refuse_unfinished_conversions(root, "shard_store")

    arrays_sharded = 0
    chunks_packed = 0
    final_shard_shape: tuple[int, ...] | None = None

    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        for array_name in _list_array_names(level, arrays):
            if not level.array_exists(array_name):
                continue

            import zarr
            existing = level.zarr_group[array_name]
            if not isinstance(existing, zarr.Array):
                # The walk only yields arrays, so a group here means an
                # explicit ``arrays=`` named a container rather than an
                # array — e.g. the ``links/<delta>`` family group instead
                # of one of its ``<offsets>`` children.  Its children
                # shard fine; name them.
                continue
            # Reuse the source array's grid shape + origin verbatim —
            # re-sharding only repacks cells into shards, it does not
            # change the grid.
            grid_shape = tuple(int(s) for s in existing.shape)
            raw_origin = existing.attrs.get("chunk_grid_origin")
            origin: tuple[int, ...] | None = (
                tuple(int(x) for x in raw_origin) if raw_origin else None
            )
            ndim = len(grid_shape)
            this_shard_shape = _normalise_shard_shape(shard_shape, ndim)
            if final_shard_shape is None:
                final_shard_shape = this_shard_shape

            # Already native-sharded with the right shape → skip.
            # Tested BEFORE the manifest walk below: on a store that was
            # born sharded this is every array, and the answer is one
            # zarr.json read rather than a presence enumeration per array.
            if _is_native_sharded(existing) and existing.shards == this_shard_shape:
                continue

            chunk_keys = [
                k for k in level.list_chunks(array_name)
                if _parse_chunk_coords(k) is not None
            ]
            if not chunk_keys:
                continue

            # Snapshot existing per-chunk payloads + array metadata so
            # we can rebuild after replacing the node at this path.  The
            # presence manifest and origin are managed by the create /
            # write calls below, so drop them from the carried-over attrs.
            # The reads go out in one gather: they are the whole array,
            # and one round-trip per cell is exactly what sharding is
            # being asked to fix.
            chunk_payloads: dict[str, bytes] = {}
            with level.batched_reads([(array_name, list(chunk_keys))]):
                for k in chunk_keys:
                    chunk_payloads[k] = level.read_bytes(array_name, k)
            preserved_attrs = dict(level.read_array_meta(array_name))
            preserved_attrs.pop("nonempty_chunks", None)
            preserved_attrs.pop("chunk_grid_origin", None)
            marker = _octree_chunks_marker(existing)

            # Drop the prior array before re-creating at this path.
            del level.zarr_group[array_name]

            # Allocate the native-sharded vlen-bytes array.  Deliberately
            # outside the write batch below: this call writes the array's
            # own ``zarr.json``, and the batch's deferred-metadata path
            # is for *group* nodes.
            level.create_sharded_chunk_array(
                array_name,
                grid_shape=grid_shape,
                shard_shape=this_shard_shape,
                origin=origin,
                compressors=marker,
                attributes=preserved_attrs,
            )

            n_packed = len(chunk_payloads)
            # Write every cell in one batch. Unbatched, each write_bytes
            # is a read-modify-write of the whole shard the cell lands in
            # plus a parse-insert-sort-rewrite of the array's presence
            # manifest -- ~60ms per cell measured, so packing 13,824
            # cells did not finish in 15 minutes and a million cells
            # would take the better part of a day. Batched, the cells
            # land in one ``set_coordinate_selection`` (zarr writes each
            # shard once) and presence is stamped once for the array.
            # The cells go back unchanged, so a level with octree chunks
            # stays one: let its write guard pass this repack.
            with _repacking(level), level.batched_writes():
                for k in chunk_keys:
                    data = chunk_payloads.pop(k)
                    if not data:
                        continue
                    level.write_bytes(array_name, k, data)

            arrays_sharded += 1
            chunks_packed += n_packed

    out: dict[str, Any] = {
        "arrays_sharded": arrays_sharded,
        "chunks_packed": chunks_packed,
        "shard_shape": list(final_shard_shape) if final_shard_shape else None,
    }
    if object_shard_rows is not None:
        out["object_arrays_repacked"] = sum(
            len(shard_object_layer(
                get_resolution_level(root, level_idx), object_shard_rows,
            )["arrays_repacked"])
            for level_idx in list_resolution_levels(root)
        )
    return out


def unshard_store(
    store_path: str | Path | Group,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Reverse of :func:`shard_store`: rewrite every native-sharded
    per-chunk array as an **unsharded** single vlen-bytes array — one
    storage object per spatial chunk (``<array>/c/i/j/k``).

    The logical single-array layout is unchanged; only the Zarr v3
    ``sharding_indexed`` packing is removed.  Unsharding the whole store
    (``arrays=None``) also clears the root ``shard_shape``, so arrays
    created afterwards are unsharded too.  Useful for write-heavy
    workflows where shard read-modify-write contention hurts throughput,
    or when a reader prefers one file per chunk.  On a store marked octree
    (:func:`set_octree`) the marker and its capability are removed once no
    ``zarr_octree_sharding`` array is left.
    """
    import zarr

    root = open_store(store_path, mode="r+")
    _refuse_unfinished_conversions(root, "unshard_store")

    arrays_unsharded = 0
    chunks_extracted = 0

    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        for array_name in _list_array_names(level, arrays):
            if not level.standalone_array_exists(array_name):
                continue
            existing = level.zarr_group[array_name]
            if not isinstance(existing, zarr.Array) or not _is_native_sharded(
                existing
            ):
                continue  # already unsharded

            grid_shape = tuple(int(s) for s in existing.shape)
            raw_origin = existing.attrs.get("chunk_grid_origin")
            origin = (
                tuple(int(x) for x in raw_origin) if raw_origin else None
            )
            chunk_keys = level.list_chunks(array_name)
            # One gather rather than a round trip per cell, as the shard
            # direction already does. It matters more here: every one of
            # these reads is currently a whole-shard fetch.
            with level.batched_reads([(array_name, chunk_keys)]):
                chunk_payloads: dict[str, bytes] = {
                    k: level.read_bytes(array_name, k) for k in chunk_keys
                }
            preserved_attrs = dict(level.read_array_meta(array_name))
            preserved_attrs.pop("nonempty_chunks", None)
            preserved_attrs.pop("chunk_grid_origin", None)
            marker = _octree_chunks_marker(existing)

            del level.zarr_group[array_name]

            level.create_sharded_chunk_array(
                array_name,
                grid_shape=grid_shape,
                shard_shape=None,
                origin=origin,
                compressors=marker,
                attributes=preserved_attrs,
            )

            with _repacking(level):  # cells unchanged: see shard_store
                for k, data in chunk_payloads.items():
                    if not data:
                        continue
                    level.write_bytes(array_name, k, data)

            arrays_unsharded += 1
            chunks_extracted += len(chunk_payloads)

    # The root's shard_shape is what every array allocated LATER honours
    # (create_store(shard_shape=) records it for exactly that), so leaving
    # it behind made the next write -- a pyramid level, a new attribute --
    # come out sharded again in a store just unsharded.  Only a whole-store
    # unshard clears it: one limited to some arrays leaves the rest, and
    # the store's default, as they were.
    if arrays is None:
        attrs = root.attrs.to_dict()
        block = dict(attrs.get("zarr_vectors", {}))
        if block.pop("shard_shape", None) is not None:
            root.attrs.update({"zarr_vectors": block})

    # A store marked octree whose octree arrays are gone is no longer
    # octree: the marker and its capability follow the arrays (see
    # _settle_octree_marker).  An unmarked store's root is not touched.
    from zarr_vectors.constants import CAP_OCTREE_SHARDING

    block = root.attrs.to_dict().get("zarr_vectors", {})
    if (block.get("octree_sharding") is not None
            or CAP_OCTREE_SHARDING in (block.get("format_capabilities") or [])):
        _settle_octree_marker(root, None)

    return {
        "arrays_unsharded": arrays_unsharded,
        "chunks_extracted": chunks_extracted,
    }


def shard_object_layer(level_group: Group, shard_rows: int | None) -> dict[str, Any]:
    """Repack a level's object layer into shards of ``shard_rows`` rows.

    The object layer -- ``object_index``'s arrays and every
    ``object_attributes/<name>`` column -- is 1-D along objects, chunked
    in fixed row buckets (16,384 index rows, 65,536 attribute rows), one
    storage object per bucket: about 70,000 objects per column at 10^9
    objects. Sharding packs ``shard_rows // bucket`` buckets into one
    object. ``shard_rows`` must be a multiple of 65,536, so both buckets
    tile it; ``None`` unshards.

    Nothing here is a zarr-vectors format key: sharding lives in each
    array's own ``zarr.json`` as zarr's ``sharding_indexed`` codec, which
    any Zarr v3 reader reads. Writers keep a level's object layer in the
    layout its arrays already have (see
    :func:`zarr_vectors.core.arrays.object_shard_rows`), so a column
    created after this call is sharded too.

    Each array is streamed, never held whole: copied block by block
    (blocks aligned to both layouts) into a sibling array, which then
    replaces it -- by a directory rename on a local store, by copying its
    objects elsewhere. A coordinator operation: nothing else may write the
    object layer meanwhile.

    Returns:
        ``{"arrays_repacked": [...], "shard_rows": shard_rows}``.
    """
    from zarr_vectors.core.arrays import (
        check_object_shard_rows,
        object_layer_arrays,
    )

    rows = None if shard_rows is None else check_object_shard_rows(shard_rows)
    repacked: list[str] = []
    for path in object_layer_arrays(level_group):
        arr = level_group.zarr_group[path]
        want = None if rows is None else (rows, *(int(c) for c in arr.chunks[1:]))
        have = tuple(int(s) for s in arr.shards) if arr.shards is not None else None
        if have == want:
            continue
        _repack_rows(level_group, path, arr, want)
        repacked.append(path)
    return {"arrays_repacked": repacked, "shard_rows": rows}


_REPACK_SUFFIX = "__repacking"


def _repack_rows(level_group: Group, path: str, arr: Any, shards: Any) -> None:
    """Rewrite one row-chunked array with ``shards``, streamed."""
    import math
    import warnings

    from zarr.errors import UnstableSpecificationWarning

    parent_path, _, leaf = path.rpartition("/")
    parent = level_group.zarr_group[parent_path] if parent_path else level_group.zarr_group
    tmp_leaf = f"{leaf}{_REPACK_SUFFIX}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        if tmp_leaf in parent:
            del parent[tmp_leaf]  # left by an interrupted repack
        tmp = parent.create_array(
            tmp_leaf,
            shape=arr.shape,
            chunks=arr.chunks,
            shards=shards,
            dtype=arr.metadata.data_type,
            serializer=arr.serializer,
            compressors=arr.compressors,
            filters=arr.filters,
            fill_value=arr.fill_value,
            attributes=dict(arr.attrs),
        )
        old_outer = int((arr.shards or arr.chunks)[0])
        new_outer = int((shards or arr.chunks)[0])
        step = math.lcm(old_outer, new_outer)
        n = int(arr.shape[0])
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            tmp[lo:hi] = arr[lo:hi]
    _replace_array(level_group, parent, leaf, tmp_leaf)
    level_group._invalidate_node(path)


def _replace_array(level_group: Group, parent: Any, leaf: str, tmp_leaf: str) -> None:
    """Put ``parent/tmp_leaf`` where ``parent/leaf`` was."""
    import os

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync
    from zarr.storage import LocalStore

    store = level_group.zarr_group.store
    base = parent.path.strip("/")
    src = f"{base}/{tmp_leaf}" if base else tmp_leaf
    dst = f"{base}/{leaf}" if base else leaf
    del parent[leaf]
    if isinstance(store, LocalStore):
        os.replace(Path(store.root) / src, Path(store.root) / dst)
        return

    async def _copy() -> None:
        proto = default_buffer_prototype()
        async for key in store.list_prefix(f"{src}/"):
            value = await store.get(key, prototype=proto)
            if value is not None:
                await store.set(f"{dst}/{key[len(src) + 1:]}", value)

    sync(_copy())
    del parent[tmp_leaf]


def reshard(
    store_path: str | Path | Group,
    shard_shape: int | Sequence[int] | None,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Re-layout a ZV store between unsharded and sharded packings.

    Args:
        store_path: Path or URL to the store.
        shard_shape: ``None`` → unshard (one object per chunk);
            ``int`` or
            tuple → shard with that outer-chunk shape.
        arrays: Optional list of logical array names to limit the
            operation to.

    Returns:
        Stats dict from the underlying :func:`shard_store` or
        :func:`unshard_store` call, plus ``action`` describing what
        ran.
    """
    if shard_shape is None:
        if not is_sharded(store_path):
            return {"action": "noop", "message": "already unsharded"}
        result = unshard_store(store_path, arrays=arrays)
        return {"action": "unshard", **result}

    result = shard_store(store_path, shard_shape=shard_shape, arrays=arrays)
    return {"action": "shard", **result}


def _octree_codec_matches(existing: Any, octree: dict[str, Any] | None) -> bool:
    """Whether a sharded array's codec is the one ``octree`` (the store's
    marker, or None for ``sharding_indexed``) would give it."""
    codec = existing.metadata.codecs[0]
    if octree is None:
        return _codec_name(codec) == "sharding_indexed"
    if _codec_name(codec) != "zarr_octree_sharding":
        return False
    from zarr_vectors.core.group import _shard_codec

    want = _shard_codec(len(existing.shape), [], octree).to_dict()["configuration"]
    have = codec.to_dict()["configuration"]
    return all(have.get(k) == v for k, v in want.items() if k not in ("codecs", "index_codecs"))


def set_octree(
    store_path: str | Path | Group,
    octree: bool | dict[str, Any] = True,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Mark a sharded store octree, or unmark it, and repack its shards to match.

    ``octree=True`` (or the codec's configuration, e.g. ``{"target":
    100_000}``) stamps the root's ``octree_sharding`` marker and
    :data:`~zarr_vectors.constants.CAP_OCTREE_SHARDING`, then rewrites every
    sharded per-chunk array with zarr-octree's ``zarr_octree_sharding``
    codec; ``octree=False`` rewrites them with ``sharding_indexed`` and
    then removes both -- the store as it would have been written
    unmarked.  Cells, keys, grid, each array's inner codecs and
    attributes are kept; only the bytes inside each shard change.  A
    store sharded after the fact (:func:`shard_store`, which records no
    root ``shard_shape``) is accepted: its arrays' shard shape is stamped
    on the root with the marker.

    Each array is converted into a scratch copy (``<level>/__set_octree``)
    and swapped in only once that copy is complete, so the array's cells
    exist somewhere on disk at every moment.  A run that is interrupted --
    killed, preempted, a failed write -- is finished by running the call
    again: it first completes or discards the conversion it finds in
    flight (a complete copy is swapped in, an incomplete one dropped, the
    source being untouched until then), then skips every array already in
    the wanted codec.  Between the two runs a local store reads correctly
    (each array's ``zarr.json`` says which codec it is in) unless the
    kill fell between a swap's two renames; on any other store a kill
    during a swap's copy leaves that array missing (``validate`` reports
    it) until the call is run again.  A
    coordinator operation: nothing else may write the store meanwhile,
    and two runs must not overlap.

    The marker is written first when the store is being marked (so the
    capability says zarr-octree is needed before any array needs it) and
    removed last when it is being unmarked.  With ``arrays=`` the marker
    changes only once every sharded array of the store is in the wanted
    codec; until then it is left as it was, and the capability stays on
    the root while any array is ``zarr_octree_sharding``.  Needs the
    zarr-octree package for either direction when an octree array is
    involved.

    Returns:
        ``{"octree": the configuration asked for or None, "arrays_repacked",
        "chunks_repacked"}``.
    """
    import zarr

    from zarr_vectors.core.group import check_octree_sharding
    from zarr_vectors.core.store import read_root_metadata

    root = open_store(store_path, mode="r+")
    _refuse_unfinished_conversions(root, "set_octree")
    meta = read_root_metadata(root)
    ndim = len(meta.bounds[0])
    config = (
        None if octree is False or octree is None
        # zarr-octree present and the configuration valid, before anything changes
        else check_octree_sharding({} if octree is True else octree, ndim)
    )
    stamp_shape = None
    if config is not None and meta.shard_shape is None:
        stamp_shape = _common_shard_shape(root, ndim)
    if config is not None and arrays is None:
        # The marker first: every array allocated from here on, by a
        # writer, takes its codec from it, and the capability is on the
        # root before the first octree array is.
        _write_octree_block(root, config, needs_octree=True, shard_shape=stamp_shape)

    repacked = chunks = 0
    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        _finish_interrupted_octree(level)
        for array_name in _list_array_names(level, arrays):
            if not level.standalone_array_exists(array_name):
                continue
            existing = level.zarr_group[array_name]
            if not isinstance(existing, zarr.Array) or not _is_native_sharded(existing):
                continue
            if _octree_codec_matches(existing, config):
                continue
            chunks += _convert_sharded_array(level, array_name, existing, config)
            repacked += 1

    _settle_octree_marker(root, config, shard_shape=stamp_shape)
    return {"octree": config, "arrays_repacked": repacked, "chunks_repacked": chunks}


_OCTREE_SCRATCH = "__set_octree"
"""The group, under a level, that :func:`set_octree` builds each converted
array in before swapping it in: ``<level>/__set_octree/<array path>``.
Absent except while an array is being converted, or after a run was
interrupted (the next run finishes or drops it).  No array under it is a
per-chunk array by :func:`zarr_vectors.core.arrays._is_per_chunk_array`,
so nothing that walks the level's arrays mistakes the copy for one."""

_OCTREE_SCRATCH_STATE = "set_octree"
"""The scratch group's attribute: ``{"array": path, "complete": bool,
"swapped": bool}``, each flag stamped in one write once it holds."""


def _convert_sharded_array(
    level: Group, array_name: str, existing: Any, config: dict[str, Any] | None,
) -> int:
    """Rewrite one sharded array in the codec ``config`` names, via the
    scratch copy; returns the number of cells carried over."""
    grid_shape = tuple(int(s) for s in existing.shape)
    shard_shape = tuple(int(s) for s in existing.shards)
    raw_origin = existing.attrs.get("chunk_grid_origin")
    origin = tuple(int(x) for x in raw_origin) if raw_origin else None
    inner = list(existing.metadata.codecs[0].codecs[1:])  # after vlen-bytes: the cells' compressors
    chunk_keys = [k for k in level.list_chunks(array_name) if _parse_chunk_coords(k) is not None]
    with level.batched_reads([(array_name, list(chunk_keys))]):
        payloads = {k: level.read_bytes(array_name, k) for k in chunk_keys}
    preserved_attrs = dict(level.read_array_meta(array_name))
    preserved_attrs.pop("nonempty_chunks", None)
    preserved_attrs.pop("chunk_grid_origin", None)

    # The copy is built beside the source, which is not touched until the
    # copy is complete and says so.  The scratch group is created with its
    # state: its existence alone means "a conversion of this array began".
    # Never over an existing one (_finish_interrupted_octree has dropped
    # any): one there now is another run's, which must not overlap this.
    state = {"array": array_name, "complete": False, "swapped": False}
    scratch = level.zarr_group.create_group(
        _OCTREE_SCRATCH, attributes={_OCTREE_SCRATCH_STATE: state},
    )
    tmp_name = f"{_OCTREE_SCRATCH}/{array_name}"
    level.create_sharded_chunk_array(
        tmp_name, grid_shape=grid_shape, shard_shape=shard_shape,
        origin=origin, compressors=inner, attributes=preserved_attrs, octree=config,
    )
    with _repacking(level), level.batched_writes():  # cells unchanged: see shard_store
        for k in chunk_keys:
            data = payloads.pop(k)
            if data:
                level.write_bytes(tmp_name, k, data)
    scratch.update_attributes({_OCTREE_SCRATCH_STATE: {**state, "complete": True}})
    _swap_in_octree_copy(level, scratch, array_name)
    return len(chunk_keys)


def _finish_interrupted_octree(level: Group) -> None:
    """Complete or discard the conversion an interrupted :func:`set_octree`
    left under ``level``.

    A copy stamped complete is the array's cells in the new codec: it is
    swapped in, whatever state the swap had reached (the source may be
    gone, or half deleted, or half overwritten by a copy on an object
    store).  A copy not stamped complete is dropped: the source is intact,
    since nothing touches it before that stamp.
    """
    from zarr.errors import GroupNotFoundError

    zg = level.zarr_group
    try:
        scratch = _open_scratch(zg)
    except (KeyError, GroupNotFoundError):
        # No scratch group, or only what an interrupted drop of one left
        # (objects under it, its zarr.json already gone): garbage either
        # way, since a conversion stamps its state there before anything else.
        _drop_scratch(level)
        return
    state = scratch.attrs.get(_OCTREE_SCRATCH_STATE) or {}
    name = state.get("array")
    if name and state.get("complete") and not state.get("swapped"):
        if _key_exists(zg.store, _join(zg.path, _OCTREE_SCRATCH, name, "zarr.json")):
            _swap_in_octree_copy(level, scratch, name)
            return
        # No copy and not stamped swapped: only a local swap (one rename)
        # gets here, after the rename -- the copy IS the array now.  Were
        # the array missing too, something else removed both: say so
        # rather than drop the evidence and report success.
        if not _key_exists(zg.store, _join(zg.path, name, "zarr.json")):
            from zarr_vectors.exceptions import StoreError

            raise StoreError(
                f"set_octree: {_join(zg.path, name)!r} and its converted copy under "
                f"{_join(zg.path, _OCTREE_SCRATCH)!r} are both missing; the store "
                f"needs restoring from a backup"
            )
    _drop_scratch(level)


def _open_scratch(zg: Any) -> Any:
    """The scratch group under the level's zarr group, or KeyError."""
    import zarr

    node = zg[_OCTREE_SCRATCH]
    if not isinstance(node, zarr.Group):
        raise KeyError(_OCTREE_SCRATCH)
    return node


def _swap_in_octree_copy(level: Group, scratch: Any, array_name: str) -> None:
    """Put the complete copy ``<scratch>/<array_name>`` where the array is.

    On a local store, the source's directory is renamed aside (into the
    scratch group, dropped with it) and the copy's renamed into its place:
    two renames, so the array is missing only between them, never for the
    length of a delete.  On
    any other store the copy's objects are copied over (``zarr.json``
    last, so a reader meanwhile finds no array rather than a part of one),
    ``swapped`` is stamped, and only then is the copy deleted.  Either way
    a crash at any point leaves a complete copy to swap in again, or a
    finished swap.
    """
    import os

    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync
    from zarr.storage import LocalStore

    zg = level.zarr_group
    store = zg.store
    src = _join(zg.path, _OCTREE_SCRATCH, array_name)
    dst = _join(zg.path, array_name)
    level._invalidate_node(array_name)
    level._invalidate_node(_OCTREE_SCRATCH)
    if isinstance(store, LocalStore):
        target = Path(store.root) / dst
        if target.exists():
            aside = Path(store.root) / _join(zg.path, _OCTREE_SCRATCH, _REPLACED, array_name)
            shutil.rmtree(aside, ignore_errors=True)  # an earlier, interrupted swap's
            aside.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, aside)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(Path(store.root) / src, target)
    else:
        sync(store.delete_dir(dst))
        async def _copy() -> None:
            proto = default_buffer_prototype()
            keys = [k async for k in store.list_prefix(f"{src}/")]
            keys.sort(key=lambda k: k == f"{src}/zarr.json")  # zarr.json last
            for key in keys:
                value = await store.get(key, prototype=proto)
                if value is not None:
                    await store.set(f"{dst}/{key[len(src) + 1:]}", value)

        sync(_copy())
        state = dict(scratch.attrs.get(_OCTREE_SCRATCH_STATE) or {})
        scratch.update_attributes({_OCTREE_SCRATCH_STATE: {**state, "swapped": True}})
    _drop_scratch(level)


# Where a local swap puts the array it replaces, under the scratch group.
_REPLACED = "__replaced"


def _drop_scratch(level: Group) -> None:
    from zarr.core.sync import sync

    zg = level.zarr_group
    level._invalidate_node(_OCTREE_SCRATCH)
    sync(zg.store.delete_dir(_join(zg.path, _OCTREE_SCRATCH)))


def _join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


def _key_exists(store: Any, key: str) -> bool:
    from zarr.core.sync import sync

    return bool(sync(store.exists(key)))


def _sharded_arrays(root: Group):
    """Every natively sharded per-chunk array of the store, as
    ``(level index, array path, zarr array)``."""
    import zarr

    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        for array_name in _list_array_names(level, None):
            try:
                node = level.zarr_group[array_name]
            except KeyError:
                continue
            if isinstance(node, zarr.Array) and _is_native_sharded(node):
                yield level_idx, array_name, node


def _common_shard_shape(root: Group, ndim: int) -> list[int]:
    """The spatial shard shape every sharded array of a store sharded
    after the fact shares (its root records none); ValueError if there
    is none, or more than one."""
    shapes = {
        tuple(int(s) for s in node.shards)[-ndim:]
        for _, _, node in _sharded_arrays(root)
    }
    if not shapes:
        raise ValueError(
            "set_octree needs a sharded store (the octree groups the cells of a "
            "shard): shard it first with shard_store()"
        )
    if len(shapes) > 1:
        raise ValueError(
            f"set_octree needs one shard shape to record on the root, and this "
            f"store's arrays are sharded with {sorted(shapes)}: reshard() it first"
        )
    return list(shapes.pop())


def _write_octree_block(
    root: Group,
    marker: dict[str, Any] | None,
    *,
    needs_octree: bool,
    shard_shape: list[int] | None = None,
) -> None:
    """Set the root's octree marker (``None`` removes it) and its
    capability (present iff ``needs_octree``), and with ``shard_shape``
    the root shard shape when it records none.  Writes the root only if
    something changes."""
    from zarr_vectors.constants import CAP_OCTREE_SHARDING

    block = dict(root.attrs.to_dict().get("zarr_vectors", {}))
    new = dict(block)
    if marker is None:
        new.pop("octree_sharding", None)
    else:
        new["octree_sharding"] = dict(marker)
        if new.get("shard_shape") is None and shard_shape is not None:
            new["shard_shape"] = list(shard_shape)
    caps = list(block.get("format_capabilities") or [])
    if needs_octree and CAP_OCTREE_SHARDING not in caps:
        new["format_capabilities"] = [*caps, CAP_OCTREE_SHARDING]
    elif not needs_octree and CAP_OCTREE_SHARDING in caps:
        new["format_capabilities"] = [c for c in caps if c != CAP_OCTREE_SHARDING]
    if new != block:
        root.attrs.update({"zarr_vectors": new})
    # A handle caches the marker for the arrays it allocates; this one
    # must see the new one.
    root.__dict__.pop("_octree_sharding", None)


def _settle_octree_marker(
    root: Group, wanted: dict[str, Any] | None, *, shard_shape: list[int] | None = None,
) -> None:
    """Make the root agree with the arrays after a conversion.

    The marker becomes ``wanted`` only when every sharded array of the
    store is in the codec ``wanted`` names (vacuously so when none is
    sharded); otherwise it is left as it was -- a conversion limited by
    ``arrays=``, or one interrupted, has not got there yet.  The
    capability says whether reading the store needs zarr-octree: present
    while the store is marked or any array is ``zarr_octree_sharding``.
    """
    all_match = True
    any_octree = False
    for _, _, node in _sharded_arrays(root):
        any_octree = any_octree or _codec_name(node.metadata.codecs[0]) == "zarr_octree_sharding"
        all_match = all_match and _octree_codec_matches(node, wanted)
    current = root.attrs.to_dict().get("zarr_vectors", {}).get("octree_sharding")
    marker = wanted if all_match else current
    _write_octree_block(
        root, marker, needs_octree=marker is not None or any_octree, shard_shape=shard_shape,
    )


# ===================================================================
# Status queries
# ===================================================================


def is_sharded(store_path: str | Path | Group) -> bool:
    """True iff any array in the store uses the ``sharding_indexed`` codec."""
    try:
        root = open_store(store_path)
    except Exception:
        return False
    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        for array_name in _list_array_names(level, None):
            try:
                node = level.zarr_group[array_name]
            except KeyError:
                continue
            if _is_native_sharded(node):
                return True
    return False


def get_shard_info(store_path: str | Path | Group) -> dict[str, Any]:
    """Return a summary of the store's sharding state.

    The result has keys:

    * ``sharded`` — bool, mirrors :func:`is_sharded`.
    * ``arrays`` — list of ``{name, shard_shape, grid_shape}`` dicts,
      one per native-sharded array, ordered by level and then by array
      path.  Every per-chunk array of a level -- ``links/<delta>/<offsets>``
      included -- is indexed by the level's own chunk grid (a link record
      sits in its source chunk's cell), so they share ``grid_shape``.
    * ``shard_count`` — total shard files across all arrays.
    """
    import zarr

    root = open_store(store_path)
    arrays: list[dict[str, Any]] = []
    shard_count = 0

    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        level_prefix = f"{level_idx}/"
        for array_name in _list_array_names(level, None):
            try:
                node = level.zarr_group[array_name]
            except KeyError:
                continue
            if not _is_native_sharded(node):
                continue
            assert isinstance(node, zarr.Array)
            shape = tuple(int(s) for s in node.shape)
            shards = node.shards or (1,) * len(shape)
            shards = tuple(int(s) for s in shards)
            n_shards = 1
            for grid_dim, shard_dim in zip(shape, shards):
                n_shards *= (grid_dim + shard_dim - 1) // shard_dim
            arrays.append({
                "name": level_prefix + array_name,
                "grid_shape": list(shape),
                "shard_shape": list(shards),
                "shard_count": n_shards,
            })
            shard_count += n_shards

    return {
        "sharded": len(arrays) > 0,
        "arrays": arrays,
        "shard_count": shard_count,
    }


__all__ = [
    "shard_store",
    "unshard_store",
    "reshard",
    "is_sharded",
    "get_shard_info",
]
