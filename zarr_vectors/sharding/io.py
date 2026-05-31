"""Sharding for ZV stores using Zarr v3's native ``sharding_indexed`` codec.

Sharding packs many per-chunk byte blobs into a single storage object
to reduce object count on cloud stores (S3/GCS) and inode pressure on
local filesystems.  This module owns the conversion between the two
storage layouts that share the same logical ZV data:

* **Flat** — each per-array group holds one Zarr array per chunk key
  (e.g. ``vertices/0.0.0``, ``vertices/0.0.1``, ...).  One storage
  object per ZVF chunk.
* **Sharded** — each per-array group is replaced with a single
  multidim vlen-bytes Zarr array at the same logical path.  Zarr v3's
  built-in ``sharding_indexed`` codec packs the chunk-grid cells into
  outer-chunk shards.  One storage object per shard (default 512
  ZVF chunks per shard).

Both layouts are read transparently by :class:`zarr_vectors.core.group.Group`
— ``read_bytes`` / ``write_bytes`` / ``list_chunks`` dispatch on the
node type at the array's path (Array → sharded, Group → flat).

Public API
----------

``shard_store(path, *, shard_shape=8)``
    Convert every per-array group in the store to a native-sharded
    vlen-bytes array.  ``shard_shape`` is either an int (broadcast to
    every axis) or an explicit per-axis tuple.  Idempotent: arrays
    already sharded with the requested shape are skipped.

``unshard_store(path)``
    Reverse direction: every native-sharded array is unpacked back to
    a Zarr group with one child array per chunk key.

``reshard(path, shard_shape)``
    Convenience wrapper: ``shard_shape=None`` → unshard, otherwise
    re-shard with the requested shape (round-trips through unsharded
    when the current shape differs).

``is_sharded(path) -> bool`` / ``get_shard_info(path) -> dict``
    Status queries — checks whether any array in the store uses the
    ``sharding_indexed`` codec.

The Morton / Hilbert curve indirection of older versions is gone; the
native sharding codec already clusters spatially-adjacent inner chunks
into the same shard via its C-order outer grid, giving the same
read-locality benefit without a custom mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from zarr_vectors.constants import (
    CROSS_CHUNK_LINK_ATTRIBUTES,
    LINK_ATTRIBUTES,
)
from zarr_vectors.core.group import _parse_chunk_coords
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
)


# ===================================================================
# Walking per-array groups
# ===================================================================


_DOUBLE_DESCENT_PREFIXES = frozenset({
    LINK_ATTRIBUTES,
    CROSS_CHUNK_LINK_ATTRIBUTES,
})


def _looks_like_per_chunk_group(zarr_group) -> bool:
    """Return True if ``zarr_group`` holds chunk-key-named child arrays
    (the legacy Option-G layout for per-chunk byte blobs).
    """
    for name in zarr_group.array_keys():
        if _parse_chunk_coords(name) is not None:
            return True
    return False


def _is_native_sharded(zarr_node) -> bool:
    """True iff ``zarr_node`` is a Zarr Array using sharding_indexed."""
    import zarr

    if not isinstance(zarr_node, zarr.Array):
        return False
    return any(_codec_name(c) == "sharding_indexed"
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


def _list_array_names(level_group, requested: list[str] | None) -> list[str]:
    """Enumerate per-array logical names under a resolution level.

    Walks the level's hierarchy and returns every path whose node looks
    like a per-chunk container — either the legacy "Option G" group
    (children are chunk-key-named Zarr arrays) or a native-sharded
    Zarr array.

    ``requested`` short-circuits the walk: callers pass an explicit
    list to limit the migration to specific arrays (e.g. just
    ``vertex_fragments``).
    """
    import zarr

    if requested is not None:
        return requested

    names: list[str] = []
    zg = level_group.zarr_group

    for top in zg:
        sub = zg[top]
        if isinstance(sub, zarr.Array):
            # Top-level native-sharded array (vertices, links/<delta>
            # rolled up — but the delta cases nest under a sub-group).
            names.append(top)
            continue
        if _looks_like_per_chunk_group(sub):
            names.append(top)
            continue
        # First descent: attributes/<name>, links/<delta>,
        # cross_chunk_links/<delta>.
        for child in sub.group_keys():
            child_node = sub[child]
            if _looks_like_per_chunk_group(child_node):
                names.append(f"{top}/{child}")
                continue
            if top not in _DOUBLE_DESCENT_PREFIXES:
                continue
            for grand in child_node.group_keys():
                grand_node = child_node[grand]
                if _looks_like_per_chunk_group(grand_node):
                    names.append(f"{top}/{child}/{grand}")
        # Native-sharded arrays nested one level deep (links/<delta>).
        for child in sub.array_keys():
            child_node = sub[child]
            if _is_native_sharded(child_node):
                names.append(f"{top}/{child}")
    return names


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


def _infer_grid_shape(chunk_keys: list[str]) -> tuple[int, ...]:
    """Derive an enclosing grid_shape from observed chunk keys.

    Returns ``max(coord[axis]) + 1`` per axis across all keys.  Empty
    input raises — callers should skip arrays with no chunks.
    """
    parsed: list[tuple[int, ...]] = []
    for k in chunk_keys:
        coords = _parse_chunk_coords(k)
        if coords is not None:
            parsed.append(coords)
    if not parsed:
        raise ValueError("Cannot infer grid_shape: no parseable chunk keys")
    ndim = len(parsed[0])
    if any(len(c) != ndim for c in parsed):
        raise ValueError(
            f"Chunk keys have mixed rank: {sorted({len(c) for c in parsed})}"
        )
    return tuple(max(c[axis] for c in parsed) + 1 for axis in range(ndim))


# ===================================================================
# shard_store / unshard_store / reshard
# ===================================================================


def shard_store(
    store_path: str | Path,
    *,
    shard_shape: int | Sequence[int] = 8,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Convert every per-array group in the store to a native-sharded
    vlen-bytes Zarr array.

    The new layout uses Zarr v3's built-in ``sharding_indexed`` codec
    — readable by any standards-compliant Zarr v3 implementation
    (zarrs, tensorstore, neuroglancer-precomputed/zarr3, ...).  No
    ZV-specific metadata is needed; the sharding configuration lives
    in each array's ``zarr.json``.

    Args:
        store_path: Path or URL to the ZV store.
        shard_shape: Outer-chunk shape in *inner-chunk* units (one
            inner chunk == one ZVF spatial chunk).  An ``int`` is
            broadcast to every axis (e.g. ``8`` → ``(8,8,8)`` for a
            3-D store).  A tuple sets the per-axis shard shape
            explicitly.  Default ``8`` ≈ 512 inner chunks per shard,
            a reasonable cloud default per
            :doc:`/spec/chunking/sharding`.
        arrays: Optional list of logical array names to migrate.  When
            omitted, every per-array container under every resolution
            level is converted.

    Returns:
        Stats dict with ``arrays_sharded``, ``chunks_packed``,
        ``shard_shape``.
    """
    store_path = Path(store_path) if isinstance(store_path, str) else store_path

    root = open_store(str(store_path), mode="r+")

    arrays_sharded = 0
    chunks_packed = 0
    final_shard_shape: tuple[int, ...] | None = None

    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        for array_name in _list_array_names(level, arrays):
            if not level.array_exists(array_name):
                continue
            chunk_keys = [
                k for k in level.list_chunks(array_name)
                if _parse_chunk_coords(k) is not None
            ]
            if not chunk_keys:
                continue

            grid_shape = _infer_grid_shape(chunk_keys)
            ndim = len(grid_shape)
            this_shard_shape = _normalise_shard_shape(shard_shape, ndim)
            if final_shard_shape is None:
                final_shard_shape = this_shard_shape

            # Already native-sharded with the right shape → skip.
            existing = level.zarr_group[array_name]
            import zarr
            if (
                isinstance(existing, zarr.Array)
                and _is_native_sharded(existing)
                and existing.shards == this_shard_shape
            ):
                continue

            # Snapshot existing per-chunk payloads + array metadata so
            # we can rebuild after replacing the node at this path.
            chunk_payloads: dict[str, bytes] = {}
            for k in chunk_keys:
                chunk_payloads[k] = level.read_bytes(array_name, k)
            preserved_attrs = dict(level.read_array_meta(array_name))

            # Delete the legacy group / prior array.
            del level.zarr_group[array_name]

            # Allocate the native-sharded vlen-bytes array.
            level.create_sharded_chunk_array(
                array_name,
                grid_shape=grid_shape,
                shard_shape=this_shard_shape,
                attributes=preserved_attrs,
            )

            # Write each chunk into its grid-coord cell.
            for k, data in chunk_payloads.items():
                if not data:
                    continue
                level.write_bytes(array_name, k, data)

            arrays_sharded += 1
            chunks_packed += len(chunk_payloads)

    return {
        "arrays_sharded": arrays_sharded,
        "chunks_packed": chunks_packed,
        "shard_shape": list(final_shard_shape) if final_shard_shape else None,
    }


def unshard_store(
    store_path: str | Path,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Reverse of :func:`shard_store`: rewrite every native-sharded
    Zarr array as a Zarr group of single-chunk uint8 arrays (the
    "Option G" flat layout).

    Useful for stores that need to be opened by tooling that doesn't
    understand the ``sharding_indexed`` codec, or for write-heavy
    workflows where shard contention hurts throughput.
    """
    import zarr

    store_path = Path(store_path) if isinstance(store_path, str) else store_path
    root = open_store(str(store_path), mode="r+")

    arrays_unsharded = 0
    chunks_extracted = 0

    for level_idx in list_resolution_levels(root):
        level = get_resolution_level(root, level_idx)
        for array_name in _list_array_names(level, arrays):
            if not level.array_exists(array_name):
                continue
            existing = level.zarr_group[array_name]
            if not isinstance(existing, zarr.Array):
                continue

            chunk_keys = level.list_chunks(array_name)
            chunk_payloads: dict[str, bytes] = {
                k: level.read_bytes(array_name, k) for k in chunk_keys
            }
            preserved_attrs = dict(level.read_array_meta(array_name))

            del level.zarr_group[array_name]

            level.zarr_group.require_group(array_name)
            level.write_array_meta(array_name, preserved_attrs)

            for k, data in chunk_payloads.items():
                level.write_bytes(array_name, k, data)

            arrays_unsharded += 1
            chunks_extracted += len(chunk_payloads)

    return {
        "arrays_unsharded": arrays_unsharded,
        "chunks_extracted": chunks_extracted,
    }


def reshard(
    store_path: str | Path,
    shard_shape: int | Sequence[int] | None,
    *,
    arrays: list[str] | None = None,
) -> dict[str, Any]:
    """Re-layout a ZV store between flat and native-sharded forms.

    Args:
        store_path: Path or URL to the store.
        shard_shape: ``None`` → unshard (flat layout); ``int`` or
            tuple → shard with that outer-chunk shape.
        arrays: Optional list of logical array names to limit the
            operation to.

    Returns:
        Stats dict from the underlying :func:`shard_store` or
        :func:`unshard_store` call, plus ``action`` describing what
        ran.
    """
    if shard_shape is None:
        if not is_sharded(str(store_path)):
            return {"action": "noop", "message": "already flat"}
        result = unshard_store(store_path, arrays=arrays)
        return {"action": "unshard", **result}

    result = shard_store(store_path, shard_shape=shard_shape, arrays=arrays)
    return {"action": "shard", **result}


# ===================================================================
# Status queries
# ===================================================================


def is_sharded(store_path: str | Path) -> bool:
    """True iff any array in the store uses the ``sharding_indexed`` codec."""
    try:
        root = open_store(str(store_path))
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


def get_shard_info(store_path: str | Path) -> dict[str, Any]:
    """Return a summary of the store's sharding state.

    The result has keys:

    * ``sharded`` — bool, mirrors :func:`is_sharded`.
    * ``arrays`` — list of ``{name, shard_shape, grid_shape}`` dicts,
      one per native-sharded array.
    * ``shard_count`` — total shard files across all arrays.
    """
    import zarr

    root = open_store(str(store_path))
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


# ===================================================================
# Back-compat shims (deprecation path for old positional API)
# ===================================================================


# Older versions accepted ``ShardLayout`` + ``shard_size`` positional
# args; these alias to the new ``shard_shape``-driven API.  The
# layout-curve (Morton / Hilbert) distinction has been removed because
# Zarr's native ``sharding_indexed`` codec clusters spatially-adjacent
# inner chunks into the same shard automatically.
__all__ = [
    "shard_store",
    "unshard_store",
    "reshard",
    "is_sharded",
    "get_shard_info",
]
