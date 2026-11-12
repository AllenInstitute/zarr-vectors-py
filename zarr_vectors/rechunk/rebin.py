"""Standalone re-binning workflow: change ``bin_shape`` without rechunking.

``bin_shape`` controls the supervoxel grid that subdivides each chunk
into fragments (fragments).  Unlike ``chunk_shape``, changing ``bin_shape``
does not alter the physical chunk file layout — only the in-chunk fragment
boundaries shift.  :func:`rebin_level` re-sorts vertices into new fragments
at a chosen resolution level and updates the declared bin shape on
both the root and level metadata.

Currently supported for **point clouds** only: fragments are re-derived
from positions alone, so any per-object fragment layout (polyline /
streamline / graph / mesh) cannot be preserved.  For those geometry
types use :func:`zarr_vectors.rechunk.rechunk` instead, which
re-derives object manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    FRAGMENT_ATTRIBUTES,
    GEOM_POINT_CLOUD,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_attributes,
    read_chunk_vertices,
    vertices_dtype,
    write_chunk_attributes,
    write_chunk_vertices,
)
from zarr_vectors.core.metadata import validate_bin_shape_divides_chunk
from zarr_vectors.core.multiscale import upsert_level_transform
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
    update_root_metadata,
)
from zarr_vectors.exceptions import StoreError
from zarr_vectors.spatial.chunking import assign_bins
from zarr_vectors.typing import BinShape


def rebin_level(
    store_path: str | Path,
    target_bin_shape: BinShape,
    *,
    level: int = 0,
) -> dict[str, Any]:
    """Re-sort vertices into new fragment bins at the given level.

    Reads each chunk, re-bins its vertices into ``target_bin_shape``
    supervoxel cells, and rewrites the chunk with one fragment per occupied
    bin.  Per-vertex attribute arrays are re-sorted with their vertices,
    and vertices keep the store's dtype.

    The bin shape lives in the NGFF ``multiscales`` block, not in the
    level's own attrs.  Re-binning level 0 sets the root
    ``base_bin_shape`` and the level-0 translation (``bin / 2``), and
    re-derives every coarser level's scale against the new base (their
    absolute bins are unchanged).  Re-binning a coarser level sets that
    level's scale (``bin / base``) and translation.

    Args:
        store_path: Path or URL to a writable ZV store.
        target_bin_shape: New per-axis supervoxel edge lengths.  Must
            evenly divide ``chunk_shape``.
        level: Resolution level to rebin.  Default ``0``.

    Returns:
        Summary dict with ``chunks_processed``, ``vertices_processed``,
        ``bins_created``, ``old_bin_shape``, and ``new_bin_shape``.

    Raises:
        StoreError: If the store contains a non-point-cloud geometry
            type (re-binning would destroy object structure) or the
            requested bin shape doesn't divide the chunk shape.
    """
    store_path = Path(store_path)
    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)

    # Re-binning preserves the chunk-key grid → we can only safely re-sort
    # vertices within a chunk when the on-disk fragment layout is derivable from
    # positions alone.  Point clouds with no object structure satisfy
    # this; everything else carries per-object / per-bundle fragment semantics
    # that we'd flatten irreversibly.
    if root_meta.geometry_types != [GEOM_POINT_CLOUD]:
        raise StoreError(
            f"rebin_level: only point-cloud stores are supported "
            f"(found {root_meta.geometry_types}). For other geometry "
            f"types use zarr_vectors.rechunk.rechunk(), which rebuilds "
            f"object manifests as well."
        )

    chunk_shape = root_meta.chunk_shape
    validate_bin_shape_divides_chunk(chunk_shape, target_bin_shape)

    level_group = get_resolution_level(root, level)
    from zarr_vectors.octree_chunks.guard import refuse_octree_write

    refuse_octree_write(level_group, "rebin_level")
    level_meta = read_level_metadata(root, level)
    old_bin_shape = (
        level_meta.bin_shape
        if level_meta.bin_shape is not None
        else root_meta.effective_bin_shape
    )

    ndim = root_meta.sid_ndim
    chunk_keys = list_chunk_keys(level_group, VERTICES)

    chunks_processed = 0
    vertices_processed = 0
    bins_created = 0
    bs = np.asarray(target_bin_shape, dtype=np.float64)

    # Fragments are rebuilt from positions, so anything that names a
    # fragment -- an object manifest, a per-fragment attribute -- would be
    # left pointing at the wrong vertices.
    named = [f for f in (OBJECT_INDEX, FRAGMENT_ATTRIBUTES)
             if f in level_group.children()]
    if named:
        raise StoreError(
            f"rebin_level: level {level} has {named}, which name fragments "
            f"that re-binning would rebuild; use "
            f"zarr_vectors.rechunk.rechunk_spatial instead"
        )

    vdtype = vertices_dtype(level_group)
    attr_meta = {}
    if VERTEX_ATTRIBUTES in level_group.children():
        for name in sorted(level_group[VERTEX_ATTRIBUTES].children()):
            attr_meta[name] = dict(level_group.read_array_meta(
                f"{VERTEX_ATTRIBUTES}/{name}") or {})

    for cc in chunk_keys:
        # Flatten any existing fragment structure: a point cloud's fragments are
        # re-derived from positions, so we don't care about the old
        # grouping.
        try:
            groups = read_chunk_vertices(level_group, cc, dtype=vdtype, ndim=ndim)
        except Exception:
            continue
        if not groups:
            continue
        positions = np.concatenate(groups, axis=0)
        if positions.shape[0] == 0:
            continue
        # Per-vertex attributes in the same (fragment) order as positions.
        attrs: dict[str, npt.NDArray] = {}
        for name, meta in attr_meta.items():
            adt = np.dtype(meta.get("dtype", "float32"))
            rs = meta.get("row_shape") or []
            ncols = int(rs[0]) if rs else max(
                1, len(meta.get("channel_names") or []))
            agroups = read_chunk_attributes(level_group, name, cc,
                                            dtype=adt, ncols=ncols)
            attrs[name] = np.concatenate([np.asarray(a) for a in agroups])

        # Re-bin into one fragment per occupied bin, sorted by bin key for a
        # deterministic on-disk ordering.
        bin_assignments = assign_bins(positions, tuple(float(b) for b in bs))
        order = [bin_assignments[k] for k in sorted(bin_assignments.keys())]
        new_groups = [positions[idxs] for idxs in order]

        write_chunk_vertices(level_group, cc, new_groups, dtype=vdtype)
        for name, data in attrs.items():
            write_chunk_attributes(level_group, name, cc,
                                   [data[idxs] for idxs in order],
                                   dtype=data.dtype)
        chunks_processed += 1
        vertices_processed += int(positions.shape[0])
        bins_created += len(new_groups)

    # Where the bin shape lives: level 0's is the root ``base_bin_shape``,
    # with the level-0 NGFF translation at ``bin / 2``; a coarse level keeps
    # its own ``bin_shape`` and ``bin_ratio`` (a float, ``bin / base``) in
    # its ``zarr_vectors_level`` block.  Every level's NGFF entry repeats
    # level 0's transform, because coarse vertices are stored in level-0
    # coordinates.  LevelMetadata.validate forbids a level-0 bin_shape.
    new_bin_shape = tuple(float(b) for b in target_bin_shape)
    if level == 0:
        # Through ``update_root_metadata``, which read-modify-writes the one
        # field named.
        update_root_metadata(root, base_bin_shape=new_bin_shape)
        translation0 = [b / 2.0 for b in new_bin_shape]
        upsert_level_transform(
            root, 0, scale=[1.0] * ndim, translation=translation0,
        )
        # Coarser levels keep their absolute bins; their ratio is relative
        # to the base, so it moves with it, and their NGFF entry follows
        # level 0's.
        for lv in list_resolution_levels(root):
            if lv == 0:
                continue
            coarse_bin = read_level_metadata(root, lv).bin_shape
            if coarse_bin is not None:
                _set_level_bins(root, lv, coarse_bin, new_bin_shape)
            upsert_level_transform(
                root, lv, scale=[1.0] * ndim, translation=translation0,
            )
    else:
        _set_level_bins(
            root, level, new_bin_shape,
            read_root_metadata(root).effective_bin_shape,
        )

    return {
        "chunks_processed": chunks_processed,
        "vertices_processed": vertices_processed,
        "bins_created": bins_created,
        "old_bin_shape": tuple(old_bin_shape) if old_bin_shape else None,
        "new_bin_shape": new_bin_shape,
        "level": level,
    }


def _set_level_bins(root, level, bin_shape, base_bin) -> None:
    """Write a coarse level's ``bin_shape`` and its float ``bin_ratio``
    against ``base_bin`` into the level's ``zarr_vectors_level`` block."""
    lg = get_resolution_level(root, level)
    attrs = lg.attrs.to_dict()
    block = dict(attrs.get("zarr_vectors_level", {}))
    block["bin_shape"] = [float(b) for b in bin_shape]
    block["bin_ratio"] = [
        float(b) / float(bb) if bb else 1.0 for b, bb in zip(bin_shape, base_bin)
    ]
    lg.attrs.update({"zarr_vectors_level": block})
