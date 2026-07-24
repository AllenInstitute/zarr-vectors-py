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

from zarr_vectors.constants import GEOM_POINT_CLOUD, VERTICES
from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_chunk_vertices,
    write_chunk_vertices,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    compute_bin_ratio,
    compute_bin_shape,
    validate_bin_shape_divides_chunk,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import StoreError
from zarr_vectors.spatial.chunking import assign_bins
from zarr_vectors.typing import BinShape, ChunkCoords


def rebin_level(
    store_path: str | Path,
    target_bin_shape: BinShape,
    *,
    level: int = 0,
) -> dict[str, Any]:
    """Re-sort vertices into new fragment bins at the given level.

    Reads each chunk, re-bins its vertices into ``target_bin_shape``
    supervoxel cells, and rewrites the chunk with one fragment per occupied
    bin.  Per-vertex attribute arrays are **not** touched — they remain
    aligned with the on-disk vertex order.  Because re-binning shuffles
    vertex order within a chunk, callers that have post-write per-vertex
    attribute arrays should regenerate them.

    Updates root ``base_bin_shape`` (when ``level == 0``) and the level
    metadata's ``bin_shape`` / ``bin_ratio``.

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

    for cc in chunk_keys:
        # Flatten any existing fragment structure: a point cloud's fragments are
        # re-derived from positions, so we don't care about the old
        # grouping.
        try:
            groups = read_chunk_vertices(level_group, cc, ndim=ndim)
        except Exception:
            continue
        if not groups:
            continue
        positions = np.concatenate(groups, axis=0)
        if positions.shape[0] == 0:
            continue

        # Re-bin into one fragment per occupied bin, sorted by bin key for a
        # deterministic on-disk ordering.
        bin_assignments = assign_bins(positions, tuple(float(b) for b in bs))
        new_groups: list[npt.NDArray] = []
        for bin_key in sorted(bin_assignments.keys()):
            idxs = bin_assignments[bin_key]
            new_groups.append(positions[idxs].astype(np.float32, copy=False))

        write_chunk_vertices(level_group, cc, new_groups, dtype=np.float32)
        chunks_processed += 1
        vertices_processed += int(positions.shape[0])
        bins_created += len(new_groups)

    # Update level metadata.
    new_bin_shape = tuple(float(b) for b in target_bin_shape)
    new_bin_ratio: tuple[int, ...] | None = None
    base_bin = root_meta.effective_bin_shape
    if base_bin is not None:
        try:
            new_bin_ratio = compute_bin_ratio(base_bin, new_bin_shape)
        except Exception:
            new_bin_ratio = None
    new_level_meta = LevelMetadata(
        level=level_meta.level,
        vertex_count=level_meta.vertex_count,
        arrays_present=level_meta.arrays_present,
        bin_shape=new_bin_shape,
        bin_ratio=new_bin_ratio,
        object_sparsity=level_meta.object_sparsity,
        coarsening_method=level_meta.coarsening_method,
        parent_level=level_meta.parent_level,
        chunk_dims=level_meta.chunk_dims,
        chunk_attribute_name=level_meta.chunk_attribute_name,
        chunk_attribute_values=level_meta.chunk_attribute_values,
        preserves_object_ids=level_meta.preserves_object_ids,
        inherited_num_objects=level_meta.inherited_num_objects,
        shared_fragments=level_meta.shared_fragments,
    )
    level_group.attrs.update(new_level_meta.to_dict())

    # Update root base_bin_shape when re-binning level 0 (so future
    # writes / reads see the new default).
    if level == 0:
        new_root = RootMetadata(
            spatial_index_dims=root_meta.spatial_index_dims,
            chunk_shape=root_meta.chunk_shape,
            bounds=root_meta.bounds,
            geometry_types=root_meta.geometry_types,
            zv_version=root_meta.zv_version,
            crs=root_meta.crs,
            links_convention=root_meta.links_convention,
            object_index_convention=root_meta.object_index_convention,
            cross_chunk_strategy=root_meta.cross_chunk_strategy,
            reduction_factor=root_meta.reduction_factor,
            base_bin_shape=new_bin_shape,
            cross_level_depth=root_meta.cross_level_depth,
            cross_level_storage=root_meta.cross_level_storage,
            format_capabilities=list(root_meta.format_capabilities),
        )
        root.attrs.update(new_root.to_dict())

    return {
        "chunks_processed": chunks_processed,
        "vertices_processed": vertices_processed,
        "bins_created": bins_created,
        "old_bin_shape": tuple(old_bin_shape) if old_bin_shape else None,
        "new_bin_shape": new_bin_shape,
        "level": level,
    }
