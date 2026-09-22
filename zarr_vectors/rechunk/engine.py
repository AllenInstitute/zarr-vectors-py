"""Rechunk engine — reads a source store and writes a rechunked copy.

The rechunked store has an extra prefix dimension on its chunk keys:
``(prefix_bin, z, y, x)`` instead of ``(z, y, x)``.  All objects in
the same prefix bin are physically contiguous, enabling O(1) group
or attribute-based filtering.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import VERTICES
from zarr_vectors.core.arrays import (
    create_object_index_array,
    create_vertices_array,
    read_all_object_manifests,
    read_object_vertices,
    vertices_dtype,
    write_chunk_vertices,
    write_object_index,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    FsGroup,
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.rechunk.spec import DimensionMapper, RechunkSpec
from zarr_vectors.typing import ChunkCoords, ObjectManifest


def rechunk(
    store_path: str | Path,
    spec: RechunkSpec,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Rechunk a store along a non-spatial dimension.

    Reads object data from the source store, assigns each object to
    a rechunk bin via ``DimensionMapper``, and writes the result to
    an output store where chunk keys have a prefix dimension
    ``(bin, z, y, x)``.

    Args:
        store_path: Source store path.
        spec: Rechunk specification.
        output: Output store path. If None, rechunks in-place by
            writing to a temporary store then replacing the source.

    Returns:
        Summary dict with ``objects_rechunked``, ``bins_created``,
        ``output_path``.
    """
    store_path = Path(store_path)

    # Determine output path
    in_place = output is None
    if in_place:
        output_path = store_path.parent / (store_path.name + ".rechunked")
    else:
        output_path = Path(output)

    if output_path.exists():
        shutil.rmtree(output_path)

    # Read source
    src_root = open_store(str(store_path))
    src_meta = read_root_metadata(src_root)
    ndim = src_meta.sid_ndim

    chunk_shape = spec.spatial_chunk_shape or src_meta.chunk_shape

    # Read level 0 data
    src_level = get_resolution_level(src_root, 0)

    # Read object manifests
    try:
        manifests = read_all_object_manifests(src_level)
        n_objects = len(manifests)
    except Exception:
        manifests = []
        n_objects = 0

    # Read groupings
    groupings: list[list[int]] | None = None
    try:
        from zarr_vectors.core.arrays import read_all_groupings
        groupings = read_all_groupings(src_level)
    except Exception:
        groupings = None

    # Read object attributes (for attribute-based rechunking)
    object_attributes: dict[str, npt.NDArray] | None = None
    if spec.by.startswith("attribute:"):
        # Try to read the attribute as per-object data
        attr_name = spec.by.split(":", 1)[1]
        object_attributes = {}
        try:
            from zarr_vectors.core.arrays import read_object_attributes
            obj_attr_data = read_object_attributes(src_level, attr_name)
            object_attributes[attr_name] = obj_attr_data
        except Exception:
            # Attribute might need to be computed (e.g. length for polylines)
            if attr_name == "length" and n_objects > 0:
                lengths = _compute_object_lengths(src_level, n_objects, ndim)
                object_attributes[attr_name] = lengths
            else:
                raise ValueError(
                    f"Cannot read or compute attribute '{attr_name}'"
                )

    # Map objects to rechunk bins
    mapper = DimensionMapper(spec)
    if n_objects > 0:
        raw_bins = mapper.map_objects(
            n_objects=n_objects,
            groupings=groupings,
            object_attributes=object_attributes,
        )
    else:
        # No objects — rechunk spatially only
        raw_bins = {}

    # Renumber the bins densely before anything consumes them.  The
    # mapper's indices can have holes -- edges no object falls between,
    # an empty group -- and a reader resolves a value to a bin by its
    # position in chunk_attribute_values.  With holes, the key prefixes
    # and the grid kept the raw index while that list was compacted, so a
    # query came back empty, or with another bin's data under its label.
    # Ungrouped objects (-1) go last, so group g stays ahead of them.
    used = sorted(set(raw_bins.values()) - {-1})
    if -1 in raw_bins.values():
        used.append(-1)
    renumber = {old: new for new, old in enumerate(used)}
    obj_to_bin = {oid: renumber[b] for oid, b in raw_bins.items()}
    unique_bins = list(range(len(used))) or [0]
    by_bin: dict[int, list[int]] = {b: [] for b in unique_bins}
    for oid in sorted(obj_to_bin):
        by_bin[obj_to_bin[oid]].append(oid)

    # Create output store
    spatial_dim_names = [
        a.get("name", f"dim{i}")
        for i, a in enumerate(src_meta.spatial_index_dims)
    ]
    rechunk_dims = [spec.dimension_name, *spatial_dim_names]

    out_root = create_store(
        str(output_path),
        axes=src_meta.spatial_index_dims,
        chunk_shape=chunk_shape,
        bounds=src_meta.bounds,
        geometry_types=src_meta.geometry_types,
        # Packing is a property of the store, so a rechunk of a sharded
        # store produces a sharded store. Dropping it here would have
        # quietly unsharded every rechunk output.
        shard_shape=src_meta.shard_shape,
        links_convention=src_meta.links_convention,
        object_index_convention=src_meta.object_index_convention,
        cross_chunk_strategy=src_meta.cross_chunk_strategy,
        base_bin_shape=src_meta.base_bin_shape,
    )

    # One label per bin, in bin order: what a reader's attribute_filter
    # names to select it.  Recorded for group and object-id rechunks too,
    # under the dimension's name: renumbering is otherwise the one step
    # that loses which bin holds which group.
    chunk_attribute_name: str | None = None
    chunk_attribute_values: list[Any] | None = None
    if obj_to_bin and spec.by != "spatial":
        chunk_attribute_name = (
            spec.by.split(":", 1)[1] if spec.by.startswith("attribute:")
            else spec.dimension_name
        )
        chunk_attribute_values = [mapper.labels[old] for old in used]

    # Create level 0
    level_meta = LevelMetadata(
        level=0,
        vertex_count=0,  # updated below
        arrays_present=[VERTICES, "object_index"],
        chunk_dims=rechunk_dims,
        chunk_attribute_name=chunk_attribute_name,
        chunk_attribute_values=chunk_attribute_values,
    )
    out_level = create_resolution_level(out_root, 0, level_meta)
    # Rechunk prefixes every chunk key with a leading bin index, so the
    # single vlen arrays need that extra axis on their grid.  Activate
    # the single-array layout with the right rank for the write span.
    from zarr_vectors.core.arrays import level_grid_layout
    _spatial_origin, _spatial_grid = level_grid_layout(
        src_meta.bounds, chunk_shape,
    )
    # The output grid carries a leading attribute-bin axis. A per-axis
    # declaration gets 1 prepended for it, so a shard never straddles
    # bins -- they are queried independently, and packing two into one
    # object drags each into the other's reads. A scalar broadcasts,
    # which is what a scalar means.
    from zarr_vectors.core.metadata import normalise_shard_shape

    _rechunk_grid = (len(unique_bins), *_spatial_grid)
    _rechunk_session = out_level.native_sharded_arrays(
        normalise_shard_shape(
            src_meta.shard_shape, len(_rechunk_grid), bin_axis=True,
        ),
        _rechunk_grid,
        origin=(0, *_spatial_origin),
    )

    # Read at the dtype the source declares: a float64 cell read as
    # float32 decodes to garbage at twice the row count.
    vdtype = vertices_dtype(src_level)
    cs = np.asarray(chunk_shape, dtype=np.float64)

    total_vertices = 0
    object_manifests_out: dict[int, ObjectManifest] = {}
    # Source id -> output id, for objects that were written.  An object
    # with no vertices gets no manifest, so it gets no id either.
    old_to_new: dict[int, int] = {}

    with _rechunk_session:
        create_vertices_array(out_level, dtype=vdtype.name)
        create_object_index_array(out_level)

        for bin_idx in unique_bins:
            # Each object keeps its own fragments: one per run of its
            # vertices inside one cell, as the type writers lay them out.
            # Merging a cell into one fragment made every manifest name
            # fragment 0, so reading one object returned the whole cell.
            cells: dict[ChunkCoords, list[npt.NDArray]] = {}
            for oid in by_bin[bin_idx]:
                try:
                    fragments = read_object_vertices(
                        src_level, oid, dtype=vdtype, ndim=ndim,
                    )
                except Exception:
                    fragments = []
                manifest: ObjectManifest = []
                for fragment in fragments:
                    for cell, run in _runs_by_cell(fragment, cs):
                        key: ChunkCoords = (bin_idx, *cell)
                        runs = cells.setdefault(key, [])
                        manifest.append((key, len(runs)))
                        runs.append(run)
                if not manifest:
                    continue
                old_to_new[oid] = len(object_manifests_out)
                object_manifests_out[old_to_new[oid]] = manifest

            for key in sorted(cells):
                write_chunk_vertices(out_level, key, cells[key], dtype=vdtype)
                total_vertices += sum(len(r) for r in cells[key])

        # The manifests use (prefix, z, y, x) coords — ndim+1 dimensions
        if object_manifests_out:
            write_object_index(
                out_level, object_manifests_out, sid_ndim=ndim + 1,
            )

    # Write groupings if rechunked by group (preserve group structure)
    if spec.by == "group" and groupings is not None:
        from zarr_vectors.core.arrays import (
            create_groupings_array,
            write_groupings,
        )

        new_groupings: dict[int, list[int]] = {}
        for gid, members in enumerate(groupings):
            new_members = [old_to_new[m] for m in members if m in old_to_new]
            if new_members:
                new_groupings[gid] = new_members

        if new_groupings:
            create_groupings_array(out_level)
            write_groupings(out_level, new_groupings)

    # In-place: replace source with output
    if in_place:
        backup = store_path.parent / (store_path.name + ".backup")
        store_path.rename(backup)
        output_path.rename(store_path)
        shutil.rmtree(backup)
        output_path = store_path

    return {
        "objects_rechunked": len(object_manifests_out),
        "bins_created": len(unique_bins),
        "total_vertices": total_vertices,
        "rechunk_dims": rechunk_dims,
        "output_path": str(output_path),
    }


def _runs_by_cell(
    positions: npt.NDArray, cs: npt.NDArray[np.float64],
) -> list[tuple[tuple[int, ...], npt.NDArray]]:
    """Split ``positions`` into maximal runs of vertices sharing a cell.

    A fragment is one contiguous run of an object's vertices inside one
    cell; a polyline that leaves a cell and comes back is two fragments
    there, not one with a segment invented across the gap.  Cells are
    ``floor(position / chunk_shape)``, as
    :func:`~zarr_vectors.spatial.chunking.assign_chunks` computes them.
    """
    if len(positions) == 0:
        return []
    cells = np.floor(positions / cs).astype(np.int64)
    breaks = (np.flatnonzero(np.any(cells[1:] != cells[:-1], axis=1)) + 1).tolist()
    starts = [0, *breaks]
    ends = [*breaks, len(positions)]
    return [
        (tuple(cells[a].tolist()), positions[a:b])
        for a, b in zip(starts, ends)
    ]


def _compute_object_lengths(
    level_group: FsGroup,
    n_objects: int,
    ndim: int,
) -> npt.NDArray[np.float64]:
    """Compute path length for each object (for polyline-like data)."""
    lengths = np.zeros(n_objects, dtype=np.float64)
    vdtype = vertices_dtype(level_group)
    for oid in range(n_objects):
        try:
            verts_list = read_object_vertices(
                level_group, oid, dtype=vdtype, ndim=ndim,
            )
            all_verts = np.concatenate(
                [v for v in verts_list if len(v) > 0], axis=0,
            )
            if len(all_verts) >= 2:
                diffs = np.diff(all_verts, axis=0)
                lengths[oid] = float(
                    np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
                )
        except Exception:
            pass
    return lengths
