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
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    read_object_vertices,
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
from zarr_vectors.spatial.chunking import assign_chunks
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
        obj_to_bin = mapper.map_objects(
            n_objects=n_objects,
            groupings=groupings,
            object_attributes=object_attributes,
        )
    else:
        # No objects — rechunk spatially only
        obj_to_bin = {}

    # Determine unique bins
    if obj_to_bin:
        unique_bins = sorted(set(obj_to_bin.values()))
    else:
        unique_bins = [0]

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
        links_convention=src_meta.links_convention,
        object_index_convention=src_meta.object_index_convention,
        cross_chunk_strategy=src_meta.cross_chunk_strategy,
        base_bin_shape=src_meta.base_bin_shape,
    )

    # Compute the bin → original-value list for attribute-based rechunking.
    # Only meaningful when ``by="attribute:..."`` and we have the source
    # values; non-attribute rechunks leave chunk_attribute_* unset.
    chunk_attribute_name: str | None = None
    chunk_attribute_values: list[Any] | None = None
    if spec.by.startswith("attribute:") and object_attributes is not None:
        attr_name = spec.by.split(":", 1)[1]
        src_vals = object_attributes.get(attr_name)
        if src_vals is not None and obj_to_bin:
            # Build {bin_idx: value} from (object_id → bin_idx) and the
            # source value array; an object's value picks the bin.
            bin_to_value: dict[int, Any] = {}
            for oid, b in obj_to_bin.items():
                if b not in bin_to_value:
                    v = src_vals[oid]
                    if hasattr(v, "item"):
                        v = v.item()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                    bin_to_value[b] = v
            chunk_attribute_name = attr_name
            chunk_attribute_values = [
                bin_to_value[b] for b in sorted(bin_to_value)
            ]

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
    _rechunk_session = out_level.native_sharded_arrays(
        None,
        (max(unique_bins) + 1, *_spatial_grid),
        origin=(0, *_spatial_origin),
    )
    _rechunk_session.__enter__()
    create_vertices_array(out_level, dtype="float32")
    create_object_index_array(out_level)

    # Rechunk: for each bin, gather all objects, assign to spatial chunks
    # with prefixed keys
    total_vertices = 0
    total_objects = 0
    object_manifests_out: dict[int, ObjectManifest] = {}
    global_obj_counter = 0

    for bin_idx in unique_bins:
        # Collect objects in this bin
        if obj_to_bin:
            bin_objects = sorted(
                oid for oid, b in obj_to_bin.items() if b == bin_idx
            )
        else:
            bin_objects = []

        # Read vertex data for these objects
        bin_positions: list[npt.NDArray] = []
        bin_obj_boundaries: list[int] = []  # cumulative vertex counts per object

        for oid in bin_objects:
            try:
                verts_list = read_object_vertices(
                    src_level, oid, dtype=np.float32, ndim=ndim,
                )
                obj_verts = np.concatenate(
                    [v for v in verts_list if len(v) > 0], axis=0,
                ) if any(len(v) > 0 for v in verts_list) else np.zeros((0, ndim), dtype=np.float32)
            except Exception:
                obj_verts = np.zeros((0, ndim), dtype=np.float32)

            bin_positions.append(obj_verts)
            bin_obj_boundaries.append(len(obj_verts))

        if not bin_positions or all(len(p) == 0 for p in bin_positions):
            continue

        # Assign to spatial chunks within this bin prefix
        all_pos = np.concatenate(bin_positions, axis=0)
        spatial_assignments = assign_chunks(all_pos, chunk_shape)

        # Build object-to-vertex mapping for this bin
        obj_starts = np.cumsum([0] + bin_obj_boundaries[:-1])

        for spatial_cc, global_indices in sorted(spatial_assignments.items()):
            # Prefixed chunk key: (bin_idx, z, y, x)
            prefixed_cc: ChunkCoords = (bin_idx,) + spatial_cc
            chunk_verts = all_pos[global_indices]

            write_chunk_vertices(
                out_level, prefixed_cc, [chunk_verts], dtype=np.float32,
            )
            total_vertices += len(chunk_verts)

        # Build object manifests for this bin
        for local_idx, oid in enumerate(bin_objects):
            start = int(obj_starts[local_idx])
            n_verts = bin_obj_boundaries[local_idx]
            if n_verts == 0:
                continue

            obj_positions = bin_positions[local_idx]
            obj_spatial = assign_chunks(obj_positions, chunk_shape)

            manifest: ObjectManifest = []
            for scc, _ in sorted(obj_spatial.items()):
                prefixed = (bin_idx,) + scc
                fragment_idx = 0  # single fragment per chunk in rechunked stores
                manifest.append((prefixed, fragment_idx))

            object_manifests_out[global_obj_counter] = manifest
            global_obj_counter += 1
            total_objects += 1

    # Write object index (with extended ndim for prefix dimension)
    if object_manifests_out:
        # The manifests use (prefix, z, y, x) coords — ndim+1 dimensions
        extended_ndim = ndim + 1 if obj_to_bin else ndim
        write_object_index(out_level, object_manifests_out, sid_ndim=extended_ndim)

    # Close the single-array write context; the per-chunk writes are done.
    _rechunk_session.__exit__(None, None, None)

    # Write groupings if rechunked by group (preserve group structure)
    if spec.by == "group" and groupings is not None:
        try:
            from zarr_vectors.core.arrays import (
                create_groupings_array,
                write_groupings,
            )
            # Remap group memberships to new object IDs
            old_to_new: dict[int, int] = {}
            new_idx = 0
            for bin_idx in unique_bins:
                bin_objects = sorted(
                    oid for oid, b in obj_to_bin.items() if b == bin_idx
                )
                for old_oid in bin_objects:
                    old_to_new[old_oid] = new_idx
                    new_idx += 1

            new_groupings: dict[int, list[int]] = {}
            for gid, members in enumerate(groupings):
                new_members = [
                    old_to_new[m] for m in members if m in old_to_new
                ]
                if new_members:
                    new_groupings[gid] = new_members

            if new_groupings:
                create_groupings_array(out_level)
                write_groupings(out_level, new_groupings)
        except Exception:
            pass

    # In-place: replace source with output
    if in_place:
        backup = store_path.parent / (store_path.name + ".backup")
        store_path.rename(backup)
        output_path.rename(store_path)
        shutil.rmtree(backup)
        output_path = store_path

    return {
        "objects_rechunked": total_objects,
        "bins_created": len(unique_bins),
        "total_vertices": total_vertices,
        "rechunk_dims": rechunk_dims,
        "output_path": str(output_path),
    }


def _compute_object_lengths(
    level_group: FsGroup,
    n_objects: int,
    ndim: int,
) -> npt.NDArray[np.float64]:
    """Compute path length for each object (for polyline-like data)."""
    lengths = np.zeros(n_objects, dtype=np.float64)
    for oid in range(n_objects):
        try:
            verts_list = read_object_vertices(
                level_group, oid, dtype=np.float32, ndim=ndim,
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
