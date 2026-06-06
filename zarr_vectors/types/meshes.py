"""Mesh I/O for zarr vectors stores.

Supports two encoding modes:

- **Raw**: vertex positions in ``vertices/``, face indices in ``links/``
  (L=3 for triangles, L=4 for quads).  Faces spanning chunk boundaries
  go into ``cross_chunk_links/``.

- **Draco**: each fragment is encoded as a Draco bitstream containing
  both positions and faces.  ``links/`` may be omitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    CROSS_CHUNK_LINKS,
    ENCODING_DRACO,
    ENCODING_RAW,
    GEOM_MESH,
    LINK_FRAGMENTS,
    LINKS,
    LINKS_EXPLICIT,
    OBJIDX_STANDARD,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    resolve_chunk_keys,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_object_vertices,
    write_chunk_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.attr_chunking import (
    assign_attribute_bins,
    compute_chunk_dim_names,
)
from zarr_vectors.constants import DEFAULT_OOB_POLICY
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    get_level_chunk_shape,
)
from zarr_vectors.core.store import (
    _apply_out_of_bounds_policy,
    _create_or_open_store,
    _ensure_root_metadata_for_write,
    _finalize_write,
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
    stamp_ccl_capabilities,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    partition_faces,
)
from zarr_vectors.spatial.chunking import (
    assign_bins,
    assign_chunks,
    compute_bounds,
    group_bins_by_chunk,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    ObjectManifest,
)


def write_mesh(
    store_path: str,
    vertices: npt.NDArray[np.floating],
    faces: npt.NDArray[np.integer],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    encoding: str = ENCODING_RAW,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    dtype: str = "float32",
    draco_quantization_bits: int = 11,
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
    out_of_bounds: str = DEFAULT_OOB_POLICY,
    compressor: Any = None,
) -> dict[str, Any]:
    """Write a mesh to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        vertices: ``(V, D)`` vertex positions.
        faces: ``(F, L)`` face index array (L=3 triangles, L=4 quads).
        chunk_shape: Spatial chunk size per dimension.
        encoding: ``"raw"`` or ``"draco"``.
        vertex_attributes: Per-vertex attributes ``{name: (V,) or (V,C)}``.
        object_ids: ``(V,)`` array assigning vertices to mesh objects.
            If None, all vertices belong to object 0.
        dtype: Numpy dtype for positions.
        draco_quantization_bits: For Draco encoding only.

    Returns:
        Summary dict.
    """
    np_dtype = np.dtype(dtype)
    vertices = np.asarray(vertices, dtype=np_dtype)
    faces = np.asarray(faces, dtype=np.int64)
    n_verts, ndim = vertices.shape
    n_faces, link_width = faces.shape

    if faces.ndim != 2 or link_width < 3:
        raise ArrayError(f"faces must be (F, L) with L≥3, got shape {faces.shape}")

    if object_ids is None:
        # When attribute-chunking, each vertex becomes its own object so
        # the per-object uniformity check is trivially satisfied.
        if chunk_by_attribute is not None:
            object_ids = np.arange(n_verts, dtype=np.int64)
        else:
            object_ids = np.zeros(n_verts, dtype=np.int64)

    if bounds is None:
        inferred = compute_bounds(vertices)
        bounds_list = (inferred[0].tolist(), inferred[1].tolist())
    else:
        bounds_list = (list(bounds[0]), list(bounds[1]))

    root = _create_or_open_store(
        store_path,
        backend=backend,
        bounds=bounds_list,
        chunk_shape=tuple(chunk_shape),
        ndim=ndim,
    )
    # OOB policy for mesh vertices.  "ignore" filters vertices but does
    # NOT rewrite ``faces`` to drop references to filtered vertices —
    # call set_bounds(..., force=True) afterwards for a fully consistent
    # contracted store, or pre-filter vertices+faces upstream.
    if out_of_bounds == "ignore":
        raise ArrayError(
            "out_of_bounds='ignore' is not supported for write_mesh: "
            "faces reference vertex indices and would be left dangling. "
            "Use 'raise' (default) or 'expand', or pre-filter vertices "
            "and remap faces upstream."
        )
    vertices, _ = _apply_out_of_bounds_policy(
        root, vertices, policy=out_of_bounds,
    )

    root_meta = _ensure_root_metadata_for_write(
        root,
        inferred_ndim=ndim,
        geometry_type=GEOM_MESH,
        base_bin_shape=bin_shape,
        links_convention=LINKS_EXPLICIT,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
    )
    axes = root_meta.spatial_index_dims

    # Attribute chunking: per-object uniformity required.
    vertex_attr_bins: npt.NDArray[np.int64] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        if (
            not vertex_attributes
            or chunk_by_attribute not in vertex_attributes
        ):
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a "
                f"key in `vertex_attributes`"
            )
        src_values = np.asarray(vertex_attributes[chunk_by_attribute])
        if src_values.ndim != 1 or src_values.shape[0] != n_verts:
            raise ArrayError(
                f"vertex_attributes[{chunk_by_attribute!r}] must be 1D "
                f"of length n_verts={n_verts}, got shape {src_values.shape}"
            )
        vertex_attr_bins, attr_bin_values = assign_attribute_bins(src_values)
        for oid in np.unique(object_ids):
            mask = object_ids == oid
            unique_bins = np.unique(vertex_attr_bins[mask])
            if len(unique_bins) > 1:
                raise ArrayError(
                    f"chunk_by_attribute={chunk_by_attribute!r} requires "
                    f"per-object uniformity for meshes; object {int(oid)} "
                    f"has {len(unique_bins)} distinct attribute values"
                )

    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute, ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )

    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_verts,
        arrays_present=[VERTICES, "links", "object_index"],
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)

    # Assign vertices to chunks.  When attribute-chunked, prefix each
    # spatial chunk key with the per-vertex bin.
    chunk_assignments = assign_chunks(vertices, chunk_shape)
    if vertex_attr_bins is not None:
        prefixed: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
        for spatial_cc, gi in chunk_assignments.items():
            gi = np.asarray(gi, dtype=np.int64)
            chunk_bins = vertex_attr_bins[gi]
            for ab in np.unique(chunk_bins):
                mask = chunk_bins == ab
                prefixed[(int(ab),) + spatial_cc] = gi[mask]
        chunk_assignments = prefixed
    chunk_list = sorted(chunk_assignments.keys())

    vertex_chunks, vertex_local, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_verts, chunk_list
    )

    # Partition faces
    intra_faces, cross_faces = partition_faces(
        faces, vertex_chunks, vertex_local, chunk_list
    )

    # Write vertices per chunk (one fragment per chunk for simplicity)
    object_manifests: dict[int, ObjectManifest] = {}
    idx_ndim = ndim + 1 if vertex_attr_bins is not None else ndim

    # Collapse all per-array zarr.json + per-chunk byte writes into one
    # asyncio.gather (mirrors points.py:300).
    with level_group.batched_writes(compressor=compressor):
        create_vertices_array(level_group, dtype=dtype, encoding=encoding)
        create_links_array(level_group, link_width=link_width, delta=0)
        create_object_index_array(level_group)
        create_cross_chunk_links_array(level_group, delta=0)
        if vertex_attributes:
            for name, data in vertex_attributes.items():
                create_attribute_array(level_group, name, dtype=str(data.dtype))
        if object_attributes:
            for _name in object_attributes:
                create_object_attributes_array(level_group, _name)

        for chunk_idx, chunk_coords in enumerate(chunk_list):
            global_indices = chunk_assignments[chunk_coords]
            chunk_verts = vertices[global_indices]

            if encoding == ENCODING_DRACO and ndim == 3:
                # Draco mode: encode positions + local faces together
                local_faces_arr = intra_faces.get(chunk_coords)
                _write_draco_chunk(
                    level_group, chunk_coords, chunk_verts,
                    local_faces_arr, draco_quantization_bits, np_dtype,
                )
            else:
                write_chunk_vertices(
                    level_group, chunk_coords, [chunk_verts], dtype=np_dtype,
                )

            # Track manifests: one fragment per chunk per unique object in chunk
            chunk_obj_ids = object_ids[global_indices]
            for oid in np.unique(chunk_obj_ids):
                oid_int = int(oid)
                if oid_int not in object_manifests:
                    object_manifests[oid_int] = []
                object_manifests[oid_int].append((chunk_coords, 0))

            # Write vertex attributes
            if vertex_attributes:
                for name, data in vertex_attributes.items():
                    chunk_attrs = data[global_indices]
                    write_chunk_attributes(
                        level_group, name, chunk_coords, [chunk_attrs],
                        dtype=data.dtype,
                    )

        # Write intra-chunk faces (raw mode)
        if encoding != ENCODING_DRACO:
            for chunk_coords in chunk_list:
                if chunk_coords in intra_faces:
                    write_chunk_links(
                        level_group, chunk_coords, [intra_faces[chunk_coords]], delta=0,
                    )

        # Write cross-chunk faces as variable-width records under
        # ``cross_chunk_links/<delta=0>/``.  Each record is a list of L
        # ``(chunk_coords, local_vertex_idx)`` endpoints where L is the
        # face arity (3 for triangles).  Faces of different arity are
        # rejected; meshes are uniform-arity by construction.
        if cross_faces:
            face_arities = {len(f) for f in cross_faces}
            if len(face_arities) != 1:
                raise ArrayError(
                    f"cross-chunk faces have inconsistent arities {face_arities}; "
                    "meshes must be uniform-arity"
                )
            write_cross_chunk_links(
                level_group,
                [list(face) for face in cross_faces],
                sid_ndim=idx_ndim,
                delta=0,
                link_width=face_arities.pop(),
            )
            stamp_ccl_capabilities(root)

        # Write object index
        write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

        # Per-object attributes (Tier B): {name: (O,) or (O, C)}.
        if object_attributes:
            for _name, _data in object_attributes.items():
                write_object_attributes(level_group, _name, np.asarray(_data))

    _finalize_write(root, "write_mesh")
    return {
        "vertex_count": n_verts,
        "face_count": n_faces,
        "chunk_count": len(chunk_list),
        "intra_face_count": sum(len(f) for f in intra_faces.values()),
        "cross_face_count": len(cross_faces),
        "encoding": encoding,
    }


def read_mesh(
    store_path: str,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read a mesh from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.
        chunks: Optional whitelist of chunk coordinate tuples; only data
            in those chunks is returned. AND-ed with ``bbox`` and
            ``object_ids``. ``chunks=[]`` yields an empty result;
            ``chunks=None`` (default) applies no chunk filter.

    Returns:
        Dict with:
        - ``vertices``: ``(V, D)`` positions
        - ``faces``: ``(F, L)`` face indices (remapped to output vertex order)
        - ``vertex_count``, ``face_count``
    """
    root = open_store(store_path, backend=backend)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim

    # Per-level chunk_shape may override root (v0.7+).
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:
        level_meta = None
    level_chunk_shape = get_level_chunk_shape(root_meta, level_meta)

    dtype = np.float32
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    link_width = 3
    try:
        lmeta = level_group.read_array_meta("links/0")
        link_width = lmeta.get("link_width", 3)
    except Exception:
        pass

    # Determine chunks to read (intersection of physical keys, bbox-implied
    # set, and explicit chunks whitelist).
    chunk_keys = resolve_chunk_keys(
        level_group, level_chunk_shape, bbox=bbox, chunks=chunks,
    )

    if attribute_filter:
        try:
            lm = read_level_metadata(root, level)
        except Exception:
            lm = None
        if (
            lm is None
            or lm.chunk_attribute_name is None
            or lm.chunk_attribute_values is None
        ):
            raise ArrayError(
                "attribute_filter requires a store written with chunk_by_attribute"
            )
        if len(attribute_filter) != 1:
            raise ArrayError(
                "attribute_filter must specify exactly one attribute"
            )
        fname, fvalue = next(iter(attribute_filter.items()))
        if fname != lm.chunk_attribute_name:
            raise ArrayError(
                f"attribute_filter key {fname!r} does not match the "
                f"store's chunk_attribute_name {lm.chunk_attribute_name!r}"
            )
        try:
            filter_bin = lm.chunk_attribute_values.index(fvalue)
        except ValueError:
            return _empty_mesh_result(ndim, link_width)
        chunk_keys = [k for k in chunk_keys if k and k[0] == filter_bin]

    # Prefetch every chunk (vertices, offsets, faces) and the cross-chunk
    # face records in one async gather.  Subsequent ``read_bytes`` calls
    # below hit the cache instead of paying one round-trip per chunk.
    chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_keys]
    _prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        (f"{LINKS}/0", chunk_key_strs),
        (LINK_FRAGMENTS, chunk_key_strs),
        (f"{CROSS_CHUNK_LINKS}/0", ["data"]),
    ]
    _batched_reads_cm = level_group.batched_reads(_prefetch_plan)
    _batched_reads_cm.__enter__()
    try:
        # Read vertices and build offset map
        all_positions: list[npt.NDArray] = []
        chunk_offsets: dict[ChunkCoords, int] = {}
        running = 0

        for chunk_coords in chunk_keys:
            chunk_offsets[chunk_coords] = running
            try:
                groups = read_chunk_vertices(
                    level_group, chunk_coords, dtype=dtype, ndim=ndim
                )
                for fragment in groups:
                    all_positions.append(fragment)
                    running += len(fragment)
            except ArrayError:
                pass

        if not all_positions:
            return _empty_mesh_result(ndim, link_width)

        positions_out = np.concatenate(all_positions, axis=0)

        # Read intra-chunk faces
        all_faces: list[npt.NDArray] = []
        for chunk_coords in chunk_keys:
            try:
                link_groups = read_chunk_links(
                    level_group, chunk_coords, link_width=link_width, delta=0,
                )
                offset = chunk_offsets.get(chunk_coords, 0)
                for lg in link_groups:
                    if len(lg) > 0:
                        remapped = lg.copy() + offset
                        all_faces.append(remapped)
            except ArrayError:
                pass

        # Cross-chunk faces are stored as variable-width records under
        # ``cross_chunk_links/<delta=0>/`` (link_width = face arity).  Map
        # each (chunk, local_idx) endpoint into the global vertex index
        # via ``chunk_offsets`` built above.
        cross_face_records = read_cross_chunk_links(level_group, delta=0)
        for face in cross_face_records:
            if len(face) != link_width:
                continue  # not a face record (e.g. edge-arity, ignore)
            vertex_ids: list[int] = []
            for cc, local_idx in face:
                if cc not in chunk_offsets:
                    vertex_ids = []
                    break
                vertex_ids.append(int(chunk_offsets[cc]) + int(local_idx))
            if len(vertex_ids) == link_width:
                all_faces.append(np.asarray(vertex_ids, dtype=np.int64)[None, :])

        if all_faces:
            faces_out = np.concatenate(all_faces, axis=0)
        else:
            faces_out = np.zeros((0, link_width), dtype=np.int64)

        # Apply bbox filter on vertices
        if bbox is not None:
            bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
            node_mask = np.all(
                (positions_out >= bbox_min) & (positions_out <= bbox_max),
                axis=1,
            )
            if not np.all(node_mask):
                keep = np.flatnonzero(node_mask)
                keep_set = set(keep.tolist())
                positions_out = positions_out[keep]

                old_to_new = {int(old): new for new, old in enumerate(keep)}
                filtered: list[npt.NDArray] = []
                for f in faces_out:
                    if all(int(v) in keep_set for v in f):
                        filtered.append(np.array([old_to_new[int(v)] for v in f]))
                faces_out = (
                    np.stack(filtered).astype(np.int64)
                    if filtered
                    else np.zeros((0, link_width), dtype=np.int64)
                )

        return {
            "vertices": positions_out,
            "faces": faces_out,
            "vertex_count": len(positions_out),
            "face_count": len(faces_out),
        }
    finally:
        _batched_reads_cm.__exit__(None, None, None)


def _empty_mesh_result(ndim: int, link_width: int) -> dict[str, Any]:
    return {
        "vertices": np.zeros((0, ndim), dtype=np.float32),
        "faces": np.zeros((0, link_width), dtype=np.int64),
        "vertex_count": 0,
        "face_count": 0,
    }


def _write_draco_chunk(
    level_group: Any,
    chunk_coords: ChunkCoords,
    positions: npt.NDArray,
    faces: npt.NDArray | None,
    qbits: int,
    np_dtype: np.dtype,
) -> None:
    """Encode a chunk as Draco and write as raw bytes."""
    from zarr_vectors.encoding.draco import draco_encode_mesh

    if faces is None or len(faces) == 0:
        # Point cloud mode
        from zarr_vectors.encoding.draco import draco_encode_point_cloud
        blob = draco_encode_point_cloud(positions, quantization_bits=qbits)
    else:
        blob = draco_encode_mesh(positions, faces, quantization_bits=qbits)

    # Store as raw bytes in the vertices chunk
    from zarr_vectors.constants import VERTEX_FRAGMENTS, VERTICES
    from zarr_vectors.core.arrays import _chunk_key
    from zarr_vectors.encoding.fragments import encode_fragments

    key = _chunk_key(chunk_coords)
    level_group.write_bytes(VERTICES, key, blob)

    # Single fragment spanning the whole chunk.  Draco-encoded vertex
    # blobs aren't row-addressable, so we describe the (start, count)
    # in vertex *rows* as (0, num_positions) — the read path knows to
    # treat Draco chunks as a single span.
    num_positions = int(np.asarray(positions).shape[0])
    level_group.write_bytes(
        VERTEX_FRAGMENTS, key,
        encode_fragments([(0, num_positions)]),
    )
