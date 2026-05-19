"""Point cloud I/O for ZV (Zarr Vectors) stores.

Supports three point cloud variants:

1. **Undifferentiated** — no per-point object identity.  All points in
   a chunk form a single fragment.  No object_index.
2. **Per-point objects** — each point is its own object (e.g. cell
   centroids).  One fragment per point per chunk.  Object index
   maps point ID → (chunk, fragment_index).
3. **Multi-point objects** — many points per object (e.g. transcript
   spots grouped into cells).  Points sharing an object_id are stored
   in one fragment per chunk.  Object index maps object → chunks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    FRAGMENT_ATTRIBUTES,
    GEOM_POINT_CLOUD,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_IDENTITY,
    OBJIDX_STANDARD,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_fragment_attribute_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    resolve_chunk_keys,
    read_all_groupings,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_vertices,
    read_group_object_ids,
    read_groupings_attributes,
    read_object_attributes,
    read_object_manifest,
    read_object_vertices,
    read_fragment,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_vertices,
    write_groupings,
    write_groupings_attributes,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.attr_chunking import (
    assign_attribute_bins,
    compute_chunk_dim_names,
)
from zarr_vectors.encoding.categorical import (
    DICTIONARY_ENCODING,
    decode_categorical,
    encode_categorical,
)
from zarr_vectors.constants import DEFAULT_OOB_POLICY
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    get_level_chunk_shape,
)
from zarr_vectors.core.store import (
    FsGroup,
    _apply_out_of_bounds_policy,
    _create_or_open_store,
    _ensure_root_metadata_for_write,
    _finalize_write,
    create_resolution_level,
    create_store,
    get_resolution_level,
    open_store,
    read_root_metadata,
    read_level_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.chunking import (
    assign_bins,
    assign_chunks,
    chunks_intersecting_bbox,
    compute_bounds,
    group_bins_by_chunk,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    ObjectManifest,
    FragmentRef,
)


def write_points(
    store_path: str,
    positions: npt.NDArray[np.floating],
    *,
    chunk_shape: ChunkShape | None = None,
    bin_shape: BinShape | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    fragment_attributes: dict[str, dict[ChunkCoords, npt.NDArray]] | None = None,
    groups: dict[int, list[int]] | None = None,
    group_attributes: dict[str, npt.NDArray] | None = None,
    dtype: str = "float32",
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
    out_of_bounds: str = DEFAULT_OOB_POLICY,
    compressor: Any = None,
    # Deprecated alias for ``vertex_attributes``; will be removed.
    attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Write a point cloud to a new ZV store.

    Args:
        store_path: Path for the new store (must not exist).
        positions: ``(N, D)`` vertex positions.
        chunk_shape: Spatial chunk size per dimension.  If None, a
            single chunk containing all points is used.
        bin_shape: Supervoxel bin edge lengths.  If None, defaults to
            ``chunk_shape`` (one bin per chunk — backward compatible).
        vertex_attributes: Per-vertex attributes ``{name: array}``.
            (Spec name; replaces the deprecated ``attributes`` kwarg.)
        object_ids: ``(N,)`` integer per-point object assignment.
        object_attributes: Per-object attributes ``{name: array}``.
        fragment_attributes: Per-fragment attributes, keyed per chunk:
            ``{name: {chunk_coords: ndarray}}``.  Each per-chunk ndarray
            has shape ``(num_fragments_in_chunk,)`` or
            ``(num_fragments_in_chunk, C)`` and is aligned with the
            fragment ordering this writer produces for that chunk.
            Use case (opt-in): materialize per-fragment parent-IDs
            (e.g. the OID owning each fragment as ``object_id``).
        groups: Group memberships ``{group_id: [object_id, ...]}``.
        group_attributes: Per-group attributes ``{name: array}``.
        dtype: Numpy dtype string for vertex positions.
        chunk_by_attribute: If set, the named per-vertex categorical
            attribute (which must appear in ``vertex_attributes``) becomes the
            **leading chunk axis**.  Chunk keys gain a prefix, e.g.
            ``gene_bin.z.y.x``.  Vertices with mixed values within the
            same object are split across multiple attribute chunks.
            v1 supports integer / string dtypes only.
        out_of_bounds: Policy for vertices outside the store's current
            ``bounds``.  ``"raise"`` (default) rejects the write,
            ``"ignore"`` drops out-of-bounds vertices (and their aligned
            per-vertex data), ``"expand"`` grows the store's bounds to
            include all input vertices.

    Returns:
        Summary dict.
    """
    # Back-compat: accept the legacy ``attributes`` kwarg.
    if attributes is not None:
        if vertex_attributes is not None:
            raise TypeError(
                "got both `attributes` and `vertex_attributes`; "
                "pass only `vertex_attributes` (the spec name)."
            )
        import warnings
        warnings.warn(
            "`attributes` is deprecated; use `vertex_attributes` "
            "(matches the on-disk `vertex_attributes/` directory).",
            DeprecationWarning, stacklevel=2,
        )
        vertex_attributes = attributes
    attributes = vertex_attributes  # internal alias for the rest of the body

    np_dtype = np.dtype(dtype)
    positions = np.asarray(positions, dtype=np_dtype)
    n_vertices, ndim = positions.shape

    if chunk_shape is None:
        bounds = compute_bounds(positions)
        extent = bounds[1] - bounds[0]
        max_extent = float(np.max(extent))
        side = max_extent + abs(float(np.max(bounds[1]))) + 1.0
        chunk_shape = tuple(side for _ in range(ndim))

    effective_bin = bin_shape if bin_shape is not None else chunk_shape
    bins_per_chunk = tuple(
        int(round(cs / bs)) for cs, bs in zip(chunk_shape, effective_bin)
    )
    total_bins_per_chunk = 1
    for b in bins_per_chunk:
        total_bins_per_chunk *= b

    # Attribute-chunking setup.  When enabled, every chunk key gains a
    # leading attr-bin dim, vertices are split per-vertex, and we need
    # object indexing for the prefixed read path.
    attr_bins: npt.NDArray[np.int64] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        if not attributes or chunk_by_attribute not in attributes:
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a key "
                f"in `attributes` (got: "
                f"{sorted(attributes) if attributes else []})"
            )
        attr_bins, attr_bin_values = assign_attribute_bins(
            attributes[chunk_by_attribute]
        )
        if attr_bins.shape[0] != n_vertices:
            raise ArrayError(
                f"chunk_by_attribute array length {attr_bins.shape[0]} "
                f"!= n_vertices {n_vertices}"
            )
        # The chunk-by attribute is recoverable from the leading axis;
        # don't also store it as a per-chunk attribute array.
        attributes = {
            k: v for k, v in attributes.items() if k != chunk_by_attribute
        }
        if object_ids is None:
            object_ids = np.arange(n_vertices, dtype=np.int64)

    needs_objects = (
        object_ids is not None
        or object_attributes is not None
        or groups is not None
    )
    if needs_objects and object_ids is None:
        object_ids = np.arange(n_vertices, dtype=np.int64)

    # Explicit `bounds=` kwarg overrides input-data extent.
    if bounds is None:
        inferred = compute_bounds(positions)
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

    # Apply out-of-bounds policy against the store's persisted bounds.
    # On a freshly-warmed store with default bounds (128³), points that
    # fall outside trigger this — caller picks raise / ignore / expand.
    positions, oob_mask = _apply_out_of_bounds_policy(
        root, positions, policy=out_of_bounds,
    )
    if not oob_mask.all() and out_of_bounds == "ignore":
        n_vertices = int(oob_mask.sum())
        if object_ids is not None:
            object_ids = np.asarray(object_ids)[oob_mask]
        if attributes:
            attributes = {
                k: np.asarray(v)[oob_mask] for k, v in attributes.items()
            }
        if attr_bins is not None:
            attr_bins = attr_bins[oob_mask]

    root_meta = _ensure_root_metadata_for_write(
        root,
        inferred_ndim=ndim,
        geometry_type=GEOM_POINT_CLOUD,
        base_bin_shape=bin_shape,
        links_convention=LINKS_IMPLICIT_SEQUENTIAL,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
    )
    axes = root_meta.spatial_index_dims

    bin_assignments = assign_bins(positions, effective_bin)
    chunked_bins = group_bins_by_chunk(bin_assignments, bins_per_chunk)
    n_chunks = len(chunked_bins)

    arrays_present = [VERTICES]
    if attributes:
        arrays_present.append(VERTEX_ATTRIBUTES)
    if fragment_attributes:
        arrays_present.append(FRAGMENT_ATTRIBUTES)
    if needs_objects:
        arrays_present.append("object_index")

    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute,
            ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )

    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_vertices,
        arrays_present=arrays_present,
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)

    object_manifests: dict[int, ObjectManifest] = {}

    # Dictionary-encode any string-typed attributes.  Non-numeric
    # dtypes (Unicode/bytes/object) are stored on disk as integer
    # ``codes`` arrays + a ``categories`` lookup table in the array's
    # metadata (see encoding/categorical.py).  Float NaN passes
    # through untouched as the native "missing" sentinel.
    attr_extra_meta: dict[str, dict[str, Any]] = {}
    if attributes:
        encoded: dict[str, npt.NDArray] = {}
        for attr_name, attr_data in attributes.items():
            arr = np.asarray(attr_data)
            if arr.dtype.kind in ("U", "S", "O"):
                if arr.ndim != 1:
                    raise ArrayError(
                        f"dictionary-encoded attribute {attr_name!r} must be "
                        f"1-D (got shape {arr.shape})"
                    )
                has_missing = (
                    arr.dtype.kind == "O" and any(v is None for v in arr)
                )
                codes, dict_meta = encode_categorical(
                    arr, fill_value=-1 if has_missing else None,
                )
                encoded[attr_name] = codes
                attr_extra_meta[attr_name] = dict_meta
            else:
                encoded[attr_name] = arr
        attributes = encoded

    # All per-array ``zarr.json`` PUTs *and* per-chunk byte writes
    # (vertices, offsets, attributes, the object_index sidecar) flush
    # in a single asyncio.gather against the underlying Store —
    # collapses ~N + ~K round-trips to ~1 against high-latency object
    # stores.  ``create_*_array`` calls only queue metadata while
    # batching is active.  See
    # :meth:`zarr_vectors.core.group.Group.batched_writes`.
    with level_group.batched_writes(compressor=compressor):
        create_vertices_array(level_group, dtype=dtype)

        if attributes:
            for attr_name, attr_data in attributes.items():
                channel_names = None
                if attr_data.ndim == 2:
                    channel_names = [f"ch{i}" for i in range(attr_data.shape[1])]
                create_attribute_array(
                    level_group, attr_name,
                    dtype=str(attr_data.dtype),
                    channel_names=channel_names,
                    extra_meta=attr_extra_meta.get(attr_name),
                )

        fragment_attr_dtypes: dict[str, np.dtype] = {}
        if fragment_attributes:
            for fname, per_chunk in fragment_attributes.items():
                if not per_chunk:
                    raise ArrayError(
                        f"fragment_attributes[{fname!r}] is empty; pass at "
                        f"least one chunk_coords -> ndarray entry or omit "
                        f"the attribute"
                    )
                sample = np.asarray(next(iter(per_chunk.values())))
                fragment_attr_dtypes[fname] = sample.dtype
                channel_names = None
                if sample.ndim == 2:
                    channel_names = [f"ch{i}" for i in range(sample.shape[1])]
                create_fragment_attribute_array(
                    level_group, fname,
                    dtype=str(sample.dtype),
                    channel_names=channel_names,
                )

        if needs_objects:
            create_object_index_array(level_group)

        if object_ids is not None:
            # With objects: use per-object fragments per chunk (not per-bin)
            # This preserves correct object_index semantics for read_object_vertices.
            # When ``attr_bins`` is set the chunk key gains a leading attr-bin
            # dim and a single object's vertices may land in multiple keys.
            spatial_assignments = assign_chunks(positions, chunk_shape)

            # Build {combined_chunk_coords: global_indices}.  Combined coords
            # are spatial-only when ``attr_bins is None``, else
            # ``(attr_bin, *spatial)`` after splitting each spatial chunk by
            # the per-vertex attr-bin.
            combined_assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
            for spatial_cc, global_indices in spatial_assignments.items():
                global_indices = np.asarray(global_indices, dtype=np.int64)
                if attr_bins is None:
                    combined_assignments[spatial_cc] = global_indices
                    continue
                chunk_attr_bins = attr_bins[global_indices]
                for ab in np.unique(chunk_attr_bins):
                    mask = chunk_attr_bins == ab
                    combined_assignments[(int(ab), *spatial_cc)] = global_indices[mask]

            for chunk_coords, global_indices in sorted(combined_assignments.items()):
                chunk_positions = positions[global_indices]
                chunk_obj_ids = object_ids[global_indices]
                unique_objs = np.unique(chunk_obj_ids)
                vert_groups: list[npt.NDArray] = []
                attr_groups_per_name: dict[str, list[npt.NDArray]] = {}
                if attributes:
                    for attr_name in attributes:
                        attr_groups_per_name[attr_name] = []

                fragment_idx = 0
                for obj_id in unique_objs:
                    mask = chunk_obj_ids == obj_id
                    vert_groups.append(chunk_positions[mask])
                    oid = int(obj_id)
                    if oid not in object_manifests:
                        object_manifests[oid] = []
                    object_manifests[oid].append((chunk_coords, fragment_idx))
                    if attributes:
                        obj_global = global_indices[mask]
                        for attr_name, attr_data in attributes.items():
                            attr_groups_per_name[attr_name].append(attr_data[obj_global])
                    fragment_idx += 1

                write_chunk_vertices(level_group, chunk_coords, vert_groups, dtype=np_dtype)
                if attributes:
                    for attr_name, groups_list in attr_groups_per_name.items():
                        write_chunk_attributes(level_group, attr_name, chunk_coords, groups_list,
                                               dtype=attributes[attr_name].dtype)
                if fragment_attributes:
                    _write_fragment_attrs_for_chunk(
                        level_group, fragment_attributes, fragment_attr_dtypes,
                        chunk_coords, num_fragments=len(vert_groups),
                    )
        else:
            # Undifferentiated: use per-bin fragments
            for chunk_coords, fragment_dict in sorted(chunked_bins.items()):
                vert_groups_bin: list[npt.NDArray] = [
                    np.zeros((0, ndim), dtype=np_dtype) for _ in range(total_bins_per_chunk)
                ]
                attr_groups_per_name_bin: dict[str, list[npt.NDArray]] = {}
                if attributes:
                    for attr_name, attr_data in attributes.items():
                        shape_tail = attr_data.shape[1:] if attr_data.ndim > 1 else ()
                        attr_groups_per_name_bin[attr_name] = [
                            np.zeros((0,) + shape_tail, dtype=attr_data.dtype)
                            for _ in range(total_bins_per_chunk)
                        ]

                for fragment_idx, global_indices in fragment_dict.items():
                    vert_groups_bin[fragment_idx] = positions[global_indices]
                    if attributes:
                        for attr_name, attr_data in attributes.items():
                            attr_groups_per_name_bin[attr_name][fragment_idx] = attr_data[global_indices]

                write_chunk_vertices(level_group, chunk_coords, vert_groups_bin, dtype=np_dtype)
                if attributes:
                    for attr_name, groups_list in attr_groups_per_name_bin.items():
                        write_chunk_attributes(level_group, attr_name, chunk_coords, groups_list,
                                               dtype=attributes[attr_name].dtype)
                if fragment_attributes:
                    _write_fragment_attrs_for_chunk(
                        level_group, fragment_attributes, fragment_attr_dtypes,
                        chunk_coords, num_fragments=len(vert_groups_bin),
                    )

        if needs_objects and object_manifests:
            idx_ndim = ndim + 1 if attr_bins is not None else ndim
            write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

    n_objects = len(object_manifests) if object_manifests else 0
    if object_attributes:
        for name, data in object_attributes.items():
            create_object_attributes_array(level_group, name)
            write_object_attributes(level_group, name, data)

    n_groups = 0
    if groups:
        create_groupings_array(level_group)
        write_groupings(level_group, groups)
        n_groups = len(groups)

    if group_attributes:
        for name, data in group_attributes.items():
            create_groupings_attributes_array(level_group, name)
            write_groupings_attributes(level_group, name, data)

    _finalize_write(root, "write_points")
    return {
        "vertex_count": n_vertices,
        "chunk_count": n_chunks,
        "object_count": n_objects,
        "group_count": n_groups,
        "bins_per_chunk": bins_per_chunk,
    }


def read_points(
    store_path: str,
    *,
    level: int = 0,
    bbox: BoundingBox | None = None,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_names: list[str] | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read point cloud data from a ZV store.

    Supports filtering by bounding box, object ID, group ID, or chunk
    whitelist. Filters AND together — every filter that is supplied
    must accept a point for it to appear in the output.

    Args:
        store_path: Path to the ZV store.
        level: Resolution level to read (default 0).
        bbox: Optional bounding box filter as ``(min_corner, max_corner)``.
        object_ids: Optional list of object IDs to read.
        group_ids: Optional list of group IDs — expands to their object IDs.
        chunks: Optional whitelist of chunk coordinate tuples; only data
            in those chunks is returned. ``chunks=[]`` yields an empty
            result; ``chunks=None`` (default) applies no chunk filter.
        attribute_names: Optional list of attribute names to read.
            If None, reads all available attributes.

    Returns:
        Dict with keys:
        - ``positions``: ``(M, D)`` array of vertex positions
        - ``attributes``: ``{name: array}`` of per-vertex attributes
        - ``object_ids``: ``(M,)`` array of object IDs per vertex (if objects exist)
        - ``vertex_count``: total vertices returned
    """
    root = open_store(store_path, backend=backend)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim
    dtype = np.float32  # default; could read from array meta

    # Per-level chunk_shape may override root (v0.7+).
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:
        level_meta = None
    level_chunk_shape = get_level_chunk_shape(root_meta, level_meta)

    # Read vertex dtype from metadata if available
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Resolve group_ids → object_ids
    if group_ids is not None:
        resolved_obj_ids: set[int] = set()
        for gid in group_ids:
            members = read_group_object_ids(level_group, gid)
            resolved_obj_ids.update(members)
        if object_ids is not None:
            resolved_obj_ids &= set(object_ids)
        object_ids = sorted(resolved_obj_ids)

    # Pre-resolve the chunks whitelist once (intersected with bbox-implied
    # chunks) so both code paths can reuse it.
    chunk_set: set[ChunkCoords] | None = None
    if chunks is not None:
        chunk_set = set(
            resolve_chunk_keys(
                level_group, level_chunk_shape,
                bbox=bbox, chunks=chunks,
            )
        )

    # Path 1: read by object ID (via object_index)
    if object_ids is not None:
        all_positions: list[npt.NDArray] = []
        all_obj_labels: list[npt.NDArray] = []
        all_attrs: dict[str, list[npt.NDArray]] = {}

        for oid in object_ids:
            try:
                manifest = read_object_manifest(level_group, oid)
            except ArrayError:
                continue

            # Optionally restrict the manifest to listed chunks.
            if chunk_set is not None:
                manifest = [
                    (cc, fragment_index) for (cc, fragment_index) in manifest if cc in chunk_set
                ]
            if not manifest:
                continue

            for chunk_coords, fragment_index in manifest:
                try:
                    fragment = read_fragment(
                        level_group, chunk_coords, fragment_index,
                        dtype=dtype, ndim=ndim,
                    )
                except ArrayError:
                    continue
                n_pts = len(fragment)
                if n_pts == 0:
                    continue
                all_positions.append(fragment)
                all_obj_labels.append(np.full(n_pts, oid, dtype=np.int64))

        if not all_positions:
            return _empty_result(ndim)

        positions_out = np.concatenate(all_positions, axis=0)

        # Apply bbox filter if needed
        if bbox is not None:
            mask = np.all(
                (positions_out >= bbox[0]) & (positions_out <= bbox[1]),
                axis=1,
            )
            positions_out = positions_out[mask]
            all_obj_labels = [
                np.concatenate(all_obj_labels)[mask]
            ]

        result: dict[str, Any] = {
            "positions": positions_out,
            "object_ids": np.concatenate(all_obj_labels) if all_obj_labels else np.array([], dtype=np.int64),
            "vertex_attributes": {},
            "vertex_count": len(positions_out),
        }
        return result

    # Path 2: read by bounding box or read all
    chunk_keys = list_chunk_keys(level_group)
    if chunk_set is not None:
        chunk_keys = [k for k in chunk_keys if k in chunk_set]

    # attribute_filter: restrict to chunks with the matching leading bin
    # index.  Cheap pre-filter on the chunk key — drops scans drastically
    # for stores written with ``chunk_by_attribute``.
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
                "attribute_filter requires a store written with "
                "chunk_by_attribute (level metadata has no chunk_attribute_name)"
            )
        if len(attribute_filter) != 1:
            raise ArrayError(
                "attribute_filter must specify exactly one attribute "
                f"(got {len(attribute_filter)})"
            )
        filt_name, filt_value = next(iter(attribute_filter.items()))
        if filt_name != lm.chunk_attribute_name:
            raise ArrayError(
                f"attribute_filter key {filt_name!r} does not match the "
                f"store's chunk_attribute_name "
                f"{lm.chunk_attribute_name!r}"
            )
        try:
            bin_idx = lm.chunk_attribute_values.index(filt_value)
        except ValueError:
            return _empty_result(root_meta.sid_ndim)
        chunk_keys = [k for k in chunk_keys if k and k[0] == bin_idx]

    effective_bin = root_meta.effective_bin_shape
    bins_per_chunk = root_meta.bins_per_chunk
    has_bins = any(b > 1 for b in bins_per_chunk)

    chunk_fragment_targets: dict[ChunkCoords, list[int]] | None = None
    chunk_keys_set: set[ChunkCoords] = set()

    # Build the prefetch plan: VERTICES + VERTEX_FRAGMENTS for every
    # chunk we may touch, plus each requested attribute array.  Cache
    # misses fall through to the sync ``read_bytes`` path so this is
    # purely a perf optimisation — correctness is unaffected.
    chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_keys]
    prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
    ]
    if attribute_names:
        for attr_name in attribute_names:
            prefetch_plan.append((f"{VERTEX_ATTRIBUTES}/{attr_name}", chunk_key_strs))

    with level_group.batched_reads(prefetch_plan):
        if bbox is not None and has_bins:
            # Bin-level targeting: only decode matching fragments
            from zarr_vectors.spatial.chunking import (
                bins_intersecting_bbox,
                bin_to_chunk,
                bin_to_fragment_index,
            )
            target_bins = bins_intersecting_bbox(
                np.asarray(bbox[0]), np.asarray(bbox[1]),
                effective_bin,
            )
            # Group target bins by chunk
            chunk_fragment_targets = {}
            for bc in target_bins:
                cc = bin_to_chunk(bc, bins_per_chunk)
                fragment_index = bin_to_fragment_index(bc, cc, bins_per_chunk)
                if cc not in chunk_fragment_targets:
                    chunk_fragment_targets[cc] = []
                chunk_fragment_targets[cc].append(fragment_index)

            # Only read from chunks that have data
            chunk_keys_set = set(chunk_keys)
            all_positions = []
            for cc, fragment_indices in chunk_fragment_targets.items():
                if cc not in chunk_keys_set:
                    continue
                for fragment_index in fragment_indices:
                    try:
                        fragment = read_fragment(
                            level_group, cc, fragment_index, dtype=dtype, ndim=ndim,
                        )
                        if len(fragment) > 0:
                            all_positions.append(fragment)
                    except ArrayError:
                        continue

        elif bbox is not None:
            # Chunk-level targeting (old stores without bins)
            target_chunks = set(chunks_intersecting_bbox(
                np.asarray(bbox[0]), np.asarray(bbox[1]),
                level_chunk_shape,
            ))
            chunk_keys = [k for k in chunk_keys if k in target_chunks]

            all_positions = []
            for chunk_coords in chunk_keys:
                try:
                    groups = read_chunk_vertices(
                        level_group, chunk_coords, dtype=dtype, ndim=ndim
                    )
                except ArrayError:
                    continue
                for fragment in groups:
                    if len(fragment) > 0:
                        all_positions.append(fragment)

        else:
            # Read all
            all_positions = []
            for chunk_coords in chunk_keys:
                try:
                    groups = read_chunk_vertices(
                        level_group, chunk_coords, dtype=dtype, ndim=ndim
                    )
                except ArrayError:
                    continue
                for fragment in groups:
                    if len(fragment) > 0:
                        all_positions.append(fragment)

        if not all_positions:
            return _empty_result(ndim)

        positions_out = np.concatenate(all_positions, axis=0)

        # Apply precise bbox filter (chunk-level is coarse)
        if bbox is not None:
            mask = np.all(
                (positions_out >= bbox[0]) & (positions_out <= bbox[1]),
                axis=1,
            )
            positions_out = positions_out[mask]

        # Read attributes — pull the on-disk dtype, channel count, and
        # any dictionary-encoding metadata from each attribute array's
        # ``zarr.json``.  Falls back to (float32, 1 col, no encoding)
        # if the meta blob is missing or unreadable.
        attrs_out: dict[str, npt.NDArray] = {}
        if attribute_names:
            for attr_name in attribute_names:
                attr_dtype, attr_ncols, attr_encoding = _read_attribute_meta(
                    level_group, attr_name,
                )
                attr_parts: list[npt.NDArray] = []
                if chunk_fragment_targets is not None:
                    # Bin-level bbox: read the same fragments as positions
                    for cc, fragment_indices in chunk_fragment_targets.items():
                        if cc not in chunk_keys_set:
                            continue
                        try:
                            attr_groups = read_chunk_attributes(
                                level_group, attr_name, cc,
                                dtype=attr_dtype, ncols=attr_ncols,
                            )
                            for fragment_index in fragment_indices:
                                if fragment_index < len(attr_groups) and len(attr_groups[fragment_index]) > 0:
                                    attr_parts.append(attr_groups[fragment_index])
                        except ArrayError:
                            continue
                else:
                    for chunk_coords in chunk_keys:
                        try:
                            attr_groups = read_chunk_attributes(
                                level_group, attr_name, chunk_coords,
                                dtype=attr_dtype, ncols=attr_ncols,
                            )
                            for ag in attr_groups:
                                attr_parts.append(ag)
                        except ArrayError:
                            continue
                if attr_parts:
                    attr_all = np.concatenate(attr_parts, axis=0)
                    if bbox is not None:
                        attr_all = attr_all[mask]
                    if attr_encoding is not None:
                        attr_all = decode_categorical(
                            attr_all,
                            attr_encoding["categories"],
                            fill_value=attr_encoding.get("_FillValue"),
                        )
                    attrs_out[attr_name] = attr_all

    return {
        "positions": positions_out,
        "vertex_attributes": attrs_out,
        "vertex_count": len(positions_out),
    }


def _read_attribute_meta(
    level_group: Any,
    attr_name: str,
) -> tuple[np.dtype, int, dict[str, Any] | None]:
    """Read on-disk metadata for ``vertex_attributes/<attr_name>/``.

    Returns ``(dtype, ncols, dict_encoding_meta_or_None)``.  Falls
    back to ``(float32, 1, None)`` if the metadata blob can't be read
    (e.g. legacy stores that pre-date the dtype field).
    """
    try:
        meta = level_group.read_array_meta(f"{VERTEX_ATTRIBUTES}/{attr_name}")
    except Exception:
        return np.dtype(np.float32), 1, None
    dtype = np.dtype(meta.get("dtype", "float32"))
    channel_names = meta.get("channel_names")
    ncols = len(channel_names) if channel_names else 1
    dict_meta: dict[str, Any] | None = None
    if meta.get("encoding") == DICTIONARY_ENCODING:
        dict_meta = {
            "categories": meta["categories"],
            "ordered": meta.get("ordered", False),
        }
        if "_FillValue" in meta:
            dict_meta["_FillValue"] = meta["_FillValue"]
    return dtype, ncols, dict_meta


def _empty_result(ndim: int) -> dict[str, Any]:
    """Return an empty result dict."""
    return {
        "positions": np.zeros((0, ndim), dtype=np.float32),
        "vertex_attributes": {},
        "vertex_count": 0,
    }


def _write_fragment_attrs_for_chunk(
    level_group: FsGroup,
    fragment_attributes: dict[str, dict[ChunkCoords, npt.NDArray]],
    fragment_attr_dtypes: dict[str, np.dtype],
    chunk_coords: ChunkCoords,
    num_fragments: int,
) -> None:
    """Emit per-chunk fragment-attribute blobs for one chunk.

    Validates that each per-chunk array's leading dim matches the
    number of fragments this writer produced for the chunk; mismatches
    surface at write time rather than as cryptic byte-length errors
    later.
    """
    for fname, per_chunk in fragment_attributes.items():
        if chunk_coords not in per_chunk:
            continue
        arr = np.asarray(per_chunk[chunk_coords])
        if arr.shape[0] != num_fragments:
            raise ArrayError(
                f"fragment_attributes[{fname!r}][{chunk_coords!r}] has "
                f"{arr.shape[0]} rows but chunk has {num_fragments} fragments"
            )
        write_chunk_fragment_attributes(
            level_group, fname, chunk_coords, arr,
            dtype=fragment_attr_dtypes[fname],
        )
