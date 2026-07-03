"""Finite line and vector I/O for zarr vectors stores.

A finite line is two endpoints stored as a two-vertex object.
Connectivity is implicit sequential (vertex 0 → vertex 1), so no
``links`` array is needed.

Lines that cross a chunk boundary are split into two single-vertex
fragments in their respective chunks, with the ``object_index``
tracking both and a ``cross_chunk_links`` entry bridging them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    GEOM_LINE,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_STANDARD,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_object_attributes,
    read_fragment,
    write_chunk_attributes,
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
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
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
    read_level_metadata,
    read_root_metadata,
    stamp_ccl_capabilities,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.chunking import (
    assign_chunks,
    compute_bounds,
    compute_chunk_coords,
    bin_to_chunk,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    CrossChunkLink,
    ObjectManifest,
    FragmentRef,
)


def write_lines(
    store_path: str,
    endpoints: npt.NDArray[np.floating],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    dtype: str = "float32",
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
    out_of_bounds: str = DEFAULT_OOB_POLICY,
    compressor: Any = None,
    # Deprecated aliases (will be removed):
    attributes: dict[str, npt.NDArray] | None = None,
    line_attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Write finite line segments to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        endpoints: ``(N, 2, D)`` array — N lines, each with 2 endpoints
            of D dimensions.
        chunk_shape: Spatial chunk size per dimension.
        vertex_attributes: Per-vertex attributes as
            ``{name: (N, 2) or (N, 2, C)}``.  Two values per line (one
            per endpoint).  (Spec name; replaces ``attributes``.)
        object_attributes: Per-line (per-object) attributes as
            ``{name: (N,) or (N, C)}``.  (Spec name; replaces
            ``line_attributes``.)
        dtype: Numpy dtype string for positions.

    Returns:
        Summary dict with ``line_count``, ``chunk_count``,
        ``cross_chunk_count``.
    """
    # Back-compat: accept the legacy kwarg names.
    if attributes is not None:
        if vertex_attributes is not None:
            raise TypeError(
                "got both `attributes` and `vertex_attributes`; "
                "pass only `vertex_attributes`."
            )
        import warnings
        warnings.warn(
            "`attributes` is deprecated; use `vertex_attributes`.",
            DeprecationWarning, stacklevel=2,
        )
        vertex_attributes = attributes
    if line_attributes is not None:
        if object_attributes is not None:
            raise TypeError(
                "got both `line_attributes` and `object_attributes`; "
                "pass only `object_attributes`."
            )
        import warnings
        warnings.warn(
            "`line_attributes` is deprecated; use `object_attributes`.",
            DeprecationWarning, stacklevel=2,
        )
        object_attributes = line_attributes
    # Internal aliases so the rest of the body stays unchanged.
    attributes = vertex_attributes
    line_attributes = object_attributes

    np_dtype = np.dtype(dtype)
    endpoints = np.asarray(endpoints, dtype=np_dtype)

    if endpoints.ndim != 3 or endpoints.shape[1] != 2:
        raise ArrayError(
            f"endpoints must be (N, 2, D), got shape {endpoints.shape}"
        )

    n_lines, _, ndim = endpoints.shape

    # Per-line attribute chunking: each line goes into one bin.  The
    # attribute name must appear in ``line_attributes`` (per-line) — or
    # in per-endpoint ``attributes`` with both endpoints sharing the
    # same value.
    line_attr_bins: npt.NDArray[np.int64] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        src_values: npt.NDArray | None = None
        if line_attributes and chunk_by_attribute in line_attributes:
            src_values = np.asarray(line_attributes[chunk_by_attribute])
        elif attributes and chunk_by_attribute in attributes:
            per_endpoint = np.asarray(attributes[chunk_by_attribute])
            if per_endpoint.shape[:2] != (n_lines, 2):
                raise ArrayError(
                    f"per-endpoint attribute {chunk_by_attribute!r} has "
                    f"shape {per_endpoint.shape}; expected (N, 2[, C])"
                )
            mismatched = np.any(per_endpoint[:, 0] != per_endpoint[:, 1])
            if mismatched:
                raise ArrayError(
                    f"chunk_by_attribute={chunk_by_attribute!r} on a "
                    f"per-endpoint attribute requires both endpoints of "
                    f"each line to share the same value (got mixed lines)"
                )
            src_values = per_endpoint[:, 0]
        if src_values is None:
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a "
                f"key in `line_attributes` (per-line) or in `attributes` "
                f"(per-endpoint, uniform per line)"
            )
        if src_values.shape[0] != n_lines:
            raise ArrayError(
                f"chunk_by_attribute array length {src_values.shape[0]} "
                f"!= n_lines {n_lines}"
            )
        line_attr_bins, attr_bin_values = assign_attribute_bins(src_values)

    effective_bin = bin_shape if bin_shape is not None else chunk_shape
    bins_per_chunk = tuple(
        int(round(cs / bs)) for cs, bs in zip(chunk_shape, effective_bin)
    )

    all_pts = endpoints.reshape(-1, ndim)
    if bounds is None:
        inferred = compute_bounds(all_pts)
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
    if out_of_bounds == "ignore":
        raise ArrayError(
            "out_of_bounds='ignore' is not supported for write_lines: "
            "endpoint pairing depends on full line presence. Use 'raise' "
            "(default) or 'expand'."
        )
    _apply_out_of_bounds_policy(root, all_pts, policy=out_of_bounds)

    root_meta = _ensure_root_metadata_for_write(
        root,
        inferred_ndim=ndim,
        geometry_type=GEOM_LINE,
        base_bin_shape=bin_shape,
        links_convention=LINKS_IMPLICIT_SEQUENTIAL,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
    )
    axes = root_meta.spatial_index_dims

    arrays_present = [VERTICES, "object_index"]
    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute, ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )
    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_lines * 2,
        arrays_present=arrays_present,
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)

    # Vectorized chunk classification: ``floor(pos / chunk_shape)`` for
    # both endpoints in one numpy pass.  The previous list-comprehension
    # called ``bin_to_chunk(compute_chunk_coords(...))`` once per line —
    # 2N Python-level numpy calls.
    cs = np.asarray(chunk_shape, dtype=np.float64)
    chunk_ints = np.floor(endpoints / cs).astype(np.int64)  # (N, 2, D)
    chunk_a_ints = chunk_ints[:, 0]                          # (N, D)
    chunk_b_ints = chunk_ints[:, 1]
    # Attribute-bin prefix becomes a leading column on the chunk key
    # when chunk-by-attribute is active.
    if line_attr_bins is not None:
        prefix = line_attr_bins[:, None].astype(np.int64)    # (N, 1)
        chunk_a_ints = np.concatenate([prefix, chunk_a_ints], axis=1)
        chunk_b_ints = np.concatenate([prefix, chunk_b_ints], axis=1)

    # Materialize chunk-coord tuples in one C-level pass.
    ca_tuples = [tuple(row) for row in chunk_a_ints.tolist()]
    cb_tuples = [tuple(row) for row in chunk_b_ints.tolist()]
    same_chunk = np.all(chunk_a_ints == chunk_b_ints, axis=1).tolist()

    chunk_groups: dict[ChunkCoords, list[tuple[int, npt.NDArray]]] = {}
    object_manifests: dict[int, ObjectManifest] = {}
    cross_links: list[CrossChunkLink] = []

    # One Python pass over lines is unavoidable to preserve line-id-
    # ordered fragment_idx assignment, but each iteration is now just
    # dict.setdefault + list.append — no per-line numpy indexing.
    for i in range(n_lines):
        ca = ca_tuples[i]
        if same_chunk[i]:
            bucket = chunk_groups.setdefault(ca, [])
            fragment_idx = len(bucket)
            bucket.append((i, endpoints[i]))  # (N=2, D)
            object_manifests[i] = [(ca, fragment_idx)]
        else:
            cb = cb_tuples[i]
            bucket_a = chunk_groups.setdefault(ca, [])
            fragment_idx_a = len(bucket_a)
            bucket_a.append((i, endpoints[i, 0:1]))  # (1, D)

            bucket_b = chunk_groups.setdefault(cb, [])
            fragment_idx_b = len(bucket_b)
            bucket_b.append((i, endpoints[i, 1:2]))  # (1, D)

            object_manifests[i] = [(ca, fragment_idx_a), (cb, fragment_idx_b)]
            cross_links.append(((ca, 0), (cb, 0)))

    idx_ndim = ndim + 1 if line_attr_bins is not None else ndim
    # Collapse all per-array zarr.json PUTs + per-chunk byte writes into
    # one asyncio.gather (mirrors points.py:300).  Smaller win on local
    # FS, large win against object stores.
    with level_group.batched_writes(compressor=compressor):
        create_vertices_array(level_group, dtype=dtype)
        create_object_index_array(level_group)
        create_cross_chunk_links_array(level_group, delta=0, sid_ndim=idx_ndim)
        if line_attributes:
            for name in line_attributes:
                create_object_attributes_array(level_group, name)

        for chunk_coords, groups_list in sorted(chunk_groups.items()):
            vert_arrays = [g[1] for g in groups_list]
            write_chunk_vertices(
                level_group, chunk_coords, vert_arrays, dtype=np_dtype,
            )

        write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

        if cross_links:
            write_cross_chunk_links(
                level_group, cross_links, sid_ndim=idx_ndim, delta=0,
                directed=True,
            )
            stamp_ccl_capabilities(root)

        if line_attributes:
            for name, data in line_attributes.items():
                write_object_attributes(level_group, name, np.asarray(data))

    _finalize_write(root, "write_lines")
    return {
        "line_count": n_lines,
        "chunk_count": len(chunk_groups),
        "cross_chunk_count": len(cross_links),
    }


def read_lines(
    store_path: str,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    bbox: BoundingBox | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read finite lines from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        object_ids: Optional list of line (object) IDs to read.
        bbox: Optional bounding box filter (lines with any endpoint
            inside the box are returned).

    Returns:
        Dict with:
        - ``endpoints``: ``(M, 2, D)`` array of line endpoints
        - ``line_count``: number of lines returned
    """
    root = open_store(store_path, backend=backend)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim

    dtype = np.float32
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Resolve attribute_filter → leading-bin index (per-object lines:
    # all of an object's chunks share the same leading coord).
    filter_bin: int | None = None
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
            return {
                "endpoints": np.zeros((0, 2, ndim), dtype=dtype),
                "line_count": 0,
            }

    # Determine which objects to read
    if object_ids is None:
        meta = level_group.read_array_meta("object_index")
        object_ids = list(range(meta["num_objects"]))

    # When filtering by attribute, prune object_ids whose manifest
    # entries don't start with the matching bin.
    if filter_bin is not None:
        try:
            manifests = read_all_object_manifests(level_group)
        except Exception:
            manifests = []
        kept: list[int] = []
        for oid in object_ids:
            if oid < len(manifests) and manifests[oid]:
                if all(cc and cc[0] == filter_bin for cc, _ in manifests[oid]):
                    kept.append(oid)
        object_ids = kept

    # Chunk-major read: decode each chunk's fragments *once*, then
    # dispatch them to the requesting objects.  The per-object access
    # pattern would re-decode the vertex-fragment index for every line
    # touching a chunk — O(K_per_chunk * N_per_chunk) per chunk.  Reading
    # the whole chunk once is O(K_per_chunk).  Includes OBJECT_INDEX in
    # the prefetch plan so the manifest decode below is fully cached.
    # The OBJECT_INDEX entry serves legacy ``data``/``offsets`` stores;
    # ``vlen_manifests_v1`` stores read the ragged ``manifests`` array
    # directly and the prefetch is a harmless no-op for them.
    _chunk_key_strs = [
        ".".join(str(c) for c in cc)
        for cc in list_chunk_keys(level_group, VERTICES)
    ]
    _prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, _chunk_key_strs),
        (VERTEX_FRAGMENTS, _chunk_key_strs),
        (OBJECT_INDEX, ["data", "offsets"]),
    ]
    _batched_reads_cm = level_group.batched_reads(_prefetch_plan)
    _batched_reads_cm.__enter__()
    try:
        manifests = read_all_object_manifests(level_group)

        # Build a per-chunk dispatch table: chunk → list of
        # (oid_local_idx, manifest_position, fragment_index).  ``oid_local_idx``
        # indexes into the per-object output list, not the global oid
        # (which can be sparse).
        oid_outputs: list[list[npt.NDArray | None]] = []
        oid_for_output: list[int] = []
        chunk_dispatch: dict[ChunkCoords, list[tuple[int, int, int]]] = {}
        for oid in object_ids:
            if oid < 0 or oid >= len(manifests):
                continue
            manifest = manifests[oid]
            if not manifest:
                continue
            slot = len(oid_outputs)
            oid_outputs.append([None] * len(manifest))
            oid_for_output.append(oid)
            for mi, (cc, fragment_index) in enumerate(manifest):
                chunk_dispatch.setdefault(cc, []).append((slot, mi, fragment_index))

        # One read_chunk_vertices per chunk; copy slices into output slots.
        for cc, entries in chunk_dispatch.items():
            try:
                groups = read_chunk_vertices(
                    level_group, cc, dtype=dtype, ndim=ndim,
                )
            except ArrayError:
                continue
            for slot, mi, fragment_index in entries:
                if 0 <= fragment_index < len(groups):
                    oid_outputs[slot][mi] = groups[fragment_index]

        result_endpoints: list[npt.NDArray] = []
        for slot_groups in oid_outputs:
            if any(g is None for g in slot_groups):
                continue
            all_verts = np.concatenate(slot_groups, axis=0)
            if len(all_verts) < 2:
                continue
            ep = np.stack([all_verts[0], all_verts[-1]], axis=0)
            result_endpoints.append(ep)
    finally:
        _batched_reads_cm.__exit__(None, None, None)

    if not result_endpoints:
        return {
            "endpoints": np.zeros((0, 2, ndim), dtype=dtype),
            "line_count": 0,
        }

    endpoints_out = np.stack(result_endpoints, axis=0)  # (M, 2, D)

    # Apply bbox filter
    if bbox is not None:
        bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
        # Keep line if either endpoint is inside the bbox
        in_a = np.all(
            (endpoints_out[:, 0] >= bbox_min) & (endpoints_out[:, 0] <= bbox_max),
            axis=1,
        )
        in_b = np.all(
            (endpoints_out[:, 1] >= bbox_min) & (endpoints_out[:, 1] <= bbox_max),
            axis=1,
        )
        mask = in_a | in_b
        endpoints_out = endpoints_out[mask]

    return {
        "endpoints": endpoints_out,
        "line_count": len(endpoints_out),
    }
