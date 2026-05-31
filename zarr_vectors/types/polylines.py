"""Polyline and streamline I/O for zarr vectors stores.

A polyline is an ordered sequence of vertices forming a connected path.
Streamlines (from tractography) are polylines with additional per-object
attributes like termination regions.

Polylines that cross chunk boundaries are split into segments.  The
``object_index`` stores the ordered segment sequence for each polyline,
and ``cross_chunk_links`` connects the last vertex of one segment to
the first vertex of the next.  Within each segment, connectivity is
implicit sequential (vertex i → vertex i+1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    FRAGMENT_ATTRIBUTES,
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJECT_INDEX,
    OBJIDX_STANDARD,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
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
    read_chunk_vertices,
    read_cross_chunk_links,
    read_group_object_ids,
    read_object_attributes,
    read_object_vertices,
    read_fragment,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_vertices,
    write_cross_chunk_links,
    write_groupings,
    write_groupings_attributes,
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
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import (
    cross_chunk_links_for_segments,
    split_polyline_at_boundaries,
)
from zarr_vectors.spatial.chunking import (
    chunks_intersecting_bbox,
    compute_bounds,
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


def write_polylines(
    store_path: str,
    polylines: list[npt.NDArray[np.floating]],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    vertex_attributes: dict[str, list[npt.NDArray]] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    fragment_attributes: dict[str, dict[ChunkCoords, npt.NDArray]] | None = None,
    groups: dict[int, list[int]] | None = None,
    group_attributes: dict[str, npt.NDArray] | None = None,
    dtype: str = "float32",
    geometry_type: str = GEOM_STREAMLINE,
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
    out_of_bounds: str = DEFAULT_OOB_POLICY,
    compressor: Any = None,
    shard_shape: int | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Write polylines/streamlines to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        polylines: List of arrays, each ``(N_k, D)`` — one per polyline.
        chunk_shape: Spatial chunk size per dimension.
        vertex_attributes: Per-vertex attributes aligned with polylines.
            ``{name: [array_for_polyline_0, array_for_polyline_1, ...]}``
            where each array is ``(N_k,)`` or ``(N_k, C)``.
        object_attributes: Per-polyline attributes.
            ``{name: (O,) or (O, C)}`` where O = number of polylines.
        fragment_attributes: Per-fragment attributes, keyed per chunk:
            ``{name: {chunk_coords: ndarray}}``.  Each per-chunk ndarray
            has shape ``(num_fragments_in_chunk,)`` or
            ``(num_fragments_in_chunk, C)`` and is aligned with the
            fragment ordering this writer produces for that chunk.
        groups: Group memberships ``{group_id: [polyline_indices]}``.
        group_attributes: Per-group attributes ``{name: (G,) or (G,C)}``.
        dtype: Numpy dtype for positions.
        geometry_type: ``"streamline"`` or ``"polyline"``.

    Returns:
        Summary dict.
    """
    np_dtype = np.dtype(dtype)
    n_polylines = len(polylines)

    if n_polylines == 0:
        raise ArrayError("Cannot write empty polyline list")

    # Determine dimensionality from first polyline
    ndim = polylines[0].shape[1]

    # Compute global bounds from all vertices unless caller pinned them.
    all_pts = np.concatenate(polylines, axis=0)
    if bounds is None:
        inferred = compute_bounds(all_pts)
        bounds_list = (inferred[0].tolist(), inferred[1].tolist())
    else:
        bounds_list = (list(bounds[0]), list(bounds[1]))
    total_vertices = len(all_pts)

    # bin_shape is retained in the store metadata (``base_bin_shape``)
    # for downstream attribute-chunking / spatial-index sub-binning, but
    # the polyline writer itself splits at chunk boundaries only — see
    # the call to ``split_polyline_at_boundaries`` below.

    root = _create_or_open_store(
        store_path,
        backend=backend,
        bounds=bounds_list,
        chunk_shape=tuple(chunk_shape),
        ndim=ndim,
    )
    # OOB policy for polyline vertices.  "ignore" is rejected — dropping
    # vertices would break the per-polyline ordering and connectivity.
    if out_of_bounds == "ignore":
        raise ArrayError(
            "out_of_bounds='ignore' is not supported for write_polylines: "
            "polyline connectivity depends on vertex ordering. Use "
            "'raise' (default) or 'expand'."
        )
    _apply_out_of_bounds_policy(root, all_pts, policy=out_of_bounds)

    root_meta = _ensure_root_metadata_for_write(
        root,
        inferred_ndim=ndim,
        geometry_type=geometry_type,
        base_bin_shape=bin_shape,
        links_convention=LINKS_IMPLICIT_SEQUENTIAL,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
    )
    axes = root_meta.spatial_index_dims

    # Attribute-chunking setup.  The chunk-by attribute must be a
    # per-vertex array; vertex_attributes[<name>] is the same list-of-
    # arrays shape (one per polyline).  Bin assignment is computed
    # globally across all polylines so bin indices are consistent.
    per_poly_attr_bins: list[npt.NDArray[np.int64]] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        if not vertex_attributes or chunk_by_attribute not in vertex_attributes:
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a "
                f"key in `vertex_attributes` (got: "
                f"{sorted(vertex_attributes) if vertex_attributes else []})"
            )
        attr_lists = vertex_attributes[chunk_by_attribute]
        if len(attr_lists) != n_polylines:
            raise ArrayError(
                f"chunk_by_attribute list length {len(attr_lists)} != "
                f"n_polylines {n_polylines}"
            )
        concat = np.concatenate(attr_lists)
        bins_concat, attr_bin_values = assign_attribute_bins(concat)
        per_poly_attr_bins = []
        offset = 0
        for arr in attr_lists:
            n = len(arr)
            per_poly_attr_bins.append(bins_concat[offset:offset + n])
            offset += n
        # The chunk-by attribute is implicit in the leading chunk axis.
        vertex_attributes = {
            k: v for k, v in vertex_attributes.items() if k != chunk_by_attribute
        }

    arrays_present = [VERTICES, "object_index"]
    if fragment_attributes:
        arrays_present.append(FRAGMENT_ATTRIBUTES)
    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute, ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )
    level_meta = LevelMetadata(
        level=0,
        vertex_count=total_vertices,
        arrays_present=arrays_present,
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)

    # Split each polyline at chunk boundaries and accumulate per-chunk data
    # chunk_data[chunk_coords] = list of (polyline_id, segment_vertices, segment_attrs)
    chunk_data: dict[ChunkCoords, list[tuple[int, npt.NDArray, dict[str, npt.NDArray]]]] = {}
    object_manifests: dict[int, ObjectManifest] = {}
    all_cross_links: list[CrossChunkLink] = []

    def _slice_attrs(poly_id: int, start: int, end: int) -> dict[str, npt.NDArray]:
        out: dict[str, npt.NDArray] = {}
        if not vertex_attributes:
            return out
        for attr_name, attr_list in vertex_attributes.items():
            out[attr_name] = attr_list[poly_id][start:end]
        return out

    # Per-chunk running offset into the chunk's vertices array — used
    # to compute chunk-local vertex indices for cross_chunk_links
    # endpoints below.  Each new fragment's first local vertex is the
    # current offset; its last local vertex is offset + len - 1.
    chunk_vertex_offsets: dict[ChunkCoords, int] = {}

    for poly_id, poly_verts in enumerate(polylines):
        poly_verts = np.asarray(poly_verts, dtype=np_dtype)

        # Split at SPATIAL CHUNK boundaries, not bin boundaries.  Per the
        # zarr-vectors spec, each (object, chunk) contributes ONE
        # fragment with implicit_sequential edges; intra-chunk vertices
        # are connected by implicit edges, and cross-chunk transitions
        # go in cross_chunk_links with real chunk-local vertex indices.
        # Splitting at bin boundaries would create multiple same-chunk
        # fragments per object, leaving no place for the intra-chunk
        # bin-boundary edges (implicit_sequential has no explicit links
        # array to hold them).  bin_shape remains in effect for
        # attribute-bin partitioning (poly_attr_bins) when that is
        # configured.
        segments = split_polyline_at_boundaries(poly_verts, chunk_shape)

        if not segments:
            object_manifests[poly_id] = []
            continue

        poly_attr_bins = (
            per_poly_attr_bins[poly_id]
            if per_poly_attr_bins is not None
            else None
        )

        # Build a flat list of sub-segments: each is one contiguous run
        # of polyline vertices that share (a) a spatial chunk and
        # (b) — when attribute chunking is active — an attribute bin.
        seg_lengths = [len(s[1]) for s in segments]
        seg_offsets = np.cumsum([0, *seg_lengths[:-1]]).astype(np.int64)

        sub_entries: list[tuple[ChunkCoords, npt.NDArray, dict[str, npt.NDArray]]] = []
        for seg_idx, (spatial_cc, seg_verts) in enumerate(segments):
            seg_off = int(seg_offsets[seg_idx])
            seg_len = seg_lengths[seg_idx]

            if poly_attr_bins is None:
                sa = _slice_attrs(poly_id, seg_off, seg_off + seg_len)
                sub_entries.append((spatial_cc, seg_verts, sa))
                continue

            seg_attr_bins = poly_attr_bins[seg_off:seg_off + seg_len]
            transitions = np.where(np.diff(seg_attr_bins) != 0)[0] + 1
            starts = np.concatenate([[0], transitions])
            ends = np.concatenate([transitions, [seg_len]])
            for s, e in zip(starts, ends):
                ab = int(seg_attr_bins[int(s)])
                prefixed = (ab,) + tuple(spatial_cc)
                sa = _slice_attrs(
                    poly_id, seg_off + int(s), seg_off + int(e),
                )
                sub_entries.append((prefixed, seg_verts[int(s):int(e)], sa))

        # manifest_with_indices augments each manifest entry with the
        # chunk-local vertex range of its fragment, used by the
        # cross_chunk_links writer below to record proper endpoints.
        manifest: ObjectManifest = []
        manifest_with_indices: list[
            tuple[ChunkCoords, int, int, int]
        ] = []  # (chunk_coords, fragment_idx, first_local_vert, last_local_vert)
        for chunk_coords, sub_verts, sa in sub_entries:
            if chunk_coords not in chunk_data:
                chunk_data[chunk_coords] = []
                chunk_vertex_offsets[chunk_coords] = 0
            fragment_idx = len(chunk_data[chunk_coords])
            first_local = chunk_vertex_offsets[chunk_coords]
            last_local = first_local + len(sub_verts) - 1
            chunk_vertex_offsets[chunk_coords] = first_local + len(sub_verts)
            chunk_data[chunk_coords].append((poly_id, sub_verts, sa))
            manifest.append((chunk_coords, fragment_idx))
            manifest_with_indices.append(
                (chunk_coords, fragment_idx, first_local, last_local)
            )

        object_manifests[poly_id] = manifest

        # Cross-chunk links: one record per consecutive-fragment pair in
        # different chunks.  Endpoint vertex indices are real chunk-
        # local indices (last vertex of segment k → first vertex of
        # segment k+1) so a reader can resolve the bridge without
        # consulting the manifest.  With the per-chunk-only fragment
        # split above, consecutive same-chunk entries cannot occur for
        # a single polyline (would be merged into one fragment).
        if len(manifest_with_indices) > 1:
            for i in range(len(manifest_with_indices) - 1):
                cc_a, _, _, last_a = manifest_with_indices[i]
                cc_b, _, first_b, _ = manifest_with_indices[i + 1]
                if cc_a != cc_b:
                    all_cross_links.append(((cc_a, last_a), (cc_b, first_b)))

    idx_ndim = ndim + 1 if per_poly_attr_bins is not None else ndim
    # Collapse all per-array zarr.json + per-chunk byte writes into one
    # asyncio.gather (mirrors points.py:300).  ``shard_shape`` also
    # activates native ``sharding_indexed`` for per-chunk arrays.
    from zarr_vectors.core.arrays import open_write_session
    with open_write_session(
        level_group, compressor=compressor, shard_shape=shard_shape,
        bounds=bounds_list, chunk_shape=chunk_shape,
    ):
        create_vertices_array(level_group, dtype=dtype)
        create_object_index_array(level_group)
        create_cross_chunk_links_array(level_group, delta=0)
        if vertex_attributes:
            for attr_name, attr_list in vertex_attributes.items():
                sample = attr_list[0]
                create_attribute_array(
                    level_group, attr_name,
                    dtype=str(sample.dtype),
                )
        if object_attributes:
            for name in object_attributes:
                create_object_attributes_array(level_group, name)

        fragment_attr_dtypes: dict[str, np.dtype] = {}
        if fragment_attributes:
            for fname, per_chunk in fragment_attributes.items():
                if not per_chunk:
                    raise ArrayError(
                        f"fragment_attributes[{fname!r}] is empty"
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

        for chunk_coords in sorted(chunk_data.keys()):
            entries = chunk_data[chunk_coords]
            vert_groups = [e[1] for e in entries]
            write_chunk_vertices(
                level_group, chunk_coords, vert_groups, dtype=np_dtype,
            )

            # Write attributes per chunk
            if vertex_attributes:
                for attr_name in vertex_attributes:
                    attr_groups = [e[2].get(attr_name) for e in entries]
                    # Filter out None (shouldn't happen but be safe)
                    attr_groups = [a for a in attr_groups if a is not None]
                    if attr_groups:
                        write_chunk_attributes(
                            level_group, attr_name, chunk_coords, attr_groups,
                            dtype=attr_groups[0].dtype,
                        )

            if fragment_attributes:
                num_fragments = len(vert_groups)
                for fname, per_chunk in fragment_attributes.items():
                    if chunk_coords not in per_chunk:
                        continue
                    arr = np.asarray(per_chunk[chunk_coords])
                    if arr.shape[0] != num_fragments:
                        raise ArrayError(
                            f"fragment_attributes[{fname!r}][{chunk_coords!r}] "
                            f"has {arr.shape[0]} rows but chunk has "
                            f"{num_fragments} fragments"
                        )
                    write_chunk_fragment_attributes(
                        level_group, fname, chunk_coords, arr,
                        dtype=fragment_attr_dtypes[fname],
                    )

        # Write object index — chunk coords gain a leading dim when
        # attribute-chunked, so widen sid_ndim accordingly.
        write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

        # Write cross-chunk links
        if all_cross_links:
            write_cross_chunk_links(
                level_group, all_cross_links, sid_ndim=idx_ndim, delta=0,
            )

        # Write object attributes
        if object_attributes:
            for name, data in object_attributes.items():
                write_object_attributes(level_group, name, np.asarray(data))

        # Write groupings
        if groups:
            create_groupings_array(level_group)
            write_groupings(level_group, groups)

        if group_attributes:
            for name, data in group_attributes.items():
                create_groupings_attributes_array(level_group, name)
                write_groupings_attributes(level_group, name, np.asarray(data))

    n_groups = len(groups) if groups else 0

    _finalize_write(root, "write_polylines")
    return {
        "polyline_count": n_polylines,
        "vertex_count": total_vertices,
        "chunk_count": len(chunk_data),
        "cross_chunk_link_count": len(all_cross_links),
        "group_count": n_groups,
    }


def read_polylines(
    store_path: str,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    group_ids: list[int] | None = None,
    bbox: BoundingBox | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read polylines/streamlines from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        object_ids: Optional list of polyline (object) IDs.
        group_ids: Optional group IDs — expands to their object IDs.
        bbox: Optional bounding box filter. Returns polylines that have
            at least one segment in a matching chunk.
        chunks: Optional whitelist of chunk coordinate tuples. Unlike
            ``bbox`` (which keeps the whole polyline if any segment
            matches), ``chunks`` crops at the *segment* level: only
            fragments stored in listed chunks are returned, and each
            surviving contiguous run becomes its own output polyline.
            A polyline whose middle segments lie in unlisted chunks may
            therefore appear in the result as multiple shorter polylines
            — the output ``polyline_count`` can exceed the input
            ``object_count``. AND-ed with ``bbox`` when both are given
            (the effective whitelist is the intersection of the two
            chunk sets). ``chunks=[]`` yields an empty result;
            ``chunks=None`` (default) applies no chunk filter.

    Returns:
        Dict with:
        - ``polylines``: list of lists of arrays. ``polylines[i]`` is
          a list of segment arrays for polyline i (concatenate for full path).
        - ``polyline_count``: number of polylines returned.
        - ``vertex_count``: total vertices across all returned polylines.
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

    # attribute_filter pre-resolution: convert (name, value) into the
    # leading chunk bin index.  Only sub-segments whose chunk key starts
    # with this bin will be returned.
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
                "attribute_filter requires a store written with "
                "chunk_by_attribute"
            )
        if len(attribute_filter) != 1:
            raise ArrayError(
                "attribute_filter must specify exactly one attribute"
            )
        fname, fvalue = next(iter(attribute_filter.items()))
        if fname != lm.chunk_attribute_name:
            raise ArrayError(
                f"attribute_filter key {fname!r} does not match the "
                f"store's chunk_attribute_name "
                f"{lm.chunk_attribute_name!r}"
            )
        try:
            filter_bin = lm.chunk_attribute_values.index(fvalue)
        except ValueError:
            return _empty_polyline_result()

    # Resolve group_ids → object_ids
    if group_ids is not None:
        resolved: set[int] = set()
        for gid in group_ids:
            members = read_group_object_ids(level_group, gid)
            resolved.update(members)
        if object_ids is not None:
            resolved &= set(object_ids)
        object_ids = sorted(resolved)

    # If no filter, read all objects
    if object_ids is None:
        try:
            meta = level_group.read_array_meta("object_index")
            object_ids = list(range(meta["num_objects"]))
        except Exception:
            return _empty_polyline_result()

    # If bbox, find which chunks are relevant
    target_chunks: set[ChunkCoords] | None = None
    if bbox is not None:
        target_chunks = set(chunks_intersecting_bbox(
            np.asarray(bbox[0]), np.asarray(bbox[1]),
            level_chunk_shape,
        ))

    # Explicit chunks whitelist switches read_polylines into segment-level
    # crop mode. When both `chunks` and `bbox` are given, the effective
    # whitelist is the intersection of the two chunk sets.
    chunk_whitelist: set[ChunkCoords] | None = None
    if chunks is not None:
        chunk_whitelist = set(
            resolve_chunk_keys(
                level_group, level_chunk_shape,
                bbox=bbox, chunks=chunks,
            )
        )

    result_polylines: list[list[npt.NDArray]] = []
    total_verts = 0

    # Prefetch every vertex chunk (and its offsets sidecar) + the
    # object_index sidecar in one async gather.  Including OBJECT_INDEX
    # is critical: without it, ``read_all_object_manifests`` would re-
    # read it for every call, and ``read_object_vertices`` / the bbox
    # branch below would re-read it once per oid — making the loop
    # O(N²).  With prefetch, all subsequent decode calls are cache hits.
    # The OBJECT_INDEX entry serves legacy ``data``/``offsets`` stores;
    # ``vlen_manifests_v1`` stores read the ragged ``manifests`` array
    # directly (one chunk per request) and the prefetch is a harmless
    # no-op for them.
    chunk_key_strs = [
        ".".join(str(c) for c in cc)
        for cc in list_chunk_keys(level_group, VERTICES)
    ]
    prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        (OBJECT_INDEX, ["data", "offsets"]),
    ]

    _batched_reads_cm = level_group.batched_reads(prefetch_plan)
    _batched_reads_cm.__enter__()
    try:
        # Read all manifests once.  The per-object loop below indexes
        # into this list — no per-iteration manifest read.
        try:
            manifests = read_all_object_manifests(level_group)
        except Exception:
            manifests = []

        # Decode every chunk's fragments exactly once.  The per-
        # object dispatch then slices from this cache — replacing
        # O(K_per_chunk · N_per_chunk) per-call ``_read_vertex_offsets``
        # work with O(K_per_chunk) per chunk.
        chunk_cache: dict[ChunkCoords, list[npt.NDArray]] = {}
        list_chunks = [
            tuple(int(c) for c in cc.split("."))
            for cc in chunk_key_strs
        ]
        for cc in list_chunks:
            try:
                chunk_cache[cc] = read_chunk_vertices(
                    level_group, cc, dtype=dtype, ndim=ndim,
                )
            except ArrayError:
                chunk_cache[cc] = []

        def _read_fragment(cc: ChunkCoords, fragment_idx: int) -> npt.NDArray | None:
            groups = chunk_cache.get(cc)
            if groups is None:
                return None
            if 0 <= fragment_idx < len(groups):
                return groups[fragment_idx]
            return None

        for oid in object_ids:
            # ------------------------------------------------------------------
            # Segment-level crop mode (chunks=).
            # ------------------------------------------------------------------
            if oid < 0 or oid >= len(manifests):
                continue
            obj_manifest = manifests[oid]
            if not obj_manifest:
                continue

            if chunk_whitelist is not None:
                if filter_bin is not None:
                    obj_manifest = [
                        (cc, fragment_index) for (cc, fragment_index) in obj_manifest
                        if cc and cc[0] == filter_bin
                    ]

                # Split the manifest into runs of consecutive entries
                # whose chunk lies in the whitelist.  Each surviving run
                # becomes its own output polyline.
                run: list[FragmentRef] = []
                for cc, fragment_idx in obj_manifest:
                    if cc in chunk_whitelist:
                        run.append((cc, fragment_idx))
                    else:
                        if run:
                            fragment_list = [
                                fragment for fragment in (_read_fragment(cc, fragment_index) for cc, fragment_index in run)
                                if fragment is not None
                            ]
                            if fragment_list:
                                result_polylines.append(fragment_list)
                                total_verts += sum(len(fragment) for fragment in fragment_list)
                            run = []
                if run:
                    fragment_list = [
                        fragment for fragment in (_read_fragment(cc, fragment_index) for cc, fragment_index in run)
                        if fragment is not None
                    ]
                    if fragment_list:
                        result_polylines.append(fragment_list)
                        total_verts += sum(len(fragment) for fragment in fragment_list)
                continue

            # Whole-object paths (attribute_filter, bbox, no filter).
            if filter_bin is not None:
                matching = [
                    (cc, fragment_index) for (cc, fragment_index) in obj_manifest
                    if cc and cc[0] == filter_bin
                ]
                if not matching:
                    continue
                fragment_list = [
                    fragment for fragment in (_read_fragment(cc, fragment_index) for cc, fragment_index in matching)
                    if fragment is not None
                ]
            else:
                fragment_list = [
                    fragment for fragment in (_read_fragment(cc, fragment_index) for cc, fragment_index in obj_manifest)
                    if fragment is not None
                ]

            if not fragment_list:
                continue

            # Bbox filter: keep the polyline if any of its segments lives
            # in a target chunk.
            if target_chunks is not None:
                has_match = any(
                    cc in target_chunks for cc, _ in obj_manifest
                )
                if not has_match:
                    continue

            result_polylines.append(fragment_list)
            total_verts += sum(len(fragment) for fragment in fragment_list)
    finally:
        _batched_reads_cm.__exit__(None, None, None)

    return {
        "polylines": result_polylines,
        "polyline_count": len(result_polylines),
        "vertex_count": total_verts,
    }


def _read_manifest_run(
    level_group: FsGroup,
    run: list[FragmentRef],
    dtype: np.dtype,
    ndim: int,
) -> list[npt.NDArray]:
    """Read a contiguous manifest slice into a list of fragments.

    Used by ``read_polylines`` in segment-crop mode to materialise each
    surviving run of in-whitelist manifest entries.
    """
    out: list[npt.NDArray] = []
    for cc, fragment_idx in run:
        try:
            out.append(read_fragment(
                level_group, cc, fragment_idx, dtype=dtype, ndim=ndim,
            ))
        except ArrayError:
            continue
    return out


def _empty_polyline_result() -> dict[str, Any]:
    return {
        "polylines": [],
        "polyline_count": 0,
        "vertex_count": 0,
    }
