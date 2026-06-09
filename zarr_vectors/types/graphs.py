"""Graph and skeleton I/O for zarr vectors stores.

Supports two modes:

- **General graph**: arbitrary undirected/directed edges. All edges stored
  explicitly in ``links/`` (intra-chunk) and ``cross_chunk_links/``
  (inter-chunk).  ``links_convention: "explicit"``.

- **Skeleton (tree)**: nodes reordered depth-first, parent links mostly
  sequential.  Only branch points (parent ≠ i−1) stored in ``links/``.
  ``links_convention: "implicit_sequential_with_branches"``.

Edge attributes (weight, type, etc.) are stored in ``link_attributes/``
parallel to the ``links/`` array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    CROSS_CHUNK_LINKS,
    FRAGMENT_ATTRIBUTES,
    GEOM_GRAPH,
    GEOM_SKELETON,
    LINK_FRAGMENTS,
    LINKS,
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    OBJIDX_STANDARD,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_cross_chunk_links_array,
    create_fragment_attribute_array,
    create_link_attributes_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    resolve_chunk_keys,
    read_all_object_manifests,
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
    read_object_vertices,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_link_attributes,
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
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    partition_edges,
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
    CrossChunkLink,
    ObjectManifest,
)


# ===================================================================
# Write
# ===================================================================

def write_graph(
    store_path: str,
    positions: npt.NDArray[np.floating],
    edges: npt.NDArray[np.integer],
    *,
    chunk_shape: ChunkShape,
    bin_shape: BinShape | None = None,
    bounds: tuple[list[float], list[float]] | None = None,
    kind: str = "graph",
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    link_attributes: dict[str, npt.NDArray] | None = None,
    object_attributes: dict[str, npt.NDArray] | None = None,
    fragment_attributes: dict[str, dict[ChunkCoords, npt.NDArray]] | None = None,
    object_ids: npt.NDArray[np.integer] | None = None,
    dtype: str = "float32",
    backend: str | None = None,
    chunk_by_attribute: str | None = None,
    out_of_bounds: str = DEFAULT_OOB_POLICY,
    compressor: Any = None,
    shard_shape: int | tuple[int, ...] | None = None,
    # Deprecated aliases (will be removed):
    is_tree: bool | None = None,
    node_attributes: dict[str, npt.NDArray] | None = None,
    edge_attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Write a graph or skeleton to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        positions: ``(N, D)`` node positions.
        edges: ``(M, 2)`` edge list (global vertex indices).
        chunk_shape: Spatial chunk size per dimension.
        kind: ``"graph"`` (default — general graph, explicit links
            convention) or ``"skeleton"`` (depth-first reorder, implicit
            sequential with branches convention).  Replaces the boolean
            ``is_tree`` kwarg.
        vertex_attributes: Per-node attributes ``{name: (N,) or (N,C)}``.
            (Spec name; replaces ``node_attributes``.)
        link_attributes: Per-edge attributes ``{name: (M,) or (M,C)}``.
            (Spec name; replaces ``edge_attributes``.)
        fragment_attributes: Per-fragment attributes, keyed per chunk:
            ``{name: {chunk_coords: ndarray}}``.  Each per-chunk ndarray
            has shape ``(num_fragments_in_chunk,)`` or
            ``(num_fragments_in_chunk, C)`` and is aligned with the
            fragment ordering this writer produces for that chunk.
        object_ids: ``(N,)`` array assigning nodes to objects.  If None,
            all nodes belong to object 0.
        dtype: Numpy dtype for positions.

    Returns:
        Summary dict.
    """
    # Back-compat: accept the legacy kwarg names.
    if is_tree is not None:
        if kind != "graph":
            raise TypeError(
                "got both `is_tree` and `kind`; pass only `kind`."
            )
        import warnings
        warnings.warn(
            "`is_tree` is deprecated; use `kind=\"skeleton\"` for trees, "
            "`kind=\"graph\"` (default) for general graphs.",
            DeprecationWarning, stacklevel=2,
        )
        kind = "skeleton" if is_tree else "graph"
    if node_attributes is not None:
        if vertex_attributes is not None:
            raise TypeError(
                "got both `node_attributes` and `vertex_attributes`; "
                "pass only `vertex_attributes`."
            )
        import warnings
        warnings.warn(
            "`node_attributes` is deprecated; use `vertex_attributes`.",
            DeprecationWarning, stacklevel=2,
        )
        vertex_attributes = node_attributes
    if edge_attributes is not None:
        if link_attributes is not None:
            raise TypeError(
                "got both `edge_attributes` and `link_attributes`; "
                "pass only `link_attributes`."
            )
        import warnings
        warnings.warn(
            "`edge_attributes` is deprecated; use `link_attributes`.",
            DeprecationWarning, stacklevel=2,
        )
        link_attributes = edge_attributes
    if kind not in ("graph", "skeleton"):
        raise ValueError(
            f"kind must be 'graph' or 'skeleton', got {kind!r}"
        )
    # Internal aliases so the rest of the body stays unchanged.
    is_tree = kind == "skeleton"
    node_attributes = vertex_attributes
    edge_attributes = link_attributes

    np_dtype = np.dtype(dtype)
    positions = np.asarray(positions, dtype=np_dtype)
    edges = np.asarray(edges, dtype=np.int64)
    n_nodes, ndim = positions.shape
    n_edges = len(edges)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ArrayError(f"edges must be (M, 2), got shape {edges.shape}")

    # When attribute-chunking, each node defaults to its own object so
    # the per-object uniformity check below is trivially satisfied and
    # the prefixed layout still groups multiple nodes per (bin, chunk).
    if object_ids is None:
        if chunk_by_attribute is not None:
            object_ids = np.arange(n_nodes, dtype=np.int64)
        else:
            object_ids = np.zeros(n_nodes, dtype=np.int64)

    # For skeletons: reorder depth-first
    reorder_map: npt.NDArray | None = None
    if is_tree:
        positions, edges, node_attributes, edge_attributes, reorder_map = (
            _reorder_tree(positions, edges, node_attributes, edge_attributes)
        )
        # Remap object_ids
        if reorder_map is not None:
            inv = np.empty_like(reorder_map)
            inv[reorder_map] = np.arange(len(reorder_map))
            object_ids = object_ids[reorder_map]

    geometry_type = GEOM_SKELETON if is_tree else GEOM_GRAPH
    links_conv = LINKS_IMPLICIT_BRANCHES if is_tree else LINKS_EXPLICIT

    # Compute bounds — explicit `bounds=` kwarg overrides input-data extent.
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
    if out_of_bounds == "ignore":
        raise ArrayError(
            "out_of_bounds='ignore' is not supported for write_graph: "
            "edges reference node indices and would be left dangling. "
            "Use 'raise' (default) or 'expand'."
        )
    _apply_out_of_bounds_policy(root, positions, policy=out_of_bounds)

    root_meta = _ensure_root_metadata_for_write(
        root,
        inferred_ndim=ndim,
        geometry_type=geometry_type,
        base_bin_shape=bin_shape,
        links_convention=links_conv,
        object_index_convention=OBJIDX_STANDARD,
        cross_chunk_strategy=CROSS_CHUNK_EXPLICIT,
    )
    axes = root_meta.spatial_index_dims

    # Attribute chunking: per-object uniformity required.  Compute the
    # per-node bin from node_attributes[chunk_by_attribute], then check
    # that all nodes of every object share a single value.
    node_attr_bins: npt.NDArray[np.int64] | None = None
    attr_bin_values: list[Any] | None = None
    if chunk_by_attribute is not None:
        if not node_attributes or chunk_by_attribute not in node_attributes:
            raise ArrayError(
                f"chunk_by_attribute={chunk_by_attribute!r} must name a "
                f"key in `node_attributes`"
            )
        src_values = np.asarray(node_attributes[chunk_by_attribute])
        if src_values.ndim != 1 or src_values.shape[0] != n_nodes:
            raise ArrayError(
                f"node_attributes[{chunk_by_attribute!r}] must be 1D of "
                f"length n_nodes={n_nodes}, got shape {src_values.shape}"
            )
        node_attr_bins, attr_bin_values = assign_attribute_bins(src_values)
        for oid in np.unique(object_ids):
            mask = object_ids == oid
            unique_bins = np.unique(node_attr_bins[mask])
            if len(unique_bins) > 1:
                raise ArrayError(
                    f"chunk_by_attribute={chunk_by_attribute!r} requires "
                    f"per-object uniformity for graphs; object {int(oid)} "
                    f"has {len(unique_bins)} distinct attribute values"
                )

    level_chunk_dims: list[str] | None = None
    if chunk_by_attribute is not None:
        level_chunk_dims = compute_chunk_dim_names(
            chunk_by_attribute, ndim,
            spatial_dim_names=[a["name"] for a in axes],
        )

    arrays_present_list = [VERTICES, "links", "object_index"]
    if fragment_attributes:
        arrays_present_list.append(FRAGMENT_ATTRIBUTES)
    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_nodes,
        arrays_present=arrays_present_list,
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)
    link_width = 2

    # Assign nodes to chunks.  When attribute-chunked, prefix each
    # spatial chunk key with the per-node bin so a single spatial cell
    # holding nodes from multiple bins splits into multiple entries.
    chunk_assignments = assign_chunks(positions, chunk_shape)
    if node_attr_bins is not None:
        prefixed_assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
        for spatial_cc, gi in chunk_assignments.items():
            gi = np.asarray(gi, dtype=np.int64)
            chunk_bins = node_attr_bins[gi]
            for ab in np.unique(chunk_bins):
                mask = chunk_bins == ab
                prefixed_assignments[(int(ab),) + spatial_cc] = gi[mask]
        chunk_assignments = prefixed_assignments
    chunk_list = sorted(chunk_assignments.keys())

    vertex_chunks, vertex_local, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_nodes, chunk_list
    )

    # For skeletons with implicit branches: extract only branch links
    if is_tree:
        edges_to_store, edge_attr_to_store = _extract_branch_links(
            edges, edge_attributes
        )
    else:
        edges_to_store = edges
        edge_attr_to_store = edge_attributes

    # Partition edges
    intra_edges, cross_links = partition_edges(
        edges_to_store, vertex_chunks, vertex_local, chunk_list
    )

    # Record which original edges (rows in edges_to_store) landed
    # intra-chunk in each chunk, so per-edge attributes can be sliced
    # in the same order partition_edges built `intra_edges`.
    intra_edge_orig_indices: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    if edge_attr_to_store and edge_attributes:
        e_src_chunk = vertex_chunks[edges_to_store[:, 0]]
        e_dst_chunk = vertex_chunks[edges_to_store[:, 1]]
        intra_mask = e_src_chunk == e_dst_chunk
        intra_orig_idx = np.where(intra_mask)[0]
        intra_chunk_of = e_src_chunk[intra_mask]
        for chunk_idx in np.unique(intra_chunk_of):
            sel = intra_chunk_of == chunk_idx
            coord = chunk_list[int(chunk_idx)]
            intra_edge_orig_indices[coord] = intra_orig_idx[sel]

    # Also partition the full edge set to get cross-chunk links for ALL edges
    # (not just branch links for skeletons).  We only need the cross-links
    # here — the intra branch links come from the partition above — so skip
    # the redundant intra-dict build over the full edge set.
    if is_tree:
        _, all_cross_links = partition_edges(
            edges, vertex_chunks, vertex_local, chunk_list,
            include_intra=False,
        )
    else:
        all_cross_links = cross_links

    # Write vertices per chunk (one fragment per object per chunk)
    object_manifests: dict[int, ObjectManifest] = {}
    idx_ndim = ndim + 1 if node_attr_bins is not None else ndim

    # Collapse all per-array zarr.json + per-chunk byte writes into one
    # asyncio.gather (mirrors points.py:300).  ``shard_shape`` also
    # activates native ``sharding_indexed`` for per-chunk arrays.
    from zarr_vectors.core.arrays import open_write_session
    with open_write_session(
        level_group, compressor=compressor, shard_shape=shard_shape,
        bounds=bounds_list, chunk_shape=chunk_shape,
    ):
        create_vertices_array(level_group, dtype=dtype)
        create_links_array(level_group, link_width=link_width, delta=0)
        create_object_index_array(level_group)
        create_cross_chunk_links_array(level_group, delta=0)
        if node_attributes:
            for name, data in node_attributes.items():
                create_attribute_array(level_group, name, dtype=str(data.dtype))
        if edge_attributes:
            for name, data in edge_attributes.items():
                create_link_attributes_array(
                    level_group, name, dtype=str(data.dtype), delta=0,
                )
        if object_attributes:
            for _name in object_attributes:
                create_object_attributes_array(level_group, _name)

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

        for chunk_idx, chunk_coords in enumerate(chunk_list):
            global_indices = chunk_assignments[chunk_coords]
            chunk_positions = positions[global_indices]
            chunk_obj_ids = object_ids[global_indices]

            unique_objs = np.unique(chunk_obj_ids)
            vert_groups: list[npt.NDArray] = []
            attr_groups: dict[str, list[npt.NDArray]] = {}
            if node_attributes:
                for name in node_attributes:
                    attr_groups[name] = []

            for obj_id in unique_objs:
                mask = chunk_obj_ids == obj_id
                vert_groups.append(chunk_positions[mask])
                oid = int(obj_id)
                if oid not in object_manifests:
                    object_manifests[oid] = []
                object_manifests[oid].append((chunk_coords, len(vert_groups) - 1))

                if node_attributes:
                    obj_global = global_indices[mask]
                    for name, data in node_attributes.items():
                        attr_groups[name].append(data[obj_global])

            write_chunk_vertices(level_group, chunk_coords, vert_groups, dtype=np_dtype)

            if node_attributes:
                for name, groups_list in attr_groups.items():
                    write_chunk_attributes(
                        level_group, name, chunk_coords, groups_list,
                        dtype=node_attributes[name].dtype,
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

        # Write intra-chunk links
        for chunk_coords in chunk_list:
            if chunk_coords in intra_edges:
                local_edges = intra_edges[chunk_coords]
                # One link group per chunk (all edges in one group)
                write_chunk_links(level_group, chunk_coords, [local_edges], delta=0)

                if edge_attr_to_store and edge_attributes:
                    orig_idx = intra_edge_orig_indices.get(chunk_coords)
                    if orig_idx is not None and len(orig_idx) > 0:
                        for name, data in edge_attr_to_store.items():
                            write_chunk_link_attributes(
                                level_group, name, chunk_coords,
                                [np.asarray(data[orig_idx])],
                                dtype=data.dtype,
                                delta=0,
                            )

        # Write cross-chunk links — widen sid_ndim when prefixed.
        if all_cross_links:
            write_cross_chunk_links(
                level_group, all_cross_links, sid_ndim=idx_ndim, delta=0,
            )

        # Write object index
        write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

        # Per-object attributes (Tier B): {name: (O,) or (O, C)}.
        if object_attributes:
            for _name, _data in object_attributes.items():
                write_object_attributes(level_group, _name, np.asarray(_data))

    _finalize_write(root, "write_graph" if not is_tree else "write_skeleton")
    return {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "chunk_count": len(chunk_list),
        "intra_edge_count": sum(len(e) for e in intra_edges.values()),
        "cross_edge_count": len(all_cross_links),
        "object_count": len(object_manifests),
        "kind": "skeleton" if is_tree else "graph",
    }


# ===================================================================
# Read
# ===================================================================

def read_graph(
    store_path: str,
    *,
    level: int = 0,
    object_ids: list[int] | None = None,
    bbox: BoundingBox | None = None,
    chunks: list[ChunkCoords] | None = None,
    attribute_filter: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Read a graph or skeleton from a zarr vectors store.

    Args:
        store_path: Path to the store.
        level: Resolution level.
        object_ids: Optional object ID filter.
        bbox: Optional bounding box filter.
        chunks: Optional whitelist of chunk coordinate tuples; only nodes
            (and intra-chunk edges) in those chunks are returned. AND-ed
            with ``bbox`` and ``object_ids``. Edges spanning a listed
            chunk and an unlisted chunk are dropped. ``chunks=[]`` yields
            an empty result; ``chunks=None`` (default) applies no filter.

    Returns:
        Dict with:
        - ``positions``: ``(N, D)`` node positions
        - ``edges``: ``(M, 2)`` edge list (remapped to output indices)
        - ``node_count``, ``edge_count``
    """
    root = open_store(store_path, backend=backend)
    root_meta = read_root_metadata(root)
    level_group = get_resolution_level(root, level)
    ndim = root_meta.sid_ndim
    is_tree = root_meta.links_convention == LINKS_IMPLICIT_BRANCHES

    # Per-level chunk_shape may override root (v0.7+).
    try:
        level_meta_for_cs = read_level_metadata(root, level)
    except Exception:
        level_meta_for_cs = None
    level_chunk_shape = get_level_chunk_shape(root_meta, level_meta_for_cs)

    dtype = np.float32
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
    except Exception:
        pass

    # Get link width
    link_width = 2
    try:
        lmeta = level_group.read_array_meta("links/0")
        link_width = lmeta.get("link_width", 2)
    except Exception:
        pass

    # Determine chunks to read (physical keys ∩ bbox-implied ∩ chunks
    # whitelist).
    chunk_keys = resolve_chunk_keys(
        level_group, level_chunk_shape, bbox=bbox, chunks=chunks,
    )

    # attribute_filter: drop chunks whose leading coord doesn't match.
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
            return _empty_graph_result(ndim)
        chunk_keys = [k for k in chunk_keys if k and k[0] == filter_bin]

    # Prefetch every chunk (vertices, offsets, edges) and the cross-chunk
    # edges in one async gather.  Subsequent ``read_bytes`` calls below
    # hit the cache instead of paying one round-trip per chunk.
    _chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_keys]
    _ccl_family = f"{CROSS_CHUNK_LINKS}/0"
    _ccl_cell_keys = (
        level_group.list_chunks(_ccl_family)
        if level_group.array_exists(_ccl_family) else []
    )
    _prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, _chunk_key_strs),
        (VERTEX_FRAGMENTS, _chunk_key_strs),
        (f"{LINKS}/0", _chunk_key_strs),
        (LINK_FRAGMENTS, _chunk_key_strs),
        (_ccl_family, _ccl_cell_keys),
    ]
    _batched_reads_cm = level_group.batched_reads(_prefetch_plan)
    _batched_reads_cm.__enter__()
    try:
        # Single O(K) pass: build chunk→global-offset map and concatenate
        # all fragments.  The previous implementation walked
        # ``chunk_keys`` three times (once for positions, once nested to
        # recompute offsets for each chunk = O(K²), once for cross-chunk
        # edges) — all redundant.
        chunk_offsets: dict[ChunkCoords, int] = {}
        all_positions: list[npt.NDArray] = []
        running = 0
        for chunk_coords in chunk_keys:
            chunk_offsets[chunk_coords] = running
            try:
                groups = read_chunk_vertices(
                    level_group, chunk_coords, dtype=dtype, ndim=ndim
                )
            except ArrayError:
                continue
            for fragment in groups:
                all_positions.append(fragment)
                running += len(fragment)

        if not all_positions:
            return _empty_graph_result(ndim)

        positions_out = np.concatenate(all_positions, axis=0)

        # Intra-chunk edges: O(1) offset lookup, vectorized remap.
        all_edges: list[npt.NDArray] = []
        for chunk_coords in chunk_keys:
            try:
                link_groups = read_chunk_links(
                    level_group, chunk_coords, link_width=link_width, delta=0,
                )
            except ArrayError:
                continue
            chunk_offset = chunk_offsets[chunk_coords]
            for lg in link_groups:
                if len(lg) > 0:
                    all_edges.append(
                        (lg.astype(np.int64, copy=False) + chunk_offset)
                    )

        # Cross-chunk edges (delta=0; cross-pyramid-level edges live
        # under delta != 0 and are not part of a single-level read).
        # Vectorized remap: build offset/local arrays first, then a
        # single column_stack — avoids the per-edge ``np.array([[ga, gb]])``
        # allocation in the original loop.
        try:
            ccl = read_cross_chunk_links(level_group, delta=0)
            if ccl:
                off_a: list[int] = []
                off_b: list[int] = []
                vi_a_list: list[int] = []
                vi_b_list: list[int] = []
                for (chunk_a, vi_a), (chunk_b, vi_b) in ccl:
                    oa = chunk_offsets.get(chunk_a)
                    ob = chunk_offsets.get(chunk_b)
                    if oa is None or ob is None:
                        continue
                    off_a.append(oa)
                    off_b.append(ob)
                    vi_a_list.append(int(vi_a))
                    vi_b_list.append(int(vi_b))
                if off_a:
                    remapped_cross = np.column_stack([
                        np.asarray(off_a, dtype=np.int64) + np.asarray(vi_a_list, dtype=np.int64),
                        np.asarray(off_b, dtype=np.int64) + np.asarray(vi_b_list, dtype=np.int64),
                    ])
                    all_edges.append(remapped_cross)
        except (ArrayError, Exception):
            pass

        # For skeletons: reconstruct full edge set from implicit sequential + branch links
        if is_tree:
            total_nodes = len(positions_out)
            # Start with implicit parents: parent[i] = i-1, parent[0] = -1
            parent_arr = np.arange(-1, total_nodes - 1, dtype=np.int64)
            # Vectorized branch-link override: one fancy-index per block.
            for edge_block in all_edges:
                if len(edge_block) == 0:
                    continue
                children = edge_block[:, 0].astype(np.int64, copy=False)
                parents = edge_block[:, 1].astype(np.int64, copy=False)
                valid = (children >= 0) & (children < total_nodes)
                if valid.any():
                    parent_arr[children[valid]] = parents[valid]
            # Build edge list from parent array
            all_edges = []
            mask = parent_arr >= 0
            children_idx = np.flatnonzero(mask)
            if len(children_idx) > 0:
                tree_edges = np.stack(
                    [children_idx, parent_arr[children_idx]], axis=1
                ).astype(np.int64)
                all_edges.append(tree_edges)

        if all_edges:
            edges_out = np.concatenate(all_edges, axis=0)
        else:
            edges_out = np.zeros((0, 2), dtype=np.int64)

        # Filter by bbox on positions — vectorized edge remap via LUT.
        if bbox is not None:
            bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
            node_mask = np.all(
                (positions_out >= bbox_min) & (positions_out <= bbox_max),
                axis=1,
            )
            if not np.all(node_mask):
                n_total = len(positions_out)
                keep_indices = np.flatnonzero(node_mask)
                positions_out = positions_out[keep_indices]
                if len(edges_out) > 0:
                    remap = np.full(n_total, -1, dtype=np.int64)
                    remap[keep_indices] = np.arange(
                        len(keep_indices), dtype=np.int64,
                    )
                    new_src = remap[edges_out[:, 0]]
                    new_dst = remap[edges_out[:, 1]]
                    keep_mask = (new_src >= 0) & (new_dst >= 0)
                    edges_out = np.stack(
                        [new_src[keep_mask], new_dst[keep_mask]], axis=1,
                    ).astype(np.int64, copy=False)
                else:
                    edges_out = np.zeros((0, 2), dtype=np.int64)

        return {
            "positions": positions_out,
            "edges": edges_out,
            "node_count": len(positions_out),
            "edge_count": len(edges_out),
        }
    finally:
        _batched_reads_cm.__exit__(None, None, None)


def _empty_graph_result(ndim: int) -> dict[str, Any]:
    return {
        "positions": np.zeros((0, ndim), dtype=np.float32),
        "edges": np.zeros((0, 2), dtype=np.int64),
        "node_count": 0,
        "edge_count": 0,
    }


# ===================================================================
# Tree helpers
# ===================================================================

def _reorder_tree(
    positions: npt.NDArray,
    edges: npt.NDArray,
    node_attributes: dict[str, npt.NDArray] | None,
    edge_attributes: dict[str, npt.NDArray] | None,
) -> tuple[
    npt.NDArray, npt.NDArray,
    dict[str, npt.NDArray] | None,
    dict[str, npt.NDArray] | None,
    npt.NDArray,
]:
    """Reorder nodes depth-first from root for skeleton storage.

    Returns reordered (positions, edges, node_attrs, edge_attrs, reorder_map)
    where reorder_map[new_idx] = old_idx.
    """
    n = len(positions)

    # Build adjacency from edges via numpy: sort by parent, then slice
    # per-parent runs.  The previous ``for e in edges`` loop was ~O(M)
    # Python iterations; this is one ``argsort`` + per-unique-parent
    # ``.tolist()`` slice.
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    if len(edges) > 0:
        e_children = edges[:, 0].astype(np.int64, copy=False)
        e_parents = edges[:, 1].astype(np.int64, copy=False)
        sort_idx = np.argsort(e_parents, kind="stable")
        sorted_parents = e_parents[sort_idx]
        sorted_children = e_children[sort_idx]
        unique_parents, start_indices = np.unique(
            sorted_parents, return_index=True,
        )
        end_indices = np.concatenate(
            [start_indices[1:], np.array([len(sorted_parents)], dtype=np.int64)]
        )
        for parent, lo, hi in zip(
            unique_parents.tolist(),
            start_indices.tolist(),
            end_indices.tolist(),
        ):
            children[parent] = sorted_children[lo:hi].tolist()
    child_nodes_arr = (
        edges[:, 0].astype(np.int64, copy=False) if len(edges) > 0 else np.array([], dtype=np.int64)
    )

    # Find root (node with no parent) — vectorized set difference.
    is_child = np.zeros(n, dtype=bool)
    if len(child_nodes_arr) > 0:
        in_range = (child_nodes_arr >= 0) & (child_nodes_arr < n)
        is_child[child_nodes_arr[in_range]] = True
    root_candidates = np.flatnonzero(~is_child)
    root_node = int(root_candidates[0]) if len(root_candidates) > 0 else 0

    # DFS ordering
    order: list[int] = []
    stack = [root_node]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # Push children in reverse so first child is processed first
        for child in reversed(children.get(node, [])):
            if child not in visited:
                stack.append(child)

    # Add any unvisited nodes (disconnected components)
    for i in range(n):
        if i not in visited:
            order.append(i)

    reorder_map = np.array(order, dtype=np.int64)
    inv_map = np.empty(n, dtype=np.int64)
    inv_map[reorder_map] = np.arange(n)

    # Reorder positions
    new_positions = positions[reorder_map]

    # Remap edges
    new_edges = np.empty_like(edges)
    new_edges[:, 0] = inv_map[edges[:, 0]]
    new_edges[:, 1] = inv_map[edges[:, 1]]

    # Reorder node attributes
    new_node_attrs = None
    if node_attributes:
        new_node_attrs = {
            name: data[reorder_map] for name, data in node_attributes.items()
        }

    return new_positions, new_edges, new_node_attrs, edge_attributes, reorder_map


def _extract_branch_links(
    edges: npt.NDArray,
    edge_attributes: dict[str, npt.NDArray] | None,
) -> tuple[npt.NDArray, dict[str, npt.NDArray] | None]:
    """Extract only non-sequential (branch) links from a tree edge list.

    In a depth-first ordered tree, most edges have child = parent + 1.
    Only branch points (child's parent ≠ child - 1) need explicit storage.

    Args:
        edges: ``(M, 2)`` where each row is ``[child, parent]``.
        edge_attributes: Optional per-edge attributes.

    Returns:
        branch_edges: subset of edges where parent ≠ child - 1.
        branch_edge_attributes: corresponding subset of attributes.
    """
    if len(edges) == 0:
        return edges, edge_attributes

    children = edges[:, 0]
    parents = edges[:, 1]

    # Non-sequential: parent != child - 1
    is_branch = parents != (children - 1)
    branch_edges = edges[is_branch]

    branch_attrs = None
    if edge_attributes:
        branch_attrs = {
            name: data[is_branch] for name, data in edge_attributes.items()
        }

    return branch_edges, branch_attrs
