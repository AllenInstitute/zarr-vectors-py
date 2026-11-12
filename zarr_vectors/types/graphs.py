"""Graph and skeleton I/O for zarr vectors stores.

Supports two modes:

- **General graph**: arbitrary undirected/directed edges.  Every edge is
  stored explicitly in ``links/0/<offsets>/`` — the all-zero offsets
  array holds the intra-chunk edges, the rest hold the boundary-crossing
  ones.  ``links_convention: "explicit"``.

- **Skeleton (tree)**: nodes reordered depth-first, parent links mostly
  sequential.  Only branch points (parent ≠ i−1) need an intra-chunk
  record, but *every* boundary-crossing edge is still explicit — implicit
  sequential order does not span chunks.  A vertex with no record takes
  the row before it as parent unless it opens a fragment, so the writer
  opens each fragment with its parentless vertex, if any, and stores
  every parent edge of a vertex that has several; see
  :func:`_skeleton_edges` for the read rule.
  ``links_convention: "implicit_sequential_with_branches"``.

Edge attributes (weight, type, etc.) are stored in ``link_attributes/``
mirroring the ``links/`` family cell for cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    DEFAULT_OOB_POLICY,
    FRAGMENT_ATTRIBUTES,
    GEOM_GRAPH,
    GEOM_SKELETON,
    LINK_FRAGMENTS,
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    OBJECT_INDEX,
    OBJIDX_STANDARD,
    RESOLUTION_PREFIX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_fragment_attribute_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    link_cells_within,
    list_link_offsets,
    read_chunk_vertices,
    read_link_rows,
    resolve_chunk_keys,
    stamp_fragments_tile,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_vertices,
    write_link_attributes,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.attr_chunking import (
    assign_attribute_bins,
    compute_chunk_dim_names,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    get_level_chunk_shape,
)
from zarr_vectors.core.paths import links_group_path
from zarr_vectors.core.store import (
    FsGroup,
    _apply_out_of_bounds_policy,
    _create_or_open_store,
    _ensure_root_metadata_for_write,
    _finalize_write,
    create_resolution_level,
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.spatial.boundary import build_vertex_chunk_mapping
from zarr_vectors.spatial.chunking import (
    assign_chunks,
    compute_bounds,
)
from zarr_vectors.types.meshes import _bbox_filter, read_object_links
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    ObjectManifest,
)

if TYPE_CHECKING:
    from zarr_vectors.core.store import ReadSource, WriteTarget


# ===================================================================
# Write
# ===================================================================

def write_graph(
    store_path: WriteTarget,
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
    shard_shape: int | tuple[int, ...] | None | Literal["inherit"] = "inherit",
    # Deprecated aliases (will be removed):
    is_tree: bool | None = None,
    node_attributes: dict[str, npt.NDArray] | None = None,
    edge_attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Write a graph or skeleton to a new zarr vectors store.

    Args:
        store_path: Path for the new store.
        positions: ``(N, D)`` node positions.
        edges: ``(M, 2)`` edge list (global vertex indices).  For a
            skeleton each row is ``[child, parent]``, and a parent of
            ``-1`` (SWC style) marks the child a root -- no edge.
        chunk_shape: Spatial chunk size per dimension.
        kind: ``"graph"`` (default — general graph, explicit links
            convention) or ``"skeleton"`` (depth-first reorder, implicit
            sequential with branches convention).  Replaces the boolean
            ``is_tree`` kwarg.
        vertex_attributes: Per-node attributes ``{name: (N,) or (N,C)}``.
            (Spec name; replaces ``node_attributes``.)
        link_attributes: Per-edge attributes ``{name: (M,) or (M,C)}``.
            (Spec name; replaces ``edge_attributes``.)  With
            ``kind="skeleton"`` they make every edge an explicit record
            (implied ones included), so each edge keeps its value.
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
        # A ``[child, -1]`` row is the SWC way of saying "child is a root",
        # and the skeleton spec's own example writes one.  It is no edge:
        # taken as one, ``-1`` indexed the last vertex and the store held
        # an edge from the root to it.
        if (edges[:, 0] < 0).any() or (edges[:, 1] < -1).any():
            raise ValueError(
                "skeleton edges are [child, parent] vertex indices, with a "
                "parent of -1 marking the child a root; got a negative "
                "child, or a parent below -1"
            )
        no_parent = edges[:, 1] < 0
        if no_parent.any():
            edges = edges[~no_parent]
            n_edges = len(edges)
            if edge_attributes:
                edge_attributes = {
                    name: np.asarray(data)[~no_parent]
                    for name, data in edge_attributes.items()
                }
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

    # Checked before anything is created.  It used to be rejected after
    # ``_create_or_open_store``, which left an empty store on disk for a
    # call that was never going to succeed.
    if out_of_bounds == "ignore":
        raise ArrayError(
            "out_of_bounds='ignore' is not supported for write_graph: "
            "edges reference node indices and would be left dangling. "
            "Use 'raise' (default) or 'expand'."
        )
    root = _create_or_open_store(
        store_path,
        backend=backend,
        bounds=bounds_list,
        chunk_shape=tuple(chunk_shape),
        ndim=ndim,
        # Only meaningful when this call CREATES the store; an existing
        # one keeps its own declaration. "inherit" means the caller said
        # nothing, so there is nothing to declare on a fresh store.
        shard_shape=None if shard_shape == "inherit" else shard_shape,
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

    # Group each chunk's vertices by object BEFORE the local-index mapping
    # is built.  The write loop below emits one fragment per object per
    # chunk, in ``np.unique`` order, and a link's chunk-local index is an
    # offset into that concatenation -- so a mapping built from the
    # unregrouped order numbers vertices differently from the way they are
    # actually stored, and every edge in a chunk holding more than one
    # object then points at the wrong vertex.  ``meshes.py`` already does
    # this, for the same reason and with the same comment.
    #
    # A skeleton also puts each fragment's parentless vertex first.  The
    # reader gives every vertex but a fragment's first the row before it
    # as parent, unless a record says otherwise -- and a vertex with no
    # parent has no record to say so.  At a fragment start it needs none;
    # anywhere else it read back joined to its predecessor.  Depth-first
    # order puts it there already when an object is one whole tree, but
    # not when the object's other vertices hang off another object's tree
    # and are visited first.  ``_check_one_root_per_object`` leaves at
    # most one such vertex per object, so per fragment.  Only fragments
    # that need the move are re-sorted.
    n_child_edges: npt.NDArray[np.int64] | None = None
    if is_tree:
        n_child_edges = _child_edge_counts(edges, n_nodes)
        parentless = n_child_edges == 0
    for _cc, _gi in list(chunk_assignments.items()):
        _gi = np.asarray(_gi, dtype=np.int64)
        if _gi.size > 1:
            _gi = _gi[np.argsort(object_ids[_gi], kind="stable")]
            if is_tree:
                _gi = _parentless_first(_gi, object_ids[_gi], parentless[_gi])
        chunk_assignments[_cc] = _gi
    chunk_list = sorted(chunk_assignments.keys())

    vertex_chunks, vertex_local, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_nodes, chunk_list
    )

    # Read order: the row each vertex will occupy once a reader
    # concatenates this level's chunks (``chunk_list`` order) and, within
    # each chunk, its fragments.  The implicit-sequential convention is
    # stated in that frame, so the store/skip decision below has to be
    # made in it too.
    _chunk_sizes = np.array(
        [len(chunk_assignments[cc]) for cc in chunk_list], dtype=np.int64,
    )
    _chunk_start = np.zeros(len(chunk_list), dtype=np.int64)
    if len(chunk_list) > 1:
        _chunk_start[1:] = np.cumsum(_chunk_sizes[:-1])
    read_idx = _chunk_start[vertex_chunks] + vertex_local

    # A fragment's first vertex has no implicit predecessor: the row
    # before it belongs to another object or another chunk, so a reader
    # cannot assume it is the parent.
    is_fragment_start = np.zeros(n_nodes, dtype=bool)
    for _cc in chunk_list:
        _gi = chunk_assignments[_cc]
        if _gi.size == 0:
            continue
        _oids = object_ids[_gi]
        _starts = np.concatenate(([True], _oids[1:] != _oids[:-1]))
        is_fragment_start[_gi[_starts]] = True

    # Select the edges that need an explicit record, then hand them to
    # ``write_links`` in global ``(chunk, vertex)`` form and let it route
    # each to the offsets array that names where its endpoints sit.  A
    # record is intra-chunk exactly when both endpoints share a chunk —
    # the all-zero-offsets case (``paths.is_intra``).
    e_chunk = vertex_chunks[edges]                      # (M, 2)
    is_cross = e_chunk[:, 0] != e_chunk[:, 1]
    if is_tree:
        # Store exactly the edges the reader's implicit rule does not
        # already give it.  That rule is ``parent[i] = i - 1`` over the
        # concatenated read order, suspended at a fragment start -- so an
        # edge is implied precisely when the child sits one row after its
        # parent AND is not itself a fragment start.
        #
        # This replaces ``is_cross | _branch_mask(edges)``, which was
        # stated in DFS-index space.  DFS order and read order diverge
        # whenever a chunk sorts ahead of the one holding the root, and
        # there the old rule dropped a real edge and let the root inherit
        # the previous chunk's last vertex as its parent -- reading back
        # an edge nobody ever wrote.
        #
        # And only when it is the child's ONE parent edge.  The reader
        # gives the implicit parent only to a vertex with no record, so a
        # vertex with two parent edges (not a tree, but accepted: a cycle
        # closing on the root, a repeated edge) needs every one stored --
        # an implied one left out was silently replaced by its sibling.
        _child, _parent = edges[:, 0], edges[:, 1]
        implied = (
            (read_idx[_child] == read_idx[_parent] + 1)
            & ~is_fragment_start[_child]
            & (n_child_edges[_child] == 1)
        )
        store_mask = ~implied
        _check_one_root_per_object(edges, object_ids, n_nodes)
        if edge_attributes:
            # ``link_attributes/`` mirrors the stored records row for row,
            # so an implied edge has nowhere to keep its values -- skipping
            # them silently dropped every attribute row but the branch and
            # cross-chunk ones.  Store every edge explicitly instead: a
            # record restating the implied parent is legal under
            # ``implicit_sequential_with_branches`` (readers apply it as an
            # override that changes nothing).  Stores written without edge
            # attributes are unaffected.
            store_mask = np.ones(n_edges, dtype=bool)
    else:
        store_mask = np.ones(n_edges, dtype=bool)
    store_rows = np.flatnonzero(store_mask)

    # Declared once the stored edges are known: ``link_attributes`` is
    # only written when some edge is stored rather than implied.  The
    # links family and its ``link_fragments`` sidecar always exist (the
    # backstop below creates the intra-chunk array).
    arrays_present_list = [
        VERTICES, VERTEX_FRAGMENTS, "links", LINK_FRAGMENTS, "object_index",
    ]
    if node_attributes:
        arrays_present_list.append(VERTEX_ATTRIBUTES)
    if fragment_attributes:
        arrays_present_list.append(FRAGMENT_ATTRIBUTES)
    if edge_attributes and len(store_rows) > 0:
        arrays_present_list.append("link_attributes")
    if object_attributes:
        arrays_present_list.append("object_attributes")
    level_meta = LevelMetadata(
        level=0,
        vertex_count=n_nodes,
        arrays_present=arrays_present_list,
        chunk_dims=level_chunk_dims,
        chunk_attribute_name=chunk_by_attribute,
        chunk_attribute_values=attr_bin_values,
    )
    level_group = create_resolution_level(root, 0, level_meta)

    # Records stay in edge order, which the partitioner preserves within
    # each cell — that is what lets the per-edge attribute rows below be
    # a plain ``[store_rows]`` slice.  Held as arrays -- ``(M, 2, D)``
    # endpoint chunks and ``(M, 2)`` local indices -- rather than one
    # list of tuples per edge.
    link_chunks = np.asarray(chunk_list, dtype=np.int64)[e_chunk[store_rows]]
    link_vi = vertex_local[edges[store_rows]]

    # Write vertices per chunk (one fragment per object per chunk)
    object_manifests: dict[int, ObjectManifest] = {}
    idx_ndim = ndim + 1 if node_attr_bins is not None else ndim

    # Collapse all per-array zarr.json + per-chunk byte writes into one
    # asyncio.gather (mirrors points.py:300).  ``shard_shape`` also
    # activates native ``sharding_indexed`` for per-chunk arrays.
    from zarr_vectors.core.arrays import open_write_session
    # chunk_by_attribute prepends a leading attr-bin axis to every chunk
    # key; size the vlen array grid to match that rank.
    session_bin_count = (
        len(attr_bin_values) if node_attr_bins is not None else None
    )
    with open_write_session(
        level_group, compressor=compressor, shard_shape=shard_shape,
        bounds=bounds_list, chunk_shape=chunk_shape,
        bin_count=session_bin_count,
    ):
        create_vertices_array(level_group, dtype=dtype)
        create_object_index_array(level_group)
        # The links and link_attributes families are created by the writes
        # below, which know the offsets arrays each record lands in.
        if node_attributes:
            for name, data in node_attributes.items():
                # Channel names carry the row shape: without them a (V, C)
                # column was declared scalar and read back as V*C values.
                create_attribute_array(
                    level_group, name, dtype=str(data.dtype),
                    channel_names=(
                        [f"ch{i}" for i in range(data.shape[1])]
                        if np.ndim(data) == 2 else None
                    ),
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

            # One fragment per object, ordered by object id -- built by
            # splitting the chunk at the owner boundaries rather than
            # masking it once per object, which costs objects x vertices
            # for the chunk (500k objects in one cell: 633s).
            #
            # ``chunk_assignments`` was already stably sorted by owner
            # above, so this sort is the identity today; it is kept
            # because ``build_vertex_chunk_mapping`` numbers vertices in
            # exactly this order, and a split that silently disagreed
            # with it would mis-point every edge in a multi-object chunk
            # rather than fail.
            vert_groups: list[npt.NDArray] = []
            attr_groups: dict[str, list[npt.NDArray]] = {}
            if node_attributes:
                for name in node_attributes:
                    attr_groups[name] = []

            if len(global_indices):
                order = np.argsort(chunk_obj_ids, kind="stable")
                ordered_global = global_indices[order]
                unique_objs, first_at = np.unique(
                    chunk_obj_ids[order], return_index=True,
                )
                split_at = first_at[1:]
                vert_groups = list(np.split(chunk_positions[order], split_at))
                for fragment_idx, obj_id in enumerate(unique_objs):
                    object_manifests.setdefault(int(obj_id), []).append(
                        (chunk_coords, fragment_idx),
                    )
                for name, data in (node_attributes or {}).items():
                    attr_groups[name] = list(
                        np.split(data[ordered_global], split_at)
                    )

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

        # One write for every edge, intra and cross alike — sid_ndim is
        # widened when chunk keys carry an attribute-bin prefix.
        partition = write_links(
            level_group, [], idx_ndim, delta=0,
            link_width=link_width, _arrays=(link_chunks, link_vi),
        )
        # Backstop: `arrays_present` advertises the family, and the
        # per-cell editors in ops/ write into an array that must already
        # exist — a tree whose every edge is implied leaves write_links
        # nothing to create.  Must run *after* it: an array present but
        # untargeted makes write_links recount the family off disk, which
        # inside this deferred write session reads back empty.
        create_links_array(
            level_group, link_width=link_width, delta=0, sid_ndim=idx_ndim,
        )

        # Per-edge attributes follow the same partition, so cross-chunk
        # edges get theirs too — before the offset merge only the
        # intra-chunk writer had an attribute counterpart to call.
        if edge_attributes and len(store_rows) > 0:
            for name, data in edge_attributes.items():
                write_link_attributes(
                    level_group, name, np.asarray(data)[store_rows],
                    num_links=partition.num_links, partition=partition,
                    delta=0,
                )

        # Write object index
        write_object_index(level_group, object_manifests, sid_ndim=idx_ndim)

        # Per-object attributes (Tier B): {name: (O,) or (O, C)}.
        if object_attributes:
            for _name, _data in object_attributes.items():
                write_object_attributes(level_group, _name, np.asarray(_data))

        # Record the tiling layout just written, so a later bulk read
    # can return each chunk's buffer without reading its fragment
    # index.  Verified against what is on disk, and stamped after
    # the chunk writes -- see stamp_fragments_tile.
    stamp_fragments_tile(level_group, ndim)
    _finalize_write(root, "write_graph" if not is_tree else "write_skeleton")
    return {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "chunk_count": len(chunk_list),
        # Intra vs cross is now an offsets property: a stored record is
        # intra iff both endpoints share a chunk.  Trees store only their
        # branch edges intra-chunk, hence the `store_mask` term.
        "intra_edge_count": int(np.count_nonzero(store_mask & ~is_cross)),
        "cross_edge_count": int(np.count_nonzero(is_cross)),
        "object_count": len(object_manifests),
        "kind": "skeleton" if is_tree else "graph",
    }


# ===================================================================
# Read
# ===================================================================

def read_graph(
    store_path: ReadSource,
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
        store_path: URL or path to the store, a pre-built zarr Store,
            or an already-open Group.
        level: Resolution level.
        object_ids: Optional object ID filter.  Only the named objects'
            vertex fragments are read -- found through the object index,
            not by scanning chunks -- and only the edges between them.
            Ids not in the store are ignored.  For a graph and a skeleton
            alike the edges are the induced subgraph: an edge comes back
            exactly when both its ends are read, so one between two
            objects needs both asked for.  A skeleton's implicit edges
            are rebuilt within each fragment as on a full read; a vertex
            whose parent was not read (another object's, or outside a
            ``bbox`` / ``chunks`` scope) comes back as a root.
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
        - ``object_ids``: ``(N,)`` owning object of each node -- only
          when ``object_ids=`` was given.  Nodes come back grouped by
          object, in the order the ids were asked for.
    """
    root = open_store(store_path, backend=backend)
    with root.cached_nodes():
        # One node-resolution pass for the whole read, and every
        # node it needs asked for in a single gather rather than
        # resolved one at a time as the code reaches them.
        prefix = f"{RESOLUTION_PREFIX}{level}"
        nodes = [
            prefix,
            f"{prefix}/{VERTICES}",
            f"{prefix}/{VERTEX_FRAGMENTS}",
            f"{prefix}/{LINK_FRAGMENTS}",
        ]
        if object_ids is not None:
            nodes.append(f"{prefix}/{OBJECT_INDEX}")
        root.prime_nodes(nodes)
        return _read_graph(
            root,
            level=level,
            object_ids=object_ids,
            bbox=bbox,
            chunks=chunks,
            attribute_filter=attribute_filter,
        )


def _read_graph(
    root: FsGroup,
    *,
    level: int,
    object_ids: list[int] | None,
    bbox: BoundingBox | None,
    chunks: list[ChunkCoords] | None,
    attribute_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    """Body of :func:`read_graph`, against an already-open store.

    Split out only so the caller can hold a
    :meth:`Group.cached_nodes` block open across the whole read;
    the two halves are one function.
    """
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

    scoped = bbox is not None or chunks is not None or bool(attribute_filter)

    if level_meta_for_cs is not None and level_meta_for_cs.octree_chunks is not None:
        from zarr_vectors.octree_chunks.read import read_graph_octree
        return read_graph_octree(
            root, level=level, bbox=bbox, object_ids=object_ids, chunks=chunks,
            attribute_filter=attribute_filter,
        )

    # Determine chunks to read (physical keys ∩ bbox-implied ∩ chunks
    # whitelist).  An object read finds its chunks through the object
    # index, so an unscoped one needs no chunk listing at all.
    chunk_keys: list[ChunkCoords] | None = None
    if object_ids is None or scoped:
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

    if object_ids is not None:
        return _read_graph_objects(
            level_group,
            object_ids,
            scope=None if chunk_keys is None else set(chunk_keys),
            bbox=bbox,
            dtype=dtype,
            ndim=ndim,
            is_tree=is_tree,
            aligned=bool(level_meta_for_cs and level_meta_for_cs.link_fragments_aligned),
        )
    assert chunk_keys is not None

    # A read narrowed to some chunks keeps only edges wholly inside them,
    # so it reads only the link cells those chunks can reach; an
    # unscoped one wants the whole family.  A skeleton also needs the
    # records that only *touch* them: a read child whose parent was not
    # read is a root here, and only its record says so -- without it the
    # child kept its implicit ``i - 1`` and the read invented an edge.
    link_cells = (
        link_cells_within(level_group, chunk_keys, touching=is_tree)
        if scoped else None
    )
    if link_cells is None:
        link_cells = {
            seg: level_group.list_chunks(f"{links_group_path(0)}/{seg}")
            for seg in list_link_offsets(level_group, 0)
        }

    # Prefetch every chunk (vertices, offsets) and the link cells in one
    # async gather.  Subsequent ``read_bytes`` calls below hit the cache
    # instead of paying one round-trip per chunk.  ``links/0`` is a group
    # of one array per offsets segment, and each array's cells are keyed
    # by *source* chunk, which for the non-zero offsets is not the
    # ``chunk_keys`` set.
    _chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_keys]
    _prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, _chunk_key_strs),
        (VERTEX_FRAGMENTS, _chunk_key_strs),
        (LINK_FRAGMENTS, _chunk_key_strs),
        *((f"{links_group_path(0)}/{seg}", keys) for seg, keys in link_cells.items()),
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
        # Row index of each fragment's first vertex in the concatenation
        # below.  That is where the implicit-sequential convention stops
        # applying -- see the tree reconstruction further down.
        fragment_starts: list[int] = []
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
                if len(fragment):
                    fragment_starts.append(running)
                all_positions.append(fragment)
                running += len(fragment)

        if not all_positions:
            return _empty_graph_result(ndim)

        positions_out = np.concatenate(all_positions, axis=0)

        # Edges (delta=0; cross-pyramid-level edges live under delta != 0
        # and are not part of a single-level read).  One family, so intra
        # and cross records come back from one call already in global
        # ``(chunk, vertex)`` form — a single remap covers both.  A record
        # touching a chunk outside ``chunk_keys`` has no offset and is
        # dropped, which is what applies the bbox/chunks filter to edges.
        all_edges: list[npt.NDArray] = []
        # Each chunk's local rows map to output rows through its offset,
        # resolved once per link cell rather than per endpoint.
        row_of = np.arange(running, dtype=np.int64)
        starts = list(chunk_offsets.items())
        ends = [start for _, start in starts[1:]] + [running]
        rows = read_link_rows(
            level_group,
            {cc: row_of[start:end] for (cc, start), end in zip(starts, ends)},
            delta=0, cells=link_cells, touching=is_tree,
        )
        # A skeleton keeps every record whose child was read; a parent of
        # -1 (not read) makes that child a root below.
        if rows.shape[0] and rows.shape[1] == 2:
            all_edges.append(np.ascontiguousarray(rows))

        # For skeletons: reconstruct full edge set from implicit sequential + branch links
        if is_tree:
            total_nodes = len(positions_out)
            # Implicit parents: parent[i] = i-1 WITHIN a fragment.  At a
            # fragment start there is no predecessor to inherit -- the row
            # before it belongs to another object or another chunk -- so it
            # starts parentless and is filled in only by an explicit
            # record.  Without this a tree whose root did not land on row 0
            # took the previous fragment's last vertex as its parent,
            # returning an edge the writer never stored (and, for a root in
            # a later chunk, one edge too many).
            records = (
                all_edges[0] if len(all_edges) == 1
                else np.concatenate(all_edges, axis=0) if all_edges
                else np.zeros((0, 2), dtype=np.int64)
            )
            valid = (records[:, 0] >= 0) & (records[:, 0] < total_nodes)
            all_edges = [_skeleton_edges(
                total_nodes,
                np.asarray(fragment_starts, dtype=np.int64),
                records[valid].astype(np.int64, copy=False),
            )]

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


def _read_graph_objects(
    level_group: Any,
    object_ids: list[int],
    *,
    scope: set[ChunkCoords] | None,
    bbox: BoundingBox | None,
    dtype: np.dtype,
    ndim: int,
    is_tree: bool,
    aligned: bool = False,
) -> dict[str, Any]:
    """:func:`read_graph` for ``object_ids=``: the named objects and nothing else.

    The rows and link records come from
    :func:`~zarr_vectors.types.meshes.read_object_links`, as for a mesh.
    A graph's records are its edges, kept when both ends are ours.  A
    skeleton's are the edges the implicit rule does not give, so its
    edges are rebuilt as on a full read (:func:`_skeleton_edges`):
    ``parent[i] = i - 1`` within a fragment (a fragment is read whole, so
    its rows keep their order) for a vertex without a record, none at a
    fragment start, and every record otherwise -- no edge, but a root,
    when its parent was not read.  So the records
    are read from every link cell touching the objects' chunks, not only
    those wholly inside them: a record whose parent is in a chunk not
    read (outside the scope, or another object's) is the only thing that
    stops its child inheriting the previous row as parent.
    """
    read = read_object_links(
        level_group, object_ids, scope=scope, dtype=dtype, ndim=ndim,
        link_width=2, aligned=aligned, touching=is_tree,
    )
    if read is None:
        result = _empty_graph_result(ndim)
        result["object_ids"] = np.zeros(0, dtype=np.int64)
        return result
    positions_out, owners_out, rows, starts = read

    if is_tree:
        edges_out = _skeleton_edges(
            len(positions_out), starts, rows[rows[:, 0] >= 0],
        )
    else:
        edges_out = np.ascontiguousarray(rows[(rows >= 0).all(axis=1)])

    if bbox is not None:
        positions_out, edges_out, keep = _bbox_filter(positions_out, edges_out, bbox, 2)
        if keep is not None:
            owners_out = owners_out[keep]

    return {
        "positions": positions_out,
        "edges": edges_out,
        "node_count": len(positions_out),
        "edge_count": len(edges_out),
        "object_ids": owners_out,
    }


def _skeleton_edges(
    n: int,
    fragment_starts: npt.NDArray[np.int64],
    records: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """A skeleton read's ``(child, parent)`` edges over ``n`` read rows.

    The ``implicit_sequential_with_branches`` rule, stated whole:

    - a vertex with no record whose child it is, and that does not open
      a fragment, has the row before it as parent -- the only implicit
      edge;
    - every record ``[child, parent]`` is an edge, and suppresses its
      child's implicit one.  A record whose parent was not read
      (``-1``: outside the scope, or another object's) is no edge here,
      but still suppresses -- the child is a root of this read.

    So a vertex with several records (a cycle, a repeated edge -- not a
    tree, but the writer accepts them) gets every one.  The parent array
    alone kept the last.  Edges come out in child order, with any such
    vertex's records after the rest.

    ``records`` are output rows, each child in ``[0, n)``.
    """
    parent_arr = np.arange(-1, n - 1, dtype=np.int64)
    if len(fragment_starts):
        parent_arr[fragment_starts] = -1
    extra: npt.NDArray[np.int64] | None = None
    if len(records):
        children, parents = records[:, 0], records[:, 1]
        parent_arr[children] = parents
        # Several records for one child?  Fewer distinct children than
        # records says so -- a scatter and a count, no sort; which ones
        # is worked out only then.
        hit = np.zeros(n, dtype=bool)
        hit[children] = True
        if int(np.count_nonzero(hit)) != len(children):
            multi = np.bincount(children, minlength=n)[children] > 1
            parent_arr[children[multi]] = -1
            extra = records[multi & (parents >= 0)]
    kids = np.flatnonzero(parent_arr >= 0)
    edges = np.stack([kids, parent_arr[kids]], axis=1).astype(np.int64, copy=False)
    if extra is not None and len(extra):
        edges = np.concatenate([edges, extra.astype(np.int64, copy=False)], axis=0)
    return edges


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

    # Adjacency as CSR: edges sorted by parent (stably, so each parent's
    # children keep their edge order), ``first[v]:first[v + 1]`` the run
    # of ``v``'s children.  Plain lists, as the walk below indexes them
    # one vertex at a time.  This replaces a dict of one list per vertex,
    # filled by a ``.tolist()`` per parent -- most of what the old
    # reorder cost, and the reason walking every tree now costs no more
    # than walking the first.
    e_children = edges[:, 0].astype(np.int64, copy=False)
    e_parents = edges[:, 1].astype(np.int64, copy=False)
    sort_idx = np.argsort(e_parents, kind="stable")
    sorted_parents = e_parents[sort_idx]
    child_list: list[int] = e_children[sort_idx].tolist()
    first: list[int] = np.searchsorted(
        sorted_parents, np.arange(n + 1, dtype=np.int64), side="left",
    ).tolist()

    # Roots (nodes with no parent) — vectorized set difference.
    is_child = np.zeros(n, dtype=bool)
    if len(e_children) > 0:
        in_range = (e_children >= 0) & (e_children < n)
        is_child[e_children[in_range]] = True
    root_candidates = np.flatnonzero(~is_child)

    # DFS ordering from every root, in index order, then from whatever
    # no root reaches (a cycle) -- not from the first root alone with the
    # rest appended as given.  That left every tree after the first in
    # input order, so a forest given in any other order stored nearly all
    # of its edges.  Input already in depth-first order comes out
    # unchanged either way.  A child already visited is pushed anyway and
    # skipped when popped, which visits in the same order as not pushing
    # it.
    order: list[int] = []
    visited = bytearray(n)
    for start in (*root_candidates.tolist(), *range(n)):
        if visited[start]:
            continue
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = 1
            order.append(node)
            lo, hi = first[node], first[node + 1]
            if hi - lo == 1:
                stack.append(child_list[lo])
            elif hi > lo:
                # Push children in reverse so first child is processed first
                stack.extend(reversed(child_list[lo:hi]))

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


def _child_edge_counts(edges: npt.NDArray, n_nodes: int) -> npt.NDArray[np.int64]:
    """How many ``[child, parent]`` rows name each vertex as the child.

    One for every vertex of a tree but its root, which has none.  Rows
    whose child is out of range are not counted.
    """
    children = edges[:, 0].astype(np.int64, copy=False)
    in_range = (children >= 0) & (children < n_nodes)
    if not in_range.all():
        children = children[in_range]
    return np.bincount(children, minlength=n_nodes)


def _parentless_first(
    chunk_vertices: npt.NDArray[np.int64],
    chunk_oids: npt.NDArray,
    chunk_parentless: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int64]:
    """Move each fragment's parentless vertex to the fragment's start.

    ``chunk_vertices`` is one chunk's vertices already grouped by object
    (``chunk_oids`` their owners, ``chunk_parentless`` which have no
    parent edge).  Returned as given when every parentless vertex already
    opens its fragment -- the depth-first case -- and otherwise re-sorted
    by (object, has a parent), stably, so the rest keep their order.
    """
    if not chunk_parentless.any():
        return chunk_vertices
    starts = np.empty(len(chunk_oids), dtype=bool)
    starts[0] = True
    np.not_equal(chunk_oids[1:], chunk_oids[:-1], out=starts[1:])
    if not (chunk_parentless & ~starts).any():
        return chunk_vertices
    order = np.lexsort((
        np.arange(len(chunk_vertices)), ~chunk_parentless, chunk_oids,
    ))
    return chunk_vertices[order]


def _check_one_root_per_object(
    edges: npt.NDArray,
    object_ids: npt.NDArray,
    n_nodes: int,
) -> None:
    """Reject a skeleton whose object holds more than one tree.

    ``implicit_sequential_with_branches`` stores a parent only where the
    reader cannot imply one, and the reader implies ``parent[i] = i - 1``
    everywhere except a fragment start.  A second root inside one object's
    fragment is therefore inexpressible: it has no parent, yet it does
    have a predecessor the reader will hand it, so it reads back joined to
    the vertex before it -- an edge nobody wrote.

    One rooted tree per object id is the fix, and it is the shape the
    object model wants anyway.  Say so rather than silently inventing
    connectivity; a forest that genuinely has no object structure can be
    written with ``kind="graph"``, which stores every edge explicitly.
    """
    has_parent = np.zeros(n_nodes, dtype=bool)
    if len(edges):
        children = edges[:, 0].astype(np.int64, copy=False)
        in_range = (children >= 0) & (children < n_nodes)
        has_parent[children[in_range]] = True
    roots = np.flatnonzero(~has_parent)
    if roots.size <= 1:
        return
    root_oids = np.asarray(object_ids)[roots]
    sorted_oids = np.sort(root_oids)
    repeated = sorted_oids[1:][sorted_oids[1:] == sorted_oids[:-1]]
    if repeated.size == 0:
        return
    offender = int(repeated[0])
    n_roots = int(np.count_nonzero(root_oids == offender))
    raise ArrayError(
        f"kind='skeleton' stores one rooted tree per object, but object "
        f"{offender} has {n_roots} roots (a forest). The "
        f"implicit_sequential_with_branches convention records a parent "
        f"only where the reader cannot imply one, so a second root in the "
        f"same object reads back with the preceding vertex as its parent "
        f"-- an edge that was never written. Give each tree its own "
        f"object_ids entry, or write it with kind='graph'."
    )


def _branch_mask(edges: npt.NDArray) -> npt.NDArray[np.bool_]:
    """Rows of a tree edge list that depth-first order does not imply.

    ``edges`` is ``(M, 2)`` ``[child, parent]``.  In a depth-first
    ordered tree most edges have ``parent == child - 1``; the rest are
    branch points.

    NOT what :func:`write_graph` selects on any more.  DFS index and read
    order are different numberings, and the store/skip decision has to be
    made in read order -- see the ``is_tree`` branch there.  This remains
    the predicate over DFS indices, which is what
    :func:`_extract_branch_links` wants.
    """
    if len(edges) == 0:
        return np.zeros(0, dtype=bool)
    return edges[:, 1] != (edges[:, 0] - 1)


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

    is_branch = _branch_mask(edges)
    branch_edges = edges[is_branch]

    branch_attrs = None
    if edge_attributes:
        branch_attrs = {
            name: data[is_branch] for name, data in edge_attributes.items()
        }

    return branch_edges, branch_attrs
