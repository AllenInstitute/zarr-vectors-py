"""Polyline and streamline I/O for zarr vectors stores.

A polyline is an ordered sequence of vertices forming a connected path.
Streamlines (from tractography) are polylines with additional per-object
attributes like termination regions.

Polylines that cross chunk boundaries are split into segments.  The
``object_index`` stores the ordered segment sequence for each polyline,
and a ``links/0/<offsets>/`` record connects the last vertex of one
segment to the first vertex of the next.  Within each segment,
connectivity is implicit sequential (vertex i → vertex i+1), so only
the segment-to-segment bridges are ever stored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    DEFAULT_OOB_POLICY,
    FRAGMENT_ATTRIBUTES,
    GEOM_STREAMLINE,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJECT_INDEX,
    OBJIDX_STANDARD,
    RESOLUTION_PREFIX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    _read_chunk_fragments,
    _reshape_vertex_buffer,
    attribute_layout,
    create_attribute_array,
    create_fragment_attribute_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    read_all_object_manifests_csr,
    read_chunk_attributes,
    read_fragment,
    read_group_object_ids,
    read_object_manifest_rows,
    read_object_manifests,
    read_vertex_fragment_index,
    resolve_chunk_keys,
    stamp_fragments_tile,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_vertex_runs,
    write_groupings,
    write_groupings_attributes,
    write_links,
    write_object_attributes,
    write_object_index_arrays,
)
from zarr_vectors.core.attr_chunking import (
    assign_attribute_bins,
    compute_chunk_dim_names,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    get_level_chunk_shape,
)
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
from zarr_vectors.exceptions import ArrayError, ChunkingError, StoreError
from zarr_vectors.spatial.boundary import (
    _pack_rows,
    split_polyline_at_boundaries,
)
from zarr_vectors.spatial.chunking import (
    compute_bounds,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    FragmentRef,
    ObjectManifest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_vectors.core.store import ReadSource, WriteTarget


class _FragmentPlan(NamedTuple):
    """Where every polyline's fragments land, as arrays.

    A fragment (``S`` of them) is a run of one polyline's vertices that
    share a chunk key.  ``frag`` is its index within its chunk and
    ``first`` its first row there; fragments fill a chunk in polyline
    order.  ``rows`` gathers the vertices chunk by chunk, so chunk ``c``
    is rows ``chunk_rows[c]:chunk_rows[c + 1]`` of ``flat[rows]`` and
    holds fragments ``chunk_frags[c]:chunk_frags[c + 1]`` of
    ``sorted_len``.
    """

    seg_poly: npt.NDArray[np.int64]
    seg_key: npt.NDArray[np.int64]
    seg_len: npt.NDArray[np.int64]
    frag: npt.NDArray[np.int64]
    first: npt.NDArray[np.int64]
    rows: npt.NDArray[np.int64]
    sorted_len: npt.NDArray[np.int64]
    chunk_keys: npt.NDArray[np.int64]
    chunk_rows: npt.NDArray[np.int64]
    chunk_frags: npt.NDArray[np.int64]


def _plan_fragments(
    flat: npt.NDArray[np.floating],
    lengths: npt.NDArray[np.int64],
    chunk_shape: Any,
    attr_bins: npt.NDArray[np.int64] | None = None,
) -> _FragmentPlan:
    """Split every polyline at chunk boundaries at once.

    ``flat`` is the polylines concatenated, ``lengths`` their vertex
    counts.  A vertex's chunk key is its spatial chunk, behind its
    attribute bin when ``attr_bins`` is given; a fragment starts where
    a polyline starts or the key changes.  Grouping fragments by key
    with a stable sort keeps them in polyline order inside each chunk,
    which is the order the per-polyline loop this replaces appended
    them in.
    """
    n, ndim = flat.shape
    if len(chunk_shape) != ndim:
        raise ChunkingError(
            f"chunk_shape length {len(chunk_shape)} != vertex ndim {ndim}"
        )
    cs = np.array(chunk_shape, dtype=np.float64)
    keys = np.floor(flat / cs).astype(np.int64)
    if attr_bins is not None:
        keys = np.concatenate([attr_bins.astype(np.int64)[:, None], keys], axis=1)
    poly_starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=poly_starts[1:])
    new_seg = np.zeros(n, dtype=bool)
    new_seg[poly_starts[lengths > 0]] = True
    new_seg[1:] |= np.any(keys[1:] != keys[:-1], axis=1)
    seg_start = np.flatnonzero(new_seg)
    seg_len = np.diff(np.append(seg_start, n))
    seg_poly = np.searchsorted(poly_starts, seg_start, side="right") - 1
    seg_key = keys[seg_start]
    n_seg = len(seg_start)

    packed = _pack_rows(seg_key)
    if packed is not None:
        order = np.argsort(packed, kind="stable")
        sorted_packed = packed[order]
        new_chunk = np.ones(n_seg, dtype=bool)
        new_chunk[1:] = sorted_packed[1:] != sorted_packed[:-1]
    else:
        order = np.lexsort(seg_key.T[::-1])
        sorted_key = seg_key[order]
        new_chunk = np.ones(n_seg, dtype=bool)
        new_chunk[1:] = np.any(sorted_key[1:] != sorted_key[:-1], axis=1)
    chunk_first = np.flatnonzero(new_chunk)
    chunk_of = np.cumsum(new_chunk) - 1
    sorted_len = seg_len[order]
    row_start = np.cumsum(sorted_len) - sorted_len
    frag = np.empty(n_seg, dtype=np.int64)
    frag[order] = np.arange(n_seg) - chunk_first[chunk_of]
    first = np.empty(n_seg, dtype=np.int64)
    first[order] = row_start - row_start[chunk_first][chunk_of]
    rows = np.repeat(seg_start[order] - row_start, sorted_len) + np.arange(n)
    return _FragmentPlan(
        seg_poly=seg_poly,
        seg_key=seg_key,
        seg_len=seg_len,
        frag=frag,
        first=first,
        rows=rows,
        sorted_len=sorted_len,
        chunk_keys=seg_key[order[chunk_first]],
        chunk_rows=np.append(row_start[chunk_first], n),
        chunk_frags=np.append(chunk_first, n_seg),
    )


def _check_vertex_attribute_rows(
    name: str, arrays: list[npt.NDArray], lengths: npt.NDArray[np.int64],
) -> None:
    """Raise unless ``arrays`` holds one entry per polyline, each with
    one row per vertex."""
    if len(arrays) != len(lengths):
        raise ArrayError(
            f"vertex_attributes[{name!r}] has {len(arrays)} entries but "
            f"there are {len(lengths)} polylines"
        )
    counts = np.fromiter((len(a) for a in arrays), dtype=np.int64, count=len(arrays))
    if not np.array_equal(counts, lengths):
        k = int(np.flatnonzero(counts != lengths)[0])
        raise ArrayError(
            f"vertex_attributes[{name!r}][{k}] has {counts[k]} rows but "
            f"polyline {k} has {lengths[k]} vertices"
        )


def _concat_vertex_attribute(
    name: str, arrays: list[npt.NDArray], lengths: npt.NDArray[np.int64],
) -> npt.NDArray:
    """A vertex attribute given one array per polyline, as one array
    cast to the dtype of its first entry -- the dtype the attribute
    array is created with."""
    _check_vertex_attribute_rows(name, arrays, lengths)
    try:
        return np.concatenate(
            arrays, axis=0, dtype=np.asarray(arrays[0]).dtype, casting="unsafe",
        )
    except ValueError as exc:
        raise ArrayError(
            f"vertex_attributes[{name!r}] entries differ in shape past "
            f"the first axis"
        ) from exc


def write_polylines(
    store_path: WriteTarget,
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
    shard_shape: int | tuple[int, ...] | None | Literal["inherit"] = "inherit",
    manifest_layout: str | None = None,
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
        manifest_layout: ``"dense"`` to write the object index as
            fixed-width integer arrays rather than one vlen blob per
            polyline (see :func:`~zarr_vectors.core.store.create_store`),
            which makes a whole-index scan such as a ``bbox`` read a
            numeric gather.  The store then needs a 0.9.4+ reader.  Only
            meaningful when this call creates the store; ``None`` keeps
            the vlen default.

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
    # the call to ``_plan_fragments`` below.

    # Checked before anything is created.  It used to be rejected after
    # ``_create_or_open_store``, which left an empty store on disk for a
    # call that was never going to succeed.
    # OOB policy for polyline vertices.  "ignore" is rejected — dropping
    # vertices would break the per-polyline ordering and connectivity.
    if out_of_bounds == "ignore":
        raise ArrayError(
            "out_of_bounds='ignore' is not supported for write_polylines: "
            "polyline connectivity depends on vertex ordering. Use "
            "'raise' (default) or 'expand'."
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
        manifest_layout=manifest_layout,
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
    lengths = np.fromiter(
        (len(p) for p in polylines), dtype=np.int64, count=n_polylines,
    )
    attr_bins: npt.NDArray[np.int64] | None = None
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
        _check_vertex_attribute_rows(chunk_by_attribute, attr_lists, lengths)
        attr_bins, attr_bin_values = assign_attribute_bins(
            np.concatenate(attr_lists),
        )
        # The chunk-by attribute is implicit in the leading chunk axis.
        vertex_attributes = {
            k: v for k, v in vertex_attributes.items() if k != chunk_by_attribute
        }

    # Every polyline split into per-chunk fragments at once: each (object,
    # chunk key) run is ONE fragment with implicit_sequential edges, and
    # each step from one fragment to the next becomes a cross-chunk link
    # (a non-zero-offset record in the links family) between real
    # chunk-local vertex indices.  The split is at spatial CHUNK
    # boundaries, and at attribute-bin changes when attribute chunking is
    # on, never at bin_shape boundaries: those would make several
    # same-chunk fragments per object with nowhere to keep the edges
    # between them (implicit_sequential has no explicit links array).
    flat_verts = np.ascontiguousarray(all_pts, dtype=np_dtype)
    plan = _plan_fragments(flat_verts, lengths, chunk_shape, attr_bins)
    flat_attrs = {
        name: _concat_vertex_attribute(name, attr_list, lengths)[plan.rows]
        for name, attr_list in (vertex_attributes or {}).items()
    }
    chunk_verts = flat_verts[plan.rows]

    # Consecutive fragments of a polyline always differ in chunk key, so
    # each such pair is a bridge: last vertex of one -> first of the next.
    bridge = np.flatnonzero(plan.seg_poly[1:] == plan.seg_poly[:-1])
    bridge_chunks = np.stack(
        [plan.seg_key[bridge], plan.seg_key[bridge + 1]], axis=1,
    )
    bridge_rows = np.stack(
        [plan.first[bridge] + plan.seg_len[bridge] - 1, plan.first[bridge + 1]],
        axis=1,
    )

    # Declared after the fragment plan, so ``links`` is listed exactly
    # when a polyline crosses a chunk boundary (only the bridges make a
    # links array).
    arrays_present = [VERTICES, VERTEX_FRAGMENTS, "object_index"]
    if vertex_attributes:
        arrays_present.append(VERTEX_ATTRIBUTES)
    if fragment_attributes:
        arrays_present.append(FRAGMENT_ATTRIBUTES)
    if object_attributes:
        arrays_present.append("object_attributes")
    if groups:
        arrays_present.append("groups")
    if group_attributes:
        arrays_present.append("group_attributes")
    if len(bridge):
        arrays_present.append("links")
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

    idx_ndim = ndim + 1 if attr_bins is not None else ndim
    # Collapse all per-array zarr.json + per-chunk byte writes into one
    # asyncio.gather (mirrors points.py:300).  ``shard_shape`` also
    # activates native ``sharding_indexed`` for per-chunk arrays.
    from zarr_vectors.core.arrays import open_write_session
    # chunk_by_attribute prepends a leading attr-bin axis to every chunk
    # key; size the vlen array grid to match that rank.
    session_bin_count = (
        len(attr_bin_values) if attr_bins is not None else None
    )
    with open_write_session(
        level_group, compressor=compressor, shard_shape=shard_shape,
        bounds=bounds_list, chunk_shape=chunk_shape,
        bin_count=session_bin_count,
    ):
        create_vertices_array(level_group, dtype=dtype)
        create_object_index_array(level_group)
        # No links array is created up front: within a segment connectivity
        # is implicit_sequential, so the all-zero (intra-chunk) offsets
        # array would never hold a row.  ``write_links`` below creates
        # exactly the non-zero-offset arrays the bridges land in.
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

        # Chunks in key order, each one buffer of its fragments' rows.
        chunk_rows = plan.chunk_rows.tolist()
        chunk_frags = plan.chunk_frags.tolist()
        for c, key in enumerate(plan.chunk_keys.tolist()):
            chunk_coords = tuple(key)
            r0, r1 = chunk_rows[c], chunk_rows[c + 1]
            write_chunk_vertex_runs(
                level_group, chunk_coords, chunk_verts[r0:r1],
                plan.sorted_len[chunk_frags[c]:chunk_frags[c + 1]],
                dtype=np_dtype,
            )
            for attr_name, attr_rows in flat_attrs.items():
                write_chunk_attributes(
                    level_group, attr_name, chunk_coords, [attr_rows[r0:r1]],
                    dtype=attr_rows.dtype,
                )

            if fragment_attributes:
                num_fragments = chunk_frags[c + 1] - chunk_frags[c]
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
        # attribute-chunked, so widen sid_ndim accordingly.  Object k's
        # manifest is its fragments in order, which are consecutive.
        manifest_offsets = np.zeros(n_polylines + 1, dtype=np.int64)
        np.cumsum(
            np.bincount(plan.seg_poly, minlength=n_polylines),
            out=manifest_offsets[1:],
        )
        write_object_index_arrays(
            level_group, manifest_offsets, plan.seg_key, plan.frag,
            sid_ndim=idx_ndim,
        )

        # Write the segment-to-segment bridges.  ``write_links`` routes
        # each one to the offsets array naming where its far endpoint sits
        # relative to the near one.
        if len(bridge):
            write_links(
                level_group, [], idx_ndim, delta=0, link_width=2,
                _arrays=(bridge_chunks, bridge_rows),
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

        # Record the tiling layout just written, so a later bulk read
    # can return each chunk's buffer without reading its fragment
    # index.  Verified against what is on disk, and stamped after
    # the chunk writes -- see stamp_fragments_tile.
    stamp_fragments_tile(level_group, ndim)
    _finalize_write(root, "write_polylines")
    return {
        "polyline_count": n_polylines,
        "vertex_count": total_vertices,
        "chunk_count": len(plan.chunk_keys),
        # Every record is boundary-crossing by construction — one is
        # appended only where consecutive fragments landed in different
        # chunks, which is exactly the non-``is_intra`` offsets case.
        "cross_chunk_link_count": len(bridge),
        "group_count": n_groups,
    }


def read_polylines(
    store_path: ReadSource,
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
        store_path: URL or path to the store, a pre-built zarr Store,
            or an already-open Group.
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
        - ``object_ids``: the source object ID of each returned polyline,
          same length and order as ``polylines``.  In the whole-object
          modes these are unique.  Under ``chunks`` (segment-level crop) a
          single object can yield several output polylines, so **IDs
          repeat** — one entry per emitted polyline, not per object.  This
          is what lets a caller re-associate the cropped runs of one
          object.
        - ``polyline_count``: number of polylines returned.
        - ``vertex_count``: total vertices across all returned polylines.
    """
    root = open_store(store_path, backend=backend)
    with root.cached_nodes():
        # One node-resolution pass for the whole read, and every
        # node it needs asked for in a single gather rather than
        # resolved one at a time as the code reaches them.
        prefix = f"{RESOLUTION_PREFIX}{level}"
        root.prime_nodes([
            prefix,
            f"{prefix}/{VERTICES}",
            f"{prefix}/{VERTEX_FRAGMENTS}",
            f"{prefix}/{OBJECT_INDEX}",
            # Probed even on a store that has none: a speculative miss
            # caches as absent, which is the answer the reader wants, and
            # leaving it out meant a warm block still paid one lookup.
            f"{prefix}/{VERTEX_ATTRIBUTES}",
        ])
        return _read_polylines(
            root,
            level=level,
            object_ids=object_ids,
            group_ids=group_ids,
            bbox=bbox,
            chunks=chunks,
            attribute_filter=attribute_filter,
        )


def _read_polylines(
    root: FsGroup,
    *,
    level: int,
    object_ids: list[int] | None,
    group_ids: list[int] | None,
    bbox: BoundingBox | None,
    chunks: list[ChunkCoords] | None,
    attribute_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    """Body of :func:`read_polylines`, against an already-open store.

    Split out only so the caller can hold a
    :meth:`Group.cached_nodes` block open across the whole read;
    the two halves are one function.
    """
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
            return _empty_polyline_result(ndim)

    # An explicitly-named object/group subset lets us read only those
    # objects' manifests and the chunks they reference, instead of the
    # whole store (see the selective path below).  Captured before the
    # group resolution / range fallback mutates ``object_ids``.
    explicit_subset = object_ids is not None or group_ids is not None

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
            return _empty_polyline_result(ndim)

    # If bbox, find which chunks are relevant.  Resolved against the
    # level rather than enumerated from the grid: the box is a cartesian
    # product with no clamp, so on a sparse store -- a specimen bounding
    # box with data in part of it -- a whole-domain query materialised
    # one tuple per *allocated* cell (a million of them for a thousand
    # occupied) purely to test membership against manifests that can
    # only name occupied ones.  The resolved set is the same set: a
    # manifest never references a chunk the level does not hold.
    target_chunks: set[ChunkCoords] | None = None
    if bbox is not None:
        target_chunks = set(resolve_chunk_keys(
            level_group, level_chunk_shape, bbox=bbox,
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

    # A box alone names no objects, so the read below would take its
    # whole-level branch: decode every vertex chunk in the level, then
    # keep the polylines that touch the box -- a full read, whatever the
    # box.  The manifests already say which objects touch it, as flat
    # arrays and without per-object Python, so narrow to those objects
    # and read only their chunks.  Same objects, same order.
    touching_manifests: dict[int, ObjectManifest] | None = None
    if (
        target_chunks is not None
        and not explicit_subset
        and chunk_whitelist is None
        and filter_bin is None
    ):
        touching = _objects_touching(level_group, target_chunks)
        if touching is not None:
            object_ids, touching_manifests = touching
            explicit_subset = True

    result_polylines: list[list[npt.NDArray]] = []
    # One entry per emitted polyline, mirroring ``result_polylines``:
    # ``{name: [rows_per_fragment, ...]}``.
    result_attrs: list[dict[str, list[npt.NDArray | None]]] = []
    result_object_ids: list[int] = []
    total_verts = 0

    # Every per-vertex attribute the level carries.  Unlike read_points,
    # this reader has no ``attribute_names`` term for a caller to narrow
    # with, so it reads what is there -- the same choice read_lines makes.
    attr_layouts: dict[str, tuple[np.dtype, int]] = {}
    try:
        _attr_names = sorted(level_group[VERTEX_ATTRIBUTES].children())
    except Exception:
        _attr_names = []
    if _attr_names:
        level_group.prime_nodes(
            [f"{VERTEX_ATTRIBUTES}/{_n}" for _n in _attr_names],
        )
    for _name in _attr_names:
        try:
            attr_layouts[_name] = attribute_layout(level_group, _name)
        except Exception:
            continue

    # The whole level, unfiltered: assembled from the manifests as flat
    # arrays rather than object by object.  Anything that route does not
    # cover sends it back here, to the per-object loop below.
    if (
        not explicit_subset
        and target_chunks is None
        and chunk_whitelist is None
        and filter_bin is None
        and not attr_layouts
    ):
        whole = _read_whole_level(level_group, len(object_ids), dtype, ndim)
        if whole is not None:
            return whole

    # Choose between a selective read (an explicit object/group subset —
    # read only those objects' manifests and the chunks they reference,
    # O(subset)) and a full read (decode every manifest + vertex chunk
    # once, O(store), best when returning the whole store).
    manifest_by_oid: dict[int, ObjectManifest] = {}
    if explicit_subset:
        needed_chunks: set[ChunkCoords] = set()
        # One coordinate selection for the whole subset. Read one id at a
        # time, each call decodes a full 16,384-row manifest bucket, and
        # 10,000 ids cost 64s against a 50k-object store -- 75x the 0.85s
        # it takes to read every polyline in it. Missing and
        # out-of-range ids are simply absent from the result, which is
        # what the per-id ``except: continue`` was for.
        if touching_manifests is not None:
            manifest_by_oid = touching_manifests
        else:
            try:
                manifest_by_oid = read_object_manifests(
                    level_group, ids=[int(o) for o in object_ids],
                )
            except Exception:
                manifest_by_oid = {}
        for m in manifest_by_oid.values():
            for cc, _fi in m:
                needed_chunks.add(cc)
        # Only whitelist chunks are ever read from the cache in crop mode,
        # so don't bother materialising the rest.
        if chunk_whitelist is not None:
            needed_chunks &= chunk_whitelist
        chunk_iter = sorted(needed_chunks)
        chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_iter]
        prefetch_plan: list[tuple[str, list[str]]] = [
            (VERTICES, chunk_key_strs),
            (VERTEX_FRAGMENTS, chunk_key_strs),
        ]
    else:
        # Full read: prefetch every vertex chunk (+ offsets sidecar) and the
        # legacy OBJECT_INDEX ``data``/``offsets`` sidecar in one async
        # gather.  ``vlen_manifests_v1`` stores read the ragged ``manifests``
        # array directly (one chunk per request); the OBJECT_INDEX entry is
        # a harmless no-op for them.
        #
        # The chunk set is the UNION of the object manifests' referenced
        # chunks, not ``list_chunk_keys(VERTICES)``.  The manifests are the
        # authority on which chunks a polyline's fragments live in; the
        # VERTICES ``nonempty_chunks`` presence attribute is a derived index
        # that can under-report (decentralized writers set
        # ``record_presence=False`` and rebuild it out of band — see
        # ``Group.write_bytes`` / ``derive_nonempty_chunks``).  Driving the
        # read off ``nonempty_chunks`` would silently skip any referenced
        # chunk missing from it: ``_read_fragment`` returns None, the
        # fragment is filtered out below, and the polyline's surrounding
        # fragments get concatenated across the gap — a spurious connection
        # between non-adjacent points.  Reading off the manifests makes this
        # full read assemble exactly what the ``object_ids=`` subset read
        # does.
        try:
            _mids, manifests = read_object_manifest_rows(level_group)
            by_id = {int(o): m for o, m in zip(_mids.tolist(), manifests)}
        except Exception:
            manifests, by_id = [], {}

        needed_chunks: set[ChunkCoords] = set()
        for m in manifests:
            for cc, _fi in m:
                needed_chunks.add(cc)
        chunk_iter = sorted(needed_chunks)
        chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_iter]
        prefetch_plan = [
            (VERTICES, chunk_key_strs),
            (VERTEX_FRAGMENTS, chunk_key_strs),
            (OBJECT_INDEX, ["data", "offsets"]),
        ]

    _batched_reads_cm = level_group.batched_reads(prefetch_plan)
    _batched_reads_cm.__enter__()
    try:
        if explicit_subset:
            def _get_manifest(oid: int) -> ObjectManifest | None:
                return manifest_by_oid.get(oid)
        else:
            # ``manifests`` was read above to derive the chunk set; the
            # per-object loop indexes into it — no per-iteration read.
            def _get_manifest(oid: int) -> ObjectManifest | None:
                return by_id.get(int(oid))

        # Decode each materialised chunk's fragments exactly once.  The
        # per-object dispatch then slices from this cache — O(K_per_chunk)
        # per chunk instead of per-call offset recomputation.
        chunk_cache: dict[ChunkCoords, Sequence[npt.NDArray]] = {}
        for cc in chunk_iter:
            try:
                chunk_cache[cc] = _read_chunk_fragments(
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

        # Per-vertex attributes, decoded once per chunk like the vertices
        # above.  This reader returned none at all, so the facade fell back
        # to a level-ordered gather -- which it then had to refuse for any
        # narrowed read, because a level-ordered column cannot be aligned
        # to a by-object assembly.  Read here and the two orders are the
        # same order by construction.
        attr_cache: dict[tuple[str, ChunkCoords], list[npt.NDArray]] = {}

        def _read_attr_fragment(
            name: str, cc: ChunkCoords, fragment_idx: int,
        ) -> npt.NDArray | None:
            key = (name, cc)
            if key not in attr_cache:
                a_dtype, a_ncols = attr_layouts[name]
                try:
                    attr_cache[key] = read_chunk_attributes(
                        level_group, name, cc, dtype=a_dtype, ncols=a_ncols,
                    )
                except (ArrayError, StoreError):
                    attr_cache[key] = []
            rows = attr_cache[key]
            if 0 <= fragment_idx < len(rows):
                return rows[fragment_idx]
            return None

        def _read_run(
            entries: list[FragmentRef],
        ) -> tuple[list[npt.NDArray], dict[str, list[npt.NDArray | None]]]:
            """One emitted polyline's fragments, with its attribute rows.

            Kept in lockstep deliberately: a fragment that fails to read
            is skipped, and skipping it in one list but not the other is
            exactly how a column ends up describing the wrong vertices.
            A fragment whose attribute rows are missing or the wrong
            length records ``None``, which drops that column rather than
            misaligning it.
            """
            frags: list[npt.NDArray] = []
            attrs: dict[str, list[npt.NDArray | None]] = {
                name: [] for name in attr_layouts
            }
            for cc, fragment_index in entries:
                fragment = _read_fragment(cc, fragment_index)
                if fragment is None:
                    continue
                frags.append(fragment)
                for name in attr_layouts:
                    rows = _read_attr_fragment(name, cc, fragment_index)
                    attrs[name].append(
                        None if rows is None or len(rows) != len(fragment)
                        else rows
                    )
            return frags, attrs

        for oid in object_ids:
            # ------------------------------------------------------------------
            # Segment-level crop mode (chunks=).
            # ------------------------------------------------------------------
            obj_manifest = _get_manifest(oid)
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
                            fragment_list, attr_list = _read_run(run)
                            if fragment_list:
                                result_polylines.append(fragment_list)
                                result_attrs.append(attr_list)
                                result_object_ids.append(oid)
                                total_verts += sum(len(fragment) for fragment in fragment_list)
                            run = []
                if run:
                    fragment_list, attr_list = _read_run(run)
                    if fragment_list:
                        result_polylines.append(fragment_list)
                        result_attrs.append(attr_list)
                        result_object_ids.append(oid)
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
                fragment_list, attr_list = _read_run(matching)
            else:
                fragment_list, attr_list = _read_run(obj_manifest)

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
            result_attrs.append(attr_list)
            result_object_ids.append(oid)
            total_verts += sum(len(fragment) for fragment in fragment_list)
    finally:
        _batched_reads_cm.__exit__(None, None, None)

    # A column is returned only when every fragment of every emitted
    # polyline supplied its rows, so it lines up with ``polylines``
    # flattened in the same order.  Anything short is dropped rather than
    # misaligned.
    attrs_out: dict[str, npt.NDArray] = {}
    for _name in attr_layouts:
        parts: list[npt.NDArray] = []
        complete = bool(result_attrs)
        for per_poly in result_attrs:
            rows = per_poly.get(_name) or []
            if not rows or any(r is None for r in rows):
                complete = False
                break
            parts.extend(rows)
        if complete and parts:
            attrs_out[_name] = np.concatenate(parts, axis=0)

    return {
        "polylines": result_polylines,
        "object_ids": result_object_ids,
        "vertex_attributes": attrs_out,
        "ndim": ndim,
        "polyline_count": len(result_polylines),
        "vertex_count": total_verts,
    }


def _objects_touching(
    level_group: Any, target_chunks: set[ChunkCoords],
) -> tuple[list[int], dict[int, ObjectManifest] | None] | None:
    """Ids of the objects with a fragment in ``target_chunks``, ascending,
    with their manifests.

    The manifests are cut from the decode that found the ids, so the
    caller need not read them again by id: the ids are spread over the
    whole level, so that read decodes nearly every manifest bucket a
    second time.  They are ``None`` when two rows share an id, which a
    read by id resolves its own way.  None when the level's manifests
    cannot be read as arrays, and the caller keeps its whole-level read.
    """
    from zarr_vectors.core.arrays import (
        _manifest_csr_arrays,
        _object_id_lookup,
        read_object_id_table,
    )
    from zarr_vectors.core.dense_manifests import manifests_from_csr

    if not target_chunks:
        return [], {}
    try:
        offsets, coords, frags = _manifest_csr_arrays(level_group)
        offsets = np.asarray(offsets, dtype=np.int64)
        coords = np.asarray(coords, dtype=np.int64)
        frags = np.asarray(frags, dtype=np.int64)
        # Row -> id. An identity table is proved from its stamp without
        # reading it.
        lookup = _object_id_lookup(level_group)
        ids = None
        if lookup is not None and lookup[0] != "identity":
            ids = read_object_id_table(level_group)
    except Exception:
        return None
    n_obj = int(offsets.shape[0]) - 1
    if n_obj <= 0 or coords.shape[0] == 0:
        return [], {}
    targets = np.asarray(sorted(target_chunks), dtype=np.int64)
    if targets.ndim != 2 or targets.shape[1] != coords.shape[1]:
        return None
    # Each chunk coordinate as one integer over the range both span.  One
    # column at a time: an axis-0 reduction over a narrow C-order array
    # is several times slower.
    cols = range(coords.shape[1])
    lo = [min(int(coords[:, d].min()), int(targets[:, d].min())) for d in cols]
    span = [
        max(int(coords[:, d].max()), int(targets[:, d].max())) - lo[d] + 1
        for d in cols
    ]
    total = 1
    for v in span:
        total *= v
    if total >= (1 << 62):
        return None

    def pack(a: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        out = np.zeros(a.shape[0], dtype=np.int64)
        for d in cols:
            out *= span[d]
            out += a[:, d] - lo[d]
        return out

    hit = np.isin(pack(coords), pack(targets))
    owner = np.repeat(np.arange(n_obj, dtype=np.int64), np.diff(offsets))
    rows = np.unique(owner[hit])
    if ids is not None:
        ids = ids[:n_obj]
        oids = ids[rows]
        if lookup[1].size > 1 and not np.all(lookup[1][1:] != lookup[1][:-1]):
            return np.sort(oids).tolist(), None
    else:
        oids = rows
    # These rows' manifests, gathered out of the CSR in hand.
    starts = offsets[rows]
    counts = offsets[rows + 1] - starts
    bounds = np.zeros(rows.size + 1, dtype=np.int64)
    np.cumsum(counts, out=bounds[1:])
    sel = np.repeat(starts - bounds[:-1], counts) + np.arange(int(bounds[-1]))
    manifests = manifests_from_csr(bounds, coords[sel], frags[sel])
    by_id = dict(zip(oids.tolist(), manifests))
    return np.sort(oids).tolist(), by_id


def _packable(span: npt.NDArray[np.int64]) -> bool:
    """Whether coordinates over ``span`` pack into one int64 each."""
    total = 1
    for v in span.tolist():
        total *= int(v)
    return total < (1 << 62)


def _pack_coords(
    coords: npt.NDArray[np.int64],
    lo: npt.NDArray[np.int64],
    span: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Each ``(D,)`` row of ``coords`` as one integer, row-major over ``span``."""
    out = np.zeros(coords.shape[0], dtype=np.int64)
    for d in range(coords.shape[1]):
        out *= int(span[d])
        out += coords[:, d] - lo[d]
    return out


def _read_whole_level(
    level_group: Any,
    num_objects: int,
    dtype: np.dtype,
    ndim: int,
) -> dict[str, Any] | None:
    """Every polyline of a level with no per-vertex attributes.

    The result of the per-object full read in :func:`_read_polylines`,
    built from the manifests as CSR arrays: one slice of the chunks'
    vertices per fragment, with no per-object manifest decode or lookup.

    None for anything that loop handles specially -- unreadable
    manifests or chunks, explicit (non-range) fragments, a fragment
    the chunk does not hold, ids named twice -- and the caller takes
    that loop, which stays the definition.
    """
    try:
        csr = read_all_object_manifests_csr(level_group)
    except Exception:
        return None
    offsets = np.asarray(csr.offsets, dtype=np.int64)
    coords = np.asarray(csr.chunk_coords, dtype=np.int64)
    fragment_idx = np.asarray(csr.fragment_idx, dtype=np.int64)
    n_rows = int(offsets.shape[0]) - 1
    per_row = np.diff(offsets)
    if csr.object_ids is None:
        ids = np.arange(n_rows, dtype=np.int64)
        listed = ids < num_objects
    else:
        ids = np.asarray(csr.object_ids, dtype=np.int64)
        if ids.shape[0] != n_rows:
            return None
        listed = (ids >= 0) & (ids < num_objects)
        named = np.sort(ids[listed])
        if named.size > 1 and bool((named[1:] == named[:-1]).any()):
            return None
    # Rows of the objects returned, ascending by id, which is the order
    # the per-object loop emits them in.
    rows = np.flatnonzero(listed & (per_row > 0))
    rows = rows[np.argsort(ids[rows], kind="stable")]
    if rows.size == 0:
        return _empty_polyline_result(ndim)

    # Manifest entries in output order, and the chunks they name.
    per = per_row[rows]
    ends = np.cumsum(per)
    entries = np.repeat(offsets[rows] - (ends - per), per) + np.arange(
        int(ends[-1]), dtype=np.int64,
    )
    entry_coords = coords[entries]
    lo = entry_coords.min(axis=0)
    span = entry_coords.max(axis=0) - lo + 1
    if not _packable(span):
        return None
    packed, entry_chunk = np.unique(
        _pack_coords(entry_coords, lo, span), return_inverse=True,
    )
    chunk_coords = np.empty((packed.shape[0], coords.shape[1]), dtype=np.int64)
    rest = packed
    for d in range(coords.shape[1] - 1, -1, -1):
        chunk_coords[:, d] = rest % span[d] + lo[d]
        rest = rest // span[d]
    chunk_list = [tuple(cc) for cc in chunk_coords.tolist()]
    keys = [".".join(str(c) for c in cc) for cc in chunk_list]

    # Every chunk's vertices end to end, with each fragment's rows as a
    # (start, count) into that one buffer.
    buffers: list[npt.NDArray] = []
    tables: list[npt.NDArray[np.int64]] = []
    base = 0
    with level_group.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)]):
        for cc, key in zip(chunk_list, keys):
            try:
                raw = level_group.read_bytes(VERTICES, key)
                fi = read_vertex_fragment_index(level_group, cc)
            except Exception:
                return None
            table = fi.ranges()
            if table is None:
                return None
            full = _reshape_vertex_buffer(raw, dtype, ndim)
            table = np.asarray(table, dtype=np.int64).reshape(-1, 2)
            if table.shape[0] and (
                int(table[:, 0].min()) < 0
                or int(table[:, 1].min()) < 0
                or int((table[:, 0] + table[:, 1]).max()) > full.shape[0]
            ):
                return None
            buffers.append(full)
            tables.append(table + np.array([base, 0], dtype=np.int64))
            base += full.shape[0]
    n_fragments = np.array([t.shape[0] for t in tables], dtype=np.int64)
    entry_fragment = fragment_idx[entries]
    if bool(
        ((entry_fragment < 0) | (entry_fragment >= n_fragments[entry_chunk])).any()
    ):
        return None
    first = np.concatenate(([0], np.cumsum(n_fragments)[:-1]))
    spans = np.concatenate(tables)[first[entry_chunk] + entry_fragment]
    vertices = np.concatenate(buffers) if len(buffers) > 1 else buffers[0]
    # Read-only like the per-chunk buffers it joins, so a caller writing
    # into a returned polyline fails here as it does on the other paths.
    vertices.flags.writeable = False

    starts = spans[:, 0]
    stops = starts + spans[:, 1]
    fragments = [
        vertices[a:b] for a, b in zip(starts.tolist(), stops.tolist())
    ]
    bounds = [0, *ends.tolist()]
    polylines = [
        fragments[a:b] for a, b in zip(bounds[:-1], bounds[1:])
    ]
    return {
        "polylines": polylines,
        "object_ids": ids[rows].tolist(),
        "vertex_attributes": {},
        "ndim": ndim,
        "polyline_count": len(polylines),
        "vertex_count": int(spans[:, 1].sum()),
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


def _empty_polyline_result(ndim: int = 3) -> dict[str, Any]:
    return {
        "polylines": [],
        "object_ids": [],
        "vertex_attributes": {},
        # Carried so an empty result still knows how wide the store is;
        # the adapter cannot infer it from no data.
        "ndim": int(ndim),
        "polyline_count": 0,
        "vertex_count": 0,
    }
