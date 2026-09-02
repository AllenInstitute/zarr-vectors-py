"""Single-level skeleton geometry type: per-chunk write + pull-by-id read.

The scale-out building blocks the ``write_graph`` whole-dataset path lacks,
for skeletons (rooted trees of ``[child, parent]`` edges):

- :func:`write_skeleton_chunk` — write one spatial chunk's skeleton
  fragments (one fragment per piece) using the core per-chunk
  primitives.  Links are stored ``[child_local, parent_local]`` in
  **chunk-local** indices with an explicit ``[root_local, -1]`` row per
  fragment (per the skeleton spec), so multiple skeletons coexist in one
  chunk without their implicit-sequential parents chaining across
  fragment boundaries.  Returns the ``(object_id, chunk, fragment_idx)``
  records that a downstream reduce groups into a dense ``object_index``.

- :func:`write_skeleton_cross_chunk_links` — store parent→child edges that
  cross a chunk boundary.  They land in the non-zero-offset arrays of the
  same ``links/0/`` family the branch links above use, which is declared
  ``directed=True`` because parent→child order is data.

- :func:`read_skeleton_by_segment_id` — resolve a segment ID to its
  object, follow the manifest, and reconstruct the skeleton (positions +
  ``[child, parent]`` edges + per-vertex attributes).

The object-index *reduce* that groups the per-chunk records into a dense
``object_index`` + ``object_attributes/segment_id`` is coordination that
lives outside this repo (``zarr_vectors_tools``); this module provides only
the single-level primitives the reduce and the reader build on.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    LINKS_IMPLICIT_BRANCHES,
    OBJIDX_STANDARD,
    VERTEX_ATTRIBUTES,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_fragment_attribute_array,
    create_links_array,
    create_object_index_array,
    create_vertices_array,
    read_chunk_link_fragment,
    read_fragment,
    read_object_manifest,
    read_object_attributes,
    read_vertex_fragment_index,
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_links,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.multiscale import upsert_level_transform
from zarr_vectors.core.store import (
    _create_or_open_store,
    _ensure_root_metadata_for_write,
    _finalize_write,
    create_resolution_level,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.types.graphs import _extract_branch_links, _reorder_tree
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.core.store import ReadSource


SEGMENT_ID_ATTR = "segment_id"

# Stored vertices are shifted by this per-axis offset so the chunk grid
# (spec convention: chunk = floor(pos / chunk_shape), origin 0) aligns to
# the source ``.frag`` grid.  World position = stored + coordinate_offset;
# also mirrored into the NGFF ``translation`` transform on every level.
COORDINATE_OFFSET_KEY = "coordinate_offset"


def get_coordinate_offset(root, ndim: int) -> npt.NDArray[np.float64]:
    """Per-axis world offset of the stored coordinate frame (zeros if none)."""
    zv = root.attrs.to_dict().get("zarr_vectors", {})
    off = zv.get(COORDINATE_OFFSET_KEY)
    if off is None:
        return np.zeros(ndim, dtype=np.float64)
    return np.asarray(off, dtype=np.float64)


def set_coordinate_offset(root, offset: Sequence[float]) -> None:
    """Record the stored→world coordinate offset + mirror to NGFF translation."""
    attrs = root.attrs.to_dict()
    zv = dict(attrs.get("zarr_vectors", {}))
    zv[COORDINATE_OFFSET_KEY] = [float(x) for x in offset]
    attrs["zarr_vectors"] = zv
    root.attrs.update(attrs)


def decompose_tree_to_paths(
    piece: dict[str, Any],
) -> tuple[npt.NDArray, dict[str, npt.NDArray], list[tuple[int, int]],
           list[tuple[int, int]], npt.NDArray]:
    """Decompose a rooted tree into maximal linear paths + branch links.

    This is the layout the ``implicit_sequential_with_branches`` skeleton
    convention expects (and that neuroglancer reconstructs): each
    *fragment* is a linear chain whose consecutive vertices are connected
    implicitly (i, i+1); the points where the tree branches are stored as
    explicit ``links`` entries connecting a new path's first vertex to the
    branch vertex in the path it left.

    Args:
        piece: ``{"positions": (N, D), "edges": (M, 2) [child, parent]
            (root omitted), "attributes": {name: (N, ...)}}``.

    Returns:
        ``(ordered_positions, ordered_attributes, frag_ranges,
        branch_links, new_of_old)`` where ``frag_ranges`` is a list of
        ``(start, count)`` over the reordered vertex array (one per path),
        ``branch_links`` are ``(child_order_idx, parent_order_idx)`` over
        the reordered indices (one per non-root path), and
        ``new_of_old[i]`` is the reordered index of input vertex ``i``.
    """
    pos = np.asarray(piece["positions"])
    n = len(pos)
    edges = np.asarray(piece["edges"], dtype=np.int64).reshape(-1, 2)
    attrs = piece.get("attributes") or {}

    parent = np.full(n, -1, dtype=np.int64)
    if len(edges) > 0:
        parent[edges[:, 0]] = edges[:, 1]
    children: dict[int, list[int]] = defaultdict(list)
    for c in range(n):
        p = int(parent[c])
        if p >= 0:
            children[p].append(c)
    roots = [i for i in range(n) if parent[i] < 0]

    order: list[int] = []
    new_of_old = np.full(n, -1, dtype=np.int64)
    frag_ranges: list[tuple[int, int]] = []
    branch_links: list[tuple[int, int]] = []

    # DFS path decomposition: each maximal chain is one fragment; the
    # first child continues the current path, every other child starts a
    # new path with a branch link back to the current (branch) vertex.
    stack: list[tuple[int, int]] = [(r, -1) for r in reversed(roots)]
    while stack:
        node, par_out = stack.pop()
        fstart = len(order)
        cur: int | None = node
        first = True
        while cur is not None:
            o = len(order)
            order.append(cur)
            new_of_old[cur] = o
            if first and par_out >= 0:
                branch_links.append((o, par_out))
            first = False
            kids = children.get(cur)
            if not kids:
                cur = None
            else:
                for k in reversed(kids[1:]):
                    stack.append((k, o))
                cur = kids[0]
        frag_ranges.append((fstart, len(order) - fstart))

    order_arr = np.asarray(order, dtype=np.int64)
    opos = pos[order_arr] if n else pos
    oattrs = {k: np.asarray(v)[order_arr] for k, v in attrs.items()}
    return opos, oattrs, frag_ranges, branch_links, new_of_old


# ===================================================================
# Per-chunk level-0 writer
# ===================================================================

def write_skeleton_chunk(
    level_group,
    chunk_coords: ChunkCoords,
    pieces: list[dict[str, Any]],
    *,
    attr_dtypes: dict[str, np.dtype] | None = None,
    dtype: np.dtype | str = np.float32,
    record_presence: bool = True,
) -> tuple[list[tuple[int, ChunkCoords, int]], dict[Any, tuple[ChunkCoords, int]]]:
    """Write one spatial chunk's skeleton fragments.

    Args:
        record_presence: Threaded into every per-chunk write this makes
            (vertices, links, attributes, fragment attributes).  Pass
            ``False`` from concurrent per-chunk writers — ``nonempty_chunks``
            is array-wide state, so stamping it is a read-modify-write that
            two workers writing *disjoint* chunks still race on — and
            re-derive the manifests once afterwards from the coordinator
            (``derive_nonempty_chunks`` / ``finalize_links``).
        level_group: Target resolution-level group (arrays must already
            be created — see :func:`init_skeleton_level`).
        chunk_coords: Spatial chunk coordinates.
        pieces: One dict per fragment with keys ``segment_id`` (int),
            ``positions`` ``(n, D)``, ``edges`` ``(m, 2)`` ``[child,
            parent]`` (root omitted), and ``attributes`` ``{name: (n,
            ...)}``.  Pieces should already be single rooted trees (use
            :func:`split_components`).  An optional ``"anchors"`` dict
            ``{tag: input_local_idx}`` requests the final chunk-local
            index of those input vertices back in the return value (used
            to resolve cross-chunk-link endpoints).
        attr_dtypes: Per-attribute numpy dtypes; defaults to each
            piece's array dtype.
        dtype: Vertex position dtype.

    Returns:
        ``(records, anchor_locs)`` where ``records`` is
        ``[(segment_id, chunk_coords, fragment_index), ...]`` for the
        object-index reduce and ``anchor_locs`` maps each anchor ``tag``
        to ``(chunk_coords, chunk_local_index)``.
    """
    dtype = np.dtype(dtype)
    attr_dtypes = attr_dtypes or {}
    attr_names: list[str] = []
    for p in pieces:
        for name in (p.get("attributes") or {}):
            if name not in attr_names:
                attr_names.append(name)

    vert_groups: list[npt.NDArray] = []
    link_groups: list[npt.NDArray] = []
    attr_groups: dict[str, list[npt.NDArray]] = {n: [] for n in attr_names}
    records: list[tuple[int, ChunkCoords, int]] = []
    frag_seg_ids: list[int] = []
    frag_obj_ids: list[int] = []
    frag_has_obj_id: list[bool] = []
    anchor_locs: dict[Any, tuple[ChunkCoords, int]] = {}
    cc_tuple = tuple(int(c) for c in chunk_coords)

    chunk_offset = 0
    fragment_idx = 0
    for piece in pieces:
        opos, oattrs, frag_ranges, blinks, new_of_old = decompose_tree_to_paths(piece)
        # The per-fragment ``segment_id`` attribute is the original (e.g.
        # flywire) id.  The object-index *records* are keyed separately by
        # ``object_id`` when supplied (the coarsener passes the dense OID to
        # preserve the per-object index), else they fall back to
        # ``segment_id`` (level 0, where the tools-side object-index reduce
        # remaps segment ids → OIDs).
        seg = int(piece["segment_id"])
        piece_has_obj_id = "object_id" in piece
        obj_key = int(piece["object_id"]) if piece_has_obj_id else seg
        piece_base = chunk_offset
        # Each non-root path's start carries one branch link (chunk-local).
        blink_at_start = {ch_o: (ch_o, par_o) for ch_o, par_o in blinks}
        for (s, c) in frag_ranges:
            vert_groups.append(opos[s:s + c].astype(dtype, copy=False))
            for name in attr_names:
                data = oattrs.get(name)
                if data is None:
                    data = np.zeros((c,), dtype=attr_dtypes.get(name, np.float32))
                else:
                    data = data[s:s + c]
                attr_groups[name].append(np.asarray(data))
            if s in blink_at_start:
                ch_o, par_o = blink_at_start[s]
                lg = [[piece_base + ch_o, piece_base + par_o]]
            else:
                lg = []
            link_groups.append(np.asarray(lg, dtype=np.int64).reshape(-1, 2))
            records.append((obj_key, cc_tuple, fragment_idx))
            frag_seg_ids.append(seg)
            frag_has_obj_id.append(piece_has_obj_id)
            if piece_has_obj_id:
                frag_obj_ids.append(obj_key)
            fragment_idx += 1
        anchors = piece.get("anchors")
        if anchors:
            for tag, input_idx in anchors.items():
                anchor_locs[tag] = (cc_tuple, piece_base + int(new_of_old[int(input_idx)]))
        chunk_offset += len(opos)

    write_chunk_vertices(
        level_group, chunk_coords, vert_groups, dtype=dtype,
        record_presence=record_presence,
    )
    # Per-cell writer, not ``write_links``: these branch links are already
    # chunk-local and all-zero-offset, and they must stay one group per
    # fragment for ``read_chunk_link_fragment`` to slice them back out.
    # ``write_links`` files a cell's records as a single group.
    write_chunk_links(
        level_group, chunk_coords, link_groups, delta=0,
        record_presence=record_presence,
    )
    for name in attr_names:
        write_chunk_attributes(
            level_group, name, chunk_coords, attr_groups[name],
            dtype=attr_dtypes.get(name, attr_groups[name][0].dtype),
            record_presence=record_presence,
        )
    # Per-fragment ``segment_id`` (uint64): one original (flywire) id per
    # fragment, in fragment order (``frag_seg_ids`` is appended in lockstep
    # with ``fragment_idx``).  Loaded with the chunk so the renderer can
    # colour each fragment by its owning segment via the normal segment
    # palette and surface the global id on pick — at every pyramid level.
    if frag_seg_ids:
        seg_ids = np.asarray(frag_seg_ids, dtype=np.uint64)
        write_chunk_fragment_attributes(
            level_group, "segment_id", chunk_coords, seg_ids, dtype=np.uint64,
            record_presence=record_presence,
        )
    if any(frag_has_obj_id):
        if not all(frag_has_obj_id):
            raise ValueError(
                "write_skeleton_chunk received mixed pieces with and without object_id"
            )
        obj_ids = np.asarray(frag_obj_ids, dtype=np.uint64)
        write_chunk_fragment_attributes(
            level_group, "object_id", chunk_coords, obj_ids, dtype=np.uint64,
            record_presence=record_presence,
        )
    return records, anchor_locs


def init_skeleton_store(
    store_path: str | Path,
    *,
    chunk_shape: tuple[float, ...],
    bounds: tuple[list[float], list[float]],
    ndim: int,
    attribute_dtypes: dict[str, str],
    backend: str | None = None,
    coordinate_offset: Sequence[float] | None = None,
    compressor: Any = None,
    shard_shape: int | tuple[int, ...] | None = None,
):
    """Create a new skeleton store + an empty level 0 with its arrays.

    ``coordinate_offset`` (optional) is the world position of the stored
    coordinate origin: stored vertices are expected to be
    ``world - coordinate_offset`` so the spec's origin-0 chunk grid
    aligns to the source grid.  It is recorded in metadata and mirrored
    to the NGFF ``translation`` transform.

    ``shard_shape`` (optional, in units of spatial chunks) wraps each
    per-chunk array's cells in Zarr v3's ``sharding_indexed`` codec, so
    many chunks share one storage object — the same knob the whole-store
    writers (``write_points``, ``write_graph``, …) take.  ``compressor``
    sets the codec pipeline for those arrays.

    Skeletons are written by streaming rather than in one call, so the
    arrays are allocated here and the caller's subsequent
    :func:`write_skeleton_chunk` calls reuse them as-is — an existing
    array is never re-created out from under a streaming writer.  A
    caller that wants its streamed writes batched should open its own
    :func:`~zarr_vectors.core.arrays.open_write_session` with the SAME
    ``shard_shape`` / ``bounds`` / ``chunk_shape`` passed here.

    Returns ``(root, level0_group)``.  Callers then stream
    :func:`write_skeleton_chunk` over chunks, reduce the records into an
    object index (``zarr_vectors_tools.multiresolution.object_index``), and
    finish with :func:`finalize_skeleton_store`.
    """
    # Tag axes as nanometers so neuroglancer treats positions as physical
    # (not unitless) coordinates.
    _axis_names = ["x", "y", "z", "t"][:ndim] if ndim <= 4 else [f"d{i}" for i in range(ndim)]
    axes = [{"name": n, "type": "space", "unit": "nanometer"} for n in _axis_names]
    root = _create_or_open_store(
        str(store_path), backend=backend,
        bounds=(list(bounds[0]), list(bounds[1])),
        chunk_shape=tuple(chunk_shape), ndim=ndim, axes=axes,
    )
    _ensure_root_metadata_for_write(
        root, inferred_ndim=ndim, geometry_type="skeleton",
        links_convention=LINKS_IMPLICIT_BRANCHES,
        object_index_convention=OBJIDX_STANDARD,
    )
    level_meta = LevelMetadata(
        level=0, vertex_count=0,
        arrays_present=[VERTICES, "links", "object_index"],
    )
    level_group = create_resolution_level(root, 0, level_meta)
    from zarr_vectors.core.arrays import open_write_session

    # Allocate every per-chunk array inside a session so ``shard_shape``
    # and ``compressor`` reach the array creation.  Without one the
    # arrays fall back to the derived (unsharded, uncompressed) layout,
    # which is why sharding never applied to skeleton stores.
    with open_write_session(
        level_group,
        compressor=compressor,
        shard_shape=shard_shape,
        bounds=(list(bounds[0]), list(bounds[1])),
        chunk_shape=tuple(chunk_shape),
    ):
        create_vertices_array(level_group, dtype="float32")
        # One links family holds both the intra-chunk branch links and the
        # boundary-crossing parent→child edges.  ``directed=True`` is family
        # policy: it stops the cross-chunk arrays canonical-sorting endpoints
        # (which would swap parent and child).  Intra-chunk records are stored
        # in input order regardless, so branch links are unaffected.
        create_links_array(
            level_group, link_width=2, delta=0, sid_ndim=ndim, directed=True,
        )
        create_object_index_array(level_group)
        create_fragment_attribute_array(level_group, "segment_id", dtype="uint64")
        for name, dt in attribute_dtypes.items():
            create_attribute_array(level_group, name, dtype=dt)
    if coordinate_offset is not None and any(float(x) != 0 for x in coordinate_offset):
        set_coordinate_offset(root, coordinate_offset)
        upsert_level_transform(
            root, 0, scale=[1.0] * ndim,
            translation=[float(x) for x in coordinate_offset],
        )
    return root, level_group


def write_skeleton_cross_chunk_links(
    level_group,
    links: list[tuple[tuple[ChunkCoords, int], tuple[ChunkCoords, int]]],
    *,
    ndim: int,
) -> None:
    """Write parent→child edges that cross a chunk boundary.

    Each link is ``((chunkA, viA), (chunkB, viB))`` with chunk-local
    vertex indices, endpoint 0 the parent and endpoint 1 the child.
    These are the connections the coarsener uses to merge an object's
    fragments once both endpoints fall in the same (coarser) chunk.

    Stored ``directed=True`` so the parent→child order survives — a
    canonical sort by chunk coord would otherwise silently swap
    endpoints whenever the child chunk sorts before the parent chunk.
    Each link lands in the offsets array naming where the child chunk
    sits relative to the parent's, within the same ``links/0/`` family
    :func:`write_skeleton_chunk` writes branch links into.
    """
    if not links:
        return
    write_links(
        level_group, links, ndim, delta=0, link_width=2, directed=True,
    )


def finalize_skeleton_store(root) -> None:
    # No ``fragments_tile`` stamp here, deliberately.  This finalises a
    # DECENTRALISED write -- many workers, each owning some chunks -- so
    # the layout is exactly the one whose tiling nothing can vouch for
    # from here.  The claim stays absent, which costs a coarser read and
    # promises nothing untrue.
    _finalize_write(root, "write_skeleton_chunked")


# ===================================================================
# Pull-by-id read
# ===================================================================

def _path_sequential_edges(n: int) -> npt.NDArray[np.int64]:
    """Implicit ``(i, i-1)`` edges for one linear-path fragment of length n."""
    if n <= 1:
        return np.zeros((0, 2), dtype=np.int64)
    i = np.arange(1, n, dtype=np.int64)
    return np.stack([i, i - 1], axis=1)


def read_skeleton_by_segment_id(
    store_path: ReadSource,
    segment_id: int,
    *,
    level: int = 0,
    backend: str | None = None,
    attributes: list[str] | None = None,
) -> dict[str, Any] | None:
    """Read one skeleton by its original (flywire) segment ID.

    Resolves ``segment_id`` → object via ``object_attributes/segment_id``
    (sorted, binary search), follows the manifest, and reconstructs the
    merged skeleton (fragments concatenated; edges offset per fragment).

    Returns ``None`` when the segment ID is absent.  Otherwise a dict
    with ``positions`` ``(N, D)``, ``edges`` ``(M, 2)`` ``[child,
    parent]``, ``attributes`` ``{name: (N, ...)}``, and
    ``fragment_count``.
    """
    root = open_store(store_path, backend=backend)
    root_meta = read_root_metadata(root)
    ndim = root_meta.sid_ndim
    level_group = get_resolution_level(root, level)

    seg_ids = read_object_attributes(level_group, SEGMENT_ID_ATTR).astype(np.uint64)
    pos = int(np.searchsorted(seg_ids, np.uint64(segment_id)))
    if pos >= len(seg_ids) or int(seg_ids[pos]) != int(segment_id):
        return None
    oid = pos

    manifest = read_object_manifest(level_group, oid)
    if not manifest:
        return {
            "positions": np.zeros((0, ndim), np.float32),
            "edges": np.zeros((0, 2), np.int64),
            "attributes": {}, "fragment_count": 0,
        }

    if attributes is None:
        attributes = _list_vertex_attributes(level_group)

    all_pos: list[npt.NDArray] = []
    all_edges: list[npt.NDArray] = []
    attr_acc: dict[str, list[npt.NDArray]] = {a: [] for a in attributes}
    running = 0
    # Per chunk, the object's path fragments as (chunk_local_start, count,
    # global_start) so branch links (chunk-local) map to object-global.
    cc_frags: dict[ChunkCoords, list[tuple[int, int, int]]] = defaultdict(list)
    for cc, fidx in manifest:
        frag = read_fragment(level_group, cc, fidx, dtype=np.float32, ndim=ndim)
        n = len(frag)
        fi = read_vertex_fragment_index(level_group, cc)
        cl_start, _count = fi.range(fidx)
        all_pos.append(frag)
        e = _path_sequential_edges(n)  # implicit within-path edges
        if len(e) > 0:
            all_edges.append(e + running)
        for a in attributes:
            grp = _read_attr_fragment(level_group, a, cc, fidx)
            if grp is not None:
                attr_acc[a].append(grp)
        cc_frags[cc].append((int(cl_start), n, running))
        running += n

    # Branch links (chunk-local) connect this object's fragments within a
    # chunk; map both endpoints to object-global and add as edges.
    def _to_global(cc, cl):
        for cl_start, n, g_start in cc_frags.get(cc, ()):
            if cl_start <= cl < cl_start + n:
                return g_start + (cl - cl_start)
        return None
    for cc, fidx in manifest:
        links = read_chunk_link_fragment(
            level_group, cc, fidx, link_width=2, default=None,
        )
        if links is None:
            continue
        for ch_cl, par_cl in np.asarray(links, dtype=np.int64).reshape(-1, 2):
            a = _to_global(cc, int(ch_cl))
            b = _to_global(cc, int(par_cl))
            if a is not None and b is not None:
                all_edges.append(np.array([[a, b]], dtype=np.int64))

    positions = np.concatenate(all_pos, axis=0) if all_pos else np.zeros((0, ndim), np.float32)
    offset = get_coordinate_offset(root, ndim)
    if positions.size and np.any(offset != 0):
        positions = positions + offset.astype(positions.dtype)
    edges_out = np.concatenate(all_edges, axis=0) if all_edges else np.zeros((0, 2), np.int64)
    out_attrs = {
        a: np.concatenate(v, axis=0) for a, v in attr_acc.items() if v
    }
    return {
        "positions": positions,
        "edges": edges_out,
        "attributes": out_attrs,
        "fragment_count": len(manifest),
    }


def _list_vertex_attributes(level_group) -> list[str]:
    try:
        if VERTEX_ATTRIBUTES in level_group:
            # Each attribute is a single vlen array, so enumerate via
            # ``children()`` (array + group keys), not ``__iter__``
            # which yields sub-groups only.
            return level_group[VERTEX_ATTRIBUTES].children()
    except Exception:
        pass
    return []


def _read_attr_fragment(level_group, name, cc, fidx):
    from zarr_vectors.core.arrays import read_chunk_attributes
    try:
        meta = level_group.read_array_meta(f"{VERTEX_ATTRIBUTES}/{name}")
        dt = np.dtype(meta.get("dtype", "float32"))
        groups = read_chunk_attributes(level_group, name, cc, dtype=dt)
    except (ArrayError, Exception):
        return None
    if fidx < len(groups):
        return groups[fidx]
    return None
