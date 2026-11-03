"""Mesh I/O for zarr vectors stores.

Supports two encoding modes:

- **Raw**: vertex positions in ``vertices/``, face indices in
  ``links/0/<offsets>/`` (L=3 for triangles, L=4 for quads).  Faces
  within one chunk land in the all-zero offsets array; faces spanning
  chunk boundaries land in the array naming the offsets between them.

- **Draco**: each fragment is encoded as a Draco bitstream containing
  both positions and its own intra-chunk faces.  Only the boundary-
  crossing faces remain in ``links/`` — the bitstream is per-chunk and
  cannot carry them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CROSS_CHUNK_EXPLICIT,
    DEFAULT_OOB_POLICY,
    ENCODING_DRACO,
    ENCODING_RAW,
    GEOM_MESH,
    LINK_FRAGMENTS,
    LINKS_EXPLICIT,
    OBJECT_INDEX,
    OBJIDX_STANDARD,
    RESOLUTION_PREFIX,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_links_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    link_cells_within,
    link_endpoints_to_rows,
    list_link_offsets,
    read_chunk_vertex_buffer,
    read_chunk_vertices,
    read_link_arrays,
    read_object_manifests,
    read_vertex_fragment_index,
    resolve_chunk_keys,
    stamp_fragments_tile,
    write_chunk_attributes,
    write_chunk_vertices,
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
from zarr_vectors.spatial.boundary import (
    build_vertex_chunk_mapping,
    partition_faces,
)
from zarr_vectors.spatial.chunking import (
    assign_chunks,
    compute_bounds,
)
from zarr_vectors.typing import (
    BinShape,
    BoundingBox,
    ChunkCoords,
    ChunkShape,
    ObjectManifest,
)

if TYPE_CHECKING:
    from zarr_vectors.core.store import ReadSource, WriteTarget


def write_mesh(
    store_path: WriteTarget,
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
    shard_shape: int | tuple[int, ...] | None = None,
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

    # Checked before anything is created.  It used to be rejected after
    # ``_create_or_open_store``, which left an empty store on disk for a
    # call that was never going to succeed.
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
    root = _create_or_open_store(
        store_path,
        backend=backend,
        bounds=bounds_list,
        chunk_shape=tuple(chunk_shape),
        ndim=ndim,
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
    # Group each chunk's vertices by object, so that every fragment written
    # below belongs to exactly one object and a manifest entry can name
    # ``(chunk, fragment)`` rather than the whole chunk.  Without this an
    # object's manifest names chunk-wide fragments shared with every other
    # object in that chunk, and any per-object read -- or any per-object
    # coarsener -- pulls in all of them.
    #
    # Done BEFORE ``build_vertex_chunk_mapping``: the chunk-local index a face
    # record carries is an offset into that chunk's fragments concatenated in
    # write order (see ``_read_mesh``), so the ordering used here and the
    # ordering written below have to be the same one.
    for _cc, _gi in list(chunk_assignments.items()):
        _gi = np.asarray(_gi, dtype=np.int64)
        if _gi.size > 1:
            _gi = _gi[np.argsort(object_ids[_gi], kind="stable")]
        chunk_assignments[_cc] = _gi
    chunk_list = sorted(chunk_assignments.keys())

    vertex_chunks, vertex_local, chunk_list = build_vertex_chunk_mapping(
        chunk_assignments, n_verts, chunk_list
    )

    # A face is intra-chunk exactly when all L of its vertices share a
    # chunk — the all-zero-offsets case (``paths.is_intra``) once the
    # record reaches ``write_links``.
    f_chunk = vertex_chunks[faces]                      # (F, L)
    f_local = vertex_local[faces]                       # (F, L)
    is_intra_face = np.all(f_chunk == f_chunk[:, :1], axis=1)
    is_draco = encoding == ENCODING_DRACO and ndim == 3

    # Draco embeds a chunk's intra-chunk faces in that chunk's own
    # bitstream, in chunk-local indices, so only the boundary-crossing
    # faces stay in the links family.  Raw stores every face there.
    intra_faces: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    if is_draco:
        intra_faces, _ = partition_faces(
            faces, vertex_chunks, vertex_local, chunk_list
        )
        store_rows = np.flatnonzero(~is_intra_face)
    else:
        store_rows = np.arange(n_faces, dtype=np.int64)

    # Records stay in face order, which the partitioner preserves within
    # each cell.  Held as arrays -- ``(F, L, D)`` endpoint chunks and
    # ``(F, L)`` local indices -- rather than one list of tuples per
    # face: building those and taking them apart again was 3 s of a
    # half-million-face write before anything was partitioned.
    link_chunks = np.asarray(chunk_list, dtype=np.int64)[f_chunk[store_rows]]
    link_vi = f_local[store_rows]

    # Write vertices per chunk (one fragment per chunk for simplicity)
    object_manifests: dict[int, ObjectManifest] = {}
    idx_ndim = ndim + 1 if vertex_attr_bins is not None else ndim

    # Collapse all per-array zarr.json + per-chunk byte writes into one
    # asyncio.gather (mirrors points.py:300).  ``shard_shape`` also
    # activates native ``sharding_indexed`` for per-chunk arrays.
    from zarr_vectors.core.arrays import open_write_session
    # chunk_by_attribute prepends a leading attr-bin axis to every chunk
    # key; size the vlen array grid to match that rank.
    session_bin_count = (
        len(attr_bin_values) if vertex_attr_bins is not None else None
    )
    with open_write_session(
        level_group, compressor=compressor, shard_shape=shard_shape,
        bounds=bounds_list, chunk_shape=chunk_shape,
        bin_count=session_bin_count,
    ):
        create_vertices_array(level_group, dtype=dtype, encoding=encoding)
        create_object_index_array(level_group)
        # The links family is created by ``write_links`` below, stamped
        # with the real face arity.  The pre-merge cross-chunk family was
        # created here with the default link_width=2 and mis-declared L=3
        # geometry whenever no face happened to cross a boundary.

        if vertex_attributes:
            for name, data in vertex_attributes.items():
                create_attribute_array(level_group, name, dtype=str(data.dtype))
        if object_attributes:
            for _name in object_attributes:
                create_object_attributes_array(level_group, _name)

        for chunk_idx, chunk_coords in enumerate(chunk_list):
            global_indices = chunk_assignments[chunk_coords]
            chunk_verts = vertices[global_indices]
            chunk_obj_ids = object_ids[global_indices]
            # Object-grouped above, so every change of id starts a fragment.
            frag_starts = np.concatenate((
                [0], np.flatnonzero(np.diff(chunk_obj_ids)) + 1,
            )).astype(np.int64)
            split_at = frag_starts[1:]

            if is_draco:
                # Draco encodes one bitstream per chunk, so the per-object
                # split cannot be expressed here: the whole chunk stays a
                # single fragment and every object in it names fragment 0.
                local_faces_arr = intra_faces.get(chunk_coords)
                _write_draco_chunk(
                    level_group, chunk_coords, chunk_verts,
                    local_faces_arr, draco_quantization_bits, np_dtype,
                )
                for oid in np.unique(chunk_obj_ids):
                    object_manifests.setdefault(int(oid), []).append(
                        (chunk_coords, 0),
                    )
            else:
                write_chunk_vertices(
                    level_group, chunk_coords,
                    np.split(chunk_verts, split_at), dtype=np_dtype,
                )
                # One fragment per object, in the order just written; the
                # fragments still tile [0, N) so the bulk-read fast path
                # (stamp_fragments_tile) keeps its claim.
                for frag_idx, start in enumerate(frag_starts):
                    object_manifests.setdefault(
                        int(chunk_obj_ids[start]), [],
                    ).append((chunk_coords, frag_idx))

            # Write vertex attributes
            if vertex_attributes:
                for name, data in vertex_attributes.items():
                    chunk_attrs = data[global_indices]
                    write_chunk_attributes(
                        level_group, name, chunk_coords,
                        ([chunk_attrs] if is_draco
                         else np.split(chunk_attrs, split_at)),
                        dtype=data.dtype,
                    )

        # One write for every face record: each is L ``(chunk_coords,
        # local_vertex_idx)`` endpoints, and ``write_links`` routes it to
        # the offsets array naming where those endpoints sit.  Winding is
        # recovered on read from ``perm_idx``, so the family is undirected.
        if len(link_vi):
            write_links(
                level_group, [], idx_ndim, delta=0,
                link_width=link_width, _arrays=(link_chunks, link_vi),
            )
        # Backstop: `arrays_present` advertises the family, and the
        # per-cell editors in ops/ write into an array that must already
        # exist — Draco leaves write_links nothing to create when no face
        # crosses.  Must run *after* it: an array present but untargeted
        # makes write_links recount the family off disk, which inside this
        # deferred write session reads back empty.
        create_links_array(
            level_group, link_width=link_width, delta=0, sid_ndim=idx_ndim,
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
    _finalize_write(root, "write_mesh")
    return {
        "vertex_count": n_verts,
        "face_count": n_faces,
        "chunk_count": len(chunk_list),
        # Geometry, not storage: Draco keeps its intra-chunk faces in the
        # bitstream rather than the links family, and these counts stay
        # the same either way.
        "intra_face_count": int(np.count_nonzero(is_intra_face)),
        "cross_face_count": int(np.count_nonzero(~is_intra_face)),
        "encoding": encoding,
    }


def read_mesh(
    store_path: ReadSource,
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
        store_path: URL or path to the store, a pre-built zarr Store,
            or an already-open Group.
        level: Resolution level.
        bbox: Optional bounding box filter.
        object_ids: Optional object ID filter.  Only the named objects'
            vertex fragments are read -- found through the object index,
            not by scanning chunks -- and only the faces whose every
            vertex belongs to one of them.  Ids not in the store are
            ignored.  Not available on Draco-encoded stores, where a
            chunk is one bitstream shared by every object in it.
        chunks: Optional whitelist of chunk coordinate tuples; only data
            in those chunks is returned. AND-ed with ``bbox`` and
            ``object_ids``. ``chunks=[]`` yields an empty result;
            ``chunks=None`` (default) applies no chunk filter.

    Returns:
        Dict with:
        - ``vertices``: ``(V, D)`` positions
        - ``faces``: ``(F, L)`` face indices (remapped to output vertex order)
        - ``vertex_count``, ``face_count``
        - ``object_ids``: ``(V,)`` owning object of each vertex -- only
          when ``object_ids=`` was given.  Vertices come back grouped by
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
        return _read_mesh(
            root,
            level=level,
            bbox=bbox,
            object_ids=object_ids,
            chunks=chunks,
            attribute_filter=attribute_filter,
        )


def _read_mesh(
    root: FsGroup,
    *,
    level: int,
    bbox: BoundingBox | None,
    object_ids: list[int] | None,
    chunks: list[ChunkCoords] | None,
    attribute_filter: dict[str, Any] | None,
) -> dict[str, Any]:
    """Body of :func:`read_mesh`, against an already-open store.

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
    encoding = ENCODING_RAW
    try:
        vmeta = level_group.read_array_meta(VERTICES)
        dtype = np.dtype(vmeta.get("dtype", "float32"))
        encoding = vmeta.get("encoding", ENCODING_RAW)
    except Exception:
        pass

    if object_ids is not None and encoding == ENCODING_DRACO:
        raise ArrayError(
            "read_mesh(object_ids=...) is not available on a Draco-encoded "
            "store: each chunk is a single bitstream shared by every object "
            "in it, so no object's vertices can be read on their own"
        )

    # Face arity is family-wide policy, stamped on the ``links/<delta>/``
    # group rather than on the per-offsets arrays under it.
    link_width = 3
    try:
        lmeta = level_group.read_array_meta(links_group_path(0))
        link_width = lmeta.get("link_width", 3)
    except Exception:
        pass

    # Is the read narrowed to some chunks?  If so, faces are read only
    # from the link cells those chunks can reach (link_cells_within);
    # if not, the whole family is wanted and listing it is cheaper than
    # proving every cell qualifies.
    scoped = bbox is not None or chunks is not None or bool(attribute_filter)

    # An object read finds its chunks through the object index, so an
    # unscoped one needs no chunk listing at all.
    chunk_keys: list[ChunkCoords] | None = None
    if object_ids is None or scoped:
        # Intersection of physical keys, bbox-implied set, and explicit
        # chunks whitelist.
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

    if object_ids is not None:
        return _read_mesh_objects(
            level_group,
            object_ids,
            scope=None if chunk_keys is None else set(chunk_keys),
            bbox=bbox,
            dtype=dtype,
            ndim=ndim,
            link_width=link_width,
        )
    assert chunk_keys is not None

    # Faces come only from the link cells whose every endpoint chunk is
    # being read; ``None`` (unscoped) reads the whole family.
    link_cells = link_cells_within(level_group, chunk_keys) if scoped else None

    # Prefetch every chunk (vertices, offsets) and the face cells in one
    # async gather.  Subsequent ``read_bytes`` calls below hit the cache
    # instead of paying one round-trip per chunk.  ``links/0`` is a group
    # of one array per offsets segment, and each array's cells are keyed
    # by *source* chunk, which for the non-zero offsets is not the
    # ``chunk_keys`` set.
    chunk_key_strs = [".".join(str(c) for c in cc) for cc in chunk_keys]
    _prefetch_plan: list[tuple[str, list[str]]] = [
        (VERTICES, chunk_key_strs),
        (VERTEX_FRAGMENTS, chunk_key_strs),
        (LINK_FRAGMENTS, chunk_key_strs),
        *_link_plan(level_group, link_cells),
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

        # Faces: one family, so intra and cross records come back from one
        # call already in global ``(chunk, local_idx)`` form — a single
        # remap covers both, with winding restored from ``perm_idx``.  A
        # face touching a chunk outside ``chunk_keys`` has no offset and is
        # dropped, which is what applies the bbox/chunks filter to faces.
        all_faces: list[npt.NDArray] = []
        # As arrays, not one tuple per face: the remap is a gather.
        face_chunks, face_vi = read_link_arrays(
            level_group, delta=0, cells=link_cells,
        )
        if face_vi.shape[0] and face_vi.shape[1] == link_width:
            rows = link_endpoints_to_rows(face_chunks, face_vi, chunk_offsets)
            keep = (rows >= 0).all(axis=1)
            if keep.any():
                all_faces.append(rows[keep])

        if all_faces:
            faces_out = np.concatenate(all_faces, axis=0)
        else:
            faces_out = np.zeros((0, link_width), dtype=np.int64)

        if bbox is not None:
            positions_out, faces_out, _ = _bbox_filter(
                positions_out, faces_out, bbox, link_width,
            )

        return {
            "vertices": positions_out,
            "faces": faces_out,
            "vertex_count": len(positions_out),
            "face_count": len(faces_out),
        }
    finally:
        _batched_reads_cm.__exit__(None, None, None)


def _read_mesh_objects(
    level_group: FsGroup,
    object_ids: list[int],
    *,
    scope: set[ChunkCoords] | None,
    bbox: BoundingBox | None,
    dtype: np.dtype,
    ndim: int,
    link_width: int,
) -> dict[str, Any]:
    """:func:`read_mesh` for ``object_ids=``: the named objects and nothing else.

    The object index names each object's ``(chunk, fragment)`` pairs, and
    ``write_mesh`` writes one fragment per object per chunk, so the
    vertices are exactly those fragments' rows.  Faces come from the
    link cells wholly inside the objects' chunks
    (:func:`link_cells_within`) and are kept only when every endpoint is
    one of those rows.  Nothing outside the objects' chunks is read.

    ``scope`` is the chunk set a ``bbox`` / ``chunks`` / attribute filter
    narrowed the read to, or ``None``; fragments outside it are dropped.
    """
    wanted: list[int] = list(dict.fromkeys(int(o) for o in object_ids))
    manifests = read_object_manifests(level_group, ids=wanted) if wanted else {}

    entries: list[tuple[int, ChunkCoords, int]] = []
    for oid in wanted:
        for cc, frag in manifests.get(oid, ()):
            cc = tuple(int(c) for c in cc)
            if scope is None or cc in scope:
                entries.append((oid, cc, int(frag)))
    if not entries:
        result = _empty_mesh_result(ndim, link_width)
        result["object_ids"] = np.zeros(0, dtype=np.int64)
        return result

    obj_chunks = sorted({cc for _, cc, _ in entries})
    link_cells = link_cells_within(level_group, obj_chunks)
    key_strs = [".".join(str(c) for c in cc) for cc in obj_chunks]
    plan: list[tuple[str, list[str]]] = [
        (VERTICES, key_strs),
        (VERTEX_FRAGMENTS, key_strs),
        *_link_plan(level_group, link_cells),
    ]

    positions: list[npt.NDArray] = []
    owners: list[npt.NDArray[np.int64]] = []
    # Chunk-local row -> output row, -1 where the row is not ours.  This
    # is the whole face filter: a face survives when every endpoint maps.
    row_maps: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    chunk_state: dict[ChunkCoords, tuple[Any, npt.NDArray]] = {}
    running = 0
    with level_group.batched_reads(plan):
        for oid, cc, frag in entries:
            state = chunk_state.get(cc)
            if state is None:
                fi = read_vertex_fragment_index(level_group, cc)
                full = read_chunk_vertex_buffer(
                    level_group, cc, dtype=dtype, ndim=ndim,
                )
                state = chunk_state[cc] = (fi, full)
                row_maps[cc] = np.full(len(full), -1, dtype=np.int64)
            fi, full = state
            if frag >= fi.num_fragments:
                raise ArrayError(
                    f"object {oid}'s manifest names fragment {frag} of chunk "
                    f"{cc}, which has {fi.num_fragments}"
                )
            if fi.is_range(frag):
                start, count = fi.range(frag)
                rows = np.arange(start, start + count, dtype=np.int64)
            else:
                rows = np.asarray(fi.indices(frag), dtype=np.int64)
            row_maps[cc][rows] = np.arange(
                running, running + len(rows), dtype=np.int64,
            )
            positions.append(full[rows])
            owners.append(np.full(len(rows), oid, dtype=np.int64))
            running += len(rows)

        face_chunks, face_vi = read_link_arrays(
            level_group, delta=0, cells=link_cells,
        )

    positions_out = np.concatenate(positions, axis=0)
    owners_out = np.concatenate(owners)
    faces_out = np.zeros((0, link_width), dtype=np.int64)
    if face_vi.shape[0] and face_vi.shape[1] == link_width:
        rows = _endpoints_through_row_maps(face_chunks, face_vi, row_maps)
        faces_out = rows[(rows >= 0).all(axis=1)]

    if bbox is not None:
        positions_out, faces_out, keep = _bbox_filter(
            positions_out, faces_out, bbox, link_width,
        )
        if keep is not None:
            owners_out = owners_out[keep]

    return {
        "vertices": positions_out,
        "faces": faces_out,
        "vertex_count": len(positions_out),
        "face_count": len(faces_out),
        "object_ids": owners_out,
    }


def _link_plan(
    level_group: FsGroup, link_cells: dict[str, list[str]] | None,
) -> list[tuple[str, list[str]]]:
    """Prefetch-plan entries for the face cells a read will touch --
    ``link_cells`` from :func:`link_cells_within`, or, for ``None``,
    every cell of every offsets array."""
    family = links_group_path(0)
    if link_cells is None:
        link_cells = {
            seg: level_group.list_chunks(f"{family}/{seg}")
            for seg in list_link_offsets(level_group, 0)
        }
    return [(f"{family}/{seg}", keys) for seg, keys in link_cells.items()]


def _endpoints_through_row_maps(
    chunks: npt.NDArray[np.int64],
    vi: npt.NDArray[np.int64],
    row_maps: dict[ChunkCoords, npt.NDArray[np.int64]],
) -> npt.NDArray[np.int64]:
    """Output row of every endpoint, ``-1`` where it is not a row we took.

    :func:`link_endpoints_to_rows` for a read that takes *some* rows of a
    chunk rather than all of them, so a chunk offset is not enough: each
    chunk carries its own row map.  The maps are laid end to end so the
    whole remap is one gather; chunks are matched by raveled coordinate
    against a sorted key list rather than by ``np.unique(axis=0)``, which
    was half of a one-object read.
    """
    m, width, sid_ndim = chunks.shape
    if m == 0 or not row_maps:
        return np.full((m, width), -1, dtype=np.int64)
    known = np.asarray(list(row_maps), dtype=np.int64).reshape(-1, sid_ndim)
    tables = list(row_maps.values())
    lengths = np.fromiter((len(t) for t in tables), dtype=np.int64, count=len(tables))
    base = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    table = np.concatenate(tables) if tables else np.zeros(0, dtype=np.int64)

    flat = chunks.reshape(-1, sid_ndim)
    flat_vi = vi.reshape(-1)
    lo = np.minimum(known.min(axis=0), flat.min(axis=0))
    ext = np.maximum(known.max(axis=0), flat.max(axis=0)) - lo + 1
    known_keys = np.ravel_multi_index((known - lo).T, ext)
    order = np.argsort(known_keys)
    sorted_keys = known_keys[order]
    keys = np.ravel_multi_index((flat - lo).T, ext)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    which = order[pos]
    ok = (sorted_keys[pos] == keys) & (flat_vi >= 0) & (flat_vi < lengths[which])
    out = np.full(flat_vi.shape, -1, dtype=np.int64)
    out[ok] = table[base[which[ok]] + flat_vi[ok]]
    return out.reshape(m, width)


def _bbox_filter(
    positions: npt.NDArray,
    faces: npt.NDArray[np.int64],
    bbox: BoundingBox,
    link_width: int,
) -> tuple[npt.NDArray, npt.NDArray[np.int64], npt.NDArray[np.int64] | None]:
    """Keep the vertices inside ``bbox`` and the faces wholly on them.

    Returns ``(positions, faces, keep)``, ``keep`` being the surviving
    input rows -- or ``None`` when every vertex survived and nothing was
    reindexed.
    """
    bbox_min, bbox_max = np.asarray(bbox[0]), np.asarray(bbox[1])
    node_mask = np.all((positions >= bbox_min) & (positions <= bbox_max), axis=1)
    if np.all(node_mask):
        return positions, faces, None
    n_total = len(positions)
    keep = np.flatnonzero(node_mask)
    # Vectorized remap via a lookup table (mirrors read_graph): dropped
    # vertices map to -1; a face survives only if every endpoint survives.
    if len(faces) > 0:
        remap = np.full(n_total, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        new_faces = remap[faces]
        faces = new_faces[(new_faces >= 0).all(axis=1)].astype(np.int64, copy=False)
    else:
        faces = np.zeros((0, link_width), dtype=np.int64)
    return positions[keep], faces, keep


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
