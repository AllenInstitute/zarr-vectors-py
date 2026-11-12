"""Convert a level between uniform chunks and octree chunks, both ways.

Octree chunks are optional: every writer still writes uniform chunks, and
a level becomes octree-chunked only by :func:`to_octree`, which plans
leaves over the level's own cells (the uniform chunk is the finest leaf)
and rewrites the per-chunk arrays leaf by leaf.  :func:`to_uniform` undoes
it: every cell gets back its vertex rows in their order, its fragments,
its objects' manifest entries (under their own object ids) and its links.

A leaf's vertex block is its cells' blocks laid end to end (cells in Z
order) and its fragments are theirs, so the conversion moves rows without
regrouping them; ``octree_chunks/cells`` records which cell each run came
from, which is what makes the way back exact.

On a sharded level no leaf is larger than a shard or crosses a shard
boundary (:func:`~zarr_vectors.octree_chunks.table.plan_leaves` with the
level's shard grid), and both conversions keep the level's packing: the
rewritten arrays are sharded as the level's own ``vertices`` are.

Both conversions are safe to interrupt.  Everything that can fail on the
input -- unsupported content, links longer than a cell, manifests or
links naming rows that are not there -- is checked while the level is
only being read.  The new layout is then written into a staging group
under the level (``<level>/octree_chunks_staging``), which readers never
look at, and swapped in array by array.  The swap starts by marking the
level ``converting``, so a reader refuses it rather than mixing the two
layouts; running the same conversion (or the other one) again finishes
an interrupted swap, and discards a staging group whose swap never
started.  So a failure or kill at any point leaves the old level intact,
or a level the next call completes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_OCTREE_CHUNKS,
    FRAGMENT_ATTRIBUTES,
    GEOM_GRAPH,
    GEOM_POINT_CLOUD,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    LINKS,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    _chunk_key,
    _ensure_chunk_arrays,
    _parse_chunk_key,
    create_links_array,
    create_object_index_array,
    create_vertices_array,
    level_grid_layout,
    list_link_offsets,
    object_shard_rows,
    open_write_session,
    read_all_object_manifests_csr,
    read_chunk_vertices,
    read_link_arrays,
    read_vertex_fragment_index,
    stamp_fragments_tile,
    write_chunk_vertices,
    write_links,
    write_object_index,
)
from zarr_vectors.core.metadata import get_level_chunk_shape, normalise_shard_shape
from zarr_vectors.core.paths import parse_delta
from zarr_vectors.core.store import (
    _finalize_write,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.encoding.compression import codecs_for_create_array, resolve_compressor
from zarr_vectors.encoding.octree_marker import OCTREE_MARKER_CODEC, is_marker
from zarr_vectors.encoding.ragged import decode_ragged_blob, encode_ragged_blob
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.octree_chunks.guard import LEVEL_KEY, LEVEL_META_KEY, converting
from zarr_vectors.octree_chunks.table import (
    LeafTable,
    OctreeError,
    _morton,
    format_link_segment,
    leaf_cap,
    parse_link_segment,
    plan_leaves,
)

TABLES = "octree_chunks"             # group holding the leaf and cell tables
LEAVES_TABLE = f"{TABLES}/leaves"    # (N, 5): depth, i, j, k, weight
CELLS_TABLE = f"{TABLES}/cells"      # (C, 6): cell i, j, k, leaf, vertices, fragments
STAGING = "octree_chunks_staging"    # group a conversion writes into before the swap
_STAGING_KEY = "octree_chunks_staging"  # its record: direction, arrays, final metadata
_LEVEL_META_KEY = LEVEL_META_KEY
_PER_CHUNK = (VERTICES, VERTEX_FRAGMENTS, LINKS, LINK_FRAGMENTS, OBJECT_INDEX)
_SWAPPED = (*_PER_CHUNK, TABLES)     # every array a conversion replaces or removes


# -- reading the level's octree state -----------------------------------------

def octree_meta(level_group: Any) -> dict[str, Any] | None:
    """The level's octree block, or ``None`` for a uniform level.

    A level whose conversion is part way through its swap holds
    ``{"converting": "to_octree" | "to_uniform"}`` here.
    """
    level = level_group.attrs.get(_LEVEL_META_KEY)
    block = level.get(LEVEL_KEY) if isinstance(level, dict) else None
    return dict(block) if isinstance(block, dict) else None


def leaf_table(level_group: Any) -> LeafTable:
    """The level's :class:`LeafTable`; raises on a uniform level, and on
    one an interrupted conversion left part way through its swap."""
    meta = octree_meta(level_group)
    if meta is None:
        raise OctreeError("this level has uniform chunks, not octree chunks")
    if "converting" in meta:
        raise OctreeError(
            f"this level is part way through {meta['converting']} (interrupted?); "
            f"run zarr_vectors.octree_chunks.{meta['converting']} on it again to finish"
        )
    cache = level_group.__dict__.setdefault("_octree_leaf_table", {})
    key = (meta.get("max_depth"), tuple(meta.get("origin_cell", ())), meta.get("leaves"))
    if key not in cache:
        rows = np.asarray(level_group._zarr[LEAVES_TABLE][...])
        cache.clear()
        cache[key] = LeafTable.from_rows(rows, meta)
    return cache[key]


def cell_table(level_group: Any) -> npt.NDArray[np.int64]:
    """``(C, 6)`` rows ``(i, j, k, leaf, vertices, fragments)``, in storage order."""
    return np.asarray(level_group._zarr[CELLS_TABLE][...], dtype=np.int64).reshape(-1, 6)


def array_shard_grid(level_group: Any, name: str = VERTICES
                     ) -> tuple[tuple[int, ...] | None, tuple[int, ...]] | None:
    """``(shard_shape, origin)`` of the level's per-chunk array ``name``, in
    cells: its cells per shard (``None`` when unsharded) and the cell its
    grid starts at, where the first shard starts.  ``None`` when the level
    has no such array."""
    import zarr

    try:
        node = level_group._zarr[name]
    except KeyError:
        return None
    if not isinstance(node, zarr.Array):
        return None
    meta = node.metadata
    codecs = tuple(getattr(meta, "codecs", ()))
    shard = None
    if codecs and type(codecs[0]).__name__ in ("ShardingCodec", "OctreeShardingCodec"):
        # Zarr-vectors cells are the inner chunks, one per cell, so the
        # outer (stored) chunk is the shard.
        shard = tuple(int(c) for c in meta.chunk_grid.chunk_shape)
    raw = node.attrs.get("chunk_grid_origin")
    origin = tuple(int(o) for o in raw) if raw else (0,) * len(meta.shape)
    return shard, origin


def _level_packing(root_meta: Any, level_group: Any, chunk_shape: tuple[float, ...]
                   ) -> tuple[tuple[int, ...] | None, tuple[int, ...]]:
    """The shard grid a conversion writes the level in: the shard shape of
    the level's ``vertices`` (the store's declaration when it has none) and
    the grid origin the level's bounds give, where the rewritten arrays
    start.  A conversion keeps the level's packing: a level sharded by
    ``shard_store`` without a root declaration stays sharded."""
    origin, _ = level_grid_layout(root_meta.bounds, chunk_shape)
    found = array_shard_grid(level_group)
    shard = found[0] if found is not None else normalise_shard_shape(root_meta.shard_shape, len(origin))
    return (None if shard is None else tuple(int(c) for c in shard)), tuple(int(o) for o in origin)


def _check_staged_packing(stage: Any, shard: tuple[int, ...] | None, origin: tuple[int, ...]) -> None:
    """Refuse, before the swap, staged arrays not packed on the shard grid
    the leaves were planned for (their leaves could straddle its shards)."""
    found = array_shard_grid(stage)
    if found is None:
        return
    if found[0] != shard or (shard is not None and found[1] != origin):
        raise OctreeError(
            f"the rewritten level came out with shards of {found[0]} cells from {found[1]}, "
            f"not the {shard} from {origin} its leaves were planned for"
        )


# -- helpers -------------------------------------------------------------------

def _level_compressors(level_group: Any) -> list[dict[str, Any]] | None:
    """The cell compressors of the level's ``vertices`` array, to keep on
    rewrite (without the octree marker codec, which :func:`_marked` sets)."""
    try:
        codecs = level_group._zarr[VERTICES].metadata.to_dict()["codecs"]
    except Exception:
        return None
    if codecs and "sharding" in str(codecs[0].get("name", "")):
        codecs = codecs[0]["configuration"]["codecs"]
    out = [c for c in codecs
           if c.get("name") not in ("vlen-bytes", "bytes", "vlen-utf8") and not is_marker(c)]
    return out or None


def _marked(compressor: Any, octree: bool) -> list[dict[str, Any]] | None:
    """``compressor`` as a codec list, ending in the octree marker codec
    (:mod:`zarr_vectors.encoding.octree_marker`) when ``octree``, without it
    otherwise.  The marker makes zarr readers that do not know octree
    chunks refuse the level's arrays instead of reading leaves as cells."""
    codecs = [c for c in codecs_for_create_array(resolve_compressor(compressor)) if not is_marker(c)]
    if octree:
        codecs.append({"name": OCTREE_MARKER_CODEC})
    return codecs or None


def _check_convertible(root: Any, level_group: Any, level: int) -> tuple[Any, tuple[float, ...]]:
    meta = read_root_metadata(root)
    kinds = set(meta.geometry_types)
    if not kinds <= {GEOM_POINT_CLOUD, GEOM_GRAPH}:
        raise OctreeError(
            f"octree chunks support point clouds and graphs so far; this store holds {sorted(kinds)}"
        )
    if meta.sid_ndim != 3:
        raise OctreeError(f"octree chunks need 3 spatial index dimensions, not {meta.sid_ndim}")
    for name in (VERTEX_ATTRIBUTES, FRAGMENT_ATTRIBUTES, LINK_ATTRIBUTES):
        if level_group.array_exists(name):
            raise OctreeError(f"converting a level with {name} is not supported yet")
    if level_group.array_exists(f"{LINKS}/0"):
        fam = level_group.read_array_meta(f"{LINKS}/0") or {}
        if fam.get("directed") or fam.get("store", "canonical") != "canonical" or int(fam.get("link_width", 2)) != 2:
            raise OctreeError("octree chunks support undirected two-endpoint links stored once (canonical)")
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:
        level_meta = None
    if level_meta is not None and level_meta.chunk_dims:
        raise OctreeError("an attribute-chunked level cannot be converted")
    return meta, tuple(get_level_chunk_shape(meta, level_meta))


def _check_no_cross_level_links(root: Any, level: int) -> None:
    """Refuse when any level holds links into or out of ``level``: they
    address its vertices by cell and row, which a conversion changes."""
    for m in list_resolution_levels(root):
        lg = get_resolution_level(root, m)
        if not lg.array_exists(LINKS):
            continue
        for seg in lg[LINKS].child_names():
            try:
                delta = parse_delta(seg)
            except ValueError:
                continue
            if delta == 0 or (m != level and m + delta != level):
                continue
            if list_link_offsets(lg, delta):
                raise OctreeError(
                    f"level {m} holds links to level {m + delta}; converting level {level} "
                    f"would leave them naming the wrong vertices (links between levels are "
                    f"not supported with octree chunks yet)"
                )


def _set_level_block(level_group: Any, block: dict[str, Any] | None) -> None:
    level = dict(level_group.attrs.get(_LEVEL_META_KEY) or {})
    if block is None:
        level.pop(LEVEL_KEY, None)
    else:
        level[LEVEL_KEY] = block
    level_group.attrs.update({_LEVEL_META_KEY: level})


def _set_capability(root: Any, on: bool) -> None:
    block = dict(root.attrs.to_dict().get("zarr_vectors", {}))
    caps = [c for c in block.get("format_capabilities") or [] if c != CAP_OCTREE_CHUNKS]
    if on:
        caps.append(CAP_OCTREE_CHUNKS)
    block["format_capabilities"] = caps
    root.attrs.update({"zarr_vectors": block})


def _any_octree_level(root: Any) -> bool:
    return any(octree_meta(get_resolution_level(root, m)) is not None
               for m in list_resolution_levels(root))


def _vertex_dtype(level_group: Any) -> np.dtype:
    try:
        return np.dtype(level_group.read_array_meta(VERTICES).get("dtype", "float32"))
    except Exception:
        return np.dtype("float32")


def _locate(table: npt.ArrayLike, query: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Row of ``table`` (unique ``(N, 3)`` rows) equal to each query row, ``-1`` if none."""
    table = np.asarray(table, dtype=np.int64).reshape(-1, 3)
    query = np.asarray(query, dtype=np.int64).reshape(-1, 3)
    out = np.full(len(query), -1, dtype=np.int64)
    if not len(table) or not len(query):
        return out
    lo = table.min(axis=0)
    span = table.max(axis=0) - lo + 1

    def code(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        rel = rows - lo
        return (rel[:, 0] * span[1] + rel[:, 1]) * span[2] + rel[:, 2]

    tc = code(table)
    order = np.argsort(tc)
    rel = query - lo
    ok = ((rel >= 0) & (rel < span)).all(axis=1)
    qc = code(np.where(ok[:, None], query, lo))
    pos = np.minimum(np.searchsorted(tc[order], qc), len(tc) - 1)
    hit = ok & (tc[order][pos] == qc)
    out[hit] = order[pos[hit]]
    return out


def _read_cells(level_group: Any, cells: list[tuple[int, ...]], dtype: np.dtype) -> list[list[npt.NDArray]]:
    """Each cell's fragments, checked to tile its rows in order."""
    keys = [_chunk_key(c) for c in cells]
    out: list[list[npt.NDArray]] = []
    with level_group.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)]):
        for cc in cells:
            fi = read_vertex_fragment_index(level_group, cc)
            start = 0
            for f in range(fi.num_fragments):
                if not fi.is_range(f) or fi.range(f)[0] != start:
                    raise OctreeError(
                        f"cell {cc}: its fragments do not tile its rows in order, which "
                        f"octree chunks need (one run per fragment)"
                    )
                start += fi.range(f)[1]
            out.append(read_chunk_vertices(level_group, cc, dtype=dtype, ndim=3, _leaf=True))
    return out


# -- the object layer ------------------------------------------------------------

@dataclass
class _Objects:
    """A level's object index as CSR rows, with each row's real object id."""

    meta: dict[str, Any]                # the index's metadata as found
    ids: npt.NDArray[np.int64]          # (n,) object id of each row
    offsets: npt.NDArray[np.int64]      # (n + 1,) row -> its entries
    coords: npt.NDArray[np.int64]       # (M, 3) chunk of each entry
    frags: npt.NDArray[np.int64]        # (M,) fragment of each entry
    layout: str | None                  # "vlen" / "dense"; None: no rows, kept as found
    block_dtype: Any
    shard_rows: int | None


def _read_objects(level_group: Any) -> _Objects | None:
    """The level's object index, or ``None`` when it has none.

    Ids come from the index's id table, so sparse ids (``7, 1007, ...``)
    survive the rewrite; rows must be in increasing id order, the order
    :func:`write_object_index` writes, so object attribute rows stay
    aligned with them.
    """
    from zarr_vectors.core import dense_manifests as dense

    if not level_group.array_exists(OBJECT_INDEX):
        return None
    meta = dict(level_group.read_array_meta(OBJECT_INDEX) or {})
    shard_rows = object_shard_rows(level_group)
    if not int(meta.get("num_objects", 0) or 0) or "sid_ndim" not in meta:
        z = np.zeros(0, np.int64)
        return _Objects(meta, z, np.zeros(1, np.int64), np.zeros((0, 3), np.int64), z, None, None, shard_rows)
    if int(meta["sid_ndim"]) != 3:
        raise OctreeError(f"the object index has {meta['sid_ndim']} spatial index dimensions, not 3")
    csr = read_all_object_manifests_csr(level_group)
    offsets = np.asarray(csr.offsets, dtype=np.int64).reshape(-1)
    n = len(offsets) - 1
    ids = np.arange(n, dtype=np.int64) if csr.object_ids is None else np.asarray(csr.object_ids, np.int64)
    if len(ids) != n:
        raise OctreeError(f"the object id table has {len(ids)} rows for {n} objects")
    if n > 1 and (np.diff(ids) <= 0).any():
        raise OctreeError("the object id table is not in increasing order, which converting needs")
    layout = "dense" if dense.is_dense(level_group, meta) else "vlen"
    return _Objects(
        meta, ids, offsets,
        np.asarray(csr.chunk_coords, dtype=np.int64).reshape(-1, 3),
        np.asarray(csr.fragment_idx, dtype=np.int64).reshape(-1),
        layout, meta.get(dense.BLOCK_DTYPE_ATTR) if layout == "dense" else None, shard_rows,
    )


def _write_objects(group: Any, objs: _Objects, coords: npt.NDArray[np.int64],
                   frags: npt.NDArray[np.int64]) -> None:
    """Write ``objs`` with its entries moved to ``coords`` / ``frags``, keyed by real id."""
    create_object_index_array(group)
    if objs.layout is None:
        group.write_array_meta(OBJECT_INDEX, objs.meta)
        return
    cl, fl, off = coords.tolist(), frags.tolist(), objs.offsets.tolist()
    manifests = {
        oid: [(tuple(cl[m]), int(fl[m])) for m in range(off[o], off[o + 1])]
        for o, oid in enumerate(objs.ids.tolist())
    }
    write_object_index(group, manifests, sid_ndim=3, layout=objs.layout, block_dtype=objs.block_dtype)


def _shard_objects_like(group: Any, objs: _Objects | None) -> None:
    """Give the rewritten object layer the row sharding the old one had."""
    if objs is not None and objs.shard_rows is not None and objs.layout is not None:
        from zarr_vectors.sharding.io import shard_object_layer

        shard_object_layer(group, objs.shard_rows)


# -- octree links ----------------------------------------------------------------

@dataclass
class _LinkPlan:
    """Links filed under their source leaf, each with its ``(offset, Δ)`` segment."""

    src: npt.NDArray[np.int64]
    r_src: npt.NDArray[np.int64]
    off: npt.NDArray[np.int64]
    dlt: npt.NDArray[np.int64]
    r_oth: npt.NDArray[np.int64]


def _plan_octree_links(table: LeafTable, leaf_a: npt.NDArray[np.int64], row_a: npt.NDArray[np.int64],
                       leaf_b: npt.NDArray[np.int64], row_b: npt.NDArray[np.int64]) -> _LinkPlan:
    """Choose each link's source leaf and segment; raises (before anything is
    written) when a link joins leaves that are not adjacent."""
    leaf_a, leaf_b = np.asarray(leaf_a, np.int64), np.asarray(leaf_b, np.int64)
    a_src = (leaf_a == leaf_b) | (table.link_source(leaf_a, leaf_b) if len(leaf_a) else np.zeros(0, bool))
    src = np.where(a_src, leaf_a, leaf_b)
    oth = np.where(a_src, leaf_b, leaf_a)
    off = np.zeros((len(src), 3), dtype=np.int64)
    dlt = np.zeros(len(src), dtype=np.int64)
    cross = src != oth
    if cross.any():
        off[cross], dlt[cross] = table.link_offsets(src[cross], oth[cross])
    return _LinkPlan(src, np.where(a_src, row_a, row_b).astype(np.int64), off, dlt,
                     np.where(a_src, row_b, row_a).astype(np.int64))


def _write_octree_links(level_group: Any, table: LeafTable, plan: _LinkPlan) -> int:
    """File every link under its source leaf, in the array its ``(offset, Δ)`` names."""
    src = plan.src
    family = f"{LINKS}/0"
    level_group.write_array_meta(family, {
        "zv_array": "links_family", "level_delta": 0, "link_width": 2,
        "directed": False, "store": "canonical", "sid_ndim": 3,
        "num_links": int(len(src)), "num_physical_records": int(len(src)),
        "octree_chunks": True,
    })
    if not len(src):
        return 0
    groups = np.column_stack([plan.off, plan.dlt])
    seg_rows, seg_of = np.unique(groups, axis=0, return_inverse=True)
    seg_of = seg_of.reshape(-1)
    wanted = []
    for g in seg_rows.tolist():
        seg = format_link_segment(g[:3], g[3])
        wanted.append((f"{family}/{seg}", {
            "zv_array": "links", "dtype": "int64", "offsets": [g[:3]],
            "depth_deltas": [int(g[3])], "has_perm": False, "link_width": 2, "level_delta": 0,
        }))
    _ensure_chunk_arrays(level_group, wanted)
    anchors = table.anchors()
    for s, (path, _meta) in enumerate(wanted):
        m = np.flatnonzero(seg_of == s)
        order = m[np.argsort(src[m], kind="stable")]
        leaves, starts = np.unique(src[order], return_index=True)
        cells = []
        for leaf, rows_idx in zip(leaves.tolist(), np.split(order, starts[1:])):
            rows = np.column_stack([plan.r_src[rows_idx], plan.r_oth[rows_idx]]).astype(np.int64)
            cells.append((_chunk_key(tuple(anchors[leaf].tolist())), encode_ragged_blob([rows], np.dtype(np.int64))))
        level_group.write_cells(path, cells)
    return int(len(src))


def read_octree_link_cells(level_group: Any, table: LeafTable, leaves: npt.ArrayLike | None = None
                           ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Links filed under ``leaves`` (all when ``None``) as ``(src_leaf, src_row,
    other_leaf, other_row)``; ``other_leaf`` is ``-1`` where it is not stored."""
    family = f"{LINKS}/0"
    segments = list_link_offsets(level_group, 0)
    anchors = table.anchors()
    want = None if leaves is None else set(np.asarray(leaves, dtype=np.int64).tolist())
    anchor_leaf = {tuple(a): n for n, a in enumerate(anchors.tolist())}
    plan, per_seg = [], []
    for seg in segments:
        path = f"{family}/{seg}"
        present = level_group.list_chunks(path)
        keys = [k for k in present if want is None or anchor_leaf.get(_parse_chunk_key(k)) in want]
        if keys:
            plan.append((path, keys))
            per_seg.append((seg, path, keys))
    out = ([], [], [], [])
    with level_group.batched_reads(plan):
        for seg, path, keys in per_seg:
            off, dlt = parse_link_segment(seg)
            intra = off == (0, 0, 0) and dlt == 0
            for key in keys:
                src = anchor_leaf[_parse_chunk_key(key)]
                blocks = decode_ragged_blob(level_group.read_bytes(path, key), np.dtype(np.int64), ncols=2)
                if not blocks:
                    continue
                rows = np.concatenate([np.asarray(b).reshape(-1, 2) for b in blocks])
                oth = src if intra else table.resolve(src, off, dlt)
                out[0].append(np.full(len(rows), src, dtype=np.int64))
                out[1].append(rows[:, 0])
                out[2].append(np.full(len(rows), oth, dtype=np.int64))
                out[3].append(rows[:, 1])
    if not out[0]:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    return tuple(np.concatenate(x).astype(np.int64) for x in out)  # type: ignore[return-value]


# -- staging and the swap ------------------------------------------------------------

def _open_staging(level_group: Any) -> Any:
    """A fresh staging group under the level, carrying the level's metadata
    (without its octree block), so writers size and pack arrays as the
    level's own."""
    level_group.delete_subtree(STAGING)
    stage = level_group.require_group(STAGING)
    attrs = dict(level_group.attrs.to_dict())
    level = dict(attrs.get(_LEVEL_META_KEY) or {})
    level.pop(LEVEL_KEY, None)
    attrs[_LEVEL_META_KEY] = level
    stage.attrs.update(attrs)
    return stage


def _seal(stage: Any, direction: str, block: dict[str, Any] | None, summary: dict[str, Any]) -> None:
    """Record that ``stage`` is complete, with the level metadata it implies."""
    level = dict(stage.attrs.get(_LEVEL_META_KEY) or {})
    level.pop(LEVEL_KEY, None)
    if block is not None:
        level[LEVEL_KEY] = block
    stage.attrs.update({_STAGING_KEY: {
        "direction": direction, "complete": True,
        "arrays": [n for n in _SWAPPED if n in stage],
        "level": level,
        "summary": {k: ([[int(a), int(b)] for a, b in v.items()] if k == "depths" else v)
                    for k, v in summary.items()},
    }})


def _summary(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: ({int(a): int(b) for a, b in v} if k == "depths" else v)
            for k, v in (rec.get("summary") or {}).items()}


def _move(level_group: Any, stage: Any, name: str) -> None:
    """Put ``<level>/<STAGING>/<name>`` where ``<level>/<name>`` was.

    Idempotent across a kill: the staged node stays until it is in
    place (a rename on a local store; elsewhere a copy, then ``name``
    stamped ``moved`` in the staging record, then a delete -- a delete
    cut short leaves a partial staged node that :func:`_swap` must not
    copy again).
    """
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync
    from zarr.storage import LocalStore

    store = level_group.zarr_group.store
    base = level_group.zarr_group.path.strip("/")
    src = "/".join(p for p in (base, STAGING, name) if p)
    dst = "/".join(p for p in (base, name) if p)
    level_group.delete_subtree(name)
    stage._invalidate_node(name)
    if isinstance(store, LocalStore):
        os.replace(Path(store.root) / src, Path(store.root) / dst)
    else:
        async def _copy() -> None:
            proto = default_buffer_prototype()
            async for key in store.list_prefix(f"{src}/"):
                value = await store.get(key, prototype=proto)
                if value is not None:
                    await store.set(f"{dst}/{key[len(src) + 1:]}", value)

        sync(_copy())
        rec = dict(stage.attrs.get(_STAGING_KEY))
        rec["moved"] = [*rec.get("moved", []), name]
        stage.attrs.update({_STAGING_KEY: rec})
        stage.delete_subtree(name)
    level_group._invalidate_node(name)


def _swap(root: Any, level_group: Any, stage: Any) -> dict[str, Any]:
    """Replace the level's arrays with the staged ones.

    Marking the level ``converting`` is the commit point: from there on
    readers refuse the level and :func:`_resume` finishes the swap
    instead of discarding the staging group.
    """
    rec = dict(stage.attrs.get(_STAGING_KEY))
    with converting(level_group, stage, root):
        _set_level_block(level_group, {"converting": rec["direction"]})
        _set_capability(root, True)
        for name in _SWAPPED:
            if name in rec["arrays"]:
                if name in stage and name not in rec.get("moved", ()):
                    _move(level_group, stage, name)
            else:
                level_group.delete_subtree(name)
        level_group.attrs.update({_LEVEL_META_KEY: rec["level"]})
        level_group.delete_subtree(STAGING)
        level_group.__dict__.pop("_octree_leaf_table", None)
        _set_capability(root, _any_octree_level(root))
    return rec


def _resume(root: Any, level_group: Any) -> dict[str, Any] | None:
    """Deal with what an interrupted conversion left; returns its record
    when it finished one, else ``None``.

    A staging group whose swap never started is discarded (the level is
    as it was, and may have changed since the staging copy was made); a
    swap that started is finished.
    """
    block = octree_meta(level_group)
    marked = block is not None and "converting" in block
    if STAGING not in level_group:
        if marked:
            raise OctreeError(
                f"this level was part way through {block['converting']} but its staged "
                f"arrays ({STAGING}) are gone, so it cannot be finished"
            )
        return None
    stage = level_group[STAGING]
    if not marked:
        with converting(level_group):
            level_group.delete_subtree(STAGING)
        return None
    rec = stage.attrs.get(_STAGING_KEY)
    if not (isinstance(rec, dict) and rec.get("complete") and rec.get("direction") == block["converting"]):
        raise OctreeError(
            f"this level was part way through {block['converting']} but its staged arrays "
            f"are incomplete, so it cannot be finished"
        )
    _swap(root, level_group, stage)
    _finalize_write(root, f"{rec['direction']} (resumed)")
    return rec


# -- the conversions ----------------------------------------------------------

def to_octree(store: Any, *, target: float, level: int = 0, balanced: bool = True,
              max_leaf_cells: int | None = None, compressor: Any = "inherit",
              backend: str | None = None) -> dict[str, Any]:
    """Rewrite a uniform level with octree chunks of at most ``target`` vertices.

    The uniform chunk is the finest leaf: no leaf is smaller, and a cell
    heavier than ``target`` stays one leaf.  Links must not be longer than
    a cell (they never are between adjacent cells' leaves), and no level
    may hold links into or out of this one.  Object ids are kept.  An
    empty level becomes an octree level with no leaves.

    On a sharded level every leaf lies inside one shard: the root cube
    starts on a shard boundary and no leaf is wider than the largest power
    of two dividing every shard side (so a shard of ``(6, 6, 6)`` cells
    takes leaves of up to 2).  ``max_leaf_cells`` (a power of two) caps
    the leaf side further, or on an unsharded level at all; it may not
    exceed the shard's cap.  The cap and the shard grid are recorded in
    the level's octree block (``max_leaf_cells``, ``shard_shape``,
    ``shard_origin``).  The level keeps its packing (the shard shape of
    its ``vertices``).

    Nothing is changed when the level cannot be converted.  When a
    previous conversion of this level was interrupted during its swap,
    this finishes it first (returning that conversion's summary, with
    ``"resumed": True``, if it was a ``to_octree``).

    Returns a summary: leaf count, the largest leaf, the depths used, the
    leaf side cap (``max_leaf_cells``, ``None`` when uncapped).
    """
    root = open_store(store, mode="r+", backend=backend)
    level_group = get_resolution_level(root, level)
    done = _resume(root, level_group)
    if done is not None and done["direction"] == "to_octree":
        return {**_summary(done), "resumed": True}
    if octree_meta(level_group) is not None:
        raise OctreeError("this level already has octree chunks; to_uniform first")
    meta, chunk_shape = _check_convertible(root, level_group, level)
    _check_no_cross_level_links(root, level)
    shard, grid_origin = _level_packing(meta, level_group, chunk_shape)
    leaf_cap(max_leaf_cells, shard)  # a bad cap is refused before the level is read
    dtype = _vertex_dtype(level_group)
    comp = _marked(_level_compressors(level_group) if compressor == "inherit" else compressor, True)

    # -- plan: everything that can fail on the input, before any write.
    cells = sorted(_parse_chunk_key(k) for k in level_group.list_chunks(VERTICES))
    groups = _read_cells(level_group, cells, dtype)
    counts = np.array([sum(len(g) for g in gs) for gs in groups], dtype=np.int64)
    keep = counts > 0
    cells = [c for c, k in zip(cells, keep) if k]
    groups = [g for g, k in zip(groups, keep) if k]
    counts = counts[keep]
    cell_arr = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    objs = _read_objects(level_group)
    had_links = level_group.array_exists(f"{LINKS}/0")
    links = read_link_arrays(level_group, delta=0) if list_link_offsets(level_group, 0) else None

    table = plan_leaves(cell_arr, counts, target=target, balanced=balanced,
                        max_leaf_cells=max_leaf_cells, shard_shape=shard,
                        shard_origin=grid_origin if shard is not None else None)
    leaf_of = table.leaf_of_cells(cell_arr)
    rel = cell_arr - np.asarray(table.origin_cell)
    order = np.lexsort((_morton(rel, max(table.max_depth, 1)), leaf_of))

    # Lay each leaf's cells end to end.
    n_frag = np.array([len(g) for g in groups], dtype=np.int64)
    row_off = np.zeros(len(cells), dtype=np.int64)
    frag_off = np.zeros(len(cells), dtype=np.int64)
    leaf_groups: dict[int, list[npt.NDArray]] = {}
    for c in order.tolist():
        leaf = int(leaf_of[c])
        gl = leaf_groups.setdefault(leaf, [])
        row_off[c] = sum(len(g) for g in gl)
        frag_off[c] = len(gl)
        gl.extend(groups[c])
    cell_rows = np.column_stack([cell_arr[order], leaf_of[order], counts[order], n_frag[order]])
    anchors = table.anchors()

    obj_coords = obj_frags = None
    if objs is not None:
        n = _locate(cell_arr, objs.coords)
        bad = (n < 0) | (objs.frags < 0) | (objs.frags >= n_frag[np.maximum(n, 0)] if len(n_frag) else True)
        if np.any(bad):
            m = int(np.flatnonzero(bad)[0])
            raise OctreeError(
                f"an object manifest names fragment {int(objs.frags[m])} of cell "
                f"{tuple(objs.coords[m].tolist())}, which the level does not hold"
            )
        obj_coords = anchors[leaf_of[n]] if len(n) else np.zeros((0, 3), np.int64)
        obj_frags = frag_off[n] + objs.frags

    link_plan = None
    if had_links:
        ch, vi = links if links is not None else (np.zeros((0, 2, 3), np.int64), np.zeros((0, 2), np.int64))
        ch = np.asarray(ch, dtype=np.int64).reshape(-1, 2, 3)
        vi = np.asarray(vi, dtype=np.int64).reshape(-1, 2)
        cn_a, cn_b = _locate(cell_arr, ch[:, 0]), _locate(cell_arr, ch[:, 1])
        for cn, end in ((cn_a, 0), (cn_b, 1)):
            bad = (cn < 0) | (vi[:, end] < 0) | (vi[:, end] >= counts[np.maximum(cn, 0)] if len(counts) else True)
            if np.any(bad):
                r = int(np.flatnonzero(bad)[0])
                raise OctreeError(
                    f"a link names row {int(vi[r, end])} of cell {tuple(ch[r, end].tolist())}, "
                    f"which the level does not hold"
                )
        link_plan = _plan_octree_links(table, leaf_of[cn_a], row_off[cn_a] + vi[:, 0],
                                       leaf_of[cn_b], row_off[cn_b] + vi[:, 1])

    summary = {"leaves": len(table), "cells": len(cells),
               "links": int(len(link_plan.src)) if link_plan is not None else 0,
               "max_leaf_vertices": int(table.weight.max()) if len(table) else 0,
               "max_cell_vertices": int(counts.max()) if len(counts) else 0,
               "depths": {int(d): int(k) for d, k in zip(*np.unique(table.depth, return_counts=True))},
               "max_leaf_cells": table.max_leaf_cells}
    block = {**table.meta(), "weight": "vertices", "leaves": len(table)}

    # -- write the octree layout beside the level, then swap it in.
    with converting(level_group, root):
        stage = _open_staging(level_group)
        with converting(stage):
            with open_write_session(stage, compressor=comp, shard_shape=shard,
                                    bounds=meta.bounds, chunk_shape=chunk_shape):
                create_vertices_array(stage, dtype=str(dtype))
                for leaf, gl in sorted(leaf_groups.items()):
                    write_chunk_vertices(stage, tuple(anchors[leaf].tolist()), gl, dtype=dtype)
                if link_plan is not None:
                    _write_octree_links(stage, table, link_plan)
                if objs is not None:
                    _write_objects(stage, objs, obj_coords, obj_frags)
            stamp_fragments_tile(stage, 3)
            _check_staged_packing(stage, shard, grid_origin)
            _shard_objects_like(stage, objs)
            leaf_rows = table.to_rows()
            stage.write_array(LEAVES_TABLE, leaf_rows, chunks=(max(len(leaf_rows), 1), 5))
            stage.write_array(CELLS_TABLE, cell_rows.astype(np.int64), chunks=(max(len(cell_rows), 1), 6))
            _seal(stage, "to_octree", block, summary)
        _swap(root, level_group, stage)
    _finalize_write(root, "to_octree")
    return summary


def to_uniform(store: Any, *, level: int = 0, compressor: Any = "inherit",
               backend: str | None = None) -> dict[str, Any]:
    """Rewrite an octree-chunked level with its original uniform chunks.

    Refused while any level holds links into or out of this one.  As for
    :func:`to_octree`, nothing is changed when it fails, and a conversion
    of this level interrupted during its swap is finished first.
    """
    root = open_store(store, mode="r+", backend=backend)
    level_group = get_resolution_level(root, level)
    done = _resume(root, level_group)
    if done is not None and done["direction"] == "to_uniform":
        return {**_summary(done), "resumed": True}
    table = leaf_table(level_group)
    _check_no_cross_level_links(root, level)
    cells_t = cell_table(level_group)
    meta = read_root_metadata(root)
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:
        level_meta = None
    chunk_shape = tuple(get_level_chunk_shape(meta, level_meta))
    shard, _ = _level_packing(meta, level_group, chunk_shape)
    dtype = _vertex_dtype(level_group)
    comp = _marked(_level_compressors(level_group) if compressor == "inherit" else compressor, False)
    anchors = table.anchors()

    # -- plan: everything that can fail on the input, before any write.
    # Per leaf, its cells' runs in storage order.
    leaf_col = cells_t[:, 3]
    row_off = np.zeros(len(cells_t), dtype=np.int64)
    frag_off = np.zeros(len(cells_t), dtype=np.int64)
    for leaf in np.unique(leaf_col).tolist():
        idx = np.flatnonzero(leaf_col == leaf)
        row_off[idx] = np.concatenate(([0], np.cumsum(cells_t[idx, 4])[:-1]))
        frag_off[idx] = np.concatenate(([0], np.cumsum(cells_t[idx, 5])[:-1]))

    keys = [_chunk_key(tuple(a)) for a in anchors.tolist()]
    cell_groups: dict[tuple[int, ...], list[npt.NDArray]] = {}
    with level_group.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)]):
        for leaf in range(len(table)):
            gl = read_chunk_vertices(level_group, tuple(anchors[leaf].tolist()), dtype=dtype, ndim=3, _leaf=True)
            for n in np.flatnonzero(leaf_col == leaf).tolist():
                cell_groups[tuple(cells_t[n, :3].tolist())] = gl[frag_off[n]:frag_off[n] + cells_t[n, 5]]

    def run_of(leaf: npt.NDArray[np.int64], at: npt.NDArray[np.int64], start: npt.NDArray[np.int64],
               size: npt.NDArray[np.int64], what: str) -> npt.NDArray[np.int64]:
        """Cell row holding position ``at`` (a row or a fragment) of ``leaf``."""
        if not len(leaf):
            return np.zeros(0, dtype=np.int64)
        if not len(cells_t):
            raise OctreeError(f"a {what} names leaf {int(leaf[0])}, which holds no cells")
        key = leaf * (1 << 40) + at
        starts = leaf_col * (1 << 40) + start
        o = np.argsort(starts, kind="stable")
        pos = np.searchsorted(starts[o], key, side="right") - 1
        n = o[np.maximum(pos, 0)]
        bad = (leaf < 0) | (pos < 0) | (leaf_col[n] != leaf) | (at - start[n] >= size[n]) | (at < 0)
        if bad.any():
            b = int(np.flatnonzero(bad)[0])
            raise OctreeError(f"a {what} names position {int(at[b])} of leaf {int(leaf[b])}, "
                              f"which the level does not hold")
        return n

    objs = _read_objects(level_group)
    obj_coords = obj_frags = None
    if objs is not None:
        leaf = _locate(anchors, objs.coords)
        n = run_of(leaf, objs.frags, frag_off, cells_t[:, 5], "object manifest")
        obj_coords = cells_t[n, :3] if len(n) else np.zeros((0, 3), np.int64)
        obj_frags = objs.frags - frag_off[n]

    had_links = level_group.array_exists(f"{LINKS}/0")
    src, r_src, oth, r_oth = read_octree_link_cells(level_group, table)
    if (oth < 0).any():
        raise OctreeError("a link names a leaf that is not stored")
    na = run_of(src, r_src, row_off, cells_t[:, 4], "link")
    nb = run_of(oth, r_oth, row_off, cells_t[:, 4], "link")
    summary = {"cells": len(cell_groups), "links": int(len(src))}

    # -- write the uniform layout beside the level, then swap it in.
    with converting(level_group, root):
        stage = _open_staging(level_group)
        with converting(stage):
            with open_write_session(stage, compressor=comp, shard_shape=shard,
                                    bounds=meta.bounds, chunk_shape=chunk_shape):
                create_vertices_array(stage, dtype=str(dtype))
                for cc in sorted(cell_groups):
                    write_chunk_vertices(stage, cc, cell_groups[cc], dtype=dtype)
                if len(src):
                    write_links(stage, [], 3, delta=0, link_width=2, _arrays=(
                        np.stack([cells_t[na, :3], cells_t[nb, :3]], axis=1),
                        np.column_stack([r_src - row_off[na], r_oth - row_off[nb]]),
                    ))
                if had_links:
                    create_links_array(stage, link_width=2, delta=0, sid_ndim=3)
                if objs is not None:
                    _write_objects(stage, objs, obj_coords, obj_frags)
            stamp_fragments_tile(stage, 3)
            _shard_objects_like(stage, objs)
            _seal(stage, "to_uniform", None, summary)
        _swap(root, level_group, stage)
    _finalize_write(root, "to_uniform")
    return summary


__all__ = [
    "ArrayError", "array_shard_grid", "cell_table", "leaf_table", "octree_meta", "read_octree_link_cells",
    "to_octree", "to_uniform",
]
