"""Convert a level between uniform chunks and octree chunks, both ways.

Octree chunks are optional: every writer still writes uniform chunks, and
a level becomes octree-chunked only by :func:`to_octree`, which plans
leaves over the level's own cells (the uniform chunk is the finest leaf)
and rewrites the per-chunk arrays leaf by leaf.  :func:`to_uniform` undoes
it: every cell gets back its vertex rows in their order, its fragments,
its objects' manifest entries and its links.

A leaf's vertex block is its cells' blocks laid end to end (cells in Z
order) and its fragments are theirs, so the conversion moves rows without
regrouping them; ``octree_chunks/cells`` records which cell each run came
from, which is what makes the way back exact.
"""

from __future__ import annotations

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
    list_link_offsets,
    open_write_session,
    read_all_object_manifests,
    read_chunk_vertices,
    read_link_arrays,
    read_vertex_fragment_index,
    stamp_fragments_tile,
    write_chunk_vertices,
    write_links,
    write_object_index,
)
from zarr_vectors.core.metadata import get_level_chunk_shape
from zarr_vectors.core.store import (
    _finalize_write,
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.encoding.ragged import decode_ragged_blob, encode_ragged_blob
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.octree_chunks.table import (
    LeafTable,
    OctreeError,
    _morton,
    format_link_segment,
    parse_link_segment,
    plan_leaves,
)

LEVEL_KEY = "octree_chunks"          # block in the level's zarr_vectors_level attrs
TABLES = "octree_chunks"             # group holding the leaf and cell tables
LEAVES_TABLE = f"{TABLES}/leaves"    # (N, 5): depth, i, j, k, weight
CELLS_TABLE = f"{TABLES}/cells"      # (C, 6): cell i, j, k, leaf, vertices, fragments
_LEVEL_META_KEY = "zarr_vectors_level"
_PER_CHUNK = (VERTICES, VERTEX_FRAGMENTS, LINKS, LINK_FRAGMENTS, OBJECT_INDEX)


# -- reading the level's octree state -----------------------------------------

def octree_meta(level_group: Any) -> dict[str, Any] | None:
    """The level's octree block, or ``None`` for a uniform level."""
    level = level_group.attrs.get(_LEVEL_META_KEY)
    block = level.get(LEVEL_KEY) if isinstance(level, dict) else None
    return dict(block) if isinstance(block, dict) else None


def leaf_table(level_group: Any) -> LeafTable:
    """The level's :class:`LeafTable`; raises on a uniform level."""
    meta = octree_meta(level_group)
    if meta is None:
        raise OctreeError("this level has uniform chunks, not octree chunks")
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


# -- helpers -------------------------------------------------------------------

def _level_compressors(level_group: Any) -> list[dict[str, Any]] | None:
    """The cell compressors of the level's ``vertices`` array, to keep on rewrite."""
    try:
        codecs = level_group._zarr[VERTICES].metadata.to_dict()["codecs"]
    except Exception:
        return None
    if codecs and "sharding" in str(codecs[0].get("name", "")):
        codecs = codecs[0]["configuration"]["codecs"]
    out = [c for c in codecs if c.get("name") not in ("vlen-bytes", "bytes", "vlen-utf8")]
    return out or None


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
    for delta in (-2, -1, 1, 2):
        if list_link_offsets(level_group, delta):
            raise OctreeError("converting a level with links to other levels is not supported yet")
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


def _drop_per_chunk_arrays(level_group: Any, extra: tuple[str, ...] = ()) -> None:
    for name in (*_PER_CHUNK, *extra):
        try:
            level_group.delete_subtree(name)
        except Exception:
            pass
    level_group.__dict__.pop("_octree_leaf_table", None)


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


def _vertex_dtype(level_group: Any) -> np.dtype:
    try:
        return np.dtype(level_group.read_array_meta(VERTICES).get("dtype", "float32"))
    except Exception:
        return np.dtype("float32")


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
            out.append(read_chunk_vertices(level_group, cc, dtype=dtype, ndim=3))
    return out


def _write_octree_links(level_group: Any, table: LeafTable, leaf_a: npt.NDArray[np.int64],
                        row_a: npt.NDArray[np.int64], leaf_b: npt.NDArray[np.int64],
                        row_b: npt.NDArray[np.int64]) -> int:
    """File every link under its source leaf, in the array its ``(offset, Δ)`` names."""
    leaf_a, leaf_b = np.asarray(leaf_a, np.int64), np.asarray(leaf_b, np.int64)
    a_src = (leaf_a == leaf_b) | (table.link_source(leaf_a, leaf_b) if len(leaf_a) else np.zeros(0, bool))
    src = np.where(a_src, leaf_a, leaf_b)
    oth = np.where(a_src, leaf_b, leaf_a)
    r_src = np.where(a_src, row_a, row_b)
    r_oth = np.where(a_src, row_b, row_a)
    off = np.zeros((len(src), 3), dtype=np.int64)
    dlt = np.zeros(len(src), dtype=np.int64)
    cross = src != oth
    if cross.any():
        off[cross], dlt[cross] = table.link_offsets(src[cross], oth[cross])

    family = f"{LINKS}/0"
    level_group.write_array_meta(family, {
        "zv_array": "links_family", "level_delta": 0, "link_width": 2,
        "directed": False, "store": "canonical", "sid_ndim": 3,
        "num_links": int(len(src)), "num_physical_records": int(len(src)),
        "octree_chunks": True,
    })
    if not len(src):
        return 0
    groups = np.column_stack([off, dlt])
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
            rows = np.column_stack([r_src[rows_idx], r_oth[rows_idx]]).astype(np.int64)
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


# -- the conversions ----------------------------------------------------------

def to_octree(store: Any, *, target: float, level: int = 0, balanced: bool = True,
              compressor: Any = "inherit", backend: str | None = None) -> dict[str, Any]:
    """Rewrite a uniform level with octree chunks of at most ``target`` vertices.

    The uniform chunk is the finest leaf: no leaf is smaller, and a cell
    heavier than ``target`` stays one leaf.  Links must not be longer than
    a cell (they never are between adjacent cells' leaves).

    Returns a summary: leaf count, the largest leaf, the depths used.
    """
    root = open_store(store, mode="r+", backend=backend)
    level_group = get_resolution_level(root, level)
    if octree_meta(level_group) is not None:
        raise OctreeError("this level already has octree chunks; to_uniform first")
    meta, chunk_shape = _check_convertible(root, level_group, level)
    dtype = _vertex_dtype(level_group)
    comp = _level_compressors(level_group) if compressor == "inherit" else compressor

    cells = sorted(_parse_chunk_key(k) for k in level_group.list_chunks(VERTICES))
    groups = _read_cells(level_group, cells, dtype)
    counts = np.array([sum(len(g) for g in gs) for gs in groups], dtype=np.int64)
    keep = counts > 0
    cells = [c for c, k in zip(cells, keep) if k]
    groups = [g for g, k in zip(groups, keep) if k]
    counts = counts[keep]
    cell_arr = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    manifests = read_all_object_manifests(level_group) if level_group.array_exists(OBJECT_INDEX) else []
    had_links = level_group.array_exists(f"{LINKS}/0")
    links = read_link_arrays(level_group, delta=0) if list_link_offsets(level_group, 0) else None

    table = plan_leaves(cell_arr, counts, target=target, balanced=balanced)
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
    cell_index = {c: n for n, c in enumerate(cells)}
    anchors = table.anchors()

    new_manifests: dict[int, list[tuple[tuple[int, ...], int]]] = {}
    for oid, entries in enumerate(manifests):
        if entries:
            new_manifests[oid] = [
                (tuple(anchors[leaf_of[n]].tolist()), int(frag_off[n] + f))
                for cc, f in entries
                for n in [cell_index[tuple(int(x) for x in cc)]]
            ]

    _drop_per_chunk_arrays(level_group)
    with open_write_session(level_group, compressor=comp, shard_shape="inherit",
                            bounds=meta.bounds, chunk_shape=chunk_shape):
        create_vertices_array(level_group, dtype=str(dtype))
        create_object_index_array(level_group)
        for leaf, gl in sorted(leaf_groups.items()):
            write_chunk_vertices(level_group, tuple(anchors[leaf].tolist()), gl, dtype=dtype)
        n_links = 0
        if had_links:
            ch, vi = links if links is not None else (np.zeros((0, 2, 3), np.int64), np.zeros((0, 2), np.int64))
            cn_a = np.array([cell_index[tuple(c)] for c in ch[:, 0].tolist()], dtype=np.int64)
            cn_b = np.array([cell_index[tuple(c)] for c in ch[:, 1].tolist()], dtype=np.int64)
            n_links = _write_octree_links(
                level_group, table, leaf_of[cn_a], row_off[cn_a] + vi[:, 0],
                leaf_of[cn_b], row_off[cn_b] + vi[:, 1],
            )
        write_object_index(level_group, new_manifests, sid_ndim=3, total_objects=len(manifests))
    stamp_fragments_tile(level_group, 3)

    level_group.write_array(LEAVES_TABLE, table.to_rows())
    level_group.write_array(CELLS_TABLE, cell_rows.astype(np.int64))
    _set_level_block(level_group, {**table.meta(), "weight": "vertices", "leaves": len(table)})
    _set_capability(root, True)
    _finalize_write(root, "to_octree")
    return {"leaves": len(table), "cells": len(cells), "links": n_links,
            "max_leaf_vertices": int(table.weight.max()) if len(table) else 0,
            "max_cell_vertices": int(counts.max()) if len(counts) else 0,
            "depths": {int(d): int(n) for d, n in zip(*np.unique(table.depth, return_counts=True))}}


def to_uniform(store: Any, *, level: int = 0, compressor: Any = "inherit",
               backend: str | None = None) -> dict[str, Any]:
    """Rewrite an octree-chunked level with its original uniform chunks."""
    root = open_store(store, mode="r+", backend=backend)
    level_group = get_resolution_level(root, level)
    table = leaf_table(level_group)
    cells_t = cell_table(level_group)
    meta = read_root_metadata(root)
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:
        level_meta = None
    chunk_shape = tuple(get_level_chunk_shape(meta, level_meta))
    dtype = _vertex_dtype(level_group)
    comp = _level_compressors(level_group) if compressor == "inherit" else compressor
    anchors = table.anchors()
    anchor_leaf = {tuple(a): n for n, a in enumerate(anchors.tolist())}

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
            gl = read_chunk_vertices(level_group, tuple(anchors[leaf].tolist()), dtype=dtype, ndim=3)
            for n in np.flatnonzero(leaf_col == leaf).tolist():
                cell_groups[tuple(cells_t[n, :3].tolist())] = gl[frag_off[n]:frag_off[n] + cells_t[n, 5]]

    # (leaf, row) -> (cell, row in cell), through each leaf's run starts.
    def to_cells(leaf: npt.NDArray[np.int64], row: npt.NDArray[np.int64]) -> tuple[npt.NDArray, npt.NDArray]:
        key = leaf * (1 << 40) + row
        starts = leaf_col * (1 << 40) + row_off
        o = np.argsort(starts, kind="stable")
        pos = np.searchsorted(starts[o], key, side="right") - 1
        n = o[pos]
        return cells_t[n, :3], row - row_off[n]

    manifests = read_all_object_manifests(level_group) if level_group.array_exists(OBJECT_INDEX) else []
    new_manifests: dict[int, list[tuple[tuple[int, ...], int]]] = {}
    frag_cells = {}
    for n in range(len(cells_t)):
        for f in range(int(cells_t[n, 5])):
            frag_cells[(int(leaf_col[n]), int(frag_off[n]) + f)] = (tuple(cells_t[n, :3].tolist()), f)
    for oid, entries in enumerate(manifests):
        if entries:
            new_manifests[oid] = [frag_cells[(anchor_leaf[tuple(int(x) for x in cc)], int(f))]
                                  for cc, f in entries]

    had_links = level_group.array_exists(f"{LINKS}/0")
    src, r_src, oth, r_oth = read_octree_link_cells(level_group, table)
    if (oth < 0).any():
        raise OctreeError("a link names a leaf that is not stored")

    _drop_per_chunk_arrays(level_group, extra=(TABLES,))
    _set_level_block(level_group, None)
    with open_write_session(level_group, compressor=comp, shard_shape="inherit",
                            bounds=meta.bounds, chunk_shape=chunk_shape):
        create_vertices_array(level_group, dtype=str(dtype))
        create_object_index_array(level_group)
        for cc in sorted(cell_groups):
            write_chunk_vertices(level_group, cc, cell_groups[cc], dtype=dtype)
        if len(src):
            ca, va = to_cells(src, r_src)
            cb, vb = to_cells(oth, r_oth)
            write_links(level_group, [], 3, delta=0, link_width=2,
                        _arrays=(np.stack([ca, cb], axis=1), np.column_stack([va, vb])))
        if had_links:
            create_links_array(level_group, link_width=2, delta=0, sid_ndim=3)
        write_object_index(level_group, new_manifests, sid_ndim=3, total_objects=len(manifests))
    stamp_fragments_tile(level_group, 3)
    _set_capability(root, False)
    _finalize_write(root, "to_uniform")
    return {"cells": len(cell_groups), "links": int(len(src))}


__all__ = [
    "ArrayError", "cell_table", "leaf_table", "octree_meta", "read_octree_link_cells",
    "to_octree", "to_uniform",
]
