"""Reading a level with octree chunks: by leaf, box or object, and its seams.

Every read resolves to a set of leaves.  A leaf's vertices are one chunk;
its links are those filed under it (``links/0/<offset>[^Δ]``) plus, for a
set of leaves, those filed under the others in the set.  A link is kept
when both endpoints were read, as in a uniform read.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    GEOM_GRAPH,
    OBJECT_INDEX,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    _chunk_key,
    read_chunk_vertices,
    read_object_manifests,
    read_vertex_fragment_index,
)
from zarr_vectors.core.metadata import get_level_chunk_shape
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.octree_chunks.convert import leaf_table, read_octree_link_cells
from zarr_vectors.octree_chunks.table import LeafTable, OctreeError


def _leaf_indices(table: LeafTable, leaves: Iterable[int | str]) -> npt.NDArray[np.int64]:
    out = [table.index(x) if isinstance(x, str) else int(x) for x in leaves]
    return np.unique(np.asarray(out, dtype=np.int64))


def leaves_in_box(table: LeafTable, bbox: Any, chunk_shape: Sequence[float]) -> npt.NDArray[np.int64]:
    """Leaves meeting the world-space box ``(lo, hi)``."""
    cs = np.asarray(chunk_shape, dtype=np.float64)
    lo = np.floor(np.asarray(bbox[0], dtype=np.float64) / cs).astype(np.int64)
    hi = np.floor(np.asarray(bbox[1], dtype=np.float64) / cs).astype(np.int64)
    return table.overlapping(lo, hi)


def _open(store: Any, level: int, backend: str | None) -> tuple[Any, Any, LeafTable, Any, tuple[float, ...]]:
    root = open_store(store, backend=backend)
    level_group = get_resolution_level(root, level)
    meta = read_root_metadata(root)
    try:
        level_meta = read_level_metadata(root, level)
    except Exception:
        level_meta = None
    return root, level_group, leaf_table(level_group), meta, tuple(get_level_chunk_shape(meta, level_meta))


def _dtype(level_group: Any) -> np.dtype:
    try:
        return np.dtype(level_group.read_array_meta(VERTICES).get("dtype", "float32"))
    except Exception:
        return np.dtype("float32")


def read_leaves(level_group: Any, table: LeafTable, leaves: npt.ArrayLike, *, with_links: bool = True,
                dtype: np.dtype | None = None) -> dict[str, Any]:
    """Vertices of ``leaves`` and the links wholly among them.

    Returns ``positions``, ``leaf`` (each row's leaf index), ``row`` (its
    row in that leaf), ``edges`` (output rows) and ``leaf_offsets``.
    """
    leaves = np.unique(np.asarray(leaves, dtype=np.int64))
    dtype = dtype or _dtype(level_group)
    anchors = table.anchors()
    keys = [_chunk_key(tuple(anchors[n].tolist())) for n in leaves.tolist()]
    positions, leaf_col, row_col = [], [], []
    offsets = np.full(len(table), -1, dtype=np.int64)
    running = 0
    with level_group.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)], tolerant=True):
        for n in leaves.tolist():
            gl = read_chunk_vertices(level_group, tuple(anchors[n].tolist()), dtype=dtype, ndim=3, _leaf=True)
            block = np.concatenate(gl) if gl else np.zeros((0, 3), dtype=dtype)
            offsets[n] = running
            positions.append(block)
            leaf_col.append(np.full(len(block), n, dtype=np.int64))
            row_col.append(np.arange(len(block), dtype=np.int64))
            running += len(block)
    out: dict[str, Any] = {
        "positions": np.concatenate(positions) if positions else np.zeros((0, 3), dtype=dtype),
        "leaf": np.concatenate(leaf_col) if leaf_col else np.zeros(0, np.int64),
        "row": np.concatenate(row_col) if row_col else np.zeros(0, np.int64),
        "leaf_offsets": offsets,
    }
    if with_links:
        src, r_src, oth, r_oth = read_octree_link_cells(level_group, table, leaves)
        keep = (oth >= 0) & (offsets[np.maximum(oth, 0)] >= 0)
        out["edges"] = np.column_stack([offsets[src[keep]] + r_src[keep],
                                        offsets[oth[keep]] + r_oth[keep]]).astype(np.int64)
    return out


def read_octree(store: Any, *, level: int = 0, leaves: Iterable[int | str] | None = None,
                bbox: Any = None, object_ids: Sequence[int] | None = None, halo: int = 0,
                backend: str | None = None) -> dict[str, Any]:
    """Read an octree-chunked level by leaf, box or object.

    Args:
        leaves: Leaf indices or addresses (``"3/4.5.6"``, ``"111/001/010"``).
        bbox: World-space ``(lo, hi)``; leaves meeting it are read and
            vertices outside it dropped, with the links on them.
        object_ids: Only these objects' vertices and the links among them.
        halo: With ``leaves``, also read the leaves adjacent to them
            (``1``; one ring is what a balanced plan bounds).

    Returns:
        ``positions``, ``edges`` (graphs), ``leaf`` / ``row`` (where each
        vertex is stored), and ``object_ids`` on an object read.
    """
    root, level_group, table, meta, chunk_shape = _open(store, level, backend)
    if object_ids is not None:
        return _read_objects(level_group, table, object_ids, bbox=bbox)
    if leaves is not None:
        sel = _leaf_indices(table, leaves)
        if halo:
            if halo != 1:
                raise OctreeError("halo= takes 0 or 1")
            ring = [table.neighbours(int(n)) for n in sel.tolist()]
            sel = np.unique(np.concatenate([sel, *ring])) if ring else sel
    elif bbox is not None:
        sel = leaves_in_box(table, bbox, chunk_shape)
    else:
        sel = np.arange(len(table))
    is_graph = GEOM_GRAPH in set(meta.geometry_types)
    out = read_leaves(level_group, table, sel, with_links=is_graph)
    if bbox is not None:
        _box_filter(out, bbox)
    return out


def _box_filter(out: dict[str, Any], bbox: Any) -> None:
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    keep = np.all((out["positions"] >= lo) & (out["positions"] <= hi), axis=1)
    if keep.all():
        return
    remap = np.full(len(keep), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    for name in ("positions", "leaf", "row", "object_ids"):
        if name in out:
            out[name] = out[name][keep]
    if "edges" in out and len(out["edges"]):
        e = remap[out["edges"]]
        out["edges"] = e[(e >= 0).all(axis=1)]


def _read_objects(level_group: Any, table: LeafTable, object_ids: Sequence[int], *, bbox: Any = None) -> dict[str, Any]:
    wanted = list(dict.fromkeys(int(o) for o in object_ids))
    manifests = read_object_manifests(level_group, ids=wanted) if wanted else {}
    anchors = table.anchors()
    anchor_leaf = {tuple(a): n for n, a in enumerate(anchors.tolist())}
    entries = [(oid, anchor_leaf[tuple(int(c) for c in cc)], int(f))
               for oid in wanted for cc, f in manifests.get(oid, ())]
    dtype = _dtype(level_group)
    is_graph = level_group.array_exists("links/0")
    if not entries:
        out = {"positions": np.zeros((0, 3), dtype=dtype), "leaf": np.zeros(0, np.int64),
               "row": np.zeros(0, np.int64), "object_ids": np.zeros(0, np.int64)}
        if is_graph:
            out["edges"] = np.zeros((0, 2), np.int64)
        return out
    leaves = sorted({n for _, n, _ in entries})
    keys = [_chunk_key(tuple(anchors[n].tolist())) for n in leaves]
    row_maps: dict[int, npt.NDArray[np.int64]] = {}
    positions, owners, leaf_col, row_col = [], [], [], []
    running = 0
    with level_group.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)], tolerant=True):
        blocks = {n: read_chunk_vertices(level_group, tuple(anchors[n].tolist()), dtype=dtype, ndim=3, _leaf=True)
                  for n in leaves}
        for oid, n, f in entries:
            fi = read_vertex_fragment_index(level_group, tuple(anchors[n].tolist()))
            start, count = fi.range(f)
            if n not in row_maps:
                row_maps[n] = np.full(sum(len(g) for g in blocks[n]), -1, dtype=np.int64)
            row_maps[n][start:start + count] = np.arange(running, running + count)
            positions.append(blocks[n][f])
            owners.append(np.full(count, oid, dtype=np.int64))
            leaf_col.append(np.full(count, n, dtype=np.int64))
            row_col.append(np.arange(start, start + count, dtype=np.int64))
            running += count
    out: dict[str, Any] = {"positions": np.concatenate(positions), "object_ids": np.concatenate(owners),
                           "leaf": np.concatenate(leaf_col), "row": np.concatenate(row_col)}
    if is_graph:
        src, r_src, oth, r_oth = read_octree_link_cells(level_group, table, leaves)
        a = np.array([row_maps[s][r] if s in row_maps else -1 for s, r in zip(src.tolist(), r_src.tolist())], np.int64)
        b = np.array([row_maps[o][r] if o in row_maps else -1 for o, r in zip(oth.tolist(), r_oth.tolist())], np.int64)
        keep = (a >= 0) & (b >= 0)
        out["edges"] = np.column_stack([a[keep], b[keep]]).astype(np.int64)
    if bbox is not None:
        _box_filter(out, bbox)
    return out


def read_seams(store: Any, leaf: int | str, *, level: int = 0, backend: str | None = None) -> dict[str, Any]:
    """Links between one leaf and the leaves around it.

    Returns ``row`` (the endpoint's row in the leaf), ``other_leaf`` and
    ``other_row``: what a per-leaf graph builder stitches across.
    """
    root, level_group, table, meta, _ = _open(store, level, backend)
    n = table.index(leaf) if isinstance(leaf, str) else int(leaf)
    ring = table.neighbours(n)
    src, r_src, oth, r_oth = read_octree_link_cells(level_group, table, np.concatenate([[n], ring]))
    mine_src = (src == n) & (oth != n)
    mine_oth = (oth == n) & (src != n)
    return {
        "row": np.concatenate([r_src[mine_src], r_oth[mine_oth]]),
        "other_leaf": np.concatenate([oth[mine_src], src[mine_oth]]),
        "other_row": np.concatenate([r_oth[mine_src], r_src[mine_oth]]),
    }


def _refuse(chunks: Any, attribute_filter: Any, attribute_names: Any = None) -> None:
    if chunks is not None:
        raise OctreeError("chunks= names uniform cells; on an octree level read by leaves= "
                          "(zarr_vectors.octree_chunks.read_octree) or by bbox=")
    if attribute_filter:
        raise OctreeError("attribute_filter= is not available on an octree level")
    if attribute_names:
        raise OctreeError("vertex attributes are not supported on an octree level yet")


def read_points_octree(root: Any, *, level: int, bbox: Any, object_ids: Any, chunks: Any,
                       attribute_names: Any, attribute_filter: Any) -> dict[str, Any]:
    """:func:`~zarr_vectors.types.points.read_points` on an octree level."""
    _refuse(chunks, attribute_filter, attribute_names)
    out = read_octree(root, level=level, bbox=bbox, object_ids=object_ids)
    res = {"positions": out["positions"], "attributes": {}, "vertex_count": len(out["positions"])}
    if "object_ids" in out:
        res["object_ids"] = out["object_ids"]
    return res


def read_graph_octree(root: Any, *, level: int, bbox: Any, object_ids: Any, chunks: Any,
                      attribute_filter: Any) -> dict[str, Any]:
    """:func:`~zarr_vectors.types.graphs.read_graph` on an octree level."""
    _refuse(chunks, attribute_filter)
    out = read_octree(root, level=level, bbox=bbox, object_ids=object_ids)
    res = {"positions": out["positions"], "edges": out["edges"],
           "node_count": len(out["positions"]), "edge_count": len(out["edges"])}
    if "object_ids" in out:
        res["object_ids"] = out["object_ids"]
    return res


__all__ = ["read_graph_octree", "read_points_octree", "leaves_in_box", "read_leaves", "read_octree", "read_seams", "OBJECT_INDEX"]
