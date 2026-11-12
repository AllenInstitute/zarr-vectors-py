"""Octree leaves over a level's chunk grid: planning, addressing, link naming.

A level with octree chunks keeps the uniform chunk grid as its finest
grid -- the store's ``chunk_shape`` is the finest leaf -- and groups its
cells into **leaves**: aligned cubes of ``2**(D - d)`` cells per side, ``d``
the leaf's depth below a root cube of ``2**D`` cells.  Each leaf is one
chunk: it holds its vertices, fragments and links, and is stored at its
**anchor**, its minimum-corner cell, so every zarr array stays a plain
array over the finest grid.

Addresses.  A leaf is ``depth/i.j.k``: its position on the grid of its own
depth.  The path form names one octant per depth from the root, e.g.
``111/001/010`` is ``3/4.5.6``.  A cell of the uniform grid is a leaf at
depth ``D``.

Links between leaves.  A record is filed under its finer endpoint (same
depth: the one with the smaller anchor).  The other endpoint, ``Δ`` depths
coarser, is named by the offset of its leaf from the source's ancestor at
its own depth -- always one grid step, in ``{-1, 0, 1}**3`` -- plus ``Δ``::

    +1.0.0      same depth, one leaf along +x
    +1.0.0^1    one depth coarser, the leaf along +x of the source's parent

so the other endpoint is ``(d - Δ, (ijk >> Δ) + o)``.  A 2:1-balanced plan
(adjacent leaves differ by at most one depth) needs at most 26 + 26
offset arrays besides the intra-leaf one.

Leaves and shards.  On a sharded level no leaf is larger than a shard or
crosses a shard boundary, so each leaf's data lives in one shard and a
shard can be written, rewritten or read on its own.  The plan's root
cube starts on a shard boundary and no leaf is wider than
:func:`shard_leaf_cap` -- the largest power of two dividing every shard
side -- so leaves tile shards exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.paths import format_delta, parse_delta

_DIRECTIONS = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
     if (x, y, z) != (0, 0, 0)],
    dtype=np.int64,
)


class OctreeError(ValueError):
    """A plan, address or link the octree chunk layout cannot express."""


def _codes(ijk: npt.NDArray[np.int64], depth: int) -> npt.NDArray[np.int64]:
    """Raveled index of grid positions at ``depth`` (``2**depth`` per side)."""
    side = np.int64(1) << np.int64(depth)
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    return (ijk[:, 0] * side + ijk[:, 1]) * side + ijk[:, 2]


def _morton(ijk: npt.NDArray[np.int64], bits: int) -> npt.NDArray[np.int64]:
    """Z-order code of ``(N, 3)`` non-negative positions of ``bits`` bits."""
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    out = np.zeros(len(ijk), dtype=np.int64)
    for b in range(bits):
        for a in range(3):
            out |= ((ijk[:, a] >> b) & 1) << (3 * b + (2 - a))
    return out


def depth_for(extent_cells: Sequence[int]) -> int:
    """Smallest ``D`` with ``2**D`` cells per side covering ``extent_cells``."""
    n = max(1, int(max(extent_cells)))
    return int(np.ceil(np.log2(n))) if n > 1 else 0


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and n & (n - 1) == 0


def shard_leaf_cap(shard_shape: Sequence[int]) -> int:
    """Widest leaf (cells per side) that tiles shards of ``shard_shape``.

    The largest power of two dividing every shard side: ``(4, 4, 4)`` gives
    4, ``(4, 2, 8)`` gives 2, and ``(6, 6, 6)`` gives 2, as a leaf of 4 cells
    would cross the boundary at 6.  With the root cube on a shard boundary,
    every leaf of this side or less lies in one shard.
    """
    sides = [int(s) for s in shard_shape]
    if not sides or min(sides) < 1:
        raise OctreeError(f"bad shard shape {tuple(sides)}")
    return min(s & -s for s in sides)


def leaf_cap(max_leaf_cells: int | None, shard_shape: Sequence[int] | None) -> int | None:
    """The leaf side cap a plan uses: ``max_leaf_cells``, or on a sharded level
    the shard's (:func:`shard_leaf_cap`) when that is not given or smaller.

    Raises when ``max_leaf_cells`` is not a power of two, or is wider than a
    sharded level allows (a leaf must never be larger than a shard).
    """
    if max_leaf_cells is not None and not _is_power_of_two(int(max_leaf_cells)):
        raise OctreeError(f"max_leaf_cells must be a power of two, got {max_leaf_cells}")
    if shard_shape is None:
        return None if max_leaf_cells is None else int(max_leaf_cells)
    cap = shard_leaf_cap(shard_shape)
    if max_leaf_cells is not None and int(max_leaf_cells) > cap:
        raise OctreeError(
            f"max_leaf_cells={max_leaf_cells} is wider than shards of {tuple(shard_shape)} "
            f"cells allow ({cap}): a leaf must lie inside one shard"
        )
    return cap if max_leaf_cells is None else int(max_leaf_cells)


def plan_leaves(
    cells: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    target: float,
    origin_cell: Sequence[int] | None = None,
    max_depth: int | None = None,
    balanced: bool = True,
    max_leaf_cells: int | None = None,
    shard_shape: Sequence[int] | None = None,
    shard_origin: Sequence[int] | None = None,
) -> LeafTable:
    """Split the root cube until every leaf weighs at most ``target``.

    Args:
        cells: ``(N, 3)`` cell coordinates on the uniform grid.
        weights: ``(N,)`` weight of each cell (e.g. its vertex count).
        target: Largest leaf weight wanted.  A single cell heavier than it
            cannot be split and stays one leaf.
        origin_cell: Grid position of the root cube's minimum corner;
            defaults to the cells' minimum, or ``0`` where that is
            non-negative -- moved down to a shard boundary when
            ``shard_shape`` is given.
        max_depth: ``D``; defaults to the smallest root cube covering the
            cells.
        balanced: Split further until adjacent leaves (faces, edges and
            corners) differ by at most one depth.
        max_leaf_cells: Widest leaf allowed, in cells per side (a power of
            two).  ``None``: the shard's cap when ``shard_shape`` is given
            (:func:`shard_leaf_cap`), no cap otherwise.
        shard_shape: Cells per shard per axis when the level is sharded:
            the plan then keeps every leaf inside one shard.
        shard_origin: The cell a shard starts at (the arrays' grid
            origin); defaults to ``0``.

    Returns:
        The non-empty leaves.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(cells) != len(weights):
        raise OctreeError(f"{len(cells)} cells but {len(weights)} weights")
    if target <= 0:
        raise OctreeError(f"target must be positive, got {target}")
    cap = leaf_cap(max_leaf_cells, shard_shape)
    shards = None if shard_shape is None else np.asarray(shard_shape, dtype=np.int64).reshape(3)
    s_origin = np.zeros(3, np.int64) if shard_origin is None else np.asarray(shard_origin, np.int64).reshape(3)
    if origin_cell is None:
        origin_cell = np.minimum(cells.min(axis=0), 0) if len(cells) else np.zeros(3, np.int64)
        if shards is not None:
            origin_cell = s_origin + ((np.asarray(origin_cell, np.int64) - s_origin) // shards) * shards
    origin = np.asarray(origin_cell, dtype=np.int64).reshape(3)
    if shards is not None and ((origin - s_origin) % shards).any():
        raise OctreeError(
            f"origin_cell {tuple(origin.tolist())} is not on a shard boundary "
            f"(shards of {tuple(shards.tolist())} cells from {tuple(s_origin.tolist())})"
        )
    rel = cells - origin
    if len(rel) and rel.min() < 0:
        raise OctreeError("a cell lies below origin_cell")
    need = depth_for((rel.max(axis=0) + 1) if len(rel) else (1,))
    D = need if max_depth is None else int(max_depth)
    if D < need:
        raise OctreeError(f"max_depth={D} does not cover the cells (needs {need})")

    # Weight of every occupied node at every depth.
    node_weight: list[dict[int, float]] = []
    for d in range(D + 1):
        codes = _codes(rel >> (D - d), d)
        u, inv = np.unique(codes, return_inverse=True)
        node_weight.append(dict(zip(u.tolist(), np.bincount(inv, weights).tolist())))

    # A leaf is split while too heavy or, holding anything, wider than the
    # cap; empty space is never stored, so it is left coarse.
    leaves: list[set[int]] = [set() for _ in range(D + 1)]
    stack = [(0, 0, 0, 0)]
    while stack:
        d, i, j, k = stack.pop()
        code = (i << (2 * d)) | (j << d) | k
        w = node_weight[d].get(code, 0.0)
        if d < D and w > 0 and (w > target or (cap is not None and (1 << (D - d)) > cap)):
            for c in range(8):
                stack.append((d + 1, 2 * i + (c >> 2), 2 * j + ((c >> 1) & 1), 2 * k + (c & 1)))
        else:
            leaves[d].add(code)

    if balanced:
        # Balancing only ever splits a leaf into its children, so it keeps
        # every leaf within the cap.
        _balance(leaves, D)

    out_d, out_ijk, out_w = [], [], []
    for d in range(D + 1):
        side = 1 << d
        for code in leaves[d]:
            w = node_weight[d].get(code, 0.0)
            if w <= 0:
                continue
            out_d.append(d)
            out_ijk.append((code // (side * side), (code // side) % side, code % side))
            out_w.append(w)
    return LeafTable.from_arrays(
        max_depth=D, origin_cell=origin.tolist(),
        depth=np.asarray(out_d, dtype=np.int64),
        ijk=np.asarray(out_ijk, dtype=np.int64).reshape(-1, 3),
        weight=np.asarray(out_w, dtype=np.float64),
        target=float(target), balanced=balanced, max_leaf_cells=cap,
        shard_shape=None if shards is None else shards.tolist(),
        shard_origin=None if shards is None else s_origin.tolist(),
    )


def _balance(leaves: list[set[int]], D: int) -> None:
    """Split leaves in place until adjacent leaves differ by at most one depth."""
    def decode(code: int, d: int) -> tuple[int, int, int]:
        side = 1 << d
        return code // (side * side), (code // side) % side, code % side

    def covering(i: int, j: int, k: int, d: int) -> int:
        """Depth of the leaf covering grid position ``(i, j, k)`` at depth
        ``d``, or ``d + 1`` when the region is split finer than ``d``."""
        for dd in range(d, -1, -1):
            s = d - dd
            ii, jj, kk = i >> s, j >> s, k >> s
            if ((ii << (2 * dd)) | (jj << dd) | kk) in leaves[dd]:
                return dd
        return d + 1

    work = [(d, c) for d in range(D + 1) for c in leaves[d]]
    while work:
        d, code = work.pop()
        if d < 2 or code not in leaves[d]:
            continue
        i, j, k = decode(code, d)
        side = 1 << d
        for di, dj, dk in _DIRECTIONS.tolist():
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < side and 0 <= nj < side and 0 <= nk < side):
                continue
            dd = covering(ni, nj, nk, d)
            if dd >= d - 1:
                continue
            s = d - dd
            pi, pj, pk = ni >> s, nj >> s, nk >> s
            leaves[dd].discard((pi << (2 * dd)) | (pj << dd) | pk)
            for c in range(8):
                ci, cj, ck = 2 * pi + (c >> 2), 2 * pj + ((c >> 1) & 1), 2 * pk + (c & 1)
                child = (ci << (2 * (dd + 1))) | (cj << (dd + 1)) | ck
                leaves[dd + 1].add(child)
                work.append((dd + 1, child))
            work.append((d, code))
            break


@dataclass
class LeafTable:
    """The non-empty leaves of one level, in Z order of their anchors."""

    max_depth: int
    origin_cell: tuple[int, int, int]
    depth: npt.NDArray[np.int64]
    ijk: npt.NDArray[np.int64]
    weight: npt.NDArray[np.float64]
    target: float | None = None
    balanced: bool = True
    max_leaf_cells: int | None = None              # leaf side cap; None: uncapped
    shard_shape: tuple[int, int, int] | None = None  # the shards the plan was fitted to
    shard_origin: tuple[int, int, int] | None = None
    _by_depth: dict[int, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = field(
        default_factory=dict, repr=False,
    )

    @classmethod
    def from_arrays(cls, *, max_depth: int, origin_cell: Sequence[int], depth: npt.ArrayLike,
                    ijk: npt.ArrayLike, weight: npt.ArrayLike, target: float | None = None,
                    balanced: bool = True, max_leaf_cells: int | None = None,
                    shard_shape: Sequence[int] | None = None,
                    shard_origin: Sequence[int] | None = None) -> LeafTable:
        depth = np.asarray(depth, dtype=np.int64).reshape(-1)
        ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
        weight = np.asarray(weight, dtype=np.float64).reshape(-1)
        anchors_rel = ijk << (max_depth - depth)[:, None]
        order = np.lexsort((depth, _morton(anchors_rel, max(max_depth, 1))))
        t = cls(max_depth=int(max_depth), origin_cell=tuple(int(c) for c in origin_cell),
                depth=depth[order], ijk=ijk[order], weight=weight[order],
                target=target, balanced=balanced,
                max_leaf_cells=None if max_leaf_cells is None else int(max_leaf_cells),
                shard_shape=None if shard_shape is None else tuple(int(c) for c in shard_shape),
                shard_origin=None if shard_origin is None else tuple(int(c) for c in shard_origin))
        for d in np.unique(t.depth).tolist():
            idx = np.flatnonzero(t.depth == d)
            codes = _codes(t.ijk[idx], d)
            o = np.argsort(codes)
            t._by_depth[int(d)] = (codes[o], idx[o])
        return t

    # -- storage ---------------------------------------------------------

    def to_rows(self) -> npt.NDArray[np.int64]:
        """``(N, 5)`` rows ``(depth, i, j, k, weight)`` for the level's table."""
        return np.column_stack([self.depth, self.ijk, np.rint(self.weight).astype(np.int64)])

    @classmethod
    def from_rows(cls, rows: npt.ArrayLike, meta: dict[str, Any]) -> LeafTable:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 5)
        return cls.from_arrays(
            max_depth=int(meta["max_depth"]), origin_cell=meta["origin_cell"],
            depth=rows[:, 0], ijk=rows[:, 1:4], weight=rows[:, 4],
            target=meta.get("target"), balanced=bool(meta.get("balanced", True)),
            max_leaf_cells=meta.get("max_leaf_cells"), shard_shape=meta.get("shard_shape"),
            shard_origin=meta.get("shard_origin"),
        )

    def meta(self) -> dict[str, Any]:
        return {"max_depth": self.max_depth, "origin_cell": list(self.origin_cell),
                "target": self.target, "balanced": self.balanced,
                "max_leaf_cells": self.max_leaf_cells,
                "shard_shape": None if self.shard_shape is None else list(self.shard_shape),
                "shard_origin": None if self.shard_origin is None else list(self.shard_origin)}

    # -- addressing ------------------------------------------------------

    def __len__(self) -> int:
        return int(len(self.depth))

    def anchors(self) -> npt.NDArray[np.int64]:
        """``(N, 3)`` cell of each leaf's minimum corner, on the uniform grid."""
        return (self.ijk << (self.max_depth - self.depth)[:, None]) + np.asarray(self.origin_cell)

    def sides(self) -> npt.NDArray[np.int64]:
        """Cells per side of each leaf."""
        return np.int64(1) << (self.max_depth - self.depth)

    def straddling(self, shard_shape: Sequence[int], shard_origin: Sequence[int] | None = None
                   ) -> npt.NDArray[np.int64]:
        """Leaves not inside one shard of ``shard_shape`` cells (shards start
        at ``shard_origin``, default ``0``): wider than a shard, or across a
        shard boundary."""
        shards = np.asarray(shard_shape, dtype=np.int64).reshape(3)
        o = np.zeros(3, np.int64) if shard_origin is None else np.asarray(shard_origin, np.int64).reshape(3)
        lo = self.anchors() - o
        hi = lo + self.sides()[:, None] - 1
        return np.flatnonzero((lo // shards != hi // shards).any(axis=1))

    def key(self, n: int) -> str:
        i, j, k = self.ijk[n].tolist()
        return f"{int(self.depth[n])}/{i}.{j}.{k}"

    def path(self, n: int) -> str:
        d = int(self.depth[n])
        i, j, k = self.ijk[n].tolist()
        return "/".join(
            f"{(i >> s) & 1}{(j >> s) & 1}{(k >> s) & 1}" for s in range(d - 1, -1, -1)
        )

    def find(self, depth: int, ijk: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Leaf index of each ``(depth, ijk)``, ``-1`` where no stored leaf is there."""
        ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
        out = np.full(len(ijk), -1, dtype=np.int64)
        entry = self._by_depth.get(int(depth))
        if entry is None or not len(ijk):
            return out
        codes, idx = entry
        side = 1 << int(depth)
        ok = ((ijk >= 0) & (ijk < side)).all(axis=1)
        q = _codes(np.where(ok[:, None], ijk, 0), depth)
        pos = np.minimum(np.searchsorted(codes, q), len(codes) - 1)
        hit = ok & (codes[pos] == q)
        out[hit] = idx[pos[hit]]
        return out

    def index(self, address: str) -> int:
        """Leaf index of ``"d/i.j.k"`` or of a path such as ``"111/001/010"``."""
        address = address.strip()
        if "." in address:
            d_s, ijk_s = address.split("/", 1)
            d, ijk = int(d_s), [int(c) for c in ijk_s.split(".")]
        else:
            steps = [s for s in address.split("/") if s] if address else []
            d, ijk = len(steps), [0, 0, 0]
            for s in steps:
                if len(s) != 3 or set(s) - {"0", "1"}:
                    raise OctreeError(f"bad octant {s!r} in path {address!r}")
                ijk = [2 * ijk[a] + int(s[a]) for a in range(3)]
        n = int(self.find(d, [ijk])[0])
        if n < 0:
            raise KeyError(f"no stored leaf at {address!r}")
        return n

    def leaf_of_cells(self, cells: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Leaf index holding each uniform-grid cell, ``-1`` if none."""
        rel = np.asarray(cells, dtype=np.int64).reshape(-1, 3) - np.asarray(self.origin_cell)
        out = np.full(len(rel), -1, dtype=np.int64)
        for d in self._by_depth:
            todo = out < 0
            if not todo.any():
                break
            hit = self.find(d, rel[todo] >> (self.max_depth - d))
            out[np.flatnonzero(todo)] = hit
        return out

    def overlapping(self, lo_cell: Sequence[int], hi_cell: Sequence[int]) -> npt.NDArray[np.int64]:
        """Leaves whose cells meet the inclusive cell box ``[lo_cell, hi_cell]``."""
        a = self.anchors()
        b = a + self.sides()[:, None] - 1
        lo, hi = np.asarray(lo_cell), np.asarray(hi_cell)
        return np.flatnonzero(((a <= hi) & (b >= lo)).all(axis=1))

    def neighbours(self, n: int) -> npt.NDArray[np.int64]:
        """Stored leaves sharing a face, edge or corner with leaf ``n``."""
        a, s = self.anchors()[n], int(self.sides()[n])
        lo, hi = a - 1, a + s
        cand = self.overlapping(lo, hi)
        return cand[cand != n]

    # -- link naming -----------------------------------------------------

    def link_source(self, a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Whether endpoint ``a`` (not ``b``) is the source a link is filed under:
        the deeper leaf, or at equal depth the one with the smaller anchor."""
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        da, db = self.depth[a], self.depth[b]
        ka, kb = self.ijk[a], self.ijk[b]
        first_diff = np.argmax(ka != kb, axis=1)
        rows = np.arange(len(a))
        smaller = ka[rows, first_diff] < kb[rows, first_diff]
        return (da > db) | ((da == db) & smaller)

    def link_offsets(self, src: npt.ArrayLike, other: npt.ArrayLike) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """``(offset (M, 3), Δ (M,))`` naming ``other`` from ``src``; raises when
        the leaves are not adjacent or ``other`` is the finer of the two."""
        src, other = np.asarray(src, dtype=np.int64), np.asarray(other, dtype=np.int64)
        delta = self.depth[src] - self.depth[other]
        if (delta < 0).any():
            raise OctreeError("a link is filed under its coarser endpoint")
        o = self.ijk[other] - (self.ijk[src] >> delta[:, None])
        if (np.abs(o) > 1).any():
            bad = int(np.flatnonzero((np.abs(o) > 1).any(axis=1))[0])
            raise OctreeError(
                f"a link joins leaves {self.key(int(src[bad]))} and {self.key(int(other[bad]))}, "
                f"which are not adjacent: links must not be longer than the smallest leaf "
                f"(the store's chunk_shape)"
            )
        return o, delta

    def resolve(self, src: int, offset: Sequence[int], delta: int) -> int:
        """Leaf named by ``(offset, Δ)`` from ``src``, ``-1`` if not stored."""
        d = int(self.depth[src]) - int(delta)
        ijk = (self.ijk[src] >> int(delta)) + np.asarray(offset, dtype=np.int64)
        return int(self.find(d, [ijk])[0])


def format_link_segment(offset: Sequence[int], delta: int) -> str:
    """``"+1.0.0"``, or ``"+1.0.0^1"`` for a neighbour ``delta`` depths coarser."""
    seg = ".".join(format_delta(int(c)) for c in offset)
    return f"{seg}^{int(delta)}" if delta else seg


def parse_link_segment(segment: str) -> tuple[tuple[int, int, int], int]:
    """Inverse of :func:`format_link_segment`."""
    base, _, d = segment.partition("^")
    parts = base.split(".")
    if len(parts) != 3:
        raise OctreeError(f"bad octree link segment {segment!r}")
    return tuple(parse_delta(p) for p in parts), int(d) if d else 0  # type: ignore[return-value]


def iter_segments(table: LeafTable) -> Iterable[str]:  # pragma: no cover - documentation helper
    """Every segment a plan can produce (26 per depth change, plus intra)."""
    yield format_link_segment((0, 0, 0), 0)
    for delta in (0, 1) if table.balanced else range(table.max_depth + 1):
        for o in _DIRECTIONS.tolist():
            yield format_link_segment(o, delta)
