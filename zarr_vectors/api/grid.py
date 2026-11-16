"""The chunk grid, exposed as geometry rather than as coordinates.

The grid is a storage concept, and most callers should never meet it.
Two questions about it are nonetheless real:

*"Will this fit?"*  A store allocates a fixed number of cells per axis
from its bounds and cell size, and a write whose coordinates fall outside
that allocation is rejected.  One consumer predicts the allocation by
calling ``level_grid_layout`` and ``compute_grid_shape`` itself, with a
comment noting that the guard "stays in lockstep with the real array
shape" — which is to say, it does not, and cannot.

*"Which region is this?"*  A caller with an externally fixed grid needs
to name a region in grid terms.

Both are answered here, in physical units, with the grid coordinates
wrapped in an opaque :class:`CellRef` that a caller receives but never
constructs.  ``Grid.plan`` answers the first question *before the store
exists*, which is the only time the answer is useful.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CellRef", "CellSet", "Grid", "GridCapacity"]


@dataclass(frozen=True, slots=True, order=True)
class CellRef:
    """One cell of the grid.

    Opaque on purpose: a caller gets these from :meth:`Grid.cells_in` or
    :meth:`Grid.cell_of` and passes them back, but never builds one from
    integers.  That is what keeps a change in how cells are addressed
    from being a downstream break.

    When the escape hatch is taken: :attr:`coords` and :attr:`key` are
    **absolute** chunk coordinates, identical to the keys on disk — which
    is what lets a reference be handed straight to the readers' ``chunks=``
    term and to the engine's cell requests.
    """

    coords: tuple[int, ...]

    @property
    def key(self) -> str:
        """The on-disk chunk key.  An escape hatch, not the interface."""
        return ".".join(str(int(c)) for c in self.coords)

    def __repr__(self) -> str:
        return f"CellRef({self.key})"


class CellSet(frozenset):  # type: ignore[type-arg]
    """A set of :class:`CellRef`, usable as a selection term."""

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(ref.key for ref in self))

    def __repr__(self) -> str:
        return f"CellSet({len(self)} cell(s))"


@dataclass(frozen=True, slots=True)
class GridCapacity:
    """Whether a grid can hold what is planned for it."""

    cells: int
    """Total cells the allocation provides."""

    shape: tuple[int, ...]
    """Cells per axis."""

    est_bytes_per_cell: float
    """Expected payload per cell, from the size hints given."""

    fits: bool
    reason: str | None = None

    def __str__(self) -> str:
        shape = "x".join(str(s) for s in self.shape)
        verdict = "fits" if self.fits else f"does not fit: {self.reason}"
        return (
            f"{shape} = {self.cells} cells, "
            f"~{self.est_bytes_per_cell / 1e6:.1f} MB/cell -- {verdict}"
        )


@dataclass(frozen=True, slots=True)
class Grid:
    """A level's cell grid, in physical units.

    Spatial: every axis is a physical axis, and :meth:`cell_of`,
    :meth:`cells_in` and :meth:`holds` are spatial predicates.  A level
    chunked by an attribute adds a leading bin axis to its keys that this
    grid does not model; a :class:`CellRef` from it selects its cell in
    every bin of such a level.
    """

    shape: tuple[int, ...]
    """Cells per axis."""

    cell_shape: tuple[float, ...]
    """Physical size of one cell, in coordinate units."""

    origin: tuple[float, ...] = ()
    """Coordinate of the grid's lower corner, in the data's own units.

    Informational.  Cells are addressed **absolutely** — :meth:`cell_of`
    is ``floor(point / cell_shape)``, the same map
    :func:`~zarr_vectors.spatial.chunking.assign_chunks` uses to decide
    where a vertex is written — so this is not a subtrahend.
    :attr:`anchor` is the cell this corner falls in.
    """

    @classmethod
    def plan(
        cls,
        bounds: tuple[Sequence[float], Sequence[float]],
        *,
        cell_size: Sequence[float] | None = None,
        target_cells: int | Sequence[int] | None = None,
    ) -> Grid:
        """The grid a store with these bounds would get — without one.

        Callable before anything is written, which is the only moment the
        answer can change a decision.  A consumer that discovers after the
        fact that its chunk ids exceed the allocation has already written
        a store it must throw away.

        The shape comes from
        :func:`~zarr_vectors.spatial.chunking.grid_layout` — the function
        the allocator itself calls — rather than from a restatement of it.
        A restatement is what this used to be, and it was one plane short:
        ``ceil((hi - lo) / size)`` counts cell-widths across the extent,
        but bounds are inclusive, so a store spans
        ``floor(hi/size) - floor(lo/size) + 1`` cells.

        That makes ``target_cells=n`` produce ``n + 1`` cells per axis
        when the upper bound lands exactly on a cell boundary: a vertex at
        ``hi`` has to be storable, and it lands in cell ``floor(hi/size)``.
        The extra plane is what the store really allocates; reporting
        ``n`` is the failure this method exists to prevent.
        """
        lo = [float(v) for v in bounds[0]]
        hi = [float(v) for v in bounds[1]]
        extent = [h - low for low, h in zip(lo, hi)]
        if cell_size is not None:
            size = tuple(float(v) for v in cell_size)
        elif target_cells is not None:
            counts = (
                [int(target_cells)] * len(lo)
                if isinstance(target_cells, int)
                else [int(c) for c in target_cells]
            )
            size = tuple(
                (e / c if c > 0 else e) or 1.0 for e, c in zip(extent, counts)
            )
        else:
            size = tuple(e or 1.0 for e in extent)
        from zarr_vectors.spatial.chunking import grid_layout

        _origin, shape = grid_layout((lo, hi), size)
        return cls(shape=shape, cell_shape=size, origin=tuple(lo))

    # ---------------- geometry ----------------

    @property
    def cells(self) -> int:
        return int(math.prod(self.shape)) if self.shape else 0

    @property
    def anchor(self) -> tuple[int, ...]:
        """Absolute chunk coord of cell 0 — ``floor(origin / cell_shape)``.

        The same value the store writes as an array's ``chunk_grid_origin``
        attribute.  Derived rather than stored, so it cannot contradict
        :attr:`origin`.
        """
        if not self.cell_shape:
            return ()
        origin = self.origin or (0.0,) * len(self.cell_shape)
        return tuple(
            int(math.floor(o / c)) for o, c in zip(origin, self.cell_shape)
        )

    def cell_of(self, point: Sequence[float]) -> CellRef:
        """Which cell a coordinate falls in.

        Absolute, as stored: ``floor(point / cell_shape)``.  It used to
        subtract :attr:`origin`, which made it bounds-relative while
        :meth:`cells_in` — and the chunk keys on disk, and the ``chunks=``
        term a cell selection becomes — stayed absolute.  The two agreed
        only for a store whose bounds began at zero; anywhere else
        ``select(cells=[cell_of(p)])`` looked in a cell that does not hold
        ``p``, and usually does not exist.
        """
        from zarr_vectors.spatial.chunking import compute_chunk_coords

        # Delegated, not reimplemented: this is the single-vertex form of
        # ``assign_chunks``, so cell_of cannot drift from how cells are
        # written.
        return CellRef(compute_chunk_coords(
            np.asarray(point, dtype=np.float64), tuple(self.cell_shape),
        ))

    def cells_in(self, bbox: tuple[Sequence[float], Sequence[float]]) -> CellSet:
        """Every cell a bounding box touches.

        The replacement for ``chunks=[(3, 1, 2), ...]``: the caller states
        a region and receives opaque references, rather than stating grid
        coordinates they had to derive themselves.

        This is the **allocation**, not the occupancy: a grid is a value
        and holds no store handle, so it answers what the declared grid
        covers whether or not anything was written there.  On a sparse
        store the two differ by orders of magnitude, and a reference per
        allocated cell is a cost paid before any read.  Use
        :meth:`zarr_vectors.api.level.Level.cells` for the cells a level
        actually holds.
        """
        lo = np.asarray(bbox[0], dtype=np.float64)
        hi = np.asarray(bbox[1], dtype=np.float64)
        from zarr_vectors.spatial.chunking import chunks_intersecting_bbox

        coords = chunks_intersecting_bbox(lo, hi, tuple(self.cell_shape))
        return CellSet(CellRef(tuple(int(c) for c in cc)) for cc in coords)

    def __iter__(self) -> Iterator[CellRef]:
        """Every cell of the allocation, as absolute references.

        The allocation, so this is ``prod(shape)`` references however
        little of the grid holds data -- see :meth:`cells_in`.

        Yields nothing for a grid with no shape; it used to yield one
        bogus ``CellRef(())``, because ``itertools.product()`` over an
        empty axis list produces a single empty tuple.
        """
        import itertools

        if not self.shape:
            return
        anchor = self.anchor
        for offsets in itertools.product(*(range(s) for s in self.shape)):
            yield CellRef(tuple(a + o for a, o in zip(anchor, offsets)))

    # ---------------- capacity ----------------

    def capacity(
        self, *, n_vertices: int | None = None, ndim: int = 3,
        bytes_per_coordinate: int = 4, target_cell_bytes: int = 64 << 20,
    ) -> GridCapacity:
        """Whether this grid is a sensible shape for the data planned.

        Two ways a grid goes wrong: too few cells, so each one is an
        unwieldy object; or so many that the per-cell overhead dominates.
        """
        cells = self.cells
        per_cell = (
            (n_vertices / cells) * ndim * bytes_per_coordinate
            if n_vertices and cells else 0.0
        )
        reason = None
        fits = True
        if cells == 0:
            fits, reason = False, "the grid has no cells"
        elif per_cell > target_cell_bytes:
            fits = False
            reason = (
                f"~{per_cell / 1e6:.0f} MB per cell exceeds the "
                f"{target_cell_bytes / 1e6:.0f} MB target; use more cells"
            )
        return GridCapacity(
            cells=cells, shape=self.shape, est_bytes_per_cell=per_cell,
            fits=fits, reason=reason,
        )

    def holds(self, ref: CellRef) -> bool:
        """Whether ``ref`` is inside this grid's allocation.

        In the same absolute frame :meth:`cell_of` and :meth:`cells_in`
        speak, so a reference either of them produced is tested against
        the allocation it actually names.
        """
        coords = tuple(getattr(ref, "coords", ref))
        anchor = self.anchor
        if len(coords) != len(self.shape) or len(anchor) != len(self.shape):
            return False
        return all(
            a <= int(c) < a + s
            for c, a, s in zip(coords, anchor, self.shape)
        )

    def __repr__(self) -> str:
        shape = "x".join(str(s) for s in self.shape)
        return f"Grid({shape} cells of {self.cell_shape})"


def _unused(_: Any = None) -> None:  # pragma: no cover
    pass
