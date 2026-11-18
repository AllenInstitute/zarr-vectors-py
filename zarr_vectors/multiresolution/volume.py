"""Coarse cells for a volume mesh: the bin lattice of the coarse level.

Merging a fine volume mesh's vertices into per-bin metavertices and then
remapping its cells does not work: a fine cell is much smaller than a
coarse bin, so nearly every cell collapses and the level comes out empty.
A coarse level's cells are built instead, on the lattice of its bins:

* every metavertex sits in one bin of the coarse level (it is the
  centroid of the fine vertices in that bin);
* for each object, every 2 x 2 x 2 block of bins that the object
  occupies in full becomes one cell joining those eight metavertices --
  a hexahedron, or six tetrahedra (Kuhn's split along the block's main
  diagonal, which tiles space with neighbouring blocks);
* a cell whose signed volume is not positive at the metavertices' actual
  positions (folded near the surface) is dropped, never written inverted.

The cells fill the occupied volume with no gaps, and each block depends
only on its own eight bins, so the build is local.  The surface is
accurate to about one bin: the volume shrinks by up to half a bin at the
boundary, and a feature thinner than two bins has no cells at that
level.  Blocks are per object, so cells never join two objects' bins
unless both objects occupy them; a block several objects occupy yields
one set of cells.
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.elements import signed_volumes

#: The eight corners of a block, as ``(d0, d1, d2)`` offsets along the
#: three spatial axes; a corner's index is ``d0 + 2*d1 + 4*d2``.
_CORNERS = np.array([(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.int64)


def _corner(d0: int, d1: int, d2: int) -> int:
    return d0 + 2 * d1 + 4 * d2


def _orient(templates: list[list[int]], element: str) -> npt.NDArray[np.int64]:
    """Templates over the corner indices, each flipped to positive volume
    on the unit block (the sign depends on the axes' handedness)."""
    out = []
    unit = _CORNERS.astype(np.float64)
    for t in templates:
        t = list(t)
        if signed_volumes(unit, [t], element)[0] < 0:
            if element == "tetra":
                t[1], t[2] = t[2], t[1]
            else:  # hexahedron: mirror the faces' winding
                t = [t[0], t[3], t[2], t[1], t[4], t[7], t[6], t[5]]
        assert signed_volumes(unit, [t], element)[0] > 0
        out.append(t)
    return np.array(out, dtype=np.int64)


#: The block as one VTK hexahedron.
HEX_TEMPLATE = _orient([[
    _corner(0, 0, 0), _corner(1, 0, 0), _corner(1, 1, 0), _corner(0, 1, 0),
    _corner(0, 0, 1), _corner(1, 0, 1), _corner(1, 1, 1), _corner(0, 1, 1),
]], "hexahedron")

#: The block as six tetrahedra around the (0,0,0)-(1,1,1) diagonal, one
#: per order of stepping along the three axes.
TET_TEMPLATE = _orient([
    [
        _corner(0, 0, 0),
        _corner(*np.eye(3, dtype=int)[p[0]]),
        _corner(*(np.eye(3, dtype=int)[p[0]] + np.eye(3, dtype=int)[p[1]])),
        _corner(1, 1, 1),
    ]
    for p in itertools.permutations(range(3))
], "tetra")


def coarse_element(source_element: str) -> str:
    """The element a coarse level is built with: hexahedra stay
    hexahedra; every other element becomes tetrahedra."""
    return "hexahedron" if source_element == "hexahedron" else "tetra"


def lattice_cells(
    mv_bins: npt.NDArray[np.int64],
    positions: npt.NDArray[np.floating],
    objects: list[npt.NDArray[np.int64]],
    element: str,
) -> tuple[npt.NDArray[np.int64], int]:
    """Cells over the occupied 2 x 2 x 2 blocks of the bin lattice.

    Args:
        mv_bins: ``(M, 3)`` bin coordinates of each metavertex.
        positions: ``(M, 3)`` metavertex positions.
        objects: per object, the metavertex indices it occupies.
        element: ``"tetra"`` or ``"hexahedron"``.

    Returns:
        ``(cells, dropped)``: ``(K, 4)`` or ``(K, 8)`` metavertex indices
        in VTK order, positive volume, no duplicates; and how many cells
        were dropped for non-positive volume.
    """
    template = HEX_TEMPLATE if element == "hexahedron" else TET_TEMPLATE
    width = template.shape[1]
    mv_bins = np.asarray(mv_bins, dtype=np.int64)
    blocks: list[npt.NDArray[np.int64]] = []
    for idx in objects:
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        if idx.size < 8:
            continue
        bins = mv_bins[idx]
        lo = bins.min(axis=0)
        dims = bins.max(axis=0) - lo + 2
        lin = np.ravel_multi_index((bins - lo).T, dims)
        order = np.argsort(lin)
        sorted_lin = lin[order]
        corner_mv = np.empty((idx.size, 8), dtype=np.int64)
        present = np.ones(idx.size, dtype=bool)
        for c, off in enumerate(_CORNERS):
            key = np.ravel_multi_index((bins - lo + off).T, dims)
            pos = np.searchsorted(sorted_lin, key)
            pos_c = np.minimum(pos, sorted_lin.size - 1)
            hit = sorted_lin[pos_c] == key
            present &= hit
            corner_mv[:, c] = idx[order[pos_c]]
        if present.any():
            blocks.append(corner_mv[present])
    if not blocks:
        return np.zeros((0, width), dtype=np.int64), 0
    corners = np.unique(np.concatenate(blocks), axis=0)
    cells = corners[:, template].reshape(-1, width)
    vol = signed_volumes(np.asarray(positions, dtype=np.float64), cells, element)
    keep = vol > 0
    return cells[keep], int((~keep).sum())
