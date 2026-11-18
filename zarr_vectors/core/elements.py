"""Volume elements: what an ``L``-wide link record is, when it is a cell.

A links family's records are edges (``L = 2``) or polygon faces
(``L >= 3``) unless the family says otherwise.  A family whose
``element`` field names a volume element holds one solid cell per record
instead -- a tetrahedron, pyramid, wedge or hexahedron -- with its
vertices in VTK order:

=============  ===  ==================================================
``element``     L   vertex order
=============  ===  ==================================================
``tetra``       4   ``0, 1, 2`` a triangle whose right-hand normal
                    points towards vertex ``3``
``pyramid``     5   base quad ``0-3`` whose right-hand normal points
                    towards the apex ``4``
``wedge``       6   triangle ``0, 1, 2`` whose right-hand normal points
                    away from triangle ``3, 4, 5``; ``i`` and ``i + 3``
                    joined by an edge
``hexahedron``  8   quad ``0-3`` whose right-hand normal points towards
                    quad ``4-7``; ``i`` and ``i + 4`` joined by an edge
=============  ===  ==================================================

Under that order every element has positive volume.  The face tables
below list each element's faces with outward right-hand normals; they
drive both :func:`signed_volumes` and :func:`boundary_faces`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ArrayError

#: The ``links_family`` field naming what each record is.
ELEMENT_KEY = "element"

#: Vertices per element.
ELEMENT_WIDTH: dict[str, int] = {
    "tetra": 4,
    "pyramid": 5,
    "wedge": 6,
    "hexahedron": 8,
}

#: Faces of each element, outward right-hand normals (VTK's tables).
ELEMENT_FACES: dict[str, tuple[tuple[int, ...], ...]] = {
    "tetra": ((0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)),
    "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    "wedge": ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    "hexahedron": (
        (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
        (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
    ),
}


def check_element(element: str, link_width: int | None = None) -> int:
    """The width of ``element``; raises for an unknown element or one
    whose width is not ``link_width``."""
    if element not in ELEMENT_WIDTH:
        raise ArrayError(
            f"unknown element {element!r}; expected one of {sorted(ELEMENT_WIDTH)}"
        )
    width = ELEMENT_WIDTH[element]
    if link_width is not None and int(link_width) != width:
        raise ArrayError(
            f"element {element!r} has {width} vertices, but the records have "
            f"{link_width}"
        )
    return width


def signed_volumes(
    vertices: npt.ArrayLike, cells: npt.ArrayLike, element: str,
) -> npt.NDArray[np.float64]:
    """Signed volume of each cell; positive when the cell is in VTK order.

    By the divergence theorem over the element's faces, each fanned into
    triangles, so it is exact for planar faces and a consistent
    orientation test otherwise.  Needs 3-D positions.
    """
    width = check_element(element)
    pos = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ArrayError(f"signed volumes need (V, 3) positions, got {pos.shape}")
    if cells.ndim != 2 or cells.shape[1] != width:
        raise ArrayError(f"{element} cells must be (N, {width}), got {cells.shape}")
    vol = np.zeros(cells.shape[0], dtype=np.float64)
    for face in ELEMENT_FACES[element]:
        a = pos[cells[:, face[0]]]
        for i in range(1, len(face) - 1):
            b = pos[cells[:, face[i]]]
            c = pos[cells[:, face[i + 1]]]
            vol += np.einsum("ij,ij->i", a, np.cross(b, c))
    return vol / 6.0


def boundary_faces(cells: npt.ArrayLike, element: str) -> list[npt.NDArray[np.int64]]:
    """The faces of a volume mesh that belong to exactly one cell.

    Returns one ``(F, k)`` array per face size present (triangles, then
    quads), each face keeping its owning cell's outward winding -- the
    closed surface of the volume when the cells are consistently
    oriented.
    """
    width = check_element(element)
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != width:
        raise ArrayError(f"{element} cells must be (N, {width}), got {cells.shape}")
    by_size: dict[int, list[npt.NDArray[np.int64]]] = {}
    for face in ELEMENT_FACES[element]:
        by_size.setdefault(len(face), []).append(cells[:, list(face)])
    out = []
    for size in sorted(by_size):
        faces = np.concatenate(by_size[size]) if by_size[size] else np.zeros((0, size), np.int64)
        if faces.shape[0] == 0:
            out.append(faces)
            continue
        key = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        out.append(faces[counts[inverse.ravel()] == 1])
    return out


def family_element(fam_meta: dict[str, Any]) -> str | None:
    """The family's element, or None for edges / polygon faces."""
    element = fam_meta.get(ELEMENT_KEY)
    return None if element is None else str(element)
