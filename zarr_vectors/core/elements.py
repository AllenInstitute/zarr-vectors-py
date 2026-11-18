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

A family may also hold several element types at once:
``element: "mixed"``, with ``element_types`` listing the types present.
Records are as wide as the widest of them, and a shorter cell repeats
its last vertex to fill the record.  A cell's vertices are distinct, so
the trailing run of copies gives each record's real width, and the
widths (4, 5, 6, 8) name the types: a record says what it is, under any
filtered read.  Every record stays a list of real vertices, so routing,
sorting and the permutation index treat it like any other.

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


#: ``element`` of a family holding several element types.
MIXED = "mixed"

#: The family field listing a mixed family's types; ``element_type``
#: values index it.
ELEMENT_TYPES_KEY = "element_types"


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
    ``element`` may be ``"mixed"`` for padded records of several types.

    By the divergence theorem over the element's faces, each fanned into
    triangles, so it is exact for planar faces and a consistent
    orientation test otherwise.  Needs 3-D positions.
    """
    if element == MIXED:
        cells = np.asarray(cells, dtype=np.int64)
        vol = np.zeros(cells.shape[0], dtype=np.float64)
        widths = record_widths(cells) if cells.size else np.zeros(0, np.int64)
        for w in np.unique(widths).tolist():
            rows = widths == w
            vol[rows] = signed_volumes(vertices, cells[rows, :w], ELEMENT_OF_WIDTH[int(w)])
        return vol
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
    ``element`` may be ``"mixed"`` for padded records of several types.

    Returns one ``(F, k)`` array per face size present (triangles, then
    quads), each face keeping its owning cell's outward winding -- the
    closed surface of the volume when the cells are consistently
    oriented.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if element == MIXED:
        groups = list(unpack_mixed(cells).items()) if cells.size else []
    else:
        width = check_element(element)
        if cells.ndim != 2 or cells.shape[1] != width:
            raise ArrayError(f"{element} cells must be (N, {width}), got {cells.shape}")
        groups = [(element, cells)]
    by_size: dict[int, list[npt.NDArray[np.int64]]] = {}
    for el, c in groups:
        for face in ELEMENT_FACES[el]:
            by_size.setdefault(len(face), []).append(c[:, list(face)])
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


def check_mixed(element_types: list[str], link_width: int | None = None) -> int:
    """The record width of a mixed family over ``element_types``."""
    if not element_types:
        raise ArrayError("a mixed family needs at least one element type")
    if len(set(element_types)) != len(element_types):
        raise ArrayError(f"element_types repeats a type: {element_types}")
    width = max(check_element(e) for e in element_types)
    if link_width is not None and int(link_width) != width:
        raise ArrayError(
            f"element types {element_types} need records {width} wide, "
            f"but the records have {link_width}"
        )
    return width


#: Element of each width; the widths are distinct, which is what lets a
#: mixed record name its type.
ELEMENT_OF_WIDTH: dict[int, str] = {w: e for e, w in ELEMENT_WIDTH.items()}


def check_distinct(cells: npt.ArrayLike, element: str) -> None:
    """Refuse cells that repeat a vertex: degenerate, and in a mixed
    family indistinguishable from a narrower element."""
    cells = np.asarray(cells, dtype=np.int64)
    if cells.shape[0] == 0:
        return
    srt = np.sort(cells, axis=1)
    bad = np.flatnonzero((srt[:, 1:] == srt[:, :-1]).any(axis=1))
    if bad.size:
        raise ArrayError(
            f"{bad.size} {element} cells repeat a vertex (first: {bad[:5].tolist()}); "
            f"a cell's vertices must be distinct"
        )


def pack_mixed(
    cells: dict[str, npt.ArrayLike],
) -> tuple[npt.NDArray[np.int64], list[str]]:
    """``{element: (N_e, W_e)}`` -> ``(records, element_types)``: every
    type's cells, in the order given, padded to the widest type by
    repeating each cell's last vertex."""
    element_types = list(cells)
    width = check_mixed(element_types)
    rows = []
    for element in element_types:
        c = np.asarray(cells[element], dtype=np.int64).reshape(-1, ELEMENT_WIDTH[element])
        check_distinct(c, element)
        rows.append(np.concatenate([c, np.repeat(c[:, -1:], width - c.shape[1], axis=1)], axis=1))
    return np.concatenate(rows), element_types


def record_widths(records: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Each mixed record's real width: its length less the trailing
    copies of its last vertex."""
    records = np.asarray(records, dtype=np.int64)
    n, width = records.shape
    run = np.ones(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    for k in range(width - 2, -1, -1):
        alive &= records[:, k] == records[:, -1]
        run += alive
    return width - run + 1


def unpack_mixed(
    records: npt.ArrayLike, element_types: list[str] | None = None,
) -> dict[str, npt.NDArray[np.int64]]:
    """Inverse of :func:`pack_mixed`: ``{element: (N_e, W_e)}``, each
    type's cells in record order.  A record whose width names no type
    (of ``element_types``, when given) raises."""
    records = np.asarray(records, dtype=np.int64)
    widths = record_widths(records) if records.size else np.zeros(0, np.int64)
    out: dict[str, npt.NDArray[np.int64]] = {}
    for w in np.unique(widths).tolist():
        element = ELEMENT_OF_WIDTH.get(int(w))
        if element is None or (element_types is not None and element not in element_types):
            raise ArrayError(f"a mixed record has {w} distinct vertices, which names no "
                             f"element of {element_types or sorted(ELEMENT_WIDTH)}")
        out[element] = records[widths == w, :w]
    for element in element_types or []:
        out.setdefault(element, np.zeros((0, ELEMENT_WIDTH[element]), dtype=np.int64))
    return out
