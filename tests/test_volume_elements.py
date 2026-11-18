"""Volume mesh elements: a links family whose records are solid cells.

AllenInstitute/zarr_vectors#10.  ``write_mesh(element=...)`` stores one
tetrahedron (pyramid, wedge, hexahedron) per record and records the
element on the ``links/0`` family, so a reader knows four indices are a
tetrahedron and not a quad.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.core.elements import ELEMENT_FACES, ELEMENT_WIDTH
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.validate.consistency import validate_consistency

#: One unit cell of each element, in VTK order.
UNIT = {
    "tetra": (
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 2, 3]],
    ),
    "pyramid": (
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        [[0, 1, 2, 3, 4]],
    ),
    "wedge": (
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 2, 3, 4, 5]],
    ),
    "hexahedron": (
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [[0, 1, 2, 3, 4, 5, 6, 7]],
    ),
}
VOLUME = {"tetra": 1 / 6, "pyramid": 1 / 3, "wedge": 0.5, "hexahedron": 1.0}


def _tet_grid(n: int = 4, step: float = 30.0):
    """An ``n``-point lattice, each cube split into six positive tetrahedra."""
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij"), -1)
    pts = g.reshape(-1, 3) * step

    def idx(i, j, k):
        return (i * n + j) * n + k

    tets = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                # Kuhn split along the 0-7 diagonal.
                for p, q in ((1, 3), (5, 1), (3, 2), (2, 6), (4, 5), (6, 4)):
                    tets.append([c[0], c[p], c[q], c[7]])
    tets = np.array(tets)
    vol = zb.signed_volumes(pts, tets, "tetra")
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    return pts, tets


@pytest.mark.parametrize("element", sorted(ELEMENT_WIDTH))
def test_unit_elements_have_their_volume(element):
    pts, cells = UNIT[element]
    vol = zb.signed_volumes(np.array(pts, float), cells, element)
    assert vol[0] == pytest.approx(VOLUME[element])


@pytest.mark.parametrize("element", sorted(ELEMENT_WIDTH))
def test_a_single_cell_is_its_own_closed_boundary(element):
    pts, cells = UNIT[element]
    faces = zb.boundary_faces(cells, element)
    assert sum(f.shape[0] for f in faces) == len(ELEMENT_FACES[element])
    # Every boundary face points away from the cell's centroid.
    pts = np.array(pts, float)
    centre = pts.mean(axis=0)
    for f in faces:
        for face in f:
            a, b, c = pts[face[0]], pts[face[1]], pts[face[2]]
            assert np.dot(np.cross(b - a, c - a), a - centre) > 0


def test_shared_faces_are_not_boundary():
    pts, tets = _tet_grid(n=2)
    (tris,) = zb.boundary_faces(tets, "tetra")
    # A cube split into six tetrahedra has the cube's 6 faces x 2 triangles.
    assert tris.shape == (12, 3)


def test_a_tetrahedral_mesh_round_trips_with_its_element(tmp_path):
    pts, tets = _tet_grid()
    out = str(tmp_path / "tet.zv")
    info = write_mesh(out, pts, tets, chunk_shape=(40.0,) * 3, element="tetra")
    assert info["element"] == "tetra"
    m = read_mesh(out)
    assert m["element"] == "tetra"
    pos = {tuple(p): i for i, p in enumerate(pts)}
    got = np.array([[pos[tuple(m["vertices"][v])] for v in f] for f in m["faces"]])
    # Vertex order is the orientation, so the order must survive too.
    assert sorted(map(tuple, got)) == sorted(map(tuple, tets))
    fam = json.loads((Path(out) / "0" / "links" / "0" / "zarr.json").read_text())
    assert fam["attributes"]["zarr_vectors"]["element"] == "tetra"
    assert validate_consistency(out).ok


def test_a_surface_mesh_reads_with_no_element(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float) * 10
    out = str(tmp_path / "quad.zv")
    write_mesh(out, pts, np.array([[0, 1, 3, 2]]), chunk_shape=(40.0,) * 3)
    assert read_mesh(out)["element"] is None


def test_the_width_must_match_the_element(tmp_path):
    pts, tets = _tet_grid(n=2)
    with pytest.raises(ArrayError, match="5 vertices"):
        write_mesh(str(tmp_path / "x.zv"), pts, tets, chunk_shape=(40.0,) * 3,
                   element="pyramid")
    with pytest.raises(ArrayError, match="unknown element"):
        write_mesh(str(tmp_path / "y.zv"), pts, tets, chunk_shape=(40.0,) * 3,
                   element="tet")
    assert not (tmp_path / "x.zv").exists()


def test_inverted_cells_are_refused_before_anything_is_written(tmp_path):
    pts, tets = _tet_grid(n=2)
    flipped = tets.copy()
    flipped[2] = flipped[2][[0, 2, 1, 3]]
    out = tmp_path / "bad.zv"
    with pytest.raises(ArrayError, match="1 of 6 tetra cells"):
        write_mesh(str(out), pts, flipped, chunk_shape=(40.0,) * 3, element="tetra")
    assert not out.exists()
    write_mesh(str(out), pts, flipped, chunk_shape=(40.0,) * 3, element="tetra",
               check_orientation=False)
    assert read_mesh(str(out))["face_count"] == 6


def test_draco_is_refused_for_volume_cells(tmp_path):
    pts, tets = _tet_grid(n=2)
    with pytest.raises(ArrayError, match="triangles only"):
        write_mesh(str(tmp_path / "d.zv"), pts, tets, chunk_shape=(40.0,) * 3,
                   element="tetra", encoding="draco")


def test_existing_records_are_not_retyped(tmp_path):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 10
    out = str(tmp_path / "q.zv")
    write_mesh(out, pts, np.array([[0, 1, 2, 3]]), chunk_shape=(40.0,) * 3)
    lg = get_resolution_level(open_store(out, mode="r+"), 0)
    with pytest.raises(ArrayError, match="cannot redeclare"):
        zb.stamp_link_element(lg, "tetra")
    zb.stamp_link_element(lg, None)  # the same meaning is a no-op


def test_the_validator_checks_the_declaration(tmp_path):
    pts, tets = _tet_grid(n=2)
    out = str(tmp_path / "v.zv")
    write_mesh(out, pts, tets, chunk_shape=(40.0,) * 3, element="tetra")
    fam = Path(out) / "0" / "links" / "0" / "zarr.json"
    doc = json.loads(fam.read_text())
    doc["attributes"]["zarr_vectors"]["element"] = "hexahedron"
    fam.write_text(json.dumps(doc))
    result = validate_consistency(out)
    assert not result.ok
    assert any("8 vertices but link_width is 4" in e for e in result.errors)


def test_the_dataset_api_writes_a_volume_mesh(tmp_path):
    import zarr_vectors as zv

    pts, tets = _tet_grid()
    path = str(tmp_path / "ds.zv")
    ds = zv.create(path, schema=zv.Schema(
        bounds=([0, 0, 0], [120.0] * 3), kind="mesh", layout=zv.Layout(cells=3)))
    ds.add_mesh(pts, tets, element="tetra")
    assert read_mesh(path)["element"] == "tetra"


def test_rechunking_keeps_the_element(tmp_path):
    from zarr_vectors.rechunk.spatial import rechunk_spatial

    pts, tets = _tet_grid()
    src = str(tmp_path / "src.zv")
    write_mesh(src, pts, tets, chunk_shape=(40.0,) * 3, element="tetra")
    out = str(tmp_path / "out.zv")
    rechunk_spatial(src, out, (80.0,) * 3)
    m = read_mesh(out)
    assert m["element"] == "tetra"
    assert m["face_count"] == len(tets)
    assert validate_consistency(out).ok
