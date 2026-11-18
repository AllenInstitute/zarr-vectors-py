"""Mixed volume elements, face attributes, and what a mesh keeps at a
coarse level: snapped volume cells, clustered surface faces, and per-face
or per-cell values carried down (zarr_vectors#10)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_volume_elements import _tet_grid
from zarr_vectors import building as zb
from zarr_vectors.core.arrays import read_link_attributes
from zarr_vectors.core.elements import pack_mixed, record_widths, unpack_mixed
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.multiresolution.mesh_coarsen import aggregate
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.validate.consistency import validate_consistency

CHUNK = (40.0,) * 3
BIN = (10.0,) * 3


def _attr(path, name, level=0):
    return read_link_attributes(get_resolution_level(open_store(path), level), name, delta=0)


def _hex_block(n=5, step=10.0, x0=0.0):
    g = np.stack(np.meshgrid(*[np.arange(n, dtype=float)] * 3, indexing="ij"), -1)
    pts = g.reshape(-1, 3) * step + [x0, 0, 0]

    def idx(i, j, k):
        return (i * n + j) * n + k

    hexes = np.array([
        [idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
         idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
        for i in range(n - 1) for j in range(n - 1) for k in range(n - 1)
    ])
    flip = zb.signed_volumes(pts, hexes, "hexahedron") < 0
    hexes[flip] = hexes[flip][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    return pts, hexes


# ---------------------------------------------------------------------------
# Mixed families
# ---------------------------------------------------------------------------


def test_packing_names_each_record_by_its_width():
    recs, types = pack_mixed({"tetra": [[0, 1, 2, 3]], "wedge": [[4, 5, 6, 7, 8, 9]],
                              "hexahedron": [[0, 1, 2, 3, 4, 5, 6, 7]]})
    assert types == ["tetra", "wedge", "hexahedron"]
    assert recs.shape == (3, 8)
    assert record_widths(recs).tolist() == [4, 6, 8]
    back = unpack_mixed(recs, types)
    assert back["tetra"].tolist() == [[0, 1, 2, 3]]
    assert back["wedge"].tolist() == [[4, 5, 6, 7, 8, 9]]


def test_a_cell_that_repeats_a_vertex_is_refused():
    with pytest.raises(ArrayError, match="repeat a vertex"):
        pack_mixed({"tetra": [[0, 1, 1, 3]]})


def test_a_mixed_mesh_round_trips(tmp_path):
    tp, tets = _tet_grid(n=3, step=10.0)
    hp, hexes = _hex_block(n=3, x0=100.0)
    pts = np.concatenate([tp, hp])
    cells = {"tetra": tets, "hexahedron": hexes + len(tp)}
    out = str(tmp_path / "m.zv")
    info = write_mesh(out, pts, cells, chunk_shape=CHUNK, bin_shape=BIN)
    assert info["element"] == "mixed"
    m = read_mesh(out)
    assert m["element"] == "mixed"
    assert m["element_types"] == ["tetra", "hexahedron"]
    assert {k: v.shape for k, v in m["cells"].items()} == {
        "tetra": tets.shape, "hexahedron": hexes.shape,
    }
    assert np.all(zb.signed_volumes(m["vertices"], m["faces"], "mixed") > 0)
    # The boundary of the two separate blocks: 2 cubes' faces.
    tris, quads = zb.boundary_faces(m["faces"], "mixed")
    assert tris.shape[0] == 6 * 4 * 2 and quads.shape[0] == 6 * 4
    assert validate_consistency(out).ok


def test_mixed_needs_cells_by_type(tmp_path):
    tp, tets = _tet_grid(n=2)
    with pytest.raises(ArrayError, match="dict"):
        write_mesh(str(tmp_path / "x.zv"), tp, tets, chunk_shape=CHUNK, element="mixed")


# ---------------------------------------------------------------------------
# Face attributes
# ---------------------------------------------------------------------------


def test_face_attributes_follow_their_faces(tmp_path):
    pts, tets = _tet_grid(n=5, step=10.0)
    out = str(tmp_path / "a.zv")
    write_mesh(out, pts, tets, chunk_shape=CHUNK, bin_shape=BIN, element="tetra",
               face_attributes={"id": np.arange(len(tets), dtype=np.int64)})
    m = read_mesh(out)
    ids = _attr(out, "id")
    pos = {tuple(p): i for i, p in enumerate(pts)}
    got = np.array([[pos[tuple(m["vertices"][v])] for v in f] for f in m["faces"]])
    assert np.array_equal(got, tets[ids])


def test_a_face_attribute_needs_one_row_per_face(tmp_path):
    pts, tets = _tet_grid(n=2)
    with pytest.raises(ArrayError, match="one per face"):
        write_mesh(str(tmp_path / "x.zv"), pts, tets, chunk_shape=CHUNK, element="tetra",
                   face_attributes={"bad": np.zeros(3)})


# ---------------------------------------------------------------------------
# Coarse levels
# ---------------------------------------------------------------------------


def test_aggregate_means_floats_and_takes_the_majority_of_labels():
    groups = np.array([0, 0, 0, 1, -1])  # -1: maps to no coarse cell
    w = np.array([1.0, 1.0, 3.0, 1.0, 9.0])
    means = aggregate(np.array([1.0, 3.0, 5.0, 7.0, 100.0]), groups, w, 3)
    # Group 2 is empty and takes the mean over everything that mapped.
    assert means.tolist() == pytest.approx([19 / 5, 7.0, 26 / 6])
    labels = aggregate(np.array([4, 4, 9, 2, 1]), groups, w, 3)
    # 9 outweighs the two 4s (3 against 2); the empty group takes the
    # overall majority, 9 again.
    assert labels.tolist() == [9, 2, 9]
    assert labels.dtype == np.int64


def test_boundary_snapping_recovers_most_of_the_volume(tmp_path):
    pts, tets = _tet_grid(n=9, step=10.0)
    out = str(tmp_path / "s.zv")
    write_mesh(out, pts, tets, chunk_shape=CHUNK, bin_shape=BIN, element="tetra")
    spec = build_pyramid(out, factors=[(2.0, 1.0)])["level_specs"][0]
    assert spec["boundary_vertices_snapped"] > 0
    assert spec["cells_dropped_inverted"] == 0
    m = read_mesh(out, level=1)
    vol = zb.signed_volumes(m["vertices"], m["faces"], "tetra")
    assert np.all(vol > 0)
    # Unsnapped, the lattice spans bin centroids 5..80: 75**3.  Snapped,
    # the faces of the box are back at 0 and 80.
    assert vol.sum() > 0.9 * 80.0 ** 3 > 75.0 ** 3
    assert validate_consistency(out).ok


def test_cell_values_come_down_by_volume(tmp_path):
    pts, tets = _tet_grid(n=9, step=10.0)
    centroid = pts[tets].mean(axis=1)
    material = (centroid[:, 0] > 40).astype(np.int32)  # two halves
    out = str(tmp_path / "c.zv")
    write_mesh(out, pts, tets, chunk_shape=CHUNK, bin_shape=BIN, element="tetra",
               face_attributes={"material": material, "x": centroid[:, 0]})
    build_pyramid(out, factors=[(2.0, 1.0)])
    m = read_mesh(out, level=1)
    coarse_x = m["vertices"][m["faces"]].mean(axis=1)[:, 0]
    mat = _attr(out, "material", 1)
    x = _attr(out, "x", 1)
    assert mat.shape == (m["face_count"],) and mat.dtype == np.int32
    assert np.all(mat[coarse_x < 30] == 0) and np.all(mat[coarse_x > 50] == 1)
    assert np.abs(x - coarse_x).max() < 15.0  # averaged within about a block


def test_a_surface_keeps_faces_and_labels_at_coarse_levels(tmp_path):
    n = 17
    g = np.stack(np.meshgrid(np.arange(n) * 5.0, np.arange(n) * 5.0, indexing="ij"), -1)
    pts = np.c_[g.reshape(-1, 2), np.full(n * n, 12.0)]
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [[a, b, d], [a, d, c]]
    tris = np.array(tris)
    side = (pts[tris].mean(axis=1)[:, 0] > 40).astype(np.int16)
    out = str(tmp_path / "t.zv")
    write_mesh(out, pts, tris, chunk_shape=CHUNK, bin_shape=(5.0,) * 3,
               face_attributes={"side": side})
    spec = build_pyramid(out, factors=[(2.0, 1.0)])["level_specs"][0]
    m = read_mesh(out, level=1)
    assert m["element"] is None and m["face_count"] == spec["face_count"] > 0
    s = _attr(out, "side", 1)
    cx = m["vertices"][m["faces"]].mean(axis=1)[:, 0]
    assert np.all(s[cx < 30] == 0) and np.all(s[cx > 50] == 1)
    # Clustered faces keep their winding: every normal points the way
    # the input's do.
    def nz(v):
        return np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2]

    fine_sign = np.sign(nz(pts[tris]))
    assert np.all(fine_sign == fine_sign[0])
    assert np.all(np.sign(nz(m["vertices"][m["faces"]])) == fine_sign[0])
    assert validate_consistency(out).ok


def test_a_mixed_volume_coarsens_to_tetrahedra(tmp_path):
    tp, tets = _tet_grid(n=5, step=10.0)
    out = str(tmp_path / "mx.zv")
    write_mesh(out, tp, {"tetra": tets}, chunk_shape=CHUNK, bin_shape=BIN)
    build_pyramid(out, factors=[(2.0, 1.0)])
    m = read_mesh(out, level=1)
    assert m["element"] == "tetra" and m["face_count"] > 0
