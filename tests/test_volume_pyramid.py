"""Volume mesh pyramids: coarse cells on the bin lattice, and coarse
levels written from a mesh decimated elsewhere (zarr_vectors#10)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_volume_elements import _tet_grid
from zarr_vectors import building as zb
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.multiresolution.coarsen import (
    _decode_parent_from_plus_one,
    build_pyramid,
)
from zarr_vectors.multiresolution.volume import lattice_cells
from zarr_vectors.types.meshes import (
    _level_vertex_table,
    read_mesh,
    write_mesh,
    write_mesh_level,
)
from zarr_vectors.validate import validate
from zarr_vectors.validate.consistency import validate_consistency

CHUNK = (40.0,) * 3
BIN = (10.0,) * 3


def _volume(m):
    return float(zb.signed_volumes(m["vertices"], m["faces"], m["element"]).sum())


def _store(tmp_path, n=9, element="tetra"):
    pts, tets = _tet_grid(n=n, step=10.0)
    out = str(tmp_path / "v.zv")
    write_mesh(out, pts, tets, chunk_shape=CHUNK, bin_shape=BIN, element=element)
    return out, pts, tets


def test_a_volume_pyramid_keeps_cells_at_every_level(tmp_path):
    out, _, _ = _store(tmp_path)
    summary = build_pyramid(out, factors=[(2.0, 1.0), (2.0, 1.0)])
    specs = summary["level_specs"]
    assert [s["element"] for s in specs] == ["tetra", "tetra"]
    assert all(s["cells_dropped_inverted"] == 0 for s in specs)
    # Bin centroids span 5..80 (level 1, bins of 20) and 15..80 (level 2,
    # bins of 40), so the unsnapped lattice holds 75**3 and 65**3;
    # snapping the boundary out to the fine surface recovers more.
    assert _volume(read_mesh(out, level=0)) == pytest.approx(80.0 ** 3)
    for level, unsnapped in ((1, 75.0 ** 3), (2, 65.0 ** 3)):
        m = read_mesh(out, level=level)
        assert m["element"] == "tetra"
        assert m["face_count"] > 0
        assert np.all(zb.signed_volumes(m["vertices"], m["faces"], "tetra") > 0)
        assert unsnapped < _volume(m) <= 80.0 ** 3 + 1e-6
    assert _volume(read_mesh(out, level=1)) > 0.9 * 80.0 ** 3
    assert validate_consistency(out).ok


def test_hexahedra_coarsen_to_hexahedra(tmp_path):
    g = np.stack(np.meshgrid(*[np.arange(5.0)] * 3, indexing="ij"), -1).reshape(-1, 3) * 10

    def idx(i, j, k):
        return (i * 5 + j) * 5 + k

    hexes = np.array([
        [idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
         idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
        for i in range(4) for j in range(4) for k in range(4)
    ])
    flip = zb.signed_volumes(g, hexes, "hexahedron") < 0
    hexes[flip] = hexes[flip][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    out = str(tmp_path / "h.zv")
    write_mesh(out, g, hexes, chunk_shape=CHUNK, bin_shape=BIN, element="hexahedron")
    build_pyramid(out, factors=[(2.0, 1.0)])
    m = read_mesh(out, level=1)
    assert m["element"] == "hexahedron"
    assert m["faces"].shape[1] == 8 and m["face_count"] > 0


def test_cells_never_join_bins_one_object_does_not_fill():
    # Two objects side by side: each occupies a 2x2x2 block of bins, and
    # the 2x2x2 blocks straddling them belong to neither.
    b = np.stack(np.meshgrid(np.arange(4), np.arange(2), np.arange(2), indexing="ij"), -1)
    b = b.reshape(-1, 3)
    pos = (b + 0.5) * 10.0
    left = np.flatnonzero(b[:, 0] < 2)
    right = np.flatnonzero(b[:, 0] >= 2)
    cells, dropped = lattice_cells(b, pos, [left, right], "hexahedron")
    assert cells.shape == (2, 8) and dropped == 0
    together, _ = lattice_cells(b, pos, [np.arange(len(b))], "hexahedron")
    assert together.shape == (3, 8)


def test_a_surface_mesh_pyramid_is_unchanged(tmp_path):
    pts = np.array([[0, 0, 0], [30, 0, 0], [0, 30, 0], [30, 30, 0]], float) + 5
    out = str(tmp_path / "s.zv")
    write_mesh(out, pts, np.array([[0, 1, 3], [0, 3, 2]]), chunk_shape=CHUNK, bin_shape=BIN)
    summary = build_pyramid(out, factors=[(2.0, 1.0)])
    assert "element" not in summary["level_specs"][0]
    assert read_mesh(out, level=1)["element"] is None


# ---------------------------------------------------------------------------
# write_mesh_level
# ---------------------------------------------------------------------------


def test_a_level_written_from_outside_reads_and_validates(tmp_path):
    out, _, _ = _store(tmp_path)
    cp, ct = _tet_grid(n=5, step=20.0)
    info = write_mesh_level(out, 1, cp, ct, element="tetra", bin_shape=(20.0,) * 3)
    assert info["level"] == 1 and info["parent_level"] == 0
    m = read_mesh(out, level=1)
    assert m["element"] == "tetra" and m["face_count"] == len(ct)
    meta = json.loads((Path(out) / "1" / "zarr.json").read_text())["attributes"]["zarr_vectors"]
    assert meta["coarsening_method"] == "manual"
    assert meta["parent_level"] == 0
    assert meta["bin_shape"] == [20.0] * 3 and meta["bin_ratio"] == [2.0] * 3
    assert validate_consistency(out).ok
    assert validate(out).ok


def test_parent_links_name_the_nearest_coarse_vertex(tmp_path):
    out, _, _ = _store(tmp_path)
    cp, ct = _tet_grid(n=5, step=20.0)
    cp = cp + np.array([3.0, 0.0, 0.0])  # off the fine lattice, so "nearest" means something
    cp[:, 0] = np.minimum(cp[:, 0], 80.0)
    info = write_mesh_level(out, 1, cp, ct, element="tetra", bin_shape=(20.0,) * 3,
                            check_orientation=False)
    root = open_store(out)
    fine_lg, coarse_lg = get_resolution_level(root, 0), get_resolution_level(root, 1)
    f_pos, _, f_assn, n_f = _level_vertex_table(fine_lg, 3)
    c_pos, _, c_assn, _ = _level_vertex_table(coarse_lg, 3)
    parent = _decode_parent_from_plus_one(
        fine_lg, fine_assn=f_assn, coarse_assn=c_assn, n_fine=n_f,
    )
    assert info["parent_links"] == n_f
    d = ((f_pos[:, None, :] - c_pos[None, :, :]) ** 2).sum(-1)
    got = d[np.arange(n_f), parent]
    assert np.allclose(got, d.min(axis=1))


def test_write_mesh_level_refusals(tmp_path):
    out, _, _ = _store(tmp_path, n=3)
    cp, ct = _tet_grid(n=2, step=20.0)
    with pytest.raises(ArrayError, match="coarse level"):
        write_mesh_level(out, 0, cp, ct, element="tetra")
    with pytest.raises(ArrayError, match="parent_level 1"):
        write_mesh_level(out, 2, cp, ct, element="tetra")
    with pytest.raises(ArrayError, match="not objects of level 0"):
        write_mesh_level(out, 1, cp, ct, element="tetra",
                         object_ids=np.full(len(cp), 7))
    write_mesh_level(out, 1, cp, ct, element="tetra")
    with pytest.raises(ArrayError, match="already exists"):
        write_mesh_level(out, 1, cp, ct, element="tetra")
