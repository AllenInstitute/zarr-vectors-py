"""Level 4 asks whether a links array exists, not what its metadata says.

``Group.read_array_meta`` answers ``{}`` for a missing path instead of
raising, so a ``try: read_array_meta(...)`` existence test always took the
"exists" branch: every point cloud warned that it had links.
"""

from __future__ import annotations

import numpy as np
import zarr

from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import write_points
from zarr_vectors.validate import validate

LINKS_WARNING = "Point cloud but links array exists"


def _points(tmp_path):
    path = str(tmp_path / "p.zv")
    rng = np.random.default_rng(0)
    write_points(path, rng.uniform(0, 100, (500, 3)).astype("f4"),
                 chunk_shape=(50, 50, 50))
    return path


def test_point_cloud_without_links_does_not_warn(tmp_path):
    r = validate(_points(tmp_path), level=4)
    assert r.ok, r.summary()
    assert LINKS_WARNING not in r.warnings
    assert "Point cloud: no links (correct)" in r.passed


def test_point_cloud_with_links_still_warns(tmp_path):
    path = _points(tmp_path)
    zarr.open_group(path, mode="r+")["0"].require_group("links/0")
    r = validate(path, level=4)
    assert LINKS_WARNING in r.warnings


def test_mesh_link_width_is_read(tmp_path):
    path = str(tmp_path / "m.zv")
    verts = np.array([[1, 1, 1], [9, 1, 1], [1, 9, 1], [1, 1, 9]], "f4")
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    write_mesh(path, verts, faces, chunk_shape=(50, 50, 50))
    r = validate(path, level=4)
    assert r.ok, r.summary()
    assert any(p.startswith("Mesh link_width=3") for p in r.passed), r.summary()
    assert not any("no links metadata" in w for w in r.warnings)
