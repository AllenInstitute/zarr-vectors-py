"""Tests for cross-chunk face records.

Cross-chunk faces are ``link_width=3`` records in the ``links/<delta=0>/``
family rather than a separate ``cross_chunk_faces`` array.  Under the
offset layout they are not a separate family either: a face whose three
vertices straddle chunks lands in the array named by the offsets between
them, and one wholly inside a chunk lands in the all-zero offsets array.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import read_links
from zarr_vectors.core.paths import links_group_path
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
)
from zarr_vectors.types.meshes import read_mesh, write_mesh


def read_cross_links(level_group, *, delta: int = 0) -> list:
    """Records spanning more than one chunk.

    ``read_links`` returns the whole family — intra-chunk links are the
    all-zero-offsets array, not a separate family — so tests that care
    about the pre-merge ``cross_chunk_links`` set filter for it.
    """
    return [
        record for record in read_links(level_group, delta=delta)
        if len({tuple(cc) for cc, _vi in record}) > 1
    ]


def _tetra_straddling_chunks(tmp_path):
    """A 4-vertex tetrahedron whose vertices land in different 50³ chunks."""
    verts = np.array([
        [40, 40, 40],  # chunk (0,0,0)
        [60, 40, 40],  # chunk (1,0,0)
        [50, 50, 50],  # chunk (1,1,1)  (exactly on boundary; rounds up)
        [50, 50, 60],  # chunk (1,1,1)
    ], dtype="f4")
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 3],
    ], dtype=np.int64)
    store = tmp_path / "m.zv"
    write_mesh(
        str(store), verts, faces,
        chunk_shape=(50.0, 50.0, 50.0),
    )
    return store, verts, faces


def test_read_mesh_returns_cross_chunk_faces(tmp_path):
    store, _, faces_in = _tetra_straddling_chunks(tmp_path)
    out = read_mesh(str(store))
    assert out["face_count"] == len(faces_in)
    assert out["vertex_count"] == 4


def test_cross_chunk_face_records_round_trip(tmp_path):
    store, _, _ = _tetra_straddling_chunks(tmp_path)
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    records = read_cross_links(lvl, delta=0)
    # All 4 faces of the tetrahedron span chunks → all 4 appear here.
    assert len(records) == 4
    # Every face has 3 endpoints (triangle, link_width=3).
    for face in records:
        assert len(face) == 3
        for cc, local_idx in face:
            assert len(cc) == 3
            assert local_idx >= 0
    # Every face straddling chunks means nothing landed in the intra
    # (all-zero offsets) array, so the whole family is the cross set.
    assert len(read_links(lvl, delta=0)) == 4


def test_intra_chunk_mesh_writes_no_cross_chunk_links(tmp_path):
    """A mesh wholly inside one chunk writes no cross-chunk records."""
    verts = np.array([
        [10, 10, 10], [20, 10, 10], [15, 20, 10], [15, 15, 20],
    ], dtype="f4")
    faces = np.array([
        [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3],
    ], dtype=np.int64)
    store = tmp_path / "m.zv"
    write_mesh(str(store), verts, faces, chunk_shape=(50.0, 50.0, 50.0))
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    assert read_cross_links(lvl, delta=0) == []
    # The faces are still stored — under the all-zero (intra) offsets,
    # which is what "no cross_chunk_links" means now.
    assert len(read_links(lvl, delta=0)) == 4
    assert lvl[links_group_path(0)].children() == ["0.0.0_0.0.0"]
