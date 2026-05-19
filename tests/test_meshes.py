"""Step 10 tests: mesh write/read core API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.types.meshes import write_mesh, read_mesh
from zarr_vectors.exceptions import ArrayError


class TestMeshBasic:

    def test_tetrahedron(self, tmp_path: Path) -> None:
        v = np.array([[0,0,0],[10,0,0],[5,10,0],[5,5,10]], dtype=np.float32)
        f = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]], dtype=np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        assert s["vertex_count"] == 4 and s["face_count"] == 4
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["vertex_count"] == 4 and r["face_count"] == 4

    def test_large_single_chunk(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        v = rng.uniform(0, 80, size=(200, 3)).astype(np.float32)
        f = rng.integers(0, 200, size=(400, 3)).astype(np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["vertex_count"] == 200 and r["face_count"] == 400

    def test_quad_mesh(self, tmp_path: Path) -> None:
        v = np.array([[0,0,0],[10,0,0],[10,10,0],[0,10,0],
                       [0,0,10],[10,0,10],[10,10,10],[0,10,10]], dtype=np.float32)
        f = np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]], dtype=np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["face_count"] == 6 and r["faces"].shape[1] == 4


class TestMeshCrossChunk:

    def test_cross_face(self, tmp_path: Path) -> None:
        v = np.array([[10,50,50],[50,50,50],[110,50,50]], dtype=np.float32)
        f = np.array([[0,1,2]], dtype=np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        assert s["cross_face_count"] == 1

    def test_cluster_faces_intra(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        c1 = rng.uniform(0, 50, size=(50, 3)).astype(np.float32)
        c2 = rng.uniform(100, 150, size=(50, 3)).astype(np.float32)
        v = np.vstack([c1, c2])
        f = np.vstack([rng.integers(0,50,size=(30,3)), rng.integers(50,100,size=(30,3))]).astype(np.int64)
        s = write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        assert s["intra_face_count"] == 60


class TestMeshBbox:

    def test_bbox_filter(self, tmp_path: Path) -> None:
        v = np.array([[10,10,10],[20,20,20],[30,30,30],
                       [150,150,150],[160,160,160],[170,170,170]], dtype=np.float32)
        f = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
        write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(200.,200.,200.))
        r = read_mesh(str(tmp_path / "m.zv"), bbox=(np.array([0,0,0]), np.array([50,50,50])))
        assert r["vertex_count"] == 3 and r["face_count"] == 1


class TestMeshEdgeCases:

    def test_bad_face_shape(self, tmp_path: Path) -> None:
        v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        try:
            write_mesh(str(tmp_path / "m.zv"), v, np.zeros((3,2), dtype=np.int64),
                        chunk_shape=(100.,100.,100.))
            assert False
        except ArrayError:
            pass

    def test_single_triangle(self, tmp_path: Path) -> None:
        v = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
        f = np.array([[0,1,2]], dtype=np.int64)
        write_mesh(str(tmp_path / "m.zv"), v, f, chunk_shape=(100.,100.,100.))
        r = read_mesh(str(tmp_path / "m.zv"))
        assert r["vertex_count"] == 3 and r["face_count"] == 1


class TestMeshNonDefaultDtypes:
    """Spec §7.1 vertices accept any numeric dtype (float or integer);
    spec §7.5 links accept any unsigned/signed integer dtype.  This
    class round-trips both to verify the spec relaxation lands at the
    impl layer."""

    def test_uint16_link_dtype(self, tmp_path: Path) -> None:
        """Writer ``link_dtype="uint16"`` shrinks face storage 4× vs the
        default ``int64`` and the reader honours the on-disk dtype via
        the ``links/0`` zattrs.  The shape and values must round-trip."""
        v = np.array([[0,0,0],[10,0,0],[5,10,0],[5,5,10]], dtype=np.float32)
        f = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]], dtype=np.int64)
        store = str(tmp_path / "m_uint16.zv")
        write_mesh(store, v, f, chunk_shape=(100.,100.,100.), link_dtype="uint16")

        # Verify the on-disk dtype is recorded in the array meta.
        from zarr_vectors.core.store import get_resolution_level, open_store
        root = open_store(store)
        lg = get_resolution_level(root, 0)
        lmeta = lg.read_array_meta("links/0")
        assert lmeta["dtype"] == "uint16"

        # Read back with the new (metadata-honoring) reader.
        r = read_mesh(store)
        assert r["vertex_count"] == 4
        assert r["face_count"] == 4
        assert r["faces"].shape == (4, 3)
        # Output is widened to int64 (correct — chunk_offset arithmetic
        # may exceed uint16 for large stores); values themselves still
        # match the input.
        assert sorted(map(tuple, r["faces"].tolist())) == sorted(
            map(tuple, f.tolist())
        )

    def test_integer_voxel_vertices(self, tmp_path: Path) -> None:
        """Vertices declared with an integer dtype (voxel indices) must
        round-trip through write_mesh / read_mesh without being silently
        coerced to float."""
        # Voxel-indexed positions: a uint32 grid of vertex coordinates.
        v = np.array(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            dtype=np.uint32,
        )
        f = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        store = str(tmp_path / "m_voxel.zv")
        write_mesh(
            store, v, f,
            chunk_shape=(1000., 1000., 1000.),
            dtype="uint32",
            # Explicit bounds — auto-inferred bounds default to a 128³
            # box, which our voxel coords would overflow.
            bounds=([0, 0, 0], [1000, 1000, 1000]),
        )

        from zarr_vectors.core.store import get_resolution_level, open_store
        root = open_store(store)
        vmeta = get_resolution_level(root, 0).read_array_meta("vertices")
        assert vmeta["dtype"] == "uint32"

        r = read_mesh(store)
        assert r["vertices"].dtype == np.uint32
        assert r["vertex_count"] == 4
        # Values match exactly — no float roundtrip damage.
        assert sorted(map(tuple, r["vertices"].tolist())) == sorted(
            map(tuple, v.tolist())
        )
