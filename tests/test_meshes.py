"""Step 10 tests: mesh write/read core API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Scoped reads: object_ids= and chunk-narrowed faces
# ---------------------------------------------------------------------------

CHUNK = (50.0, 50.0, 50.0)
BOUNDS = ([0.0, 0.0, 0.0], [200.0, 200.0, 200.0])


def _patch_mesh(n_objects: int = 12, side: int = 6, span: float = 40.0, seed: int = 3):
    """``n_objects`` square grid patches, each ``side x side`` vertices,
    scattered so that many straddle a chunk boundary."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, span, side), np.linspace(0, span, side), indexing="ij")
    i, j = np.meshgrid(np.arange(side - 1), np.arange(side - 1), indexing="ij")
    a = (i * side + j).ravel()
    quad = np.stack([a, a + side, a + 1, a + 1, a + side, a + side + 1], axis=1)
    tri = quad.reshape(-1, 3)
    verts, faces, oids = [], [], []
    for k in range(n_objects):
        origin = rng.uniform(5, 200 - span - 5, size=3)
        normal_axis = k % 3
        p = np.zeros((side * side, 3))
        axes = [ax for ax in range(3) if ax != normal_axis]
        p[:, axes[0]] = u.ravel()
        p[:, axes[1]] = v.ravel()
        verts.append(p + origin)
        faces.append(tri + k * side * side)
        oids.append(np.full(side * side, k))
    return (np.concatenate(verts).astype(np.float32),
            np.concatenate(faces).astype(np.int64),
            np.concatenate(oids).astype(np.int64))


def _face_set(vertices, faces):
    """Faces as tuples of coordinate tuples -- winding included, row
    order and vertex numbering not."""
    return {tuple(map(tuple, vertices[f].tolist())) for f in faces}


@pytest.fixture
def patch_store(tmp_path: Path):
    v, f, o = _patch_mesh()
    path = tmp_path / "patches.zv"
    write_mesh(str(path), v, f, chunk_shape=CHUNK, bounds=BOUNDS, object_ids=o)
    return path, v, f, o


class TestMeshObjectRead:

    def test_each_object_round_trips(self, patch_store) -> None:
        path, v, f, o = patch_store
        for oid in np.unique(o):
            got = read_mesh(str(path), object_ids=[int(oid)])
            mask = o == oid
            assert got["vertex_count"] == int(mask.sum())
            assert np.array_equal(np.unique(got["vertices"], axis=0),
                                  np.unique(v[mask], axis=0))
            want_faces = f[mask[f].all(axis=1)]
            assert got["face_count"] == len(want_faces)
            assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want_faces)
            assert np.all(got["object_ids"] == oid)

    def test_store_has_objects_spanning_chunks(self, patch_store) -> None:
        """Guard on the fixture: the cross-chunk path must be exercised."""
        path, v, f, o = patch_store
        chunk = np.floor(v / np.asarray(CHUNK)).astype(int)
        spans = [len(np.unique(chunk[o == k], axis=0)) for k in np.unique(o)]
        assert max(spans) > 1

    def test_several_objects_grouped_in_request_order(self, patch_store) -> None:
        path, v, f, o = patch_store
        got = read_mesh(str(path), object_ids=[7, 2, 7])
        assert got["vertex_count"] == int(((o == 7) | (o == 2)).sum())
        ids = got["object_ids"]
        n7 = int((o == 7).sum())
        assert np.all(ids[:n7] == 7) and np.all(ids[n7:] == 2)
        want = f[((o == 7) | (o == 2))[f].all(axis=1)]
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)

    def test_unknown_id_reads_nothing(self, patch_store) -> None:
        path, *_ = patch_store
        got = read_mesh(str(path), object_ids=[10_000])
        assert got["vertex_count"] == 0 and got["face_count"] == 0
        assert got["faces"].shape == (0, 3)
        assert len(got["object_ids"]) == 0

    def test_object_ids_and_bbox(self, patch_store) -> None:
        path, v, f, o = patch_store
        oid = 4
        mask = o == oid
        lo = v[mask].min(axis=0)
        hi = (lo + v[mask].max(axis=0)) / 2
        box = (lo, hi)
        got = read_mesh(str(path), object_ids=[oid], bbox=box)
        inside = mask & np.all((v >= box[0]) & (v <= box[1]), axis=1)
        assert got["vertex_count"] == int(inside.sum())
        want = f[inside[f].all(axis=1)]
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)
        assert np.all(got["object_ids"] == oid)

    def test_object_ids_and_chunks(self, patch_store) -> None:
        path, v, f, o = patch_store
        chunk = np.floor(v / np.asarray(CHUNK)).astype(int)
        oid = next(k for k in np.unique(o)
                   if len(np.unique(chunk[o == k], axis=0)) > 1)
        first = tuple(int(c) for c in np.unique(chunk[o == oid], axis=0)[0])
        got = read_mesh(str(path), object_ids=[int(oid)], chunks=[first])
        keep = (o == oid) & np.all(chunk == first, axis=1)
        assert got["vertex_count"] == int(keep.sum())
        want = f[keep[f].all(axis=1)]
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)

    def test_draco_store_refuses_object_ids(self, tmp_path: Path) -> None:
        pytest.importorskip("DracoPy")
        v, f, o = _patch_mesh(n_objects=2)
        path = tmp_path / "d.zv"
        write_mesh(str(path), v, f, chunk_shape=CHUNK, bounds=BOUNDS,
                   object_ids=o, encoding="draco")
        with pytest.raises(ArrayError, match="Draco"):
            read_mesh(str(path), object_ids=[0])


class TestMeshScopedFaces:

    def test_chunks_read_matches_filtered_full_read(self, patch_store) -> None:
        path, v, f, o = patch_store
        chunk = np.floor(v / np.asarray(CHUNK)).astype(int)
        cells = [tuple(int(c) for c in cc) for cc in np.unique(chunk, axis=0)[:5]]
        got = read_mesh(str(path), chunks=cells)
        inside = np.zeros(len(v), dtype=bool)
        for cc in cells:
            inside |= np.all(chunk == cc, axis=1)
        want = f[inside[f].all(axis=1)]
        assert got["vertex_count"] == int(inside.sum())
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)

    def test_scoped_read_touches_only_reachable_link_cells(self, patch_store) -> None:
        """The defect this guards: a one-chunk read decoded every face in
        the level, so its cost tracked the store, not the answer."""
        from zarr_vectors.core.arrays import link_cells_within
        from zarr_vectors.core.store import open_store

        path, v, *_ = patch_store
        level = open_store(str(path))["0"]
        chunk = np.floor(v / np.asarray(CHUNK)).astype(int)
        one = tuple(int(c) for c in chunk[0])
        cells = link_cells_within(level, [one])
        # One chunk can reach only its own all-zero-offsets cell.
        assert list(cells.values()) == [[".".join(map(str, one))]]
        assert all(set(seg.replace("_", ".").split(".")) <= {"0"} for seg in cells)

    def test_link_cells_within_includes_cross_cells_inside(self, patch_store) -> None:
        from zarr_vectors.core.arrays import link_cells_within
        from zarr_vectors.core.store import open_store

        path, v, *_ = patch_store
        level = open_store(str(path))["0"]
        chunk = np.floor(v / np.asarray(CHUNK)).astype(int)
        everything = link_cells_within(
            level, [tuple(c) for c in np.unique(chunk, axis=0).tolist()],
        )
        assert any(set(seg.replace("_", ".").split(".")) - {"0"} for seg in everything)

    def test_bbox_read_faces_unchanged(self, patch_store) -> None:
        path, v, f, o = patch_store
        box = (np.array([30.0, 30.0, 30.0]), np.array([140.0, 140.0, 140.0]))
        got = read_mesh(str(path), bbox=box)
        inside = np.all((v >= box[0]) & (v <= box[1]), axis=1)
        want = f[inside[f].all(axis=1)]
        assert got["vertex_count"] == int(inside.sum())
        assert _face_set(got["vertices"], got["faces"]) == _face_set(v, want)
