"""Face ownership must partition geometry without reading unrelated cells."""

from contextlib import contextmanager

import numpy as np
import pytest

from zarr_vectors.building import MeshReader, write_mesh
from zarr_vectors.core.group import Group


@pytest.fixture
def store(tmp_path):
    # One boundary triangle, reversed winding, and one far-away triangle.
    vertices = np.array(
        [[1, 1, 1], [11, 1, 1], [1, 11, 1], [101, 1, 1], [102, 1, 1], [101, 2, 1]], dtype=np.float32
    )
    faces = np.array([[2, 1, 0], [3, 4, 5]])
    path = tmp_path / "surface.zarrvectors"
    write_mesh(path, vertices, faces, chunk_shape=(10.0, 10.0, 10.0))
    return path, vertices, faces


def test_owned_boundary_face_is_complete_and_winding_preserved(store):
    path, vertices, faces = store
    reader = MeshReader(path)
    result = reader.read([(0, 0, 0)])
    assert result["face_count"] == 1
    np.testing.assert_array_equal(result["vertices"][result["faces"]], vertices[faces[:1]])
    chunk = next(c for c in reader.chunks if c.key == (0, 0, 0))
    assert chunk.bounds == ((0.0, 0.0, 0.0), (20.0, 20.0, 10.0))


def test_disjoint_reads_partition_faces(store):
    path, vertices, faces = store
    reader = MeshReader(path)
    triangles = []
    for chunk in reader.chunks:
        result = reader.read([chunk.key])
        triangles.extend(result["vertices"][result["faces"]])
    np.testing.assert_array_equal(np.asarray(triangles), vertices[faces])
    assert reader.read([])["face_count"] == 0
    with pytest.raises(KeyError):
        reader.read([(99, 0, 0)])


def test_selected_read_never_requests_far_payload(store, monkeypatch):
    path, _, _ = store
    reader = MeshReader(path)
    requested = []
    original_batch = Group.batched_reads
    original_read = Group.read_bytes

    @contextmanager
    def batch(self, plan):
        plan = list(plan)
        requested.extend((name, key) for name, keys in plan for key in keys)
        with original_batch(self, plan):
            yield

    def read(self, name, key):
        requested.append((name, key))
        return original_read(self, name, key)

    monkeypatch.setattr(Group, "batched_reads", batch)
    monkeypatch.setattr(Group, "read_bytes", read)
    reader.read([(0, 0, 0)])
    assert requested
    assert not any(key == "10.0.0" for _, key in requested)
    assert any(name == "vertices" and key == "1.0.0" for name, key in requested)
    assert all(key == "0.0.0" for name, key in requested if name.startswith("links/"))


def test_missing_dependency_is_a_failure(store, monkeypatch):
    path, _, _ = store
    reader = MeshReader(path)
    from zarr_vectors.exceptions import ArrayError

    original = Group.read_bytes

    def read(group, name, key):
        return b"" if name == "vertices" and key == "1.0.0" else original(group, name, key)

    monkeypatch.setattr(Group, "read_bytes", read)
    with pytest.raises(ArrayError, match="missing vertex chunk"):
        reader.read([(0, 0, 0)])


def test_budget_rejected_before_vertex_payload(store, monkeypatch):
    path, _, _ = store
    reader = MeshReader(path)
    original = Group.read_bytes
    vertex_reads = []

    def read(group, name, key):
        if name == "vertices":
            vertex_reads.append(key)
        return original(group, name, key)

    monkeypatch.setattr(Group, "read_bytes", read)
    with pytest.raises(MemoryError, match="limit"):
        reader.read([(0, 0, 0)], max_bytes=1)
    assert vertex_reads == []


def test_draco_is_explicitly_rejected(tmp_path):
    pytest.importorskip("DracoPy")
    path = tmp_path / "draco.zv"
    write_mesh(
        path,
        np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float32),
        np.array([[0, 1, 2]]),
        chunk_shape=(10.0, 10.0, 10.0),
        encoding="draco",
    )
    with pytest.raises(NotImplementedError, match="Draco"):
        MeshReader(path)


def test_metadata_snapshot_is_reused_but_geometry_is_read_again(store):
    from zarr.storage import LocalStore, WrapperStore

    class Counter(WrapperStore):
        def __init__(self, path):
            super().__init__(LocalStore(path, read_only=True))
            self.metadata_reads = 0
            self.payload_reads = 0

        async def get(self, key, prototype, byte_range=None):
            result = await self._store.get(key, prototype, byte_range)
            if result is not None:
                if key.endswith("zarr.json"):
                    self.metadata_reads += 1
                else:
                    self.payload_reads += 1
            return result

    counter = Counter(store[0])
    with MeshReader(counter) as reader:
        first = reader.read([(0, 0, 0)])
        metadata_before, payload_before = counter.metadata_reads, counter.payload_reads
        second = reader.read([(0, 0, 0)])
        assert counter.metadata_reads == metadata_before
        assert counter.payload_reads > payload_before
        np.testing.assert_array_equal(first["vertices"], second["vertices"])
    with pytest.raises(RuntimeError, match="closed"):
        reader.read([(0, 0, 0)])


def test_budget_boundary_includes_all_dependency_vertices_and_faces(store):
    # Three chunks, each holding one float32 XYZ vertex, plus one int64 face.
    with MeshReader(store[0]) as reader:
        result = reader.read([(0, 0, 0)], max_bytes=3 * 3 * 4 + 3 * 8)
        assert result["face_count"] == 1
        with pytest.raises(MemoryError, match="limit"):
            reader.read([(0, 0, 0)], max_bytes=3 * 3 * 4 + 3 * 8 - 1)


def test_failed_initialization_exits_metadata_scope(store, monkeypatch):
    active = 0
    cached_nodes = Group.cached_nodes
    read_meta = Group.read_array_meta

    @contextmanager
    def track_scope(group):
        nonlocal active
        with cached_nodes(group):
            active += 1
            try:
                yield
            finally:
                active -= 1

    def fail_vertices(group, name):
        if name == "vertices":
            raise RuntimeError("fixture metadata failure")
        return read_meta(group, name)

    monkeypatch.setattr(Group, "cached_nodes", track_scope)
    monkeypatch.setattr(Group, "read_array_meta", fail_vertices)
    with pytest.raises(RuntimeError, match="fixture metadata failure"):
        MeshReader(store[0])
    assert active == 0
