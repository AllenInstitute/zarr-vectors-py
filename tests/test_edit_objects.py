"""Tests for object-level edits + free-function wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_chunk_vertices,
    read_object_manifest,
    read_object_vertices,
)
from zarr_vectors.core.store import open_store
from zarr_vectors.ops import (
    EditSession,
    ObjectRef,
    add_object,
    edit_object,
    remove_object,
)
from zarr_vectors.types.points import write_points


@pytest.fixture
def small_store(tmp_path: Path) -> str:
    path = tmp_path / "store.zv"
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, size=(6, 3)).astype(np.float32)
    write_points(
        str(path), positions,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(6, dtype=np.int64),
    )
    return str(path)


class TestAddObject:

    def test_add_object_creates_new_oid(self, small_store: str) -> None:
        root = open_store(small_store, mode="r+")
        new_vertices = np.array(
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
            dtype=np.float32,
        )
        ref, report = add_object(root, level=0, vertices=new_vertices)
        assert isinstance(ref, ObjectRef)
        assert ref.level == 0
        assert ref.object_id >= 6  # past the baseline OIDs
        # The new OID's manifest should resolve to the right vertices.
        verts = read_object_vertices(
            root["0"], ref.object_id, dtype=np.float32, ndim=3,
        )
        flat = np.concatenate(verts)
        np.testing.assert_allclose(flat, new_vertices, atol=1e-5)
        assert report.n_edits == 1

    def test_add_object_multichunk(self, tmp_path: Path) -> None:
        # Multi-chunk add: vertices span two chunks.
        path = tmp_path / "store.zv"
        seed_positions = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        write_points(
            str(path), seed_positions,
            chunk_shape=(50.0, 50.0, 50.0),
            bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            object_ids=np.array([0], dtype=np.int64),
        )
        root = open_store(str(path), mode="r+")
        vs = np.array(
            [[10.0, 10.0, 10.0], [70.0, 70.0, 70.0]], dtype=np.float32,
        )
        ref, report = add_object(root, level=0, vertices=vs)
        manifest = read_object_manifest(root["0"], ref.object_id)
        # Manifest has two entries, one per chunk.
        chunks = {tuple(cc) for cc, _ in manifest}
        assert chunks == {(0, 0, 0), (1, 1, 1)}


class TestEditObject:

    def test_edit_object_atomic_remap(self, small_store: str) -> None:
        root = open_store(small_store, mode="r+")
        old = read_object_manifest(root["0"], 0)
        new_manifest = old  # noop manifest, just to exercise the remap path
        report = edit_object(
            root, 0, level=0, new_manifest=new_manifest, atomic=True,
        )
        assert 0 in report.oid_remap
        new_oid = report.oid_remap[0]
        # Old OID still has the original manifest.
        assert read_object_manifest(root["0"], 0) == old
        # New OID is the same manifest (deep equal after tuple normalisation).
        assert read_object_manifest(root["0"], new_oid) == old

    def test_edit_object_minimal_overwrites(self, small_store: str) -> None:
        root = open_store(small_store, mode="r+")
        # Reassign object 0 to point at object 1's fragment.
        m1 = read_object_manifest(root["0"], 1)
        report = edit_object(
            root, 0, level=0, new_manifest=m1, atomic=False,
        )
        assert report.oid_remap == {}
        assert read_object_manifest(root["0"], 0) == m1


class TestRemoveObject:

    def test_remove_object_atomic_softdelete(self, small_store: str) -> None:
        root = open_store(small_store, mode="r+")
        report = remove_object(root, 0, level=0, atomic=True)
        new_oid = report.oid_remap[0]
        # New OID has an empty manifest; original OID still resolves
        # to its pre-edit manifest (atomic guarantee).
        assert read_object_manifest(root["0"], new_oid) == []
        original = read_object_manifest(root["0"], 0)
        assert original  # non-empty

    def test_remove_object_minimal_overwrites(self, small_store: str) -> None:
        root = open_store(small_store, mode="r+")
        report = remove_object(root, 3, level=0, atomic=False)
        assert report.oid_remap == {}
        assert read_object_manifest(root["0"], 3) == []


class TestSessionMultiObject:

    def test_add_and_remove_in_one_session(self, small_store: str) -> None:
        root = open_store(small_store, mode="r+")
        added_ids: list[int] = []
        with EditSession(root, atomic=True) as ed:
            ref = ed.add_object(
                level=0, vertices=np.array([[50.0, 50.0, 50.0]], dtype=np.float32),
            )
            added_ids.append(ref.object_id)
            ed.remove_object(2, level=0)
        # The added object resolves; the removed object's new OID has
        # an empty manifest.
        new_verts = read_object_vertices(
            root["0"], added_ids[0], dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            np.concatenate(new_verts), [[50.0, 50.0, 50.0]], atol=1e-5,
        )
        remap = ed.report.oid_remap
        assert 2 in remap
        assert read_object_manifest(root["0"], remap[2]) == []
