"""Tests for the vertex-edit surface.

Covers:

- Free-function ``edit_vertex(atomic=False)`` does in-place overwrite.
- Free-function ``edit_vertex(atomic=True)`` appends a new fragment
  and rewrites referring object manifests under new OIDs while
  leaving the old OID's manifest intact (non-destructive).
- ``EditSession`` coalesces multiple in-chunk edits into one chunk RMW.
- Chunk-cross move under both atomic and minimal mode; verifies the
  source-row retention rule.
- ``EditReport`` exposes touched chunks, OID remap, and edit count.
"""

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
    VertexRef,
    add_vertex,
    edit_vertex,
    remove_vertex,
)
from zarr_vectors.types.points import write_points


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def single_chunk_store(tmp_path: Path) -> tuple[str, np.ndarray]:
    """Build a 1-chunk store with a deterministic 8-point cloud.

    ``object_ids`` is supplied explicitly so ``write_points`` emits an
    object_index, giving the edit tests a manifest to inspect.
    """
    path = tmp_path / "store.zv"
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, size=(8, 3)).astype(np.float32)
    write_points(
        str(path),
        positions,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(len(positions), dtype=np.int64),
    )
    return str(path), positions


@pytest.fixture
def multi_chunk_store(tmp_path: Path) -> tuple[str, np.ndarray]:
    """Store with 4 chunks (50-unit chunks, 0..100 in each axis)."""
    path = tmp_path / "store.zv"
    positions = np.array([
        [10.0, 10.0, 10.0],   # chunk (0,0,0)
        [20.0, 20.0, 20.0],   # chunk (0,0,0)
        [70.0, 70.0, 70.0],   # chunk (1,1,1)
        [80.0, 80.0, 80.0],   # chunk (1,1,1)
    ], dtype=np.float32)
    write_points(
        str(path),
        positions,
        chunk_shape=(50.0, 50.0, 50.0),
        bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
        object_ids=np.arange(len(positions), dtype=np.int64),
    )
    return str(path), positions


# ---------------------------------------------------------------------
# Free-function: edit_vertex (atomic=False, in-chunk)
# ---------------------------------------------------------------------

class TestEditVertexMinimal:

    def test_overwrites_row_in_place(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, original = single_chunk_store
        root = open_store(path, mode="r+")

        # write_points with object_ids creates 1 fragment per object.
        # Edit object 3's vertex via from_object.
        ref = VertexRef.from_object(
            root, level=0, object_id=3, vertex_index=0,
        )
        n_frags_before = len(
            read_chunk_vertices(
                root["0"], ref.chunk, dtype=np.float32, ndim=3,
            )
        )
        report = edit_vertex(
            root, ref,
            new_pos=[1.0, 2.0, 3.0],
            atomic=False,
        )

        after = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        # atomic=False must not change the fragment count.
        assert len(after) == n_frags_before
        np.testing.assert_allclose(
            after[ref.fragment][ref.local], [1.0, 2.0, 3.0], atol=1e-5,
        )
        # Object 0's row untouched (different fragment).
        ref0 = VertexRef.from_object(
            root, level=0, object_id=0, vertex_index=0,
        )
        np.testing.assert_allclose(
            after[ref0.fragment][ref0.local], original[0], atol=1e-5,
        )
        # Report is well-formed.
        assert report.n_edits == 1
        assert (0, tuple(ref.chunk)) in report.touched_chunks
        assert report.oid_remap == {}

    def test_attribute_edit_only(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, _ = single_chunk_store
        root = open_store(path, mode="r+")
        ref = VertexRef.from_object(
            root, level=0, object_id=0, vertex_index=0,
        )
        # No new_pos, no new_attrs — should be a noop.
        report = edit_vertex(root, ref, atomic=False)
        assert report.n_edits == 0


# ---------------------------------------------------------------------
# Free-function: edit_vertex (atomic=True, in-chunk)
# ---------------------------------------------------------------------

class TestEditVertexAtomic:

    def test_appends_fragment_keeps_old_data(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, original = single_chunk_store
        root = open_store(path, mode="r+")
        ref = VertexRef.from_object(
            root, level=0, object_id=2, vertex_index=0,
        )
        before = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        n_frags_before = len(before)

        report = edit_vertex(
            root, ref,
            new_pos=[42.0, 42.0, 42.0],
            atomic=True,
        )

        after = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        # Atomic: one new fragment was appended.
        assert len(after) == n_frags_before + 1
        assert after[-1].shape == (1, 3)
        np.testing.assert_allclose(after[-1][0], [42.0, 42.0, 42.0], atol=1e-5)
        # Old fragment is BYTE-for-BYTE unchanged.
        np.testing.assert_allclose(
            after[ref.fragment], before[ref.fragment], atol=1e-5,
        )

        assert report.n_edits == 1

    def test_atomic_invariant_old_oid_unchanged(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        """After atomic edit, reading via old OID returns the pre-edit
        position; the new OID returns the new one.
        """
        path, original = single_chunk_store
        root = open_store(path, mode="r+")

        # write_points emits one OID per input point, each pointing at
        # one row.  Edit object 2's vertex 0.
        old_oid = 2
        manifest_before = read_object_manifest(root["0"], old_oid)
        assert len(manifest_before) == 1
        cc, frag = manifest_before[0]

        # Locate the vertex via from_object.
        ref = VertexRef.from_object(
            root, level=0, object_id=old_oid, vertex_index=0,
        )
        report = edit_vertex(
            root, ref,
            new_pos=[99.0, 99.0, 99.0],
            atomic=True,
        )

        # OID remap recorded.
        assert old_oid in report.oid_remap, (
            f"oid_remap missing entry for {old_oid}: {report.oid_remap}"
        )
        new_oid = report.oid_remap[old_oid]
        assert new_oid != old_oid

        # Old OID still resolves to the original vertex.
        verts_old = read_object_vertices(
            root["0"], old_oid, dtype=np.float32, ndim=3,
        )
        flat_old = np.concatenate(verts_old)
        np.testing.assert_allclose(flat_old[0], original[old_oid], atol=1e-5)

        # New OID resolves to the edited vertex.
        verts_new = read_object_vertices(
            root["0"], new_oid, dtype=np.float32, ndim=3,
        )
        flat_new = np.concatenate(verts_new)
        np.testing.assert_allclose(flat_new[0], [99.0, 99.0, 99.0], atol=1e-5)


# ---------------------------------------------------------------------
# EditSession coalescing
# ---------------------------------------------------------------------

class TestEditSession:

    def test_multiple_edits_one_chunk_rmw(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        """Editing two rows of the same chunk in one session must end
        with both edits applied (single coalesced RMW)."""
        path, _ = single_chunk_store
        root = open_store(path, mode="r+")

        ref0 = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        ref1 = VertexRef.from_object(root, level=0, object_id=1, vertex_index=0)
        with EditSession(root, atomic=False) as ed:
            ed.edit_vertex(ref0, new_pos=[1.0, 1.0, 1.0])
            ed.edit_vertex(ref1, new_pos=[2.0, 2.0, 2.0])

        after = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            after[ref0.fragment][ref0.local], [1.0, 1.0, 1.0], atol=1e-5,
        )
        np.testing.assert_allclose(
            after[ref1.fragment][ref1.local], [2.0, 2.0, 2.0], atol=1e-5,
        )
        assert ed.report.n_edits == 2
        # Only one chunk touched.
        unique = set(tuple(cc) for _, cc in ed.report.touched_chunks)
        assert unique == {(0, 0, 0)}

    def test_invalid_refresh_pyramid_rejected(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        from zarr_vectors.exceptions import EditError
        path, _ = single_chunk_store
        root = open_store(path, mode="r+")
        with pytest.raises(EditError):
            EditSession(root, refresh_pyramid="every")  # type: ignore[arg-type]

    def test_change_set_snapshot(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, _ = single_chunk_store
        root = open_store(path, mode="r+")
        ref = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        with EditSession(root, atomic=False) as ed:
            ed.edit_vertex(ref, new_pos=[5.0, 5.0, 5.0])
            snap = ed.change_set()
            assert snap["atomic"] is False
            assert any(
                d["chunk"] == [0, 0, 0] for d in snap["dirty_chunks"]
            ), snap


# ---------------------------------------------------------------------
# Chunk-cross move
# ---------------------------------------------------------------------

class TestChunkCrossMove:

    def test_minimal_single_object_moves_row(
        self, multi_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        """atomic=False, lone object: source row deleted, target gains
        a new fragment."""
        path, original = multi_chunk_store
        root = open_store(path, mode="r+")

        # Object 0 lives at (0,0,0); move it to (1,1,1).
        ref = VertexRef.from_object(
            root, level=0, object_id=0, vertex_index=0,
        )
        new_pos = [85.0, 85.0, 85.0]  # in chunk (1,1,1)
        report = edit_vertex(root, ref, new_pos=new_pos, atomic=False)

        # Source chunk loses the row.
        src = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        flat_src = np.concatenate(src) if src else np.empty((0, 3))
        # Object 0 was at [10, 10, 10] — should no longer appear.
        assert not np.any(
            np.all(np.isclose(flat_src, [10.0, 10.0, 10.0]), axis=1)
        ), f"source chunk still has the old row: {flat_src}"

        # Target chunk has a new fragment containing the new pos.
        dst = read_chunk_vertices(
            root["0"], (1, 1, 1), dtype=np.float32, ndim=3,
        )
        flat_dst = np.concatenate(dst)
        assert np.any(
            np.all(np.isclose(flat_dst, new_pos), axis=1)
        ), f"target chunk missing new row: {flat_dst}"

        # The propagated object resolves to the new position.
        verts = read_object_vertices(
            root["0"], 0, dtype=np.float32, ndim=3,
        )
        flat = np.concatenate(verts)
        np.testing.assert_allclose(flat[0], new_pos, atol=1e-5)
        assert report.n_edits == 1

    def test_atomic_carry_source_attrs_to_target_chunk(
        self, tmp_path: Path,
    ) -> None:
        """Regression test for the Iteration-2 parallel-attribute fix.

        When a vertex with a non-default per-vertex attribute is moved
        atomically to a new chunk, the new fragment at the target
        chunk must carry the same attribute value (not zero).
        """
        path = tmp_path / "store.zv"
        positions = np.array(
            [[10.0, 10.0, 10.0], [60.0, 60.0, 60.0]], dtype=np.float32,
        )
        intensity = np.array([3.14, 2.71], dtype=np.float32)
        write_points(
            str(path), positions,
            chunk_shape=(50.0, 50.0, 50.0),
            bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
            object_ids=np.arange(2, dtype=np.int64),
            vertex_attributes={"intensity": intensity},
        )
        root = open_store(str(path), mode="r+")

        ref = VertexRef.from_object(
            root, level=0, object_id=0, vertex_index=0,
        )
        # Atomic move from (0,0,0) into (1,1,1).
        new_pos = [85.0, 85.0, 85.0]
        report = edit_vertex(root, ref, new_pos=new_pos, atomic=True)
        new_oid = report.oid_remap[0]

        # The new OID's vertex still carries intensity=3.14.
        from zarr_vectors.core.arrays import read_object_manifest, read_chunk_attributes
        manifest = read_object_manifest(root["0"], new_oid)
        assert len(manifest) == 1
        new_cc, new_frag = manifest[0]
        groups = read_chunk_attributes(
            root["0"], "intensity", new_cc, dtype=np.float32,
        )
        np.testing.assert_allclose(
            groups[new_frag][0], 3.14, atol=1e-5,
            err_msg=(
                "atomic chunk-cross: target chunk's fragment must carry "
                "the source row's attribute value, not zero"
            ),
        )

    def test_atomic_keeps_source_row(
        self, multi_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        """atomic=True: source row stays so old OID is still readable."""
        path, original = multi_chunk_store
        root = open_store(path, mode="r+")

        old_oid = 0
        ref = VertexRef.from_object(
            root, level=0, object_id=old_oid, vertex_index=0,
        )
        new_pos = [85.0, 85.0, 85.0]
        report = edit_vertex(root, ref, new_pos=new_pos, atomic=True)

        # Source chunk STILL contains the original row.
        src = read_chunk_vertices(
            root["0"], (0, 0, 0), dtype=np.float32, ndim=3,
        )
        flat_src = np.concatenate(src)
        assert np.any(
            np.all(np.isclose(flat_src, original[old_oid]), axis=1)
        ), f"atomic must keep source row, missing in {flat_src}"

        # New OID resolves to new position.
        new_oid = report.oid_remap[old_oid]
        verts_new = read_object_vertices(
            root["0"], new_oid, dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            np.concatenate(verts_new)[0], new_pos, atol=1e-5,
        )
        # Old OID still resolves to the original.
        verts_old = read_object_vertices(
            root["0"], old_oid, dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            np.concatenate(verts_old)[0], original[old_oid], atol=1e-5,
        )


# ---------------------------------------------------------------------
# add_vertex / remove_vertex
# ---------------------------------------------------------------------

class TestAddRemove:

    def test_add_vertex_creates_fragment(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, original = single_chunk_store
        root = open_store(path, mode="r+")
        ref, report = add_vertex(
            root, level=0, pos=[150.0, 150.0, 150.0],
        )
        assert ref.level == 0
        assert ref.local == 0
        # Read back: should find the new fragment in chunk (0,0,0).
        chunk = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        assert chunk[ref.fragment].shape == (1, 3)
        np.testing.assert_allclose(
            chunk[ref.fragment][0], [150.0, 150.0, 150.0], atol=1e-5,
        )
        assert report.n_edits == 1

    def test_remove_vertex_atomic_softdelete(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, original = single_chunk_store
        root = open_store(path, mode="r+")
        # Soft-delete object 3.
        ref = VertexRef.from_object(
            root, level=0, object_id=3, vertex_index=0,
        )
        report = remove_vertex(root, ref, atomic=True)
        new_oid = report.oid_remap.get(3)
        assert new_oid is not None
        # The new OID's manifest is empty.
        manifest = read_object_manifest(root["0"], new_oid)
        assert manifest == []

    def test_remove_vertex_minimal_rejected(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        from zarr_vectors.exceptions import EditError
        path, _ = single_chunk_store
        root = open_store(path, mode="r+")
        ref = VertexRef.from_object(root, level=0, object_id=0, vertex_index=0)
        with pytest.raises(EditError):
            remove_vertex(root, ref, atomic=False)


# ---------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------

class TestVertexRef:

    def test_from_object(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, original = single_chunk_store
        root = open_store(path, mode="r+")
        ref = VertexRef.from_object(
            root, level=0, object_id=5, vertex_index=0,
        )
        assert ref.level == 0
        # Verify the resolved physical address yields original[5].
        chunk = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            chunk[ref.fragment][ref.local], original[5], atol=1e-5,
        )

    def test_from_position_nearest(
        self, single_chunk_store: tuple[str, np.ndarray],
    ) -> None:
        path, original = single_chunk_store
        root = open_store(path, mode="r+")
        # Pick a point near original[0] and require resolve.
        target = original[0]
        ref = VertexRef.from_position(
            root, level=0, pos=target + 1e-3,
        )
        chunk = read_chunk_vertices(
            root["0"], ref.chunk, dtype=np.float32, ndim=3,
        )
        np.testing.assert_allclose(
            chunk[ref.fragment][ref.local], target, atol=1e-5,
        )
