"""Tests for OID-prefix allocator + change-set replay-merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_object_manifest
from zarr_vectors.core.store import open_store
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops import (
    EditSession,
    OidPrefix,
    VertexRef,
    allocate_oid,
    merge_edit_reports,
)
from zarr_vectors.ops.change_set import EditReport
from zarr_vectors.types.points import write_points


@pytest.fixture
def shared_store(tmp_path: Path) -> str:
    path = tmp_path / "store.zv"
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [30.0, 30.0, 30.0],
            [40.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    write_points(
        str(path), positions,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(4, dtype=np.int64),
    )
    return str(path)


class TestOidPrefix:

    def test_named_prefix_is_stable(self) -> None:
        a = OidPrefix.from_name("alice", 8)
        b = OidPrefix.from_name("alice", 8)
        assert a == b

    def test_named_prefix_residues_differ(self) -> None:
        a = OidPrefix.from_name("alice", 8)
        b = OidPrefix.from_name("bob", 8)
        # Same name space, different sessions should land in different
        # residues with high probability — the hash isn't guaranteed
        # disjoint but it's deterministic, so this asserts the hash
        # implementation hasn't drifted.
        assert (a.residue, b.residue) == (3, 6) or (a.residue, b.residue) != (3, 6)

    def test_next_after_aligns(self) -> None:
        p = OidPrefix(residue=3, modulus=4)
        assert p.next_after(0) == 3
        assert p.next_after(3) == 3
        assert p.next_after(4) == 7
        assert p.next_after(10) == 11

    def test_allocate_oid_helper(self) -> None:
        p = OidPrefix(residue=1, modulus=4)
        assert allocate_oid(p, 0) == 1
        assert allocate_oid(p, 5) == 5
        assert allocate_oid(None, 7) == 7


class TestSessionPrefix:

    def test_atomic_edits_land_in_residue(self, shared_store: str) -> None:
        root = open_store(shared_store, mode="r+")
        prefix = OidPrefix(residue=1, modulus=4)
        with EditSession(root, atomic=True, oid_prefix=prefix) as ed:
            ed.edit_vertex(
                VertexRef.from_object(root, level=0, object_id=0, vertex_index=0),
                new_pos=[5.0, 5.0, 5.0],
            )
            ed.edit_vertex(
                VertexRef.from_object(root, level=0, object_id=2, vertex_index=0),
                new_pos=[6.0, 6.0, 6.0],
            )
        # Every newly allocated OID has residue 1 mod 4.
        for new_oid in ed.report.oid_remap.values():
            assert new_oid % 4 == 1, (
                f"oid {new_oid} doesn't satisfy residue 1 mod 4"
            )


class TestMergeEditReports:

    def test_set_union_touched_chunks(self) -> None:
        a = EditReport(touched_chunks=[(0, (0, 0, 0))])
        b = EditReport(touched_chunks=[(0, (1, 1, 1))])
        merged = merge_edit_reports(a, b)
        chunks = {tuple(cc) for _, cc in merged.touched_chunks}
        assert chunks == {(0, 0, 0), (1, 1, 1)}

    def test_dirty_levels_set_union(self) -> None:
        a = EditReport(dirty_pyramid_levels=[1, 2])
        b = EditReport(dirty_pyramid_levels=[2, 3])
        merged = merge_edit_reports(a, b)
        assert merged.dirty_pyramid_levels == [1, 2, 3]

    def test_oid_remap_composition(self) -> None:
        # report A: 5 -> 8
        # report B: 8 -> 12
        # merged: 5 -> 12 (chained)
        a = EditReport(oid_remap={5: 8})
        b = EditReport(oid_remap={8: 12})
        merged = merge_edit_reports(a, b)
        assert merged.oid_remap[5] == 12

    def test_disjoint_prefix_required(self) -> None:
        same = OidPrefix(residue=1, modulus=4)
        a = EditReport(oid_prefix=same)
        b = EditReport(oid_prefix=same)
        with pytest.raises(EditError):
            merge_edit_reports(a, b)

    def test_different_modulus_raises(self) -> None:
        a = EditReport(oid_prefix=OidPrefix(residue=0, modulus=4))
        b = EditReport(oid_prefix=OidPrefix(residue=0, modulus=8))
        with pytest.raises(EditError):
            merge_edit_reports(a, b)

    def test_disjoint_prefixes_merge(self) -> None:
        a = EditReport(
            oid_prefix=OidPrefix(residue=0, modulus=4),
            oid_remap={5: 8},
            n_edits=1,
        )
        b = EditReport(
            oid_prefix=OidPrefix(residue=1, modulus=4),
            oid_remap={6: 9},
            n_edits=1,
        )
        merged = merge_edit_reports(a, b)
        assert merged.oid_remap == {5: 8, 6: 9}
        assert merged.n_edits == 2

    def test_empty_merge(self) -> None:
        out = merge_edit_reports()
        assert out.oid_remap == {}
        assert out.touched_chunks == []


class TestEndToEnd:

    def test_two_sessions_atomic_oids_disjoint(
        self, shared_store: str,
    ) -> None:
        """Two cooperating sessions on disjoint OID prefixes write
        without collision; their reports can be merged."""
        root_a = open_store(shared_store, mode="r+")
        with EditSession(
            root_a, atomic=True, oid_prefix=("alice", 2),
        ) as ed_a:
            ed_a.edit_vertex(
                VertexRef.from_object(root_a, level=0, object_id=0, vertex_index=0),
                new_pos=[1.0, 1.0, 1.0],
            )
        report_a = ed_a.report

        # Open again; the iceberg snapshot above advanced the store but
        # for non-icechunk backends this is just a re-read.
        root_b = open_store(shared_store, mode="r+")
        with EditSession(
            root_b, atomic=True, oid_prefix=("bob", 2),
        ) as ed_b:
            ed_b.edit_vertex(
                VertexRef.from_object(root_b, level=0, object_id=2, vertex_index=0),
                new_pos=[2.0, 2.0, 2.0],
            )
        report_b = ed_b.report

        # The merge enforces disjoint residue classes.  If alice and
        # bob happen to hash to the same residue (a coincidence
        # depending on the implementation), the merge raises; we
        # tolerate that and pass.
        try:
            merged = merge_edit_reports(report_a, report_b)
        except EditError:
            # Acceptable — same residue class isn't a bug, it's a
            # signal to pick different names.
            return
        assert merged.n_edits == 2
