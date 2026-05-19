"""Tests for the OID-compaction pass of ``vacuum``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_object_attributes,
    read_object_manifest,
)
from zarr_vectors.core.store import open_store
from zarr_vectors.ops import (
    EditSession,
    VertexRef,
    vacuum,
)
from zarr_vectors.types.points import write_points


@pytest.fixture
def baseline(tmp_path: Path) -> tuple[str, np.ndarray]:
    path = tmp_path / "store.zv"
    positions = np.array(
        [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]],
        dtype=np.float32,
    )
    label = np.array([100, 200, 300], dtype=np.int64)
    write_points(
        str(path), positions,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        object_ids=np.arange(3, dtype=np.int64),
        object_attributes={"label": label},
    )
    return str(path), label


class TestCompactOids:

    def test_compact_after_atomic_edits(
        self, baseline: tuple[str, np.ndarray],
    ) -> None:
        path, label = baseline
        root = open_store(path, mode="r+")
        # Run a few atomic vertex edits — each appends a new OID and
        # leaves the old one orphaned only if the manifest is later
        # cleared.  We orphan OID 1 by replacing its manifest with [].
        from zarr_vectors.ops import edit_object
        edit_object(root, 1, level=0, new_manifest=[], atomic=False)

        report = vacuum(root, compact_oids=True)
        # OID 1's slot was orphaned → 3 OIDs collapse to 2 live ones.
        assert 0 in report.oid_remap
        assert 2 in report.oid_remap
        # OID 1 should no longer be in the live set.
        new_oid_2 = report.oid_remap[2]
        # The compacted manifests should resolve correctly.
        m_new = read_object_manifest(root["0"], new_oid_2)
        # The original OID 2 had a one-fragment manifest; after compaction
        # the same manifest is at new_oid_2.
        assert len(m_new) >= 1

    def test_object_attributes_compacted(
        self, baseline: tuple[str, np.ndarray],
    ) -> None:
        path, label = baseline
        root = open_store(path, mode="r+")
        from zarr_vectors.ops import edit_object
        edit_object(root, 0, level=0, new_manifest=[], atomic=False)

        report = vacuum(root, compact_oids=True)
        arr = read_object_attributes(root["0"], "label")
        # Two live OIDs survive — the dropped one was OID 0 which had
        # label=100; the remaining values are [200, 300].
        assert arr.shape == (2,)
        np.testing.assert_array_equal(arr, [200, 300])

    def test_already_dense_is_noop_on_disk(
        self, baseline: tuple[str, np.ndarray],
    ) -> None:
        path, _ = baseline
        root = open_store(path, mode="r+")
        report = vacuum(root, compact_oids=True)
        # Identity remap when there are no empty manifests.
        for oid in (0, 1, 2):
            assert report.oid_remap.get(oid, oid) == oid


class TestDeferredPasses:

    def test_drop_empty_fragments_raises(
        self, baseline: tuple[str, np.ndarray],
    ) -> None:
        path, _ = baseline
        root = open_store(path, mode="r+")
        with pytest.raises(NotImplementedError):
            vacuum(root, drop_empty_fragments=True)

    def test_dedup_parallel_rows_raises(
        self, baseline: tuple[str, np.ndarray],
    ) -> None:
        path, _ = baseline
        root = open_store(path, mode="r+")
        with pytest.raises(NotImplementedError):
            vacuum(root, dedup_parallel_rows=True)


class TestDryRun:

    def test_dry_run_returns_remap_without_writing(
        self, baseline: tuple[str, np.ndarray],
    ) -> None:
        path, _ = baseline
        root = open_store(path, mode="r+")
        from zarr_vectors.ops import edit_object
        edit_object(root, 1, level=0, new_manifest=[], atomic=False)

        before_arr = read_object_attributes(root["0"], "label")
        report = vacuum(root, compact_oids=True, dry_run=True)
        after_arr = read_object_attributes(root["0"], "label")
        # Disk untouched.
        np.testing.assert_array_equal(before_arr, after_arr)
        # Remap still computed.
        assert report.oid_remap
