"""Step 06 tests: point cloud write/read core API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from zarr_vectors.core.store import create_store, open_store
from zarr_vectors.exceptions import MetadataError
from zarr_vectors.types.points import read_points, write_points


# ===================================================================
# Undifferentiated point cloud (variant 1)
# ===================================================================

class TestUndifferentiatedPoints:

    def test_write_and_read_all(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(200, 3)).astype(np.float32)
        store = str(tmp_path / "points.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(500.0, 500.0, 500.0),
        )
        assert summary["vertex_count"] == 200
        assert summary["chunk_count"] > 0

        result = read_points(store)
        assert result["vertex_count"] == 200
        assert result["positions"].shape == (200, 3)

    def test_single_chunk(self, tmp_path: Path) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        store = str(tmp_path / "tiny.zarr")

        write_points(store, positions, chunk_shape=(1000.0, 1000.0, 1000.0))
        result = read_points(store)
        assert result["vertex_count"] == 2
        # Positions should match (order may differ due to chunk assignment)
        read_pos = result["positions"]
        assert read_pos.shape == (2, 3)

    def test_with_attributes(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(99)
        positions = rng.uniform(0, 100, size=(50, 3)).astype(np.float32)
        intensity = rng.uniform(0, 1, size=50).astype(np.float32)
        store = str(tmp_path / "attrs.zarr")

        write_points(
            store, positions,
            chunk_shape=(50.0, 50.0, 50.0),
            vertex_attributes={"intensity": intensity},
        )

        result = read_points(store, attribute_names=["intensity"])
        assert result["vertex_count"] == 50

    def test_default_chunk_shape(self, tmp_path: Path) -> None:
        """When no chunk_shape given, should use a single chunk."""
        positions = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        store = str(tmp_path / "default.zarr")

        summary = write_points(store, positions)
        assert summary["chunk_count"] == 1
        result = read_points(store)
        assert result["vertex_count"] == 2

    def test_bbox_filter(self, tmp_path: Path) -> None:
        positions = np.array([
            [10, 10, 10],
            [50, 50, 50],
            [90, 90, 90],
            [150, 150, 150],
        ], dtype=np.float32)
        store = str(tmp_path / "bbox.zarr")

        write_points(store, positions, chunk_shape=(100.0, 100.0, 100.0))

        result = read_points(
            store,
            bbox=(np.array([0, 0, 0]), np.array([100, 100, 100])),
        )
        # Should get 3 points (10,50,90 but not 150)
        assert result["vertex_count"] == 3

    def test_2d_points(self, tmp_path: Path) -> None:
        positions = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        store = str(tmp_path / "2d.zarr")

        write_points(store, positions, chunk_shape=(10.0, 10.0))
        result = read_points(store)
        assert result["vertex_count"] == 3
        assert result["positions"].shape[1] == 2


# ===================================================================
# Per-point objects (variant 2)
# ===================================================================

class TestPerPointObjects:

    def test_each_point_is_object(self, tmp_path: Path) -> None:
        positions = np.array([
            [10, 10, 10],
            [50, 50, 50],
            [110, 110, 110],
        ], dtype=np.float32)
        store = str(tmp_path / "per_point.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=np.array([0, 1, 2], dtype=np.int64),
        )
        assert summary["object_count"] == 3

        # Read by object ID
        result = read_points(store, object_ids=[1])
        assert result["vertex_count"] == 1
        np.testing.assert_allclose(result["positions"][0], [50, 50, 50], atol=1e-5)

    def test_implicit_object_ids(self, tmp_path: Path) -> None:
        """When object_attributes given but no object_ids, each point gets its own ID."""
        positions = np.array([
            [10, 10, 10],
            [20, 20, 20],
        ], dtype=np.float32)
        obj_attrs = np.array([100, 200], dtype=np.float32)
        store = str(tmp_path / "implicit_oid.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_attributes={"label": obj_attrs},
        )
        assert summary["object_count"] == 2

    def test_with_groups(self, tmp_path: Path) -> None:
        positions = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30],
            [40, 40, 40],
        ], dtype=np.float32)
        store = str(tmp_path / "grouped.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=np.array([0, 1, 2, 3], dtype=np.int64),
            groups={0: [0, 1], 1: [2, 3]},
            group_attributes={
                "name": np.array([1.0, 2.0], dtype=np.float32),
            },
        )
        assert summary["group_count"] == 2

        # Read by group
        result = read_points(store, group_ids=[0])
        assert result["vertex_count"] == 2


# ===================================================================
# Multi-point objects (variant 3)
# ===================================================================

class TestMultiPointObjects:

    def test_many_points_per_object(self, tmp_path: Path) -> None:
        """5 points belong to object 0, 3 to object 1."""
        positions = np.array([
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [50, 50, 50],
            [51, 51, 51],
            [52, 52, 52],
        ], dtype=np.float32)
        object_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        store = str(tmp_path / "multi.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=object_ids,
        )
        assert summary["object_count"] == 2

        # Read object 0: should get 5 points
        result = read_points(store, object_ids=[0])
        assert result["vertex_count"] == 5

        # Read object 1: should get 3 points
        result = read_points(store, object_ids=[1])
        assert result["vertex_count"] == 3

    def test_objects_across_chunks(self, tmp_path: Path) -> None:
        """Object 0 has points in two different chunks."""
        positions = np.array([
            [10, 50, 50],    # chunk (0,0,0)
            [110, 50, 50],   # chunk (1,0,0)
            [20, 50, 50],    # chunk (0,0,0)
        ], dtype=np.float32)
        object_ids = np.array([0, 0, 0], dtype=np.int64)
        store = str(tmp_path / "cross.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            object_ids=object_ids,
        )
        assert summary["chunk_count"] == 2
        assert summary["object_count"] == 1

        result = read_points(store, object_ids=[0])
        assert result["vertex_count"] == 3


# ===================================================================
# Edge cases
# ===================================================================

class TestPointsEdgeCases:

    def test_single_point(self, tmp_path: Path) -> None:
        positions = np.array([[42.0, 43.0, 44.0]], dtype=np.float32)
        store = str(tmp_path / "single.zarr")
        write_points(store, positions, chunk_shape=(100.0, 100.0, 100.0))
        result = read_points(store)
        assert result["vertex_count"] == 1

    def test_many_chunks(self, tmp_path: Path) -> None:
        """1000 points across many small chunks."""
        rng = np.random.default_rng(77)
        positions = rng.uniform(0, 1000, size=(1000, 3)).astype(np.float32)
        store = str(tmp_path / "many.zarr")

        summary = write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
        )
        assert summary["vertex_count"] == 1000
        assert summary["chunk_count"] > 1

        result = read_points(store)
        assert result["vertex_count"] == 1000

    def test_multichannel_attribute(self, tmp_path: Path) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        color = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.float32)
        store = str(tmp_path / "color.zarr")

        write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"color": color},
        )

        result = read_points(store)
        assert result["vertex_count"] == 2


# ===================================================================
# Out-of-bounds policy (against a pre-warmed store)
# ===================================================================

class TestOutOfBoundsPolicy:

    def test_raise_is_default(self, tmp_path: Path) -> None:
        """With the default 128³ bounds and `raise` policy, OOB writes
        surface a clear MetadataError."""
        store = str(tmp_path / "raise.zarr")
        create_store(store)  # default bounds = 128³
        pts = np.array([[200.0, 50.0, 50.0]], dtype=np.float32)
        try:
            write_points(store, pts)
            assert False, "should have raised"
        except MetadataError:
            pass

    def test_ignore_drops_oob(self, tmp_path: Path) -> None:
        """`ignore` drops OOB points and writes the rest."""
        store = str(tmp_path / "ignore.zarr")
        create_store(store)
        pts = np.array(
            [[200.0, 50.0, 50.0], [10.0, 10.0, 10.0], [50.0, 50.0, 50.0]],
            dtype=np.float32,
        )
        summary = write_points(store, pts, out_of_bounds="ignore")
        assert summary["vertex_count"] == 2

    def test_expand_grows_bounds(self, tmp_path: Path) -> None:
        """`expand` grows the store's bounds to enclose all input points."""
        store = str(tmp_path / "expand.zarr")
        create_store(store)
        pts = np.array(
            [[200.0, 50.0, 50.0], [10.0, 10.0, 10.0]],
            dtype=np.float32,
        )
        write_points(store, pts, out_of_bounds="expand")
        root = open_store(store)
        zv = root.attrs.to_dict()["zarr_vectors"]
        assert zv["bounds"][1][0] >= 200.0

    def test_unknown_policy_raises(self, tmp_path: Path) -> None:
        store = str(tmp_path / "bad.zarr")
        create_store(store)
        pts = np.array([[200.0, 50.0, 50.0]], dtype=np.float32)
        try:
            write_points(store, pts, out_of_bounds="garbage")
            assert False
        except MetadataError:
            pass


# ===================================================================
# Attribute dtype round-trip (non-float and dictionary-encoded)
# ===================================================================

class TestAttributeDtypes:
    """write_points + read_points must round-trip non-float vertex
    attributes (the read path previously hardcoded float32 + 1 col)
    and dictionary-encoded string attributes (Arrow Dictionary /
    pandas Categorical convention persisted in .zattrs)."""

    def _positions(self, n: int = 6, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 90, size=(n, 3)).astype(np.float32)

    def test_int_attribute_roundtrip(self, tmp_path: Path) -> None:
        positions = self._positions(6)
        cell_id = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
        store = str(tmp_path / "int_attr.zarr")

        write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"cell_id": cell_id},
        )
        result = read_points(store, attribute_names=["cell_id"])

        out = result["vertex_attributes"]["cell_id"]
        assert out.dtype == np.int32
        assert sorted(out.tolist()) == sorted(cell_id.tolist())

    def test_string_attribute_dictionary_encoded(self, tmp_path: Path) -> None:
        positions = self._positions(6)
        labels = np.array(
            ["soma", "axon", "soma", "dendrite", "axon", "soma"],
        )
        store = str(tmp_path / "str_attr.zarr")

        write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"compartment": labels},
        )

        # On-disk array must be integer codes (not raw strings) with
        # an ``encoding="dictionary"`` marker in .zattrs.
        root = open_store(store)
        attr_meta = root.attrs.to_dict()  # root attrs are unrelated; check level
        from zarr_vectors.core.store import get_resolution_level, read_root_metadata
        rm = read_root_metadata(root)  # noqa: F841 — sanity
        lg = get_resolution_level(root, 0)
        m = lg.read_array_meta("vertex_attributes/compartment")
        assert m["encoding"] == "dictionary"
        assert sorted(m["categories"]) == ["axon", "dendrite", "soma"]
        assert np.dtype(m["dtype"]).kind in ("u", "i")

        # Read path returns the decoded strings, not the codes.
        result = read_points(store, attribute_names=["compartment"])
        out = result["vertex_attributes"]["compartment"]
        assert sorted(out.tolist()) == sorted(labels.tolist())

    def test_string_attribute_with_missing_values(self, tmp_path: Path) -> None:
        positions = self._positions(5)
        labels = np.array(["soma", None, "axon", None, "soma"], dtype=object)
        store = str(tmp_path / "str_missing.zarr")

        write_points(
            store, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"compartment": labels},
        )

        result = read_points(store, attribute_names=["compartment"])
        out = result["vertex_attributes"]["compartment"]
        # Two Nones round-trip as None (object dtype).
        assert out.dtype == object
        assert sum(1 for v in out if v is None) == 2
        assert sorted(v for v in out if v is not None) == ["axon", "soma", "soma"]


# ===================================================================
# Remote-URL clobber regression
# ===================================================================

class TestRemoteUrlPreservesMetadata:
    """``write_points(<gs://url>, ...)`` against an existing remote
    store must NOT silently re-create it and clobber the root NGFF
    metadata (axes, units, ...).  Before the fix in
    ``_create_or_open_store``, non-local URL schemes fell through to
    ``create_store(mode="w")`` which dropped any ``axes[*].unit``
    field the writer didn't re-supply.
    """

    def test_axis_units_survive_write_points_via_url_string(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from zarr_vectors.core import store as store_module
        from zarr_vectors.core.metadata import NgffAxis

        store_path = str(tmp_path / "remote_mock.zarr")

        # Create the store with explicit axis units, just like the user's
        # ``write_mfish_vectors.ipynb`` notebook does.
        create_store(
            store_path,
            bounds=([0, 0, 0], [100, 100, 100]),
            chunk_shape=(100.0, 100.0, 100.0),
            axes=[
                NgffAxis(name="x", type="space", unit="micrometer"),
                NgffAxis(name="y", type="space", unit="micrometer"),
                NgffAxis(name="z", type="space", unit="micrometer"),
            ],
        )

        # Force ``_create_or_open_store`` to take the "non-local URL"
        # branch even though the path is a local tmp dir.  This exercises
        # the bug-fix branch without needing real GCS / S3 credentials.
        real_detect = store_module._detect_scheme
        def fake_detect(p):
            if isinstance(p, str) and p == store_path:
                return "gs"
            return real_detect(p)
        monkeypatch.setattr(store_module, "_detect_scheme", fake_detect)

        # Pass the *URL string*, not the Group handle — this is the path
        # that hits the bug.
        positions = np.array(
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]],
            dtype=np.float32,
        )
        write_points(
            store_path, positions, chunk_shape=(100.0, 100.0, 100.0),
        )

        # Axes (with units) must survive the write.
        root = open_store(store_path)
        axes_after = root.attrs.to_dict()["multiscales"][0]["axes"]
        for ax in axes_after:
            assert ax.get("unit") == "micrometer", (
                f"write_points clobbered axis unit; got {axes_after}"
            )


# ===================================================================
# fragment_attributes integration
# ===================================================================

class TestFragmentAttributesIntegration:

    def test_per_chunk_fragment_attrs_roundtrip(self, tmp_path: Path) -> None:
        """Pass per-chunk fragment attributes through write_points and
        confirm they roundtrip via read_chunk_fragment_attributes."""
        from zarr_vectors.core.arrays import (
            list_chunk_keys,
            read_chunk_fragment_attributes,
            read_object_manifest,
        )

        # Two chunks (100^3 each), one object per point — 4 points → 2 in
        # each chunk → 2 fragments per chunk (one per object_id).
        positions = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [110, 110, 110],
            [120, 120, 120],
        ], dtype=np.float32)
        object_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        store_path = str(tmp_path / "frag_attrs.zarr")

        # Build per-chunk fragment-attribute arrays by mirroring the
        # writer's ordering: each chunk holds one fragment per unique
        # object_id, sorted by oid.  We materialize the OID as a
        # fragment_attribute — the canonical opt-in parent-ID use case.
        per_chunk_oids: dict = {
            (0, 0, 0): np.array([0, 1], dtype=np.int64),
            (1, 1, 1): np.array([2, 3], dtype=np.int64),
        }

        write_points(
            store_path, positions,
            chunk_shape=(100.0, 100.0, 100.0),
            bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
            object_ids=object_ids,
            fragment_attributes={"owner_oid": per_chunk_oids},
        )

        root = open_store(store_path)
        lg = root["0"]

        keys = sorted(list_chunk_keys(lg))
        assert (0, 0, 0) in keys and (1, 1, 1) in keys

        for chunk, expected_oids in per_chunk_oids.items():
            data = read_chunk_fragment_attributes(
                lg, "owner_oid", chunk, dtype=np.int64,
            )
            np.testing.assert_array_equal(data, expected_oids)

        # Cross-check against object_index: the OID we materialized as
        # a per-fragment attribute must match the manifest reverse-lookup.
        for oid in range(4):
            manifest = read_object_manifest(lg, oid)
            assert len(manifest) == 1
            chunk, fragment_idx = manifest[0]
            data = read_chunk_fragment_attributes(
                lg, "owner_oid", chunk, dtype=np.int64,
            )
            assert int(data[fragment_idx]) == oid

    def test_fragment_count_mismatch_raises(self, tmp_path: Path) -> None:
        from zarr_vectors.exceptions import ArrayError
        positions = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
        store_path = str(tmp_path / "bad_frag_attrs.zarr")
        # Two points, two distinct OIDs → two fragments in one chunk, but
        # caller supplies only one row.
        try:
            write_points(
                store_path, positions,
                chunk_shape=(100.0, 100.0, 100.0),
                bounds=([0.0, 0.0, 0.0], [100.0, 100.0, 100.0]),
                object_ids=np.array([0, 1], dtype=np.int64),
                fragment_attributes={
                    "owner_oid": {(0, 0, 0): np.array([0], dtype=np.int64)},
                },
            )
        except ArrayError:
            pass
        else:
            raise AssertionError(
                "expected ArrayError on fragment-count mismatch"
            )
