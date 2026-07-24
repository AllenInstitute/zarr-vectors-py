"""Step 08 tests: polyline/streamline write, read, cross-chunk, groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.types.polylines import write_polylines, read_polylines
from zarr_vectors.exceptions import ArrayError


# ===================================================================
# Basic write/read
# ===================================================================

class TestPolylineBasic:

    def test_single_polyline_single_chunk(self, tmp_path: Path) -> None:
        poly = [np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.float32)]
        store = str(tmp_path / "single.zarrvectors")

        summary = write_polylines(store, poly, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["polyline_count"] == 1
        assert summary["vertex_count"] == 3
        assert summary["cross_chunk_link_count"] == 0

        result = read_polylines(store)
        assert result["polyline_count"] == 1
        assert result["vertex_count"] == 3
        full = np.concatenate(result["polylines"][0])
        np.testing.assert_allclose(full, poly[0], atol=1e-5)

    def test_multiple_polylines(self, tmp_path: Path) -> None:
        polys = [
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),
            np.array([[30, 30, 30], [40, 40, 40], [50, 50, 50]], dtype=np.float32),
            np.array([[60, 60, 60], [70, 70, 70]], dtype=np.float32),
        ]
        store = str(tmp_path / "multi.zarrvectors")

        summary = write_polylines(store, polys, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["polyline_count"] == 3
        assert summary["vertex_count"] == 7

        result = read_polylines(store)
        assert result["polyline_count"] == 3
        assert result["vertex_count"] == 7

    def test_read_by_object_id(self, tmp_path: Path) -> None:
        polys = [
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),
            np.array([[30, 30, 30], [40, 40, 40]], dtype=np.float32),
            np.array([[50, 50, 50], [60, 60, 60]], dtype=np.float32),
        ]
        store = str(tmp_path / "byid.zarrvectors")
        write_polylines(store, polys, chunk_shape=(100.0, 100.0, 100.0))

        result = read_polylines(store, object_ids=[1])
        assert result["polyline_count"] == 1
        full = np.concatenate(result["polylines"][0])
        np.testing.assert_allclose(full[0], [30, 30, 30], atol=1e-5)


# ===================================================================
# Cross-chunk splitting
# ===================================================================

class TestPolylineCrossChunk:

    def test_streamline_crosses_one_boundary(self, tmp_path: Path) -> None:
        poly = [np.array([
            [10, 50, 50],
            [50, 50, 50],
            [90, 50, 50],   # still in chunk (0,0,0)
            [110, 50, 50],  # chunk (1,0,0)
            [150, 50, 50],
        ], dtype=np.float32)]
        store = str(tmp_path / "cross1.zarrvectors")

        summary = write_polylines(store, poly, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["cross_chunk_link_count"] == 1
        assert summary["chunk_count"] == 2

        result = read_polylines(store)
        assert result["polyline_count"] == 1
        # Should get 2 segments
        assert len(result["polylines"][0]) == 2
        # Concatenated should have 5 vertices
        full = np.concatenate(result["polylines"][0])
        assert full.shape == (5, 3)
        np.testing.assert_allclose(full[0], [10, 50, 50], atol=1e-5)
        np.testing.assert_allclose(full[4], [150, 50, 50], atol=1e-5)

    def test_streamline_crosses_two_boundaries(self, tmp_path: Path) -> None:
        poly = [np.array([
            [10, 50, 50],    # chunk (0,0,0)
            [50, 50, 50],    # chunk (0,0,0)
            [110, 50, 50],   # chunk (1,0,0)
            [150, 50, 50],   # chunk (1,0,0)
            [210, 50, 50],   # chunk (2,0,0)
        ], dtype=np.float32)]
        store = str(tmp_path / "cross2.zarrvectors")

        summary = write_polylines(store, poly, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["cross_chunk_link_count"] == 2
        assert summary["chunk_count"] == 3

        result = read_polylines(store)
        assert len(result["polylines"][0]) == 3  # 3 segments
        full = np.concatenate(result["polylines"][0])
        assert full.shape == (5, 3)

    def test_multiple_polylines_mixed_chunks(self, tmp_path: Path) -> None:
        polys = [
            # Polyline 0: single chunk
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),
            # Polyline 1: crosses boundary
            np.array([[10, 50, 50], [110, 50, 50]], dtype=np.float32),
            # Polyline 2: single chunk
            np.array([[50, 50, 50], [60, 60, 60]], dtype=np.float32),
        ]
        store = str(tmp_path / "mixed.zarrvectors")

        summary = write_polylines(store, polys, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["polyline_count"] == 3
        assert summary["cross_chunk_link_count"] == 1

        result = read_polylines(store)
        assert result["polyline_count"] == 3
        # Polyline 0: 1 segment
        assert len(result["polylines"][0]) == 1
        # Polyline 1: 2 segments (crosses boundary)
        assert len(result["polylines"][1]) == 2
        # Polyline 2: 1 segment
        assert len(result["polylines"][2]) == 1

    def test_reconstruction_preserves_order(self, tmp_path: Path) -> None:
        """Verify segment order matches polyline order after cross-chunk split."""
        pts = np.array([
            [10, 50, 50],
            [30, 50, 50],
            [50, 50, 50],
            [70, 50, 50],
            [90, 50, 50],
            [110, 50, 50],
            [130, 50, 50],
            [150, 50, 50],
        ], dtype=np.float32)
        poly = [pts]
        store = str(tmp_path / "order.zarrvectors")

        write_polylines(store, poly, chunk_shape=(100.0, 100.0, 100.0))
        result = read_polylines(store)
        full = np.concatenate(result["polylines"][0])
        np.testing.assert_allclose(full, pts, atol=1e-5)


# ===================================================================
# Vertex attributes
# ===================================================================

class TestPolylineAttributes:

    def test_per_vertex_scalar(self, tmp_path: Path) -> None:
        poly = [np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.float32)]
        fa = [np.array([0.1, 0.5, 0.9], dtype=np.float32)]
        store = str(tmp_path / "attr.zarrvectors")

        write_polylines(
            store, poly,
            chunk_shape=(100.0, 100.0, 100.0),
            vertex_attributes={"fa": fa},
        )

        result = read_polylines(store)
        assert result["polyline_count"] == 1


# ===================================================================
# Object attributes
# ===================================================================

class TestPolylineObjectAttributes:

    def test_termination(self, tmp_path: Path) -> None:
        polys = [
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),
            np.array([[30, 30, 30], [40, 40, 40]], dtype=np.float32),
        ]
        termination = np.array([[1, 5], [2, 7]], dtype=np.int32)
        store = str(tmp_path / "term.zarrvectors")

        write_polylines(
            store, polys,
            chunk_shape=(100.0, 100.0, 100.0),
            object_attributes={"termination": termination},
        )

        result = read_polylines(store)
        assert result["polyline_count"] == 2


# ===================================================================
# Groups (tracts)
# ===================================================================

class TestPolylineGroups:

    def test_tract_groups(self, tmp_path: Path) -> None:
        polys = [
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),
            np.array([[30, 30, 30], [40, 40, 40]], dtype=np.float32),
            np.array([[50, 50, 50], [60, 60, 60]], dtype=np.float32),
            np.array([[70, 70, 70], [80, 80, 80]], dtype=np.float32),
        ]
        groups = {0: [0, 1], 1: [2, 3]}
        store = str(tmp_path / "tracts.zarrvectors")

        summary = write_polylines(
            store, polys,
            chunk_shape=(100.0, 100.0, 100.0),
            groups=groups,
            group_attributes={"name": np.array([0.0, 1.0], dtype=np.float32)},
        )
        assert summary["group_count"] == 2

        # Read by group
        result = read_polylines(store, group_ids=[0])
        assert result["polyline_count"] == 2

        result = read_polylines(store, group_ids=[1])
        assert result["polyline_count"] == 2


# ===================================================================
# Bbox filter
# ===================================================================

class TestPolylineBbox:

    def test_bbox_filter(self, tmp_path: Path) -> None:
        polys = [
            np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),   # chunk (0,0,0)
            np.array([[110, 110, 110], [120, 120, 120]], dtype=np.float32),  # chunk (1,1,1)
        ]
        store = str(tmp_path / "bbox.zarrvectors")
        write_polylines(store, polys, chunk_shape=(100.0, 100.0, 100.0))

        result = read_polylines(
            store,
            bbox=(np.array([0, 0, 0]), np.array([50, 50, 50])),
        )
        assert result["polyline_count"] == 1


# ===================================================================
# Edge cases
# ===================================================================

class TestPolylineEdgeCases:

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        try:
            write_polylines(str(tmp_path / "e.zarrvectors"), [], chunk_shape=(100.0, 100.0, 100.0))
            assert False
        except ArrayError:
            pass

    def test_single_point_polyline(self, tmp_path: Path) -> None:
        """A polyline with just 1 vertex — degenerate but valid."""
        poly = [np.array([[50, 50, 50]], dtype=np.float32)]
        store = str(tmp_path / "single_pt.zarrvectors")
        summary = write_polylines(store, poly, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["polyline_count"] == 1
        assert summary["vertex_count"] == 1

        result = read_polylines(store)
        assert result["vertex_count"] == 1

    def test_many_streamlines(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        polys = []
        for _ in range(100):
            n = rng.integers(10, 50)
            start = rng.uniform(0, 200, size=(1, 3)).astype(np.float32)
            steps = rng.normal(0, 2, size=(n - 1, 3)).astype(np.float32)
            pts = np.clip(
                np.concatenate([start, start + np.cumsum(steps, axis=0)]),
                0, 299
            ).astype(np.float32)
            polys.append(pts)

        store = str(tmp_path / "many.zarrvectors")
        summary = write_polylines(store, polys, chunk_shape=(100.0, 100.0, 100.0))
        assert summary["polyline_count"] == 100

        result = read_polylines(store)
        assert result["polyline_count"] == 100

    def test_2d_polylines(self, tmp_path: Path) -> None:
        poly = [np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)]
        store = str(tmp_path / "2d.zarrvectors")
        write_polylines(store, poly, chunk_shape=(10.0, 10.0))
        result = read_polylines(store)
        assert result["vertex_count"] == 3


# ===================================================================
# object_ids provenance — which object each returned polyline came from
# ===================================================================

def _provenance_store(tmp_path: Path) -> str:
    """Object 0 spans chunks (0,0,0)/(1,0,0)/(2,0,0); objects 1 and 2 sit
    wholly inside (0,0,0) and (1,1,1) respectively."""
    polys = [
        np.array([
            [10, 50, 50], [50, 50, 50],      # chunk (0,0,0)
            [110, 50, 50], [150, 50, 50],    # chunk (1,0,0)
            [210, 50, 50], [250, 50, 50],    # chunk (2,0,0)
        ], dtype=np.float32),
        np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32),
        np.array([[110, 110, 110], [120, 120, 120]], dtype=np.float32),
    ]
    store = str(tmp_path / "prov.zarrvectors")
    write_polylines(store, polys, chunk_shape=(100.0, 100.0, 100.0))
    return store


class TestPolylineObjectIds:

    def test_whole_object_matches_requested_subset(self, tmp_path: Path) -> None:
        """Order follows the requested iteration order, not a sort."""
        store = _provenance_store(tmp_path)

        result = read_polylines(store, object_ids=[2, 0])
        assert result["object_ids"] == [2, 0]
        # polylines[0] really is object 2, not the lower-numbered object.
        first = np.concatenate(result["polylines"][0])
        np.testing.assert_allclose(first[0], [110, 110, 110], atol=1e-5)

    def test_skipped_objects_absent(self, tmp_path: Path) -> None:
        """An ID with no manifest drops out of both lists together."""
        store = _provenance_store(tmp_path)

        result = read_polylines(store, object_ids=[0, 99, 1])
        assert result["object_ids"] == [0, 1]
        assert len(result["polylines"]) == 2

    def test_bbox_filtered_objects_absent(self, tmp_path: Path) -> None:
        """Objects failing the has_match guard never reach object_ids."""
        store = _provenance_store(tmp_path)

        result = read_polylines(
            store,
            bbox=(np.array([0, 0, 0]), np.array([50, 50, 50])),
        )
        # Object 2 lives in chunk (1,1,1) — outside the bbox.
        assert result["object_ids"] == [0, 1]

    def test_crop_mode_repeats_id_per_run(self, tmp_path: Path) -> None:
        """Object 0's manifest alternates in/out of the whitelist, so it
        emits one polyline per surviving run — same ID, repeated."""
        store = _provenance_store(tmp_path)

        result = read_polylines(
            store, object_ids=[0], chunks=[(0, 0, 0), (2, 0, 0)],
        )
        assert result["object_ids"] == [0, 0]
        assert len(result["polylines"]) == result["polyline_count"] == 2

    def test_empty_result_has_object_ids(self, tmp_path: Path) -> None:
        """The short-circuit empty path carries the key too, so callers
        need no special case."""
        rng = np.random.default_rng(0)
        polys = [rng.uniform(0, 100, (10, 3)).astype("f4") for _ in range(2)]
        labels = [np.array(["A"] * 10), np.array(["B"] * 10)]
        store = str(tmp_path / "attr.zarrvectors")
        write_polylines(
            store, polys,
            chunk_shape=(50.0, 50.0, 50.0),
            bin_shape=(50.0, 50.0, 50.0),
            vertex_attributes={"bundle": labels},
            chunk_by_attribute="bundle",
        )

        result = read_polylines(store, attribute_filter={"bundle": "ZZZ"})
        assert result["object_ids"] == []
        assert result["polyline_count"] == 0

    @pytest.mark.parametrize("kwargs", [
        {},
        {"object_ids": [0, 99, 1]},
        {"object_ids": []},
        {"bbox": (np.array([0, 0, 0]), np.array([50, 50, 50]))},
        {"chunks": [(0, 0, 0), (2, 0, 0)]},
        {"chunks": []},
    ])
    def test_length_invariant(self, tmp_path: Path, kwargs: dict) -> None:
        store = _provenance_store(tmp_path)

        result = read_polylines(store, **kwargs)
        assert (
            len(result["object_ids"])
            == len(result["polylines"])
            == result["polyline_count"]
        )
