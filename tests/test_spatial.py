"""Step 05 tests: spatial chunking, boundary splitting, edge/face partitioning."""

from __future__ import annotations

import numpy as np

from zarr_vectors.exceptions import ChunkingError
from zarr_vectors.spatial.boundary import (
    build_reindex_map,
    build_vertex_chunk_mapping,
    cross_chunk_links_for_segments,
    partition_edges,
    partition_faces,
    split_polyline_at_boundaries,
)
from zarr_vectors.spatial.chunking import (
    assign_chunks,
    chunks_intersecting_bbox,
    compute_bounds,
    compute_chunk_coords,
    compute_grid_shape,
    positions_in_bbox,
)

# ===================================================================
# Chunk assignment
# ===================================================================

class TestAssignChunks:

    def test_single_chunk(self) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = assign_chunks(positions, (100.0, 100.0, 100.0))
        assert len(result) == 1
        assert (0, 0, 0) in result
        assert len(result[(0, 0, 0)]) == 2

    def test_two_chunks(self) -> None:
        positions = np.array([
            [10, 10, 10],   # chunk (0,0,0)
            [110, 10, 10],  # chunk (1,0,0)
            [50, 50, 50],   # chunk (0,0,0)
        ], dtype=np.float32)
        result = assign_chunks(positions, (100.0, 100.0, 100.0))
        assert len(result) == 2
        assert set(result[(0, 0, 0)]) == {0, 2}
        assert set(result[(1, 0, 0)]) == {1}

    def test_many_chunks_3d(self) -> None:
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 1000, size=(1000, 3)).astype(np.float32)
        result = assign_chunks(positions, (100.0, 100.0, 100.0))
        # Every vertex should appear exactly once
        all_indices = np.concatenate([v for v in result.values()])
        assert len(all_indices) == 1000
        assert len(np.unique(all_indices)) == 1000

    def test_2d_points(self) -> None:
        positions = np.array([[5, 5], [15, 5], [5, 15]], dtype=np.float32)
        result = assign_chunks(positions, (10.0, 10.0))
        assert (0, 0) in result
        assert (1, 0) in result
        assert (0, 1) in result

    def test_negative_coordinates(self) -> None:
        positions = np.array([[-5, -5, -5], [5, 5, 5]], dtype=np.float32)
        result = assign_chunks(positions, (10.0, 10.0, 10.0))
        assert (-1, -1, -1) in result
        assert (0, 0, 0) in result

    def test_boundary_vertex(self) -> None:
        # Vertex exactly on boundary x=100 should go to chunk (1,0,0)
        positions = np.array([[100.0, 50, 50]], dtype=np.float32)
        result = assign_chunks(positions, (100.0, 100.0, 100.0))
        assert (1, 0, 0) in result

    def test_empty_positions(self) -> None:
        positions = np.zeros((0, 3), dtype=np.float32)
        result = assign_chunks(positions, (100.0, 100.0, 100.0))
        assert result == {}

    def test_wrong_dimensions(self) -> None:
        positions = np.zeros((10, 3), dtype=np.float32)
        try:
            assign_chunks(positions, (100.0, 100.0))  # 2D shape, 3D points
            assert False
        except ChunkingError:
            pass

    def test_large_batch_performance(self) -> None:
        """100K points should run in reasonable time."""
        rng = np.random.default_rng(123)
        positions = rng.uniform(0, 1000, size=(100_000, 3)).astype(np.float32)
        result = assign_chunks(positions, (50.0, 50.0, 50.0))
        total = sum(len(v) for v in result.values())
        assert total == 100_000


class TestComputeChunkCoords:

    def test_basic(self) -> None:
        pos = np.array([150.0, 250.0, 350.0])
        assert compute_chunk_coords(pos, (100.0, 100.0, 100.0)) == (1, 2, 3)

    def test_origin(self) -> None:
        pos = np.array([0.0, 0.0, 0.0])
        assert compute_chunk_coords(pos, (100.0, 100.0, 100.0)) == (0, 0, 0)

    def test_negative(self) -> None:
        pos = np.array([-50.0, 50.0, 0.0])
        assert compute_chunk_coords(pos, (100.0, 100.0, 100.0)) == (-1, 0, 0)


class TestComputeBounds:

    def test_basic(self) -> None:
        positions = np.array([[1, 2, 3], [10, 20, 30], [5, 5, 5]], dtype=np.float32)
        lo, hi = compute_bounds(positions)
        np.testing.assert_array_equal(lo, [1, 2, 3])
        np.testing.assert_array_equal(hi, [10, 20, 30])

    def test_single_point(self) -> None:
        positions = np.array([[5, 5, 5]], dtype=np.float32)
        lo, hi = compute_bounds(positions)
        np.testing.assert_array_equal(lo, hi)

    def test_empty_raises(self) -> None:
        try:
            compute_bounds(np.zeros((0, 3), dtype=np.float32))
            assert False
        except ChunkingError:
            pass


class TestComputeGridShape:

    def test_basic(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([300, 200, 100]))
        shape = compute_grid_shape(bounds, (100.0, 100.0, 100.0))
        assert shape == (3, 2, 1)

    def test_non_aligned(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([150, 250, 50]))
        shape = compute_grid_shape(bounds, (100.0, 100.0, 100.0))
        assert shape == (2, 3, 1)  # ceil(150/100)=2, ceil(250/100)=3, ceil(50/100)=1

    def test_minimum_one(self) -> None:
        bounds = (np.array([0, 0, 0]), np.array([0, 0, 0]))
        shape = compute_grid_shape(bounds, (100.0, 100.0, 100.0))
        assert all(s >= 1 for s in shape)


class TestChunksIntersectingBbox:

    def test_single_chunk(self) -> None:
        result = chunks_intersecting_bbox(
            np.array([10, 10, 10]),
            np.array([90, 90, 90]),
            (100.0, 100.0, 100.0),
        )
        assert result == [(0, 0, 0)]

    def test_spanning_two(self) -> None:
        result = chunks_intersecting_bbox(
            np.array([50, 50, 50]),
            np.array([150, 50, 50]),
            (100.0, 100.0, 100.0),
        )
        assert (0, 0, 0) in result
        assert (1, 0, 0) in result

    def test_2d(self) -> None:
        result = chunks_intersecting_bbox(
            np.array([0, 0]),
            np.array([150, 50]),
            (100.0, 100.0),
        )
        assert (0, 0) in result
        assert (1, 0) in result


class TestPositionsInBbox:

    def test_basic(self) -> None:
        positions = np.array([
            [0, 0, 0],
            [50, 50, 50],
            [200, 200, 200],
            [100, 100, 100],
        ], dtype=np.float32)
        idx = positions_in_bbox(
            positions,
            np.array([10, 10, 10]),
            np.array([150, 150, 150]),
        )
        assert set(idx) == {1, 3}


# ===================================================================
# Polyline splitting
# ===================================================================

class TestSplitPolyline:

    def test_no_crossing(self) -> None:
        """All vertices in one chunk — single segment."""
        verts = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.float32)
        segments = split_polyline_at_boundaries(verts, (100.0, 100.0, 100.0))
        assert len(segments) == 1
        assert segments[0][0] == (0, 0, 0)
        np.testing.assert_array_equal(segments[0][1], verts)

    def test_single_crossing(self) -> None:
        """Polyline crosses from chunk (0,0,0) to (1,0,0)."""
        verts = np.array([
            [10, 50, 50],
            [50, 50, 50],
            [90, 50, 50],   # still in (0,0,0)
            [110, 50, 50],  # now in (1,0,0)
            [150, 50, 50],
        ], dtype=np.float32)
        segments = split_polyline_at_boundaries(verts, (100.0, 100.0, 100.0))
        assert len(segments) == 2
        assert segments[0][0] == (0, 0, 0)
        assert segments[1][0] == (1, 0, 0)
        assert len(segments[0][1]) == 3
        assert len(segments[1][1]) == 2

    def test_multiple_crossings(self) -> None:
        """Polyline crosses 3 chunks."""
        verts = np.array([
            [10, 50, 50],    # (0,0,0)
            [50, 50, 50],    # (0,0,0)
            [110, 50, 50],   # (1,0,0)
            [150, 50, 50],   # (1,0,0)
            [210, 50, 50],   # (2,0,0)
        ], dtype=np.float32)
        segments = split_polyline_at_boundaries(verts, (100.0, 100.0, 100.0))
        assert len(segments) == 3
        assert segments[0][0] == (0, 0, 0)
        assert segments[1][0] == (1, 0, 0)
        assert segments[2][0] == (2, 0, 0)

    def test_preserves_all_vertices(self) -> None:
        rng = np.random.default_rng(99)
        verts = np.cumsum(rng.uniform(0, 20, size=(50, 3)), axis=0).astype(np.float32)
        segments = split_polyline_at_boundaries(verts, (100.0, 100.0, 100.0))
        reconstructed = np.concatenate([s[1] for s in segments], axis=0)
        np.testing.assert_array_equal(reconstructed, verts)

    def test_empty(self) -> None:
        verts = np.zeros((0, 3), dtype=np.float32)
        segments = split_polyline_at_boundaries(verts, (100.0, 100.0, 100.0))
        assert segments == []

    def test_single_vertex(self) -> None:
        verts = np.array([[50, 50, 50]], dtype=np.float32)
        segments = split_polyline_at_boundaries(verts, (100.0, 100.0, 100.0))
        assert len(segments) == 1
        assert segments[0][1].shape == (1, 3)


class TestCrossChunkLinksForSegments:

    def test_two_segments(self) -> None:
        seg_a = ((0, 0, 0), np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32))
        seg_b = ((0, 0, 1), np.array([[3, 3, 3], [4, 4, 4]], dtype=np.float32))
        links = cross_chunk_links_for_segments([seg_a, seg_b], [0, 0])
        assert len(links) == 1
        # Last vertex of seg_a (index 2) → first vertex of seg_b (index 0)
        assert links[0] == (((0, 0, 0), 2), ((0, 0, 1), 0))

    def test_three_segments(self) -> None:
        segs = [
            ((0, 0, 0), np.zeros((3, 3), dtype=np.float32)),
            ((1, 0, 0), np.zeros((2, 3), dtype=np.float32)),
            ((2, 0, 0), np.zeros((4, 3), dtype=np.float32)),
        ]
        links = cross_chunk_links_for_segments(segs, [0, 0, 0])
        assert len(links) == 2
        assert links[0] == (((0, 0, 0), 2), ((1, 0, 0), 0))
        assert links[1] == (((1, 0, 0), 1), ((2, 0, 0), 0))

    def test_single_segment_no_links(self) -> None:
        segs = [((0, 0, 0), np.zeros((5, 3), dtype=np.float32))]
        links = cross_chunk_links_for_segments(segs, [0])
        assert links == []


# ===================================================================
# Edge partitioning
# ===================================================================

class TestPartitionEdges:

    def _setup_4_node_graph(self):
        """4 nodes: n0,n1 in chunk 0, n2,n3 in chunk 1."""
        chunk_assignments = {
            (0, 0, 0): np.array([0, 1]),
            (0, 0, 1): np.array([2, 3]),
        }
        chunk_list = [(0, 0, 0), (0, 0, 1)]
        v_chunks, v_local, cl = build_vertex_chunk_mapping(
            chunk_assignments, 4, chunk_list
        )
        return v_chunks, v_local, cl

    def test_all_intra(self) -> None:
        v_chunks, v_local, cl = self._setup_4_node_graph()
        edges = np.array([[0, 1]], dtype=np.int64)  # both in chunk 0
        intra, cross = partition_edges(edges, v_chunks, v_local, cl)
        assert len(cross) == 0
        assert (0, 0, 0) in intra
        np.testing.assert_array_equal(intra[(0, 0, 0)], [[0, 1]])

    def test_all_cross(self) -> None:
        v_chunks, v_local, cl = self._setup_4_node_graph()
        edges = np.array([[0, 2], [1, 3]], dtype=np.int64)
        intra, cross = partition_edges(edges, v_chunks, v_local, cl)
        assert len(intra) == 0
        assert len(cross) == 2
        # n0 (chunk 0, local 0) → n2 (chunk 1, local 0)
        assert cross[0] == (((0, 0, 0), 0), ((0, 0, 1), 0))
        # n1 (chunk 0, local 1) → n3 (chunk 1, local 1)
        assert cross[1] == (((0, 0, 0), 1), ((0, 0, 1), 1))

    def test_mixed(self) -> None:
        v_chunks, v_local, cl = self._setup_4_node_graph()
        edges = np.array([
            [0, 1],  # intra chunk 0
            [2, 3],  # intra chunk 1
            [0, 2],  # cross
            [1, 3],  # cross
            [1, 2],  # cross
        ], dtype=np.int64)
        intra, cross = partition_edges(edges, v_chunks, v_local, cl)
        assert len(cross) == 3
        total_intra = sum(len(v) for v in intra.values())
        assert total_intra == 2


# ===================================================================
# Face partitioning
# ===================================================================

class TestPartitionFaces:

    def test_all_intra(self) -> None:
        chunk_assignments = {(0, 0, 0): np.array([0, 1, 2, 3])}
        v_chunks, v_local, cl = build_vertex_chunk_mapping(
            chunk_assignments, 4
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        intra, cross = partition_faces(faces, v_chunks, v_local, cl)
        assert len(cross) == 0
        assert (0, 0, 0) in intra
        assert len(intra[(0, 0, 0)]) == 2

    def test_all_cross(self) -> None:
        chunk_assignments = {
            (0, 0, 0): np.array([0, 1]),
            (0, 0, 1): np.array([2, 3]),
        }
        v_chunks, v_local, cl = build_vertex_chunk_mapping(
            chunk_assignments, 4, [(0, 0, 0), (0, 0, 1)]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        intra, cross = partition_faces(faces, v_chunks, v_local, cl)
        assert len(intra) == 0
        assert len(cross) == 2
        # Face [0,1,2]: n0(chunk0,local0), n1(chunk0,local1), n2(chunk1,local0)
        assert cross[0][0] == ((0, 0, 0), 0)
        assert cross[0][1] == ((0, 0, 0), 1)
        assert cross[0][2] == ((0, 0, 1), 0)

    def test_mixed(self) -> None:
        chunk_assignments = {
            (0, 0, 0): np.array([0, 1, 2]),
            (0, 0, 1): np.array([3]),
        }
        v_chunks, v_local, cl = build_vertex_chunk_mapping(
            chunk_assignments, 4, [(0, 0, 0), (0, 0, 1)]
        )
        faces = np.array([
            [0, 1, 2],  # all in chunk 0 — intra
            [0, 1, 3],  # n3 in chunk 1 — cross
        ], dtype=np.int64)
        intra, cross = partition_faces(faces, v_chunks, v_local, cl)
        assert (0, 0, 0) in intra
        assert len(intra[(0, 0, 0)]) == 1
        assert len(cross) == 1


# ===================================================================
# Vertex chunk mapping helpers
# ===================================================================

class TestBuildVertexChunkMapping:

    def test_basic(self) -> None:
        chunk_assignments = {
            (0, 0, 0): np.array([0, 2]),
            (1, 0, 0): np.array([1, 3]),
        }
        v_chunks, v_local, cl = build_vertex_chunk_mapping(
            chunk_assignments, 4
        )
        assert v_chunks[0] != v_chunks[1]  # different chunks
        assert v_local[0] == 0  # first in its chunk
        assert v_local[2] == 1  # second in its chunk

    def test_missing_vertex_raises(self) -> None:
        chunk_assignments = {
            (0, 0, 0): np.array([0, 1]),
            # vertex 2 is missing
        }
        try:
            build_vertex_chunk_mapping(chunk_assignments, 3)
            assert False
        except ChunkingError:
            pass


class TestBuildReindexMap:

    def test_basic(self) -> None:
        chunk_assignments = {
            (0, 0, 0): np.array([5, 10]),
            (1, 0, 0): np.array([3]),
        }
        reindex = build_reindex_map(chunk_assignments)
        assert reindex[(0, 0, 0)][5] == 0
        assert reindex[(0, 0, 0)][10] == 1
        assert reindex[(1, 0, 0)][3] == 0
