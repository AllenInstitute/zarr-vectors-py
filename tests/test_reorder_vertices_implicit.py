"""Tests for ``reorder_vertices_implicit`` — inverse of
``materialise_object_links_explicit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    read_chunk_links,
    read_chunk_vertices,
    read_cross_chunk_links,
)
from zarr_vectors.core.metadata import RootMetadata
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops import reorder_vertices_implicit
from zarr_vectors.types.graphs import read_graph, write_graph


def _edge_set(edges: np.ndarray) -> set:
    """Undirected, frozenset-of-frozensets representation for set equality."""
    out: set = set()
    arr = np.asarray(edges).reshape(-1, 2)
    for row in arr:
        out.add(frozenset((int(row[0]), int(row[1]))))
    return out


def _edge_set_by_position(positions: np.ndarray, edges: np.ndarray) -> set:
    """Edge set keyed by (rounded) endpoint positions — permutation-invariant."""
    out: set = set()
    arr = np.asarray(edges).reshape(-1, 2)
    rounded = np.round(positions, 3)
    for row in arr:
        a = tuple(rounded[int(row[0])].tolist())
        b = tuple(rounded[int(row[1])].tolist())
        out.add(frozenset((a, b)))
    return out


# -------- fixtures ---------------------------------------------------


@pytest.fixture
def scrambled_path_one_chunk(tmp_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """A 6-vertex linear path, all in one chunk, with explicit edges in
    a scrambled vertex order."""
    path = tmp_path / "store.zv"
    # Positions are arranged so the "true" path goes [10,20,30,40,50,60]
    # but the stored vertex order is scrambled: [30, 10, 50, 60, 20, 40].
    positions = np.array(
        [
            [30.0, 30.0, 30.0],
            [10.0, 10.0, 10.0],
            [50.0, 50.0, 50.0],
            [60.0, 60.0, 60.0],
            [20.0, 20.0, 20.0],
            [40.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    # Edges in stored-index space: connect [10-20], [20-30], [30-40], [40-50], [50-60]
    # Stored indices: 10=1, 20=4, 30=0, 40=5, 50=2, 60=3
    edges = np.array(
        [
            [1, 4],  # 10 - 20
            [4, 0],  # 20 - 30
            [0, 5],  # 30 - 40
            [5, 2],  # 40 - 50
            [2, 3],  # 50 - 60
        ],
        dtype=np.int64,
    )
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="graph",  # explicit convention
    )
    return str(path), positions, edges


@pytest.fixture
def scrambled_y_skeleton(tmp_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """A 7-vertex Y-shaped skeleton: trunk [0-1-2-3] then branches
    [3-4-5] and [3-6]."""
    path = tmp_path / "store.zv"
    # Positions for vertices 0..6 are placed in scrambled storage order.
    # Storage indices: A=0, B=1, C=2, D=3 (fork), E=4, F=5, G=6
    positions = np.array(
        [
            [10.0, 10.0, 10.0],   # A
            [20.0, 10.0, 10.0],   # B
            [30.0, 10.0, 10.0],   # C
            [40.0, 10.0, 10.0],   # D fork
            [50.0, 20.0, 10.0],   # E
            [60.0, 20.0, 10.0],   # F
            [50.0, 0.0, 10.0],    # G
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [1, 0],
            [2, 1],
            [3, 2],
            [4, 3],
            [5, 4],
            [6, 3],
        ],
        dtype=np.int64,
    )
    write_graph(
        str(path), positions, edges,
        chunk_shape=(200.0, 200.0, 200.0),
        bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
        kind="graph",
    )
    return str(path), positions, edges


# -------- tests ------------------------------------------------------


class TestSourceConventionDispatch:

    def test_already_implicit_branches_is_noop(
        self, scrambled_path_one_chunk: tuple[str, np.ndarray, np.ndarray]
    ) -> None:
        path, _pos, _edges = scrambled_path_one_chunk
        root = open_store(path, mode="r+")
        # First flip via a one-shot manual write to simulate a
        # ``_with_branches`` source.
        attrs = root.attrs.to_dict()
        zv = dict(attrs.get("zarr_vectors", {}))
        zv["links_convention"] = "implicit_sequential_with_branches"
        root.attrs.update({"zarr_vectors": zv})

        report = reorder_vertices_implicit(root, level=0, flip_convention=True)
        assert report["objects_processed"] == 0
        assert report["convention_flipped"] is False

    def test_implicit_sequential_source_raises(
        self, tmp_path: Path
    ) -> None:
        # Build a skeleton store (which has _with_branches convention),
        # then poke its convention to implicit_sequential to simulate
        # the unsupported source.
        path = tmp_path / "store.zv"
        positions = np.array(
            [[float(i), 0.0, 0.0] for i in range(5)], dtype=np.float32,
        )
        edges = np.array(
            [[i + 1, i] for i in range(4)], dtype=np.int64,
        )
        write_graph(
            str(path), positions, edges,
            chunk_shape=(200.0, 200.0, 200.0),
            bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
            kind="skeleton",
        )
        root = open_store(str(path), mode="r+")
        attrs = root.attrs.to_dict()
        zv = dict(attrs.get("zarr_vectors", {}))
        zv["links_convention"] = "implicit_sequential"
        root.attrs.update({"zarr_vectors": zv})

        with pytest.raises(EditError, match="implicit_sequential"):
            reorder_vertices_implicit(root, level=0)


class TestSingleChunkPath:

    def test_round_trip_path(
        self, scrambled_path_one_chunk: tuple[str, np.ndarray, np.ndarray]
    ) -> None:
        path, _stored_positions, _stored_edges = scrambled_path_one_chunk
        # Read the graph before reordering to capture the edge set in
        # position space.
        before = read_graph(path)
        before_edges = _edge_set_by_position(
            np.asarray(before["positions"]),
            np.asarray(before["edges"]),
        )

        root = open_store(path, mode="r+")
        report = reorder_vertices_implicit(
            root, level=0, flip_convention=True,
        )

        assert report["objects_processed"] == 1
        assert report["objects_skipped_non_tree"] == 0
        assert report["convention_flipped"] is True
        meta = RootMetadata.from_dict(root.attrs.to_dict())
        assert meta.links_convention == "implicit_sequential_with_branches"

        # After reordering the path is the spine — no branch overrides
        # should remain.
        assert report["branch_overrides_written"] == 0
        assert report["cross_chunk_branch_overrides_written"] == 0

        # The graph reader should still reproduce the original edge set
        # (modulo vertex permutation, hence position-based comparison).
        after = read_graph(path)
        after_edges = _edge_set_by_position(
            np.asarray(after["positions"]),
            np.asarray(after["edges"]),
        )
        assert after_edges == before_edges

        # Vertices in the chunk should now be in path order:
        # [10, 20, 30, 40, 50, 60] (or its reverse — root selection is
        # deterministic but either endpoint is valid).
        level_group = get_resolution_level(root, 0)
        chunks = read_chunk_vertices(level_group, (0, 0, 0), ndim=3)
        assert len(chunks) == 1
        verts = chunks[0]
        xs = verts[:, 0].tolist()
        assert xs == sorted(xs) or xs == sorted(xs, reverse=True)


class TestBranchingTree:

    def test_y_skeleton_emits_one_branch_override(
        self, scrambled_y_skeleton: tuple[str, np.ndarray, np.ndarray]
    ) -> None:
        path, _pos, _edges = scrambled_y_skeleton
        before = read_graph(path)
        before_edges = _edge_set_by_position(
            np.asarray(before["positions"]),
            np.asarray(before["edges"]),
        )

        root = open_store(path, mode="r+")
        report = reorder_vertices_implicit(
            root, level=0, flip_convention=True,
        )

        assert report["objects_processed"] == 1
        assert report["convention_flipped"] is True
        # Y-shape has one branch point => one non-spine edge.
        total_overrides = (
            report["branch_overrides_written"]
            + report["cross_chunk_branch_overrides_written"]
        )
        assert total_overrides == 1

        after = read_graph(path)
        after_edges = _edge_set_by_position(
            np.asarray(after["positions"]),
            np.asarray(after["edges"]),
        )
        assert after_edges == before_edges

class TestNonTreeSkipped:

    def test_cycle_object_skipped_with_warning(
        self, tmp_path: Path
    ) -> None:
        # 4-vertex cycle: edges 0-1, 1-2, 2-3, 3-0.
        path = tmp_path / "store.zv"
        positions = np.array(
            [
                [10.0, 10.0, 10.0],
                [20.0, 10.0, 10.0],
                [20.0, 20.0, 10.0],
                [10.0, 20.0, 10.0],
            ],
            dtype=np.float32,
        )
        edges = np.array(
            [[1, 0], [2, 1], [3, 2], [0, 3]],
            dtype=np.int64,
        )
        write_graph(
            str(path), positions, edges,
            chunk_shape=(200.0, 200.0, 200.0),
            bounds=([0.0, 0.0, 0.0], [200.0, 200.0, 200.0]),
            kind="graph",
        )

        # Capture chunk bytes before reorder for byte-equal check.
        root = open_store(str(path), mode="r+")
        level_group = get_resolution_level(root, 0)
        chunks_before = read_chunk_vertices(level_group, (0, 0, 0), ndim=3)
        verts_before = chunks_before[0].copy()

        with pytest.warns(UserWarning, match="not a tree"):
            report = reorder_vertices_implicit(
                root, level=0, flip_convention=True,
            )

        assert report["objects_processed"] == 0
        assert report["objects_skipped_non_tree"] == 1
        assert report["skipped_oids"] == [0]
        # Convention should NOT flip when any object was skipped.
        assert report["convention_flipped"] is False
        meta = RootMetadata.from_dict(root.attrs.to_dict())
        assert meta.links_convention == "explicit"

        # The vertices should be untouched.
        chunks_after = read_chunk_vertices(level_group, (0, 0, 0), ndim=3)
        np.testing.assert_array_equal(verts_before, chunks_after[0])


class TestMultiChunkStreamline:

    def test_path_spanning_multiple_chunks(self, tmp_path: Path) -> None:
        # Path of 9 vertices stretching across 3 chunks along X.
        # Chunks are 50 wide; vertices land at x = 5, 15, 25, ...
        path = tmp_path / "store.zv"
        xs = np.array([5, 15, 25, 35, 55, 65, 75, 105, 115], dtype=np.float32)
        positions = np.column_stack(
            [xs, np.full(len(xs), 10.0, dtype=np.float32),
             np.full(len(xs), 10.0, dtype=np.float32)],
        )
        # Scramble storage order: reverse the array. Edges still
        # reference the (reversed) storage indices.
        perm = np.arange(len(xs))[::-1].copy()
        stored = positions[perm]
        # Original adjacency 0-1-2-3-4-5-6-7-8 in original index space
        # becomes (in stored index space) reversed: (n-1)-(n-2)-...-0.
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        edges = np.array(
            [[inv[i + 1], inv[i]] for i in range(len(xs) - 1)],
            dtype=np.int64,
        )

        write_graph(
            str(path), stored, edges,
            chunk_shape=(50.0, 50.0, 50.0),
            bounds=([0.0, 0.0, 0.0], [150.0, 50.0, 50.0]),
            kind="graph",
        )

        before = read_graph(str(path))
        before_edges = _edge_set_by_position(
            np.asarray(before["positions"]),
            np.asarray(before["edges"]),
        )

        root = open_store(str(path), mode="r+")
        report = reorder_vertices_implicit(
            root, level=0, flip_convention=True,
        )
        assert report["objects_processed"] == 1
        assert report["convention_flipped"] is True
        # A pure path: no branch overrides intra-chunk.
        assert report["branch_overrides_written"] == 0

        after = read_graph(str(path))
        after_edges = _edge_set_by_position(
            np.asarray(after["positions"]),
            np.asarray(after["edges"]),
        )
        assert after_edges == before_edges
