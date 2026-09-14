"""Shared pytest fixtures for zarr-vectors tests.

All fixtures produce small synthetic datasets suitable for unit and
integration testing.  Sizes are kept small enough for fast CI but
large enough to exercise multi-chunk and boundary logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Temporary store paths
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for a Zarr Vectors store."""
    return tmp_path / "test_store.zarr"


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for scratch files."""
    return tmp_path / "scratch"


# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy random generator for reproducible test data."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Point cloud fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_points_3d(rng: np.random.Generator) -> dict:
    """100 random XYZ points in [0, 1000)³ with an intensity attribute.

    Returns dict with keys: positions, vertex_attributes, chunk_shape.
    With chunk_shape=(500, 500, 500), points span ~8 chunks.
    """
    positions = rng.uniform(0, 1000, size=(100, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, size=(100,)).astype(np.float32)
    return {
        "positions": positions,
        "vertex_attributes": {"intensity": intensity},
        "chunk_shape": (500.0, 500.0, 500.0),
    }


@pytest.fixture
def two_chunk_points(rng: np.random.Generator) -> dict:
    """50 points guaranteed to land in exactly 2 chunks along the X axis.

    25 points in x=[0, 50), 25 in x=[50, 100).
    chunk_shape=(50, 100, 100) so the split is at x=50.
    """
    left = rng.uniform(0, 49.9, size=(25, 1)).astype(np.float32)
    right = rng.uniform(50.0, 99.9, size=(25, 1)).astype(np.float32)
    yz = rng.uniform(0, 50, size=(50, 2)).astype(np.float32)
    x = np.concatenate([left, right], axis=0)
    positions = np.concatenate([x, yz], axis=1)
    return {
        "positions": positions,
        "chunk_shape": (50.0, 100.0, 100.0),
    }


# ---------------------------------------------------------------------------
# Streamline / polyline fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_streamlines(rng: np.random.Generator) -> dict:
    """10 streamlines with 20-50 points each in [0, 200)³.

    Returns dict with keys: polylines, chunk_shape.
    """
    polylines: list[np.ndarray] = []
    for _ in range(10):
        n_pts = rng.integers(20, 51)
        # Random walk to make spatially coherent streamlines
        start = rng.uniform(0, 180, size=(1, 3)).astype(np.float32)
        steps = rng.normal(0, 2, size=(n_pts - 1, 3)).astype(np.float32)
        pts = np.concatenate([start, start + np.cumsum(steps, axis=0)], axis=0)
        # Clamp to [0, 200)
        pts = np.clip(pts, 0, 199.9).astype(np.float32)
        polylines.append(pts)
    return {
        "polylines": polylines,
        "chunk_shape": (100.0, 100.0, 100.0),
    }


# ---------------------------------------------------------------------------
# Skeleton / tree fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_skeleton(rng: np.random.Generator) -> dict:
    """50-node tree with 5 branch points in [0, 500)³.

    The tree is built as a depth-first structure:
    - Root at node 0
    - 5 branch points each spawning a side branch of ~5 nodes
    - Main trunk of ~25 nodes

    Returns dict with keys: positions, parents, node_attributes, chunk_shape.
    """
    n_nodes = 50
    positions = np.zeros((n_nodes, 3), dtype=np.float32)
    parents = np.zeros(n_nodes, dtype=np.int64)

    # Build a main trunk with branches
    parents[0] = -1  # root
    positions[0] = rng.uniform(0, 100, size=3)

    idx = 1
    branch_points = [0, 10, 20, 30, 40]

    for i in range(1, n_nodes):
        if i in branch_points and i > 0:
            # Branch: parent is a node further back on the trunk
            parents[i] = max(0, i - rng.integers(5, 15))
        else:
            parents[i] = i - 1

        # Position: step from parent
        step = rng.normal(0, 10, size=3).astype(np.float32)
        positions[i] = np.clip(positions[parents[i]] + step, 0, 499.9)

    radius = rng.uniform(0.5, 5.0, size=n_nodes).astype(np.float32)
    compartment = rng.integers(0, 4, size=n_nodes).astype(np.int32)

    return {
        "positions": positions,
        "parents": parents,
        "node_attributes": {"radius": radius, "compartment": compartment},
        "chunk_shape": (250.0, 250.0, 250.0),
    }


# ---------------------------------------------------------------------------
# General graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph(rng: np.random.Generator) -> dict:
    """20-node undirected graph with 40 edges in [0, 200)³.

    Returns dict with keys: positions, edges, edge_attributes, chunk_shape.
    """
    n_nodes = 20
    n_edges = 40
    positions = rng.uniform(0, 200, size=(n_nodes, 3)).astype(np.float32)

    # Random edges (no self-loops, may have duplicates — that's fine for testing)
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edges = np.stack([src, dst], axis=1).astype(np.int64)

    weights = rng.uniform(0, 1, size=len(edges)).astype(np.float32)

    return {
        "positions": positions,
        "edges": edges,
        "edge_attributes": {"weight": weights},
        "chunk_shape": (100.0, 100.0, 100.0),
    }


# ---------------------------------------------------------------------------
# Mesh fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_mesh(rng: np.random.Generator) -> dict:
    """Icosahedron-like mesh: 42 vertices, 80 triangles in [0, 100)³.

    Built by subdividing an octahedron once. Positions scaled and shifted
    to fit in [0, 100)³.

    Returns dict with keys: vertices, faces, chunk_shape.
    """
    # Start with octahedron
    verts = np.array([
        [0, 0, 1], [0, 0, -1], [1, 0, 0],
        [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    ], dtype=np.float64)
    tris = np.array([
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ], dtype=np.int64)

    # Subdivide once
    edge_midpoints: dict[tuple[int, int], int] = {}
    new_verts = list(verts)
    new_tris = []

    def get_midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in edge_midpoints:
            return edge_midpoints[key]
        mid = (new_verts[a] + new_verts[b]) / 2.0
        mid = mid / np.linalg.norm(mid)  # project to unit sphere
        idx = len(new_verts)
        new_verts.append(mid)
        edge_midpoints[key] = idx
        return idx

    for tri in tris:
        a, b, c = tri
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_tris.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca],
        ])

    vertices = np.array(new_verts, dtype=np.float32)
    faces = np.array(new_tris, dtype=np.int64)

    # Scale to [10, 90]³ (away from chunk boundaries for cleaner base tests)
    vertices = vertices * 40 + 50

    return {
        "vertices": vertices,
        "faces": faces,
        "chunk_shape": (50.0, 50.0, 50.0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_arrays_close():
    """Fixture returning a helper to compare arrays with tolerance."""
    def _assert(a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> None:
        np.testing.assert_allclose(a, b, atol=atol)
    return _assert


@pytest.fixture
def assert_arrays_equal():
    """Fixture returning a helper to compare integer/exact arrays."""
    def _assert(a: np.ndarray, b: np.ndarray) -> None:
        np.testing.assert_array_equal(a, b)
    return _assert
