"""Smoke-level performance regression tests.

These cover the hot paths fixed in commit history around the
geometry-type read/write vectorization sweep.  Each upper bound is set
~3× over expected post-fix wall time so the suite catches order-of-
magnitude regressions (e.g. a re-introduced O(N²) loop) without flaking
on slow CI.

Run with ``pytest tests/test_perf_writes.py -q`` — these run as part of
the regular test suite.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.types.graphs import read_graph, write_graph
from zarr_vectors.types.lines import read_lines, write_lines
from zarr_vectors.types.meshes import read_mesh, write_mesh
from zarr_vectors.types.polylines import read_polylines, write_polylines


CHUNK = (200.0, 200.0, 200.0)
BIN = (50.0, 50.0, 50.0)


def _new_store(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"perf_{prefix}_")) / "store.zarrvectors"


def _time(fn, *args, **kwargs) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


# Upper bounds (seconds, local FS).  Set ~3× over expected times measured
# after the vectorization sweep so the gates catch O(N²) regressions
# without flaking under load.  The cross_chunk_links arrays switched to a
# per-tuple cell layout in 0.8 (one tiny Zarr array per chunk-tuple cell);
# write budgets for geometry types with many cross-chunk records are
# loosened to reflect the per-cell write overhead — sharding (planned)
# will amortise this back down.
PERF_BUDGET = {
    "write_lines": 8.0,      # ~5s with thousands of cross-chunk cells
    "read_lines": 8.0,       # ~3-5s walking per-cell reads
    "write_polylines": 6.0,  # measured ~0.6s pre-0.8
    "read_polylines": 6.0,   # ~3s walking per-cell reads
    "write_graph": 12.0,     # ~8-9s with thousands of cross-chunk cells
    "read_graph": 25.0,      # ~17s walking thousands of CCL cells; sharding will amortise
    "write_mesh": 4.0,       # measured ~0.3s pre-0.8
    "read_mesh": 4.0,        # ~1s per-cell reads (mesh has fewer cross-chunk records)
}

# Smaller than the N used by the 02_data_types benchmark notebook (which
# lives in zarr-vectors-tools, benchmarks/internals/) so CI stays under
# 30s total.
N = 10_000


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_perf_write_read_lines(rng: np.random.Generator) -> None:
    endpoints = rng.uniform(0, 1000, (N, 2, 3)).astype(np.float32)
    store = _new_store("lines")
    tw, _ = _time(write_lines, store, endpoints, chunk_shape=CHUNK, bin_shape=BIN)
    tr, _ = _time(read_lines, store)
    assert tw < PERF_BUDGET["write_lines"], (
        f"write_lines too slow: {tw:.3f}s > {PERF_BUDGET['write_lines']}s"
    )
    assert tr < PERF_BUDGET["read_lines"], (
        f"read_lines too slow: {tr:.3f}s > {PERF_BUDGET['read_lines']}s"
    )


def test_perf_write_read_polylines(rng: np.random.Generator) -> None:
    counts = rng.integers(8, 16, size=N // 12)
    polylines = []
    for c in counts:
        steps = rng.normal(0, 5, (c, 3))
        start = rng.uniform(0, 1000, 3)
        polylines.append((start + steps.cumsum(axis=0)).astype(np.float32))
    store = _new_store("polylines")
    tw, _ = _time(
        write_polylines, store, polylines, chunk_shape=CHUNK, bin_shape=BIN,
    )
    tr, _ = _time(read_polylines, store)
    assert tw < PERF_BUDGET["write_polylines"], (
        f"write_polylines too slow: {tw:.3f}s > {PERF_BUDGET['write_polylines']}s"
    )
    assert tr < PERF_BUDGET["read_polylines"], (
        f"read_polylines too slow: {tr:.3f}s > {PERF_BUDGET['read_polylines']}s"
    )


def test_perf_write_read_graph(rng: np.random.Generator) -> None:
    positions = rng.uniform(0, 1000, (N, 3)).astype(np.float32)
    src = rng.integers(0, N, size=3 * N // 2)
    dst = rng.integers(0, N, size=3 * N // 2)
    mask = src != dst
    edges = np.stack([src[mask], dst[mask]], axis=1).astype(np.int64)
    store = _new_store("graph")
    tw, _ = _time(
        write_graph, store, positions, edges,
        chunk_shape=CHUNK, bin_shape=BIN,
    )
    tr, _ = _time(read_graph, store)
    assert tw < PERF_BUDGET["write_graph"], (
        f"write_graph too slow: {tw:.3f}s > {PERF_BUDGET['write_graph']}s"
    )
    assert tr < PERF_BUDGET["read_graph"], (
        f"read_graph too slow: {tr:.3f}s > {PERF_BUDGET['read_graph']}s"
    )


def test_perf_write_read_mesh(rng: np.random.Generator) -> None:
    side = int(np.sqrt(N))
    xs, ys = np.meshgrid(
        np.linspace(0, 1000, side), np.linspace(0, 1000, side),
    )
    zs = rng.uniform(0, 100, (side, side))
    vertices = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    i = np.arange(side - 1)
    j = np.arange(side - 1)
    ii, jj = np.meshgrid(i, j, indexing="ij")
    a = (ii * side + jj).ravel()
    b = a + 1
    c = a + side
    d = c + 1
    faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([b, d, c], axis=1)]
    ).astype(np.int64)
    store = _new_store("mesh")
    tw, _ = _time(
        write_mesh, store, vertices, faces,
        chunk_shape=CHUNK, bin_shape=BIN,
    )
    tr, _ = _time(read_mesh, store)
    assert tw < PERF_BUDGET["write_mesh"], (
        f"write_mesh too slow: {tw:.3f}s > {PERF_BUDGET['write_mesh']}s"
    )
    assert tr < PERF_BUDGET["read_mesh"], (
        f"read_mesh too slow: {tr:.3f}s > {PERF_BUDGET['read_mesh']}s"
    )
