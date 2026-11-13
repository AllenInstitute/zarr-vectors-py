"""Regression tests for reported defects in the high-level ``api`` layer."""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.exceptions import MetadataError

BOUNDS = ((0.0, 0.0, 0.0), (100.0, 100.0, 100.0))


def _pts(n: int = 1000, lo: float = 0.0, hi: float = 100.0, seed: int = 0):
    return np.random.default_rng(seed).uniform(lo, hi, (n, 3)).astype("f4")


# ---------------- Grid.capacity ----------------


def test_capacity_divides_by_covered_cells_not_allocated_planes():
    # target_cells=2 allocates 3 planes per axis (the upper bound sits on a
    # boundary), but the data's volume covers 2x2x2 cells' worth.
    grid = zv.Grid.plan(BOUNDS, target_cells=2)
    assert grid.shape == (3, 3, 3)
    cap = grid.capacity(n_vertices=8_000_000)
    assert cap.est_bytes_per_cell == pytest.approx(8_000_000 * 12 / 8)
    assert cap.cells == 27


def test_capacity_without_extent_uses_allocation():
    grid = zv.Grid(shape=(2, 2, 2), cell_shape=(1.0, 1.0, 1.0))
    cap = grid.capacity(n_vertices=800)
    assert cap.est_bytes_per_cell == pytest.approx(800 * 12 / 8)


# ---------------- stability manifest vs exports ----------------


@pytest.mark.parametrize("name", [
    "zarr_vectors", "zarr_vectors.open", "zarr_vectors.stability",
    "zarr_vectors.runtime_capabilities", "zarr_vectors.concurrency_contract",
    "zarr_vectors.Dataset",
])
def test_top_level_exports_are_supported(name):
    assert zv.stability(name) == "supported"


def test_every_name_in_dunder_all_is_supported():
    for name in zv.__all__:
        assert zv.stability(f"zarr_vectors.{name}") == "supported", name


def test_retired_top_level_reexports_are_internal():
    assert zv.stability("zarr_vectors.rechunk_by_attribute") == "internal"
    assert zv.stability("zarr_vectors.FsGroup") == "internal"


def test_a_missing_name_under_a_supported_prefix_raises():
    with pytest.raises(KeyError, match="no module or exported name"):
        zv.stability("zarr_vectors.api.nonexistent")
    assert zv.stability("zarr_vectors.api.Dataset") == "supported"


# ---------------- Schema(kind=...) ----------------


def test_schema_rejects_an_unknown_kind():
    with pytest.raises(MetadataError, match="not a geometry type"):
        zv.Schema(kind="banana")


@pytest.mark.parametrize("kind", [
    "point_cloud", "line", "polyline", "streamline", "skeleton", "graph", "mesh",
])
def test_schema_accepts_every_geometry_type(kind):
    assert zv.Schema(kind=kind).kind == kind
