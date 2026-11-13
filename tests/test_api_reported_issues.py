"""Regression tests for reported defects in the high-level ``api`` layer."""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv

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
