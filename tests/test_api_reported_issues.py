"""Regression tests for reported defects in the high-level ``api`` layer."""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.exceptions import MetadataError, StoreError

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


# ---------------- Schema.position_dtype ----------------


@pytest.mark.parametrize("kind,write", [
    ("point_cloud", lambda ds, p: ds.add_points(p)),
    ("polyline", lambda ds, p: ds.add_polylines([p[:5], p[5:9]])),
    ("mesh", lambda ds, p: ds.add_mesh(p[:3], np.array([[0, 1, 2]]))),
    ("graph", lambda ds, p: ds.add_graph(p[:5], np.array([[0, 1]]))),
])
def test_position_dtype_float64_is_honoured(tmp_path, kind, write):
    pts = np.random.default_rng(1).uniform(0, 100, (50, 3))
    ds = zv.create(
        tmp_path / "s.zv",
        schema=zv.Schema(bounds=BOUNDS, kind=kind, position_dtype="float64"),
    )
    write(ds, pts)
    got = ds.read().positions
    assert got.dtype == np.float64
    if kind == "point_cloud":
        # Full precision, not float32 widened back.
        order = np.lexsort(got.T)
        want = pts[np.lexsort(pts.T)]
        assert np.array_equal(got[order], want)


# ---------------- add_* layout= and repeat writes ----------------


def _store(tmp_path, kind="point_cloud", cells=4):
    return zv.create(
        tmp_path / "s.zv",
        schema=zv.Schema(bounds=BOUNDS, kind=kind, layout=zv.Layout(cells=cells)),
    )


def test_layout_with_a_different_grid_is_refused(tmp_path):
    ds = _store(tmp_path)
    with pytest.raises(zv.SchemaConflict, match="cell size"):
        ds.add_points(_pts(), layout=zv.Layout(cells=8))
    assert 0 not in ds.levels or ds.level(0).vertex_count == 0


def test_layout_that_only_changes_storage_is_written_on_the_store_grid(tmp_path):
    ds = _store(tmp_path)
    pts = _pts()
    ds.add_points(pts, layout=zv.Layout(compression="zstd"))
    lo, hi = np.zeros(3), np.full(3, 50.0)
    want = int(np.all((pts >= lo) & (pts <= hi), axis=1).sum())
    assert len(ds.select(bbox=(lo, hi)).read().positions) == want


def test_layout_matching_the_store_grid_is_accepted(tmp_path):
    ds = _store(tmp_path)
    ds.add_points(_pts(), layout=zv.Layout(cells=4))
    assert ds.level(0).vertex_count == 1000


@pytest.mark.parametrize("second", [
    lambda ds: ds.add_points(_pts(500, seed=1)),
    lambda ds: ds.add_points(_pts(500, 50, 100, seed=2)),
])
def test_a_second_add_points_is_refused_and_leaves_the_first(tmp_path, second):
    ds = _store(tmp_path)
    ds.add_points(_pts())
    with pytest.raises(StoreError, match="cannot append"):
        second(ds)
    assert len(ds.read().positions) == 1000
    assert ds.level(0).vertex_count == 1000


def test_a_second_add_polylines_is_refused(tmp_path):
    ds = _store(tmp_path, kind="polyline")
    lines = [_pts(5, seed=i) for i in range(3)]
    ds.add_polylines(lines)
    with pytest.raises(StoreError, match="cannot append"):
        ds.add_polylines(lines)
    assert ds.read().part_count == 3
