"""Regression tests for reported defects in the high-level ``api`` layer."""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.exceptions import ArrayError, MetadataError, StoreError

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


# ---------------- clear errors for malformed input ----------------


def _new(tmp_path, kind="point_cloud", name="s.zv"):
    return zv.create(tmp_path / name, schema=zv.Schema(bounds=BOUNDS, kind=kind))


def test_select_objects_on_a_store_without_ids_says_so(tmp_path):
    from zarr_vectors.api.select import SelectionError

    ds = _new(tmp_path)
    ds.add_points(_pts())
    with pytest.raises(SelectionError, match="holds no object ids"):
        ds.select(objects=[1]).read()
    with pytest.raises(SelectionError, match="holds no groups"):
        ds.select(groups=[1]).read()


def test_select_objects_still_works_with_ids(tmp_path):
    ds = _new(tmp_path)
    ds.add_points(_pts(), object_ids=np.arange(1000) % 5)
    assert ds.select(objects=[1]).read().vertex_count == 200


@pytest.mark.parametrize("value", [
    np.zeros(15, "f4"), np.zeros(3, "f4"), [1.0, 2.0, 3.0],
])
def test_flat_polyline_attribute_is_explained(tmp_path, value):
    ds = _new(tmp_path, "polyline")
    lines = [_pts(5, seed=i) for i in range(3)]
    with pytest.raises(ArrayError, match="one array per polyline"):
        ds.add_polylines(lines, attributes={"a": value})


def test_ragged_polyline_attribute_with_a_short_part(tmp_path):
    ds = _new(tmp_path, "polyline")
    lines = [_pts(5, seed=i) for i in range(3)]
    bad = [np.zeros(5, "f4"), np.zeros(5, "f4"), np.zeros(2, "f4")]
    with pytest.raises(ArrayError, match=r"\[2\] has 2 rows"):
        ds.add_polylines(lines, attributes={"a": bad})
    good = [np.zeros(5, "f4")] * 3
    _new(tmp_path, "polyline", "ok.zv").add_polylines(lines, attributes={"a": good})


@pytest.mark.parametrize("call", [
    lambda ds: ds.add_points(_pts(), attributes={"a": np.zeros(10, "f4")}),
    lambda ds: ds.add_points(_pts(), attributes={"a": np.zeros(2000, "f4")}),
    lambda ds: ds.add_points(_pts(), object_ids=np.zeros(10, int)),
    lambda ds: ds.add_points(
        _pts(), object_ids=np.arange(1000) % 5,
        object_attributes={"a": np.zeros(2, "f4")},
    ),
])
def test_wrong_length_point_arrays_are_explained(tmp_path, call):
    with pytest.raises(ArrayError, match="exactly one row per item"):
        call(_new(tmp_path))


def test_wrong_length_mesh_and_graph_arrays_are_explained(tmp_path):
    p = _pts(6)
    with pytest.raises(ArrayError, match="6 vertices"):
        _new(tmp_path, "mesh").add_mesh(
            p, np.array([[0, 1, 2], [3, 4, 5]]), attributes={"a": np.zeros(2, "f4")},
        )
    with pytest.raises(ArrayError, match="2 edges"):
        _new(tmp_path, "graph", "g.zv").add_graph(
            p, np.array([[0, 1], [2, 3]]), edge_attributes={"a": np.zeros(1, "f4")},
        )


def test_string_group_key_is_explained(tmp_path):
    with pytest.raises(TypeError, match="integer group ids"):
        _new(tmp_path).add_points(
            _pts(), object_ids=np.arange(1000) % 5, groups={"g": [1, 2]},
        )


def test_out_of_bounds_message_names_the_api_keyword(tmp_path):
    with pytest.raises(MetadataError, match="on_out_of_bounds="):
        _new(tmp_path).add_points(_pts() + 200)


# ---------------- Query.object_ids for every geometry ----------------

_G = np.array(
    [[10, 10, 10], [11, 11, 11], [12, 12, 12],
     [80, 80, 80], [81, 81, 81], [90, 90, 90]], "f4",
)
_LOW = ((0, 0, 0), (50, 50, 50))


def _gridded(tmp_path, kind):
    return zv.create(
        tmp_path / f"{kind}.zv",
        schema=zv.Schema(bounds=BOUNDS, kind=kind, layout=zv.Layout(cells=4)),
    )


def test_object_ids_for_points(tmp_path):
    pts = _pts()
    ids = (pts[:, 0] // 10).astype(int)
    ds = _gridded(tmp_path, "point_cloud")
    ds.add_points(pts, object_ids=ids)
    assert ds.select().object_ids().tolist() == list(range(10))
    lo, hi = np.asarray(_LOW[0]), np.asarray(_LOW[1])
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    assert ds.select(bbox=_LOW).object_ids().tolist() == sorted(set(ids[inside]))


def test_object_ids_for_points_without_ids_is_empty(tmp_path):
    ds = _gridded(tmp_path, "point_cloud")
    ds.add_points(_pts())
    assert ds.select().object_ids().size == 0
    assert ds.select(bbox=_LOW).object_ids().size == 0


@pytest.mark.parametrize("tree", [False, True])
def test_object_ids_for_graphs(tmp_path, tree):
    ds = _gridded(tmp_path, "skeleton" if tree else "graph")
    ds.add_graph(
        _G, np.array([[0, 1], [1, 2], [3, 4], [4, 5]]),
        object_ids=np.array([0, 0, 0, 1, 1, 1]), tree=tree,
    )
    assert ds.select().object_ids().tolist() == [0, 1]
    assert ds.select(bbox=_LOW).object_ids().tolist() == [0]
    assert ds.select(near=((90, 90, 90), 1.0)).object_ids().tolist() == [1]


def test_object_ids_for_meshes(tmp_path):
    ds = _gridded(tmp_path, "mesh")
    ds.add_mesh(_G, np.array([[0, 1, 2], [3, 4, 5]]), object_ids=np.array([0, 0, 0, 1, 1, 1]))
    assert ds.select().object_ids().tolist() == [0, 1]
    assert ds.select(bbox=_LOW).object_ids().tolist() == [0]


def test_object_ids_for_lines(tmp_path):
    ds = _gridded(tmp_path, "line")
    ds.add_lines(_G.reshape(3, 2, 3))
    assert ds.select().object_ids().tolist() == [0, 1, 2]
    # line 1 runs (12,12,12)-(80,80,80): one endpoint in the box.
    assert ds.select(bbox=_LOW).object_ids().tolist() == [0, 1]


# ---------------- multi-channel mesh / graph vertex attributes ----------------


@pytest.mark.parametrize("kind", ["mesh", "graph"])
def test_multichannel_vertex_attribute_is_read_back(tmp_path, kind):
    V = _pts(6)
    colour = np.arange(18, dtype="f4").reshape(6, 3)
    tag = np.arange(6, dtype="f4")
    ds = zv.create(
        tmp_path / "s.zv",
        schema=zv.Schema(bounds=BOUNDS, kind=kind, layout=zv.Layout(cells=3)),
    )
    attrs = {"colour": colour, "tag": tag}
    if kind == "mesh":
        ds.add_mesh(V, np.array([[0, 1, 2], [3, 4, 5]]), attributes=attrs)
    else:
        ds.add_graph(V, np.array([[0, 1]]), attributes=attrs)
    got = ds.read()
    assert set(got.attributes) == set(ds.level(0).attribute_names()) == {"colour", "tag"}
    order = got.attributes["tag"].astype(int)
    assert got.attributes["colour"].shape == (6, 3)
    assert np.array_equal(got.attributes["colour"], colour[order])
