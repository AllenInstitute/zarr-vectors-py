"""Regressions for the lazy layer's leftovers (TODO.md section 1).

* The lazy readers decoded vertices as float32 whatever the level declares,
  so a float64 level read back as garbage at twice the row count.
* ``add_face_attribute`` always raised: it wrote a ``face_attributes/``
  tree no reader knew, created as a group.  Face attributes now go where
  every per-record attribute goes, ``link_attributes/<name>/0/``.
* ``append_vertices`` wrote rank-3 keys into an attribute-chunked level.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from zarr_vectors.core.arrays import (
    list_chunk_keys,
    read_link_attributes,
    read_links,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.lazy import open_zv
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import write_polylines

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a[np.lexsort(a.T[::-1])]


# ===================================================================
# 1. float64 levels through every lazy read path
# ===================================================================


@pytest.fixture
def f64_points(tmp_path):
    """A binned float64 level (one fragment per bin, so the view's
    bbox read goes through the per-bin ``read_fragment`` path)."""
    rng = np.random.default_rng(7)
    # Values float32 cannot hold exactly.
    pos = rng.uniform(0, 100, (150, 3)) + 1e-9
    store = str(tmp_path / "f64.zv")
    write_points(
        store, pos, dtype="float64", chunk_shape=(50.0, 50.0, 50.0),
        bin_shape=(25.0, 25.0, 25.0),
    )
    return store, pos


@pytest.fixture
def f64_objects(tmp_path):
    """An unbinned float64 level with one object per vertex."""
    rng = np.random.default_rng(8)
    pos = rng.uniform(0, 100, (90, 3)) + 1e-9
    store = str(tmp_path / "f64o.zv")
    write_points(
        store, pos, dtype="float64", chunk_shape=(50.0, 50.0, 50.0),
        object_ids=np.arange(90),
    )
    return store, pos


def test_level_vertices_float64_exact(f64_points):
    store, pos = f64_points
    got = open_zv(store)[0].vertices.compute()
    assert got.dtype == np.float64
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(pos))


def test_view_spatial_read_float64_exact(f64_points):
    store, pos = f64_points
    lo, hi = np.zeros(3), np.full(3, 100.0)
    got = open_zv(store)[0].filter(bbox=(lo, hi)).compute()["positions"]
    assert got.dtype == np.float64
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(pos))


def test_view_chunk_read_float64_exact(f64_objects):
    # No bins: the view reads whole chunks rather than single bins.
    store, pos = f64_objects
    lo, hi = np.zeros(3), np.full(3, 100.0)
    got = open_zv(store)[0].filter(bbox=(lo, hi)).compute()["positions"]
    assert got.dtype == np.float64
    np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(pos))


def test_view_object_read_float64_exact(f64_objects):
    store, pos = f64_objects
    out = open_zv(store)[0].filter(object_ids=[3, 40, 89]).compute()
    assert out["positions"].dtype == np.float64
    for oid in (3, 40, 89):
        np.testing.assert_array_equal(
            out["positions"][out["object_ids"] == oid], pos[[oid]],
        )


def test_polyline_read_float64_exact(tmp_path):
    rng = np.random.default_rng(9)
    lines = [rng.uniform(0, 100, (n, 3)) + 1e-9 for n in (5, 12, 30)]
    store = str(tmp_path / "f64l.zv")
    write_polylines(store, lines, dtype="float64", chunk_shape=(200.0,) * 3)
    polys = open_zv(store)[0].polylines
    for oid, line in enumerate(lines):
        got = polys[oid].compute()
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, line)


# ===================================================================
# 2. add_face_attribute
# ===================================================================


def _mesh(tmp_path):
    rng = np.random.default_rng(10)
    verts = rng.uniform(0, 100, (60, 3)).astype("f4")
    faces = np.stack([rng.permutation(60)[:3] for _ in range(80)])
    store = str(tmp_path / "mesh.zv")
    write_mesh(store, verts, faces, chunk_shape=(50.0, 50.0, 50.0))
    return store


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_add_face_attribute_round_trip(tmp_path, mode):
    import asyncio

    store = _mesh(tmp_path)
    lg = get_resolution_level(open_store(store), 0)
    n_faces = len(read_links(lg, delta=0))
    assert n_faces == 80
    values = np.arange(n_faces, dtype=np.float64) * 0.5

    w = open_zv(store)[0].writer()
    if mode == "sync":
        w.add_face_attribute_sync("area", values)
    else:
        asyncio.run(w.add_face_attribute("area", values))

    lg = get_resolution_level(open_store(store), 0)
    got = read_link_attributes(lg, "area", delta=0)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, values)
    assert not lg.array_exists("face_attributes")
    lv = lg.attrs.to_dict()["zarr_vectors_level"]
    assert "link_attributes" in lv["arrays_present"]


def test_add_face_attribute_vector_and_dtype(tmp_path):
    store = _mesh(tmp_path)
    values = np.arange(80 * 3, dtype=np.float64).reshape(80, 3)
    open_zv(store)[0].writer().add_face_attribute_sync(
        "normal", values, dtype="float32",
    )
    lg = get_resolution_level(open_store(store), 0)
    got = read_link_attributes(lg, "normal", delta=0)
    assert got.dtype == np.float32 and got.shape == (80, 3)
    np.testing.assert_array_equal(got, values.astype(np.float32))


def test_add_face_attribute_length_mismatch_raises(tmp_path):
    store = _mesh(tmp_path)
    with pytest.raises(ArrayError, match="face count"):
        open_zv(store)[0].writer().add_face_attribute_sync(
            "area", np.zeros(79),
        )


def test_add_face_attribute_without_faces_raises(tmp_path):
    store = str(tmp_path / "p.zv")
    write_points(store, np.zeros((4, 3), "f4"), chunk_shape=(50.0,) * 3)
    with pytest.raises(ArrayError, match="no links/0"):
        open_zv(store)[0].writer().add_face_attribute_sync("a", np.zeros(4))


# ===================================================================
# 3. append_vertices on an attribute-chunked level
# ===================================================================


def _attr_store(tmp_path):
    rng = np.random.default_rng(11)
    pos = rng.uniform(0, 100, (40, 3)).astype("f4")
    cell = np.array(["a", "b"] * 20)
    store = str(tmp_path / "attr.zv")
    write_points(
        store, pos, chunk_shape=(50.0, 50.0, 50.0),
        vertex_attributes={"cell": cell}, chunk_by_attribute="cell",
        object_ids=np.arange(40),
    )
    return store, pos, cell


def test_append_vertices_attribute_chunked(tmp_path):
    store, pos, cell = _attr_store(tmp_path)
    new = np.array([[10.0, 10.0, 10.0], [60.0, 60.0, 60.0], [70.0, 5.0, 90.0]],
                   dtype="f4")
    new_cell = np.array(["b", "a", "b"])
    w = open_zv(store)[0].writer()
    out = w.append_vertices_sync(new, chunk_attribute=new_cell)
    w.commit_sync()
    assert out["vertices_added"] == 3

    lg = get_resolution_level(open_store(store), 0)
    keys = list_chunk_keys(lg)
    assert keys and all(len(k) == 4 for k in keys)
    for bin_idx, label in enumerate(["a", "b"]):
        got = read_points(store, attribute_filter={"cell": label})["positions"]
        want = np.concatenate([pos[cell == label], new[new_cell == label]])
        np.testing.assert_array_equal(_sorted_rows(got), _sorted_rows(want))
    # The new objects are addressable through the object index.
    got = open_zv(store)[0].filter(object_ids=[40, 41, 42]).compute()
    np.testing.assert_array_equal(
        _sorted_rows(got["positions"]), _sorted_rows(new),
    )


def test_append_vertices_attribute_chunked_requires_values(tmp_path):
    store, _, _ = _attr_store(tmp_path)
    lg = get_resolution_level(open_store(store), 0)
    before = sorted(list_chunk_keys(lg))
    w = open_zv(store)[0].writer()
    with pytest.raises(ArrayError, match="chunk_attribute="):
        w.append_vertices_sync(np.zeros((2, 3), "f4"))
    with pytest.raises(ArrayError, match="cannot add a bin"):
        w.append_vertices_sync(
            np.zeros((2, 3), "f4"), chunk_attribute=np.array(["a", "zzz"]),
        )
    lg = get_resolution_level(open_store(store), 0)
    assert sorted(list_chunk_keys(lg)) == before


def test_append_vertices_spatial_rejects_chunk_attribute(tmp_path):
    store = str(tmp_path / "p.zv")
    write_points(
        store, np.zeros((4, 3), "f4"), chunk_shape=(50.0,) * 3,
        object_ids=np.arange(4),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = open_zv(store)[0].writer()
    with pytest.raises(ArrayError, match="not attribute-chunked"):
        w.append_vertices_sync(
            np.zeros((1, 3), "f4"), chunk_attribute=np.array([1]),
        )
