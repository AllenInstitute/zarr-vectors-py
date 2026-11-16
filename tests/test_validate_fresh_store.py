"""A store created empty validates before its first write.

``api.create`` / ``create_store`` declare no ``links_convention``; the
first links write stamps the geometry's own.  RootMetadata reads the
absent key as ``implicit_sequential``, so level 4 used to fail a fresh
``Schema(kind="graph")`` store (and skeleton, mesh) until ``add_graph``.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

import zarr_vectors.api as zv
from zarr_vectors.validate import validate


@pytest.mark.parametrize("kind", ["graph", "skeleton", "mesh", "polyline", "point_cloud"])
def test_fresh_store_validates(tmp_path, kind):
    path = str(tmp_path / "s.zv")
    zv.create(path, schema=zv.Schema(kind=kind, bounds=([0] * 3, [100] * 3)))
    r = validate(path, level=5)
    assert r.ok, r.summary()


def test_graph_validates_after_add_graph(tmp_path):
    path = str(tmp_path / "g.zv")
    ds = zv.create(path, schema=zv.Schema(kind="graph", bounds=([0] * 3, [100] * 3)))
    pos = np.array([[1, 1, 1], [20, 20, 20], [60, 60, 60]], "f4")
    ds.add_graph(pos, np.array([[0, 1], [1, 2]]))
    assert zarr.open_group(path).attrs["zarr_vectors"]["links_convention"] == "explicit"
    r = validate(path, level=5)
    assert r.ok, r.summary()


def test_graph_with_links_and_wrong_declared_convention_still_fails(tmp_path):
    path = str(tmp_path / "g.zv")
    ds = zv.create(path, schema=zv.Schema(kind="graph", bounds=([0] * 3, [100] * 3)))
    pos = np.array([[1, 1, 1], [20, 20, 20], [60, 60, 60]], "f4")
    ds.add_graph(pos, np.array([[0, 1], [1, 2]]))
    g = zarr.open_group(path, mode="r+")
    block = dict(g.attrs["zarr_vectors"])
    block["links_convention"] = "implicit_sequential"
    g.attrs["zarr_vectors"] = block
    r = validate(path, level=4)
    assert any("'graph' requires links" in e for e in r.errors), r.summary()


def test_graph_with_links_and_no_declared_convention_still_fails(tmp_path):
    path = str(tmp_path / "g.zv")
    ds = zv.create(path, schema=zv.Schema(kind="graph", bounds=([0] * 3, [100] * 3)))
    pos = np.array([[1, 1, 1], [20, 20, 20], [60, 60, 60]], "f4")
    ds.add_graph(pos, np.array([[0, 1], [1, 2]]))
    g = zarr.open_group(path, mode="r+")
    block = dict(g.attrs["zarr_vectors"])
    del block["links_convention"]
    g.attrs["zarr_vectors"] = block
    r = validate(path, level=4)
    assert any("'graph' requires links" in e for e in r.errors), r.summary()
