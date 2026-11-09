"""Sharded arrays read the same through either reader, with either sharding codec.

``ZARR_VECTORS_SHARD_READER`` picks how sharded arrays are read: ``zarr``
(zarr's array API only) or ``optimised`` (each shard index once, coalesced
ranges; the default).  ``ZARR_VECTORS_SHARD_CODEC`` picks the sharding codec
they are written with: ``sharding_indexed`` (the default) or zarr-octree's
``zarr_octree_sharding``, which is optional -- without the zarr-octree package
nothing here imports it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

import zarr_vectors as zv
from zarr_vectors.core import _batch_reader, _cells_on_disk

try:
    import zarr_octree  # noqa: F401

    HAVE_OCTREE = True
except ImportError:
    HAVE_OCTREE = False

BOX = 100.0
CODECS = {
    "sharding_indexed": {},
    "zarr_octree_sharding": {"target": 2_000},
    "zarr_octree_runs": {"layout": "runs", "min_weight": 500, "max_weight": 2_000},
}


def _points(n=3_000, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(20, 80, (3, 3))
    pts = centres[rng.integers(0, 3, n)] + rng.standard_t(3, (n, 3)) * 6
    pts = np.clip(pts, 0, BOX - 1e-3).astype(np.float32)
    return pts, np.arange(n) // 50


def _write(path, codec, monkeypatch):
    if codec != "sharding_indexed" and not HAVE_OCTREE:
        pytest.skip("zarr-octree is not installed")
    name = "zarr_octree_sharding" if codec.startswith("zarr_octree") else codec
    monkeypatch.setenv("ZARR_VECTORS_SHARD_CODEC", name)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_CODEC_CONFIG", json.dumps(CODECS[codec]))
    pts, ids = _points()
    schema = zv.Schema(bounds=([0, 0, 0], [BOX, BOX, BOX]), kind="point_cloud",
                       layout=zv.Layout(cells=8, shard_cells=4, compression="zstd"))
    ds = zv.create(str(path), schema=schema)
    ds.add_points(pts, object_ids=ids)
    meta = json.loads((path / "0" / "vertices" / "zarr.json").read_text())
    assert meta["codecs"][0]["name"] == name
    monkeypatch.delenv("ZARR_VECTORS_SHARD_CODEC")
    monkeypatch.delenv("ZARR_VECTORS_SHARD_CODEC_CONFIG")
    return pts, ids


def _open(path, where):
    if where == "local":
        return zv.open(str(path))
    return zv.open(WrapperStore(LocalStore(str(path), read_only=True)))  # not a LocalStore: the object-store path


def _sorted(p):
    p = np.asarray(p)
    return p[np.lexsort(p.T[::-1])] if len(p) else p


QUERIES = [((10, 10, 10), (60, 60, 60)), ((0, 0, 0), (BOX, BOX, BOX)), ((70, 5, 30), (95, 40, 90))]
OBJECTS = [3, 17, 41]


def _answers(ds):
    boxes = [_sorted(ds.select(bbox=(list(lo), list(hi))).read().positions) for lo, hi in QUERIES]
    return boxes, _sorted(ds.level(0).objects[OBJECTS].positions)


@pytest.mark.parametrize("where", ["local", "store"])
@pytest.mark.parametrize("reader", ["optimised", "zarr"])
@pytest.mark.parametrize("codec", list(CODECS))
def test_queries_match_plain_zarr_reads(tmp_path, monkeypatch, codec, reader, where):
    """Every codec and reader returns what ``sharding_indexed`` read through zarr returns."""
    pts, ids = _write(tmp_path / "plain.zv", "sharding_indexed", monkeypatch)
    _write(tmp_path / "s.zv", codec, monkeypatch)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "zarr")
    want_boxes, want_objects = _answers(_open(tmp_path / "plain.zv", "local"))
    np.testing.assert_array_equal(want_boxes[1], _sorted(pts))  # the whole box: every point
    np.testing.assert_array_equal(want_objects, _sorted(pts[np.isin(ids, OBJECTS)]))
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", reader)
    boxes, objects = _answers(_open(tmp_path / "s.zv", where))
    for got, want in zip(boxes, want_boxes):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(objects, want_objects)


@pytest.mark.parametrize("codec", list(CODECS))
def test_zarr_reader_never_uses_the_optimised_reader(tmp_path, monkeypatch, codec):
    _write(tmp_path / "s.zv", codec, monkeypatch)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "zarr")

    def _no(*a, **k):
        raise AssertionError("the optimised reader ran")

    monkeypatch.setattr(_cells_on_disk, "read_cells", _no)
    monkeypatch.setattr(_cells_on_disk, "read_indexes", _no)
    for where in ("local", "store"):
        ds = _open(tmp_path / "s.zv", where)
        assert len(ds.select(bbox=([0, 0, 0], [BOX, BOX, BOX])).read().positions) == 3_000


@pytest.mark.parametrize("codec", list(CODECS))
def test_optimised_reader_reads_store_backed_shards_itself(tmp_path, monkeypatch, codec):
    _write(tmp_path / "s.zv", codec, monkeypatch)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "optimised")

    async def _no(*a, **k):
        raise AssertionError("a sharded cell went through zarr's per-cell read")

    monkeypatch.setattr(_batch_reader, "_async_get_sharded_cell", _no)
    ds = _open(tmp_path / "s.zv", "store")
    assert len(ds.select(bbox=([0, 0, 0], [BOX, BOX, BOX])).read().positions) == 3_000


def test_an_unknown_reader_is_an_error(monkeypatch):
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "fast")
    with pytest.raises(ValueError, match="ZARR_VECTORS_SHARD_READER"):
        _batch_reader.shard_reader()
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "optimized")
    assert _batch_reader.shard_reader() == "optimised"
    monkeypatch.delenv("ZARR_VECTORS_SHARD_READER")
    assert _batch_reader.shard_reader() == "optimised"


_WITHOUT_OCTREE = textwrap.dedent("""
    import importlib.abc, json, os, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "zarr_octree" or name.startswith("zarr_octree."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    sys.meta_path.insert(0, Block())
    import numpy as np
    import zarr_vectors as zv

    root = sys.argv[1]
    pts = np.random.default_rng(0).uniform(0, 100, (2_000, 3)).astype("float32")
    schema = zv.Schema(bounds=([0, 0, 0], [100, 100, 100]), kind="point_cloud",
                       layout=zv.Layout(cells=8, shard_cells=4, compression="zstd"))
    ds = zv.create(root + "/a.zv", schema=schema)
    ds.add_points(pts, object_ids=np.arange(2_000) // 50)
    counts = {}
    for reader in ("zarr", "optimised"):
        os.environ["ZARR_VECTORS_SHARD_READER"] = reader
        counts[reader] = len(zv.open(root + "/a.zv").select(bbox=([0, 0, 0], [50, 50, 50])).read().positions)
    os.environ["ZARR_VECTORS_SHARD_CODEC"] = "zarr_octree_sharding"
    try:
        zv.create(root + "/b.zv", schema=schema).add_points(pts, object_ids=np.arange(2_000) // 50)
        asked = "written"
    except Exception as exc:
        asked = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"counts": counts, "imported": "zarr_octree" in sys.modules, "asked": asked,
                      "want": int(np.all(pts < 50, axis=1).sum())}))
""")


def test_zarr_octree_is_optional(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZARR_VECTORS_SHARD_")}
    r = subprocess.run([sys.executable, "-c", _WITHOUT_OCTREE, str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["counts"] == {"zarr": out["want"], "optimised": out["want"]}
    assert not out["imported"]
    assert out["asked"].startswith("ValueError") and "pip install zarr-octree" in out["asked"], out["asked"]
