"""Sharded arrays read the same through every reader, with either sharding codec.

``ZARR_VECTORS_SHARD_READER`` picks how sharded arrays are read: ``standard``
(the default: as zarr-vectors always has), ``optimised`` (opt-in: each shard
index once, coalesced ranges, any store) or ``zarr`` (zarr's array API only).
A store marked octree at construction (``Layout(octree=...)``) writes its
shards with zarr-octree's ``zarr_octree_sharding`` instead of
``sharding_indexed``, and :func:`zarr_vectors.sharding.set_octree` converts
either way.  zarr-octree is optional: an unmarked store never imports it.
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
CODECS = {  # name: Layout(octree=...)
    "sharding_indexed": False,
    "zarr_octree_sharding": {"target": 2_000},
    "zarr_octree_runs": {"layout": "runs", "min_weight": 500, "max_weight": 2_000},
}
READERS = ["standard", "optimised", "zarr"]


def _points(n=3_000, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(20, 80, (3, 3))
    pts = centres[rng.integers(0, 3, n)] + rng.standard_t(3, (n, 3)) * 6
    pts = np.clip(pts, 0, BOX - 1e-3).astype(np.float32)
    return pts, np.arange(n) // 50


def _schema(octree=False, kind="point_cloud"):
    return zv.Schema(bounds=([0, 0, 0], [BOX, BOX, BOX]), kind=kind,
                     layout=zv.Layout(cells=8, shard_cells=4, compression="zstd", octree=octree))


def _codecs(path):
    """{array path: codec name} for every sharded array of the store."""
    out = {}
    for z in path.rglob("zarr.json"):
        m = json.loads(z.read_text())
        if m.get("node_type") == "array" and m["codecs"][0]["name"] in ("sharding_indexed", "zarr_octree_sharding"):
            out[str(z.parent.relative_to(path))] = m["codecs"][0]["name"]
    return out


def _root(path):
    return json.loads((path / "zarr.json").read_text())["attributes"]["zarr_vectors"]


def _write(path, codec):
    if codec != "sharding_indexed" and not HAVE_OCTREE:
        pytest.skip("zarr-octree is not installed")
    name = "zarr_octree_sharding" if codec.startswith("zarr_octree") else codec
    pts, ids = _points()
    ds = zv.create(str(path), schema=_schema(CODECS[codec]))
    ds.add_points(pts, object_ids=ids)
    assert set(_codecs(path).values()) == {name}
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
@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("codec", list(CODECS))
def test_queries_match_plain_zarr_reads(tmp_path, monkeypatch, codec, reader, where):
    """Every codec and reader returns what ``sharding_indexed`` read through zarr returns."""
    pts, ids = _write(tmp_path / "plain.zv", "sharding_indexed")
    _write(tmp_path / "s.zv", codec)
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
    _write(tmp_path / "s.zv", codec)
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
    _write(tmp_path / "s.zv", codec)
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "optimised")

    async def _no(*a, **k):
        raise AssertionError("a sharded cell went through zarr's per-cell read")

    monkeypatch.setattr(_batch_reader, "_async_get_sharded_cell", _no)
    ds = _open(tmp_path / "s.zv", "store")
    assert len(ds.select(bbox=([0, 0, 0], [BOX, BOX, BOX])).read().positions) == 3_000


@pytest.mark.parametrize("codec", list(CODECS))
def test_standard_reader_reads_store_backed_shards_through_zarr_as_before(tmp_path, monkeypatch, codec):
    """The default: one range per cell, and a store that is not local read
    through zarr, never by the optimised reader."""
    _write(tmp_path / "s.zv", codec)
    monkeypatch.delenv("ZARR_VECTORS_SHARD_READER", raising=False)
    monkeypatch.setenv("ZARR_VECTORS_COALESCE", "gap:1000000:100000000")  # ignored by the standard reader
    assert _batch_reader.shard_reader() == "standard"
    assert _cells_on_disk._coalescing()[0] == "none"
    seen = []
    real = _cells_on_disk.read_cells
    monkeypatch.setattr(_cells_on_disk, "read_cells", lambda reqs: seen.append([s.local_root for s, _ in reqs]) or real(reqs))
    for where in ("store", "local"):
        assert len(_open(tmp_path / "s.zv", where).select(bbox=([0, 0, 0], [BOX, BOX, BOX])).read().positions) == 3_000
    assert all(root is not None for batch in seen for root in batch)  # only ever local files


def test_an_unknown_reader_is_an_error(monkeypatch):
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "fast")
    with pytest.raises(ValueError, match="ZARR_VECTORS_SHARD_READER"):
        _batch_reader.shard_reader()
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "optimized")
    assert _batch_reader.shard_reader() == "optimised"
    monkeypatch.delenv("ZARR_VECTORS_SHARD_READER")
    assert _batch_reader.shard_reader() == "standard"


def test_an_unmarked_store_carries_no_marker(tmp_path):
    _write(tmp_path / "s.zv", "sharding_indexed")
    root = _root(tmp_path / "s.zv")
    assert "octree_sharding" not in root and "octree_sharding" not in root["format_capabilities"]
    assert zv.Schema.from_store(zv.open(str(tmp_path / "s.zv"))._root_meta).layout.octree is False


@pytest.mark.skipif(not HAVE_OCTREE, reason="zarr-octree is not installed")
@pytest.mark.parametrize("kind", ["point_cloud", "mesh", "graph"])
def test_a_marked_store_writes_every_sharded_array_octree(tmp_path, kind):
    """Every sharded array, the links family's included, and those a later write allocates."""
    rng = np.random.default_rng(1)
    p = tmp_path / "s.zv"
    ds = zv.create(str(p), schema=_schema({"target": 2_000}, kind))
    pos = rng.uniform(5, 95, (600, 3)).astype(np.float32)
    if kind == "point_cloud":
        ds.add_points(pos, object_ids=np.arange(600) // 50)
        zv.open(str(p), mode="r+").add_points(pos + 1, object_ids=np.arange(600) // 50 + 12)  # a later write
    elif kind == "mesh":
        faces = np.stack([np.arange(0, 597, 3), np.arange(1, 598, 3), np.arange(2, 599, 3)], 1)
        ds.add_mesh(pos, faces, object_ids=np.arange(600) // 60)
    else:
        edges = np.stack([np.arange(599), np.arange(1, 600)], 1)
        ds.add_graph(pos, edges, object_ids=np.zeros(600, dtype=np.int64))
    codecs = _codecs(p)
    assert codecs and set(codecs.values()) == {"zarr_octree_sharding"}, codecs
    if kind != "point_cloud":
        assert any("/links/" in a for a in codecs)
    root = _root(p)
    assert root["octree_sharding"] == {"target": 2_000} and "octree_sharding" in root["format_capabilities"]
    assert zv.Schema.from_store(zv.open(str(p))._root_meta).layout.octree == {"target": 2_000}


def test_octree_needs_a_sharded_store(tmp_path):
    with pytest.raises(Exception, match="sharded"):
        zv.create(str(tmp_path / "s.zv"), schema=zv.Schema(
            bounds=([0, 0, 0], [BOX, BOX, BOX]), kind="point_cloud",
            layout=zv.Layout(cells=8, shard_cells=None, octree=True)))


@pytest.mark.skipif(not HAVE_OCTREE, reason="zarr-octree is not installed")
@pytest.mark.parametrize("kind", ["point_cloud", "mesh"])
def test_set_octree_converts_both_ways_and_back_is_the_unmarked_store(tmp_path, kind):
    """Unmarked -> octree -> unmarked: the same answers throughout, and the
    store converted back is byte for byte the store written unmarked."""
    import filecmp
    import shutil

    from zarr_vectors.sharding import set_octree

    rng = np.random.default_rng(2)
    pos = rng.uniform(5, 95, (900, 3)).astype(np.float32)
    faces = np.stack([np.arange(0, 897, 3), np.arange(1, 898, 3), np.arange(2, 899, 3)], 1)
    a = tmp_path / "a.zv"
    ds = zv.create(str(a), schema=_schema(False, kind))
    if kind == "point_cloud":
        ds.add_points(pos, object_ids=np.arange(900) // 30)
    else:
        ds.add_mesh(pos, faces, object_ids=np.arange(900) // 90)
    b = tmp_path / "b.zv"
    shutil.copytree(a, b)
    want = _answers(zv.open(str(a)))

    out = set_octree(str(b), {"target": 3_000})
    assert out["arrays_repacked"] == len(_codecs(a)) > 0
    assert set(_codecs(b).values()) == {"zarr_octree_sharding"}
    assert _root(b)["octree_sharding"] == {"target": 3_000}
    got = _answers(zv.open(str(b)))
    assert all(np.array_equal(x, y) for x, y in zip(got[0], want[0])) and np.array_equal(got[1], want[1])
    assert set_octree(str(b), {"target": 3_000})["arrays_repacked"] == 0  # re-running finishes, never redoes

    set_octree(str(b), False)
    for f in (p.relative_to(a) for p in a.rglob("*") if p.is_file()):
        if str(f) == "zarr.json":  # the root names its store
            ra, rb = json.loads((a / f).read_text()), json.loads((b / f).read_text())
            ra["attributes"]["ome"].pop("name", None), rb["attributes"]["ome"].pop("name", None)
            assert ra == rb
        else:
            assert filecmp.cmp(a / f, b / f, shallow=False), f
    assert sorted(p.relative_to(a) for p in a.rglob("*")) == sorted(p.relative_to(b) for p in b.rglob("*"))


def test_set_octree_needs_a_sharded_store(tmp_path):
    from zarr_vectors.sharding import set_octree

    p = tmp_path / "s.zv"
    zv.create(str(p), schema=zv.Schema(bounds=([0, 0, 0], [BOX, BOX, BOX]), kind="point_cloud",
                                       layout=zv.Layout(cells=8, shard_cells=None))).add_points(*_points()[:1])
    with pytest.raises(ValueError, match="sharded"):
        set_octree(str(p))


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
    for reader in ("standard", "zarr", "optimised"):
        os.environ["ZARR_VECTORS_SHARD_READER"] = reader
        counts[reader] = len(zv.open(root + "/a.zv").select(bbox=([0, 0, 0], [50, 50, 50])).read().positions)
    marked = zv.Schema(bounds=([0, 0, 0], [100, 100, 100]), kind="point_cloud",
                       layout=zv.Layout(cells=8, shard_cells=4, compression="zstd", octree=True))
    try:
        zv.create(root + "/b.zv", schema=marked).add_points(pts, object_ids=np.arange(2_000) // 50)
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
    assert out["counts"] == {"standard": out["want"], "zarr": out["want"], "optimised": out["want"]}
    assert not out["imported"]
    assert out["asked"].startswith("ValueError") and "pip install zarr-octree" in out["asked"], out["asked"]
