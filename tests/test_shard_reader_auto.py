"""The default shard reader, ``auto``: the optimised reader for every sharded array.

``ZARR_VECTORS_SHARD_READER`` unset (or ``auto``) reads every sharded
array -- ``sharding_indexed`` or zarr-octree's ``octree_sharding`` --
with the optimised reader: each shard's index once, the wanted cells'
ranges coalesced.  Behind an object store the ``standard`` reader goes
through zarr's sharding codec one cell at a time, so the default makes
fewer requests than ``standard`` on either codec and answers the same.

Every request a read makes is recorded -- each ``get`` of a store that is
not local (a :class:`~zarr.storage.WrapperStore`), and on a local store
each ``LocalStore.get`` and each file range, shard index and whole file
zarr-vectors reads itself -- and compared between the default and an
explicit reader.  ``standard``, ``optimised`` and ``zarr`` keep their
meaning when set explicitly.
"""

from __future__ import annotations

import collections
import shutil

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

import zarr_vectors as zv
from zarr_vectors.constants import OCTREE_CODEC_NAMES
from zarr_vectors.core import _batch_reader, _cells_on_disk

try:
    import zarr_octree  # noqa: F401

    HAVE_OCTREE = True
except ImportError:
    HAVE_OCTREE = False

needs_octree = pytest.mark.skipif(not HAVE_OCTREE, reason="zarr-octree is not installed")

BOX = 100.0
TARGET = {"target": 2_000}
OBJECTS = [3, 17, 41]
CELLS = np.stack(np.meshgrid(*[np.arange(2, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def _schema(kind, octree):
    return zv.Schema(bounds=([0, 0, 0], [BOX, BOX, BOX]), kind=kind,
                     layout=zv.Layout(cells=8, shard_cells=4, compression="zstd", octree=octree))


def _write(path, kind="point_cloud", octree=False, n=3_000):
    rng = np.random.default_rng(0)
    centres = rng.uniform(20, 80, (3, 3))
    pos = centres[rng.integers(0, 3, n)] + rng.standard_t(3, (n, 3)) * 6
    pos = np.clip(pos, 0, BOX - 1e-3).astype(np.float32)
    ds = zv.create(str(path), schema=_schema(kind, octree))
    if kind == "point_cloud":
        ds.add_points(pos, object_ids=np.arange(n) // 50)
    elif kind == "mesh":
        faces = np.stack([np.arange(0, n - 2, 3), np.arange(1, n - 1, 3), np.arange(2, n, 3)], 1)
        ds.add_mesh(pos, faces, object_ids=np.arange(n) // 60)
    else:
        edges = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
        edges = edges[(edges[:, 0] + 1) % 50 != 0]  # a chain per object
        ds.add_graph(pos, edges, object_ids=np.arange(n) // 50)
    return path


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """{"indexed": path, "octree": path}: the same point cloud, either codec."""
    root = tmp_path_factory.mktemp("auto")
    out = {"indexed": _write(root / "indexed.zv")}
    if HAVE_OCTREE:
        out["octree"] = _write(root / "octree.zv", octree=TARGET)
    return out


def _open(path, where, log=None):
    if where == "local":
        return zv.open(str(path))
    return zv.open(_Counting(LocalStore(str(path), read_only=True), log))  # not a LocalStore


# ---------------------------------------------------------------------------
# Recording every request
# ---------------------------------------------------------------------------


class _Counting(WrapperStore):
    """A store that is not a LocalStore, noting each ``get`` while recording."""

    def __init__(self, store, log=None):
        super().__init__(store)
        self.log = log

    def _with_store(self, store):
        return type(self)(store, self.log)

    async def get(self, key, prototype, byte_range=None):
        buf = await self._store.get(key, prototype, byte_range)
        _note("store get", key, byte_range, buf)
        return buf

    async def get_partial_values(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        bufs = await self._store.get_partial_values(prototype, key_ranges)
        for (key, byte_range), buf in zip(key_ranges, bufs):
            _note("store get", key, byte_range, buf)
        return bufs


_LOG: list | None = None


def _note(kind, key, where, data):
    if _LOG is not None:
        n = len(data) if isinstance(data, (bytes, bytearray)) else (0 if data is None or isinstance(data, Exception) else len(data))
        _LOG.append((kind, str(key), repr(where), n))


@pytest.fixture
def record(monkeypatch):
    """``record(fn)`` -> ``(fn(), [(kind, key, range, nbytes), ...])``: every
    store ``get``, ``LocalStore.get`` and file read ``fn`` made."""
    real_get = LocalStore.get
    real_ranges = _cells_on_disk._read_ranges
    real_index = _cells_on_disk._read_index
    real_file = _batch_reader._read_file

    async def local_get(self, key, prototype=None, byte_range=None):
        buf = await real_get(self, key, prototype, byte_range)
        _note("local get", key, byte_range, buf)
        return buf

    def read_ranges(items):
        out = real_ranges(items)
        for (path, offset, size), data in zip(items, out):
            _note("file range", path, (offset, size), data)
        return out

    def read_index(item):
        out = real_index(item)
        _note("file index", item[1], None, None)
        return out

    def read_file(path):
        out = real_file(path)
        _note("file", path, None, out)
        return out

    monkeypatch.setattr(LocalStore, "get", local_get)
    monkeypatch.setattr(_cells_on_disk, "_read_ranges", read_ranges)
    monkeypatch.setattr(_cells_on_disk, "_read_index", read_index)
    monkeypatch.setattr(_batch_reader, "_read_file", read_file)

    def run(fn):
        global _LOG
        _LOG = []
        try:
            got = fn()
            return got, collections.Counter(_LOG)
        finally:
            _LOG = None

    return run


def _reader(monkeypatch, reader):
    if reader is None:
        monkeypatch.delenv("ZARR_VECTORS_SHARD_READER", raising=False)
    else:
        monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", reader)


def _arrays(result):
    """Every array of a read result, in its order: the readers must agree exactly."""
    if isinstance(result, np.ndarray):
        return [result]
    return [getattr(result, f) for f in ("positions", "object_ids", "edges", "faces")]


def _equal(a, b):
    for x, y in zip(_arrays(a), _arrays(b)):
        assert (x is None) == (y is None)
        if x is not None:
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _cells(ds):
    """``core.cells.read_cells``: the ``batched_reads`` prefetch path."""
    from zarr_vectors.core.cells import read_cells

    got = read_cells(ds.level(0).store, CELLS, ("vertices",), on_error="raise", fragments=True)
    frag = got.fragments
    return np.concatenate([
        np.asarray(a, dtype=np.float64).ravel()
        for a in (got["vertices"].data, got["vertices"].offsets, frag.cell_offsets, frag.starts, frag.counts)
    ])


QUERIES = {
    "level": lambda ds: ds.level(0).read(),
    "bbox": lambda ds: ds.select(bbox=([10, 10, 10], [60, 60, 60])).read(),
    "objects": lambda ds: ds.level(0).objects[OBJECTS],
    "cells": _cells,
}


def _read(record, monkeypatch, path, where, reader, query):
    _reader(monkeypatch, reader)
    ds = _open(path, where)
    return record(lambda: QUERIES[query](ds))


def _total(log, kinds=("store get",)):
    return sum(c for k, c in log.items() if k[0] in kinds), sum(k[3] * c for k, c in log.items() if k[0] in kinds)


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


@needs_octree
def test_the_default_is_auto_and_reads_every_array_optimised(stores, monkeypatch):
    from zarr_vectors.core._cells_on_disk import cell_source

    src = {name: cell_source("", zv.open(str(p)).level(0).store.zarr_group["vertices"], any_codecs=True)
           for name, p in stores.items()}
    assert src["indexed"].index_format == "dense" and src["octree"].index_format == "zarr_octree"

    monkeypatch.delenv("ZARR_VECTORS_SHARD_READER", raising=False)
    monkeypatch.delenv("ZARR_VECTORS_COALESCE", raising=False)
    assert _batch_reader.shard_reader() == "auto"
    for s in (src["octree"], src["indexed"], None):
        assert _batch_reader.shard_reader_for(s) == "optimised"
    want = ("auto", _cells_on_disk._AUTO_GAP, _cells_on_disk._AUTO_SPAN)
    assert _cells_on_disk._coalescing(src["octree"]) == _cells_on_disk._coalescing(src["indexed"]) == want
    assert _cells_on_disk._coalescing() == want

    monkeypatch.setenv("ZARR_VECTORS_COALESCE", "gap:10:20")  # the optimised reader's setting, every array
    assert _cells_on_disk._coalescing(src["octree"]) == ("gap", 10, 20)
    assert _cells_on_disk._coalescing(src["indexed"]) == ("gap", 10, 20)

    for value, want in (("AUTO", "auto"), ("standard", "standard"), ("optimised", "optimised"),
                        ("optimized", "optimised"), ("zarr", "zarr")):
        monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", value)
        assert _batch_reader.shard_reader() == want
        assert {_batch_reader.shard_reader_for(s) for s in src.values()} == {
            "optimised" if want == "auto" else want}
    for value, octree, indexed in (("standard", "none", "none"), ("optimised", "gap", "gap"), ("zarr", "none", "none")):
        monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", value)
        assert (_cells_on_disk._coalescing(src["octree"])[0], _cells_on_disk._coalescing(src["indexed"])[0]) == (octree, indexed)

    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "fast")
    with pytest.raises(ValueError, match="ZARR_VECTORS_SHARD_READER"):
        _batch_reader.shard_reader()


# ---------------------------------------------------------------------------
# 1. sharding_indexed: the optimised reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", list(QUERIES))
@pytest.mark.parametrize("where", ["local", "store"])
def test_an_indexed_store_reads_as_optimised(stores, record, monkeypatch, where, query):
    """The optimised reader's requests exactly, the standard reader's
    answer, and on a store that is not local fewer requests."""
    path = stores["indexed"]
    want, standard = _read(record, monkeypatch, path, where, "standard", query)
    got, default = _read(record, monkeypatch, path, where, None, query)
    _, optimised = _read(record, monkeypatch, path, where, "optimised", query)
    _equal(got, want)
    assert default and default == optimised
    if where == "store":
        assert _total(default)[0] < _total(standard)[0], (_total(standard), _total(default))


def test_an_indexed_store_never_goes_through_zarrs_per_cell_read(stores, monkeypatch):
    async def _no(*a, **k):
        raise AssertionError("an indexed cell went through zarr's per-cell read")

    monkeypatch.delenv("ZARR_VECTORS_SHARD_READER", raising=False)
    monkeypatch.setattr(_batch_reader, "_async_get_sharded_cell", _no)
    ds = _open(stores["indexed"], "store")
    assert len(ds.select(bbox=([0, 0, 0], [BOX, BOX, BOX])).read().positions) == 3_000
    assert len(_cells(ds)) > 0


# ---------------------------------------------------------------------------
# 2. With zarr-octree: the optimised reader
# ---------------------------------------------------------------------------


@needs_octree
@pytest.mark.parametrize("query", list(QUERIES))
def test_an_octree_store_reads_as_optimised_and_reads_less(stores, record, monkeypatch, query):
    path = stores["octree"]
    want, standard = _read(record, monkeypatch, path, "store", "standard", query)
    got, default = _read(record, monkeypatch, path, "store", None, query)
    _, optimised = _read(record, monkeypatch, path, "store", "optimised", query)
    _equal(got, want)
    assert default == optimised  # the optimised reader's requests, exactly
    (n_std, b_std), (n_def, b_def) = _total(standard), _total(default)
    assert n_def < n_std and b_def < b_std, (n_std, b_std, n_def, b_def)


@needs_octree
def test_an_octree_store_never_goes_through_zarrs_per_cell_read(stores, monkeypatch):
    async def _no(*a, **k):
        raise AssertionError("an octree cell went through zarr's per-cell read")

    monkeypatch.delenv("ZARR_VECTORS_SHARD_READER", raising=False)
    monkeypatch.setattr(_batch_reader, "_async_get_sharded_cell", _no)
    ds = _open(stores["octree"], "store")
    assert len(ds.select(bbox=([0, 0, 0], [BOX, BOX, BOX])).read().positions) == 3_000
    assert len(_cells(ds)) > 0


@needs_octree
@pytest.mark.parametrize("query", list(QUERIES))
def test_an_octree_store_answers_as_indexed(stores, monkeypatch, query):
    _reader(monkeypatch, "zarr")
    want = QUERIES[query](_open(stores["indexed"], "local"))
    _reader(monkeypatch, None)
    for where in ("local", "store"):
        _equal(QUERIES[query](_open(stores["octree"], where)), want)


# ---------------------------------------------------------------------------
# 3. Mixed stores: per array
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed(stores, tmp_path_factory):
    """The indexed store with only ``vertices`` converted to zarr-octree."""
    if not HAVE_OCTREE:
        pytest.skip("zarr-octree is not installed")
    from zarr_vectors.sharding import set_octree

    path = tmp_path_factory.mktemp("mixed") / "mixed.zv"
    shutil.copytree(stores["indexed"], path)
    set_octree(str(path), TARGET, arrays=["vertices"])
    return path


@needs_octree
@pytest.mark.parametrize("query", list(QUERIES))
@pytest.mark.parametrize("where", ["local", "store"])
def test_a_mixed_store_reads_every_array_optimised(stores, mixed, record, monkeypatch, where, query):
    import json

    codecs = {a: json.loads((mixed / "0" / a / "zarr.json").read_text())["codecs"][0]["name"]
              for a in ("vertices", "vertex_fragments")}
    assert codecs["vertices"] in OCTREE_CODEC_NAMES and codecs["vertex_fragments"] == "sharding_indexed"

    _reader(monkeypatch, "zarr")
    plain = QUERIES[query](_open(stores["indexed"], "local"))
    want, standard = _read(record, monkeypatch, mixed, where, "standard", query)
    got, default = _read(record, monkeypatch, mixed, where, None, query)
    _, optimised = _read(record, monkeypatch, mixed, where, "optimised", query)
    _equal(want, plain)
    _equal(got, plain)
    assert default and default == optimised
    if where == "store":
        assert _total(default)[0] < _total(standard)[0], (_total(standard), _total(default))


# ---------------------------------------------------------------------------
# Every geometry, every codec, under the default
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("octree", [False, True], ids=["indexed", "octree"])
@pytest.mark.parametrize("kind", ["point_cloud", "mesh", "graph"])
def test_every_kind_reads_the_same_under_the_default(tmp_path, monkeypatch, kind, octree):
    if octree and not HAVE_OCTREE:
        pytest.skip("zarr-octree is not installed")
    path = _write(tmp_path / f"{kind}.zv", kind, TARGET if octree else False, n=1_500)
    queries = {
        "level": QUERIES["level"],
        "bbox": QUERIES["bbox"],
        "objects": lambda ds: ds.level(0).objects[[1, 8]],
    }
    for name, query in queries.items():
        _reader(monkeypatch, "standard")
        want = query(_open(path, "local"))
        _reader(monkeypatch, None)
        for where in ("local", "store"):
            _equal(query(_open(path, where)), want)
