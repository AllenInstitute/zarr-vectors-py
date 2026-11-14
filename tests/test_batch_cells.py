"""Experiment ``batch_cells``: the sharded reader's per-cell work done as arrays.

With ``ZARR_VECTORS_EXPERIMENTS=batch_cells`` the optimised sharded reader
parses keys, locates cells in their shards' indexes and plans its reads
for all cells at once, and decompresses each cell straight out of its
read. It must return the same cells and make the same reads -- the same
ranges of the same files, the same store requests -- as without it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from zarr.storage import LocalStore, WrapperStore

import zarr_vectors as zv
from zarr_vectors.core import _batch_reader, _cells_on_disk
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import write_points

try:
    import zarr_octree  # noqa: F401

    HAVE_OCTREE = True
except ImportError:
    HAVE_OCTREE = False


@pytest.fixture(autouse=True)
def _optimised_reader(monkeypatch):
    """The experiment is in the optimised reader, so pin it."""
    monkeypatch.setenv("ZARR_VECTORS_SHARD_READER", "optimised")
    monkeypatch.delenv("ZARR_VECTORS_EXPERIMENTS", raising=False)


def _on(monkeypatch, on):
    if on:
        monkeypatch.setenv("ZARR_VECTORS_EXPERIMENTS", "batch_cells")
    else:
        monkeypatch.delenv("ZARR_VECTORS_EXPERIMENTS", raising=False)


@pytest.fixture
def ranges(monkeypatch):
    """Every ``(file, offset, size)`` the local reader reads."""
    log = []
    real = _cells_on_disk._read_ranges

    def spy(items):
        log.extend(items)
        return real(items)

    monkeypatch.setattr(_cells_on_disk, "_read_ranges", spy)
    return log


def _level(path, compressor="zstd", n=700):
    rng = np.random.default_rng(0)
    write_points(
        str(path), rng.uniform(0, 100, (n, 3)).astype("float32"),
        chunk_shape=(25.0, 25.0, 25.0), bounds=[[0, 0, 0], [100, 100, 100]],
        compressor=compressor, shard_shape=(2, 2, 2),
        vertex_attributes={"rgb": rng.integers(0, 255, (n, 3)).astype("uint8")},
    )
    return get_resolution_level(open_store(str(path)), 0)


def _plan(lg):
    keys = [".".join(map(str, k)) for k in list_chunk_keys(lg)]
    extra = ["0.3.3", "3.3.0", "9.9.9", "-1.0.0", "1.2", "1.2.3.4", "a.b.c", "not-a-key"]
    return [(name, keys + extra) for name in ("vertices", "vertex_attributes/rgb")]


def _both(monkeypatch, ranges, fn):
    """``fn()`` and the ranges it read (in any order), off and then on."""
    out = []
    for on in (False, True):
        _on(monkeypatch, on)
        ranges.clear()
        out.append((fn(), sorted(ranges)))
    return out


@pytest.mark.parametrize("compressor", [None, "zstd", "blosc"])
def test_prefetch_is_the_same_cells_and_the_same_reads(tmp_path, monkeypatch, ranges, compressor):
    lg = _level(tmp_path / "s.zv", compressor)
    plan = _plan(lg)
    batched = []
    real = _batch_reader._sharded_read_batched
    monkeypatch.setattr(_batch_reader, "_sharded_read_batched", lambda *a, **k: batched.append(1) or real(*a, **k))
    (off, off_reads), (on, on_reads) = _both(
        monkeypatch, ranges, lambda: _batch_reader.flush_prefetch(lg.zarr_group, plan),
    )
    assert batched  # the experiment ran ...
    assert on == off and any(on.values())  # ... and read the same cells
    assert on_reads == off_reads and on_reads


def test_off_is_the_per_cell_reader(tmp_path, monkeypatch):
    lg = _level(tmp_path / "s.zv")

    def _no(*a, **k):
        raise AssertionError("batch_cells ran with the experiment off")

    monkeypatch.setattr(_batch_reader, "_sharded_read_batched", _no)
    assert _batch_reader.flush_prefetch(lg.zarr_group, _plan(lg))


def test_few_cells_an_array_take_the_per_cell_reader(tmp_path, monkeypatch):
    lg = _level(tmp_path / "s.zv")
    _on(monkeypatch, True)
    called = []
    real = _batch_reader._sharded_read_batched
    monkeypatch.setattr(_batch_reader, "_sharded_read_batched", lambda *a, **k: called.append(1) or real(*a, **k))
    few = [(name, keys[:_batch_reader._BATCH_CELLS - 1]) for name, keys in _plan(lg)]
    _batch_reader.flush_prefetch(lg.zarr_group, few)
    assert not called
    _batch_reader.flush_prefetch(lg.zarr_group, _plan(lg))
    assert called


@pytest.mark.parametrize("coalesce", ["auto", "none", "runs", "gap:0:100000", "gap:65536:300", "gap:200:1"])
def test_every_coalescing_reads_the_same_ranges(tmp_path, monkeypatch, ranges, coalesce):
    """Including reads capped below one cell (``gap:200:1``), each then a cell alone."""
    monkeypatch.setenv("ZARR_VECTORS_COALESCE", coalesce)
    lg = _level(tmp_path / "s.zv", n=3_000)
    plan = _plan(lg)
    (off, off_reads), (on, on_reads) = _both(
        monkeypatch, ranges, lambda: _batch_reader.flush_prefetch(lg.zarr_group, plan),
    )
    assert on == off
    assert on_reads == off_reads


def _shard_files(lg):
    root = lg._sharded_chunk_array("vertices").store_path
    base = os.path.join(str(root.store.root), root.path, "c")
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(base) for f in fs)


def test_a_missing_shard_reads_as_empty_cells(tmp_path, monkeypatch, ranges):
    lg = _level(tmp_path / "s.zv")
    os.remove(_shard_files(lg)[0])
    plan = [p for p in _plan(lg) if p[0] == "vertices"]
    (off, _), (on, _) = _both(monkeypatch, ranges, lambda: _batch_reader.flush_prefetch(lg.zarr_group, plan))
    assert on == off and b"" in on.values()


def test_a_shard_index_that_fails_its_checksum(tmp_path, monkeypatch, ranges):
    lg = _level(tmp_path / "s.zv")
    with open(_shard_files(lg)[0], "r+b") as fh:
        fh.seek(-2, os.SEEK_END)
        fh.write(b"\xff\xff")
    plan = [p for p in _plan(lg) if p[0] == "vertices"]
    (off, _), (on, _) = _both(
        monkeypatch, ranges, lambda: _batch_reader.flush_prefetch(lg.zarr_group, plan, tolerant=True),
    )
    assert on == off and "0.0.0" not in on and on
    _on(monkeypatch, True)
    with pytest.raises(ValueError, match="crc32c"):
        _batch_reader.flush_prefetch(lg.zarr_group, plan)


def test_a_corrupt_cell_is_left_out_or_raises(tmp_path, monkeypatch, ranges):
    lg = _level(tmp_path / "s.zv")
    plan = [p for p in _plan(lg) if p[0] == "vertices"]
    victim = _shard_files(lg)[0]
    with open(victim, "r+b") as fh:
        fh.write(b"\x00" * 64)  # the first cells' zstd frames
    (off, _), (on, _) = _both(
        monkeypatch, ranges, lambda: _batch_reader.flush_prefetch(lg.zarr_group, plan, tolerant=True),
    )
    assert on == off and on
    for on_ in (False, True):
        _on(monkeypatch, on_)
        with pytest.raises(Exception) as caught:
            _batch_reader.flush_prefetch(lg.zarr_group, plan)
        if not on_:
            want = type(caught.value)
    assert type(caught.value) is want


_GETS: list = []


class _Requests(WrapperStore):
    """A store that is not a LocalStore, recording every ``get`` in ``_GETS``."""

    async def get(self, key, prototype, byte_range=None):  # type: ignore[no-untyped-def]
        _GETS.append((key, repr(byte_range)))
        return await super().get(key, prototype, byte_range)


BOX = 100.0
CODECS = {"sharding_indexed": False, "zarr_octree_sharding": {"target": 2_000}}


def _store(path, codec):
    if codec != "sharding_indexed" and not HAVE_OCTREE:
        pytest.skip("zarr-octree is not installed")
    rng = np.random.default_rng(0)
    centres = rng.uniform(20, 80, (3, 3))
    pts = np.clip(centres[rng.integers(0, 3, 3_000)] + rng.standard_t(3, (3_000, 3)) * 6, 0, BOX - 1e-3)
    schema = zv.Schema(bounds=([0, 0, 0], [BOX] * 3), kind="point_cloud",
                       layout=zv.Layout(cells=8, shard_cells=4, compression="zstd", octree=CODECS[codec]))
    zv.create(str(path), schema=schema).add_points(pts.astype(np.float32), object_ids=np.arange(3_000) // 50)


def _answers(ds):
    def flat(r):
        p = np.asarray(r.positions)
        return p[np.lexsort(p.T[::-1])].tolist()

    return [
        flat(ds.select().read()),
        flat(ds.select(bbox=([10, 10, 10], [60, 60, 60])).read()),
        flat(ds.select(objects=[3, 17, 41]).read()),
    ]


@pytest.mark.parametrize("codec", list(CODECS))
@pytest.mark.parametrize("where", ["local", "store"])
def test_queries_answer_and_read_the_same(tmp_path, monkeypatch, ranges, codec, where):
    """A level, a box and objects, read off a local store or any other."""
    monkeypatch.setattr(_batch_reader, "_BATCH_CELLS", 1)  # every read batched
    _store(tmp_path / "s.zv", codec)
    got = []
    for on in (False, True):
        _on(monkeypatch, on)
        ranges.clear()
        _GETS.clear()
        store = _Requests(LocalStore(str(tmp_path / "s.zv"), read_only=True)) if where == "store" else None
        ds = zv.open(str(tmp_path / "s.zv") if store is None else store)
        got.append((_answers(ds), sorted(ranges), sorted(_GETS)))
    assert got[1][0] == got[0][0] and got[0][0][0]
    assert got[1][1] == got[0][1] and (got[0][1] or where == "store")
    assert got[1][2] == got[0][2] and (got[0][2] or where == "local")


def test_keys_parse_as_one_by_one():
    src = SimpleNamespace(shape=(4, 5, 6), origin=(-2, 0, 1))
    keys = ["-2.0.1", "1.4.6", "2.0.1", "-3.0.1", "0.5.1", "1.1", "1.1.1.1", "x.1.1", "1..1", " 1.+2.3", "1_0.0.1", ""]
    cells, kept = _batch_reader._cells_of(src, keys)
    want = []
    for k in keys:
        c = _batch_reader._parse_coords(k)
        if c is None or len(c) != 3:
            continue
        c = tuple(a - o for a, o in zip(c, src.origin))
        if all(0 <= a < n for a, n in zip(c, src.shape)):
            want.append((k, c))
    assert kept == [k for k, _ in want]
    assert [tuple(c) for c in cells.tolist()] == [c for _, c in want]
    cells, kept = _batch_reader._cells_of(src, ["0.0.1", "1.2.3"])  # the all-integer path
    assert kept == ["0.0.1", "1.2.3"] and cells.tolist() == [[2, 0, 0], [3, 2, 2]]
    assert _batch_reader._cells_of(src, [])[1] == []
