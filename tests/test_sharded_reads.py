"""Reads of a sharded store take a shard's index once, not once per cell.

zarr reads a sharded cell through its sharding codec, one index lookup
and one range read per cell; the prefetch now locates every cell from
each shard's index, read once, and decodes each cell with its array's
own codecs. The cells must be the ones zarr returns, byte for byte, for
every codec, including cells nobody wrote, shards that do not exist and
keys off the grid.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zarr_vectors.core import _batch_reader
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.cells import read_cells
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import write_points

_CHUNK = (25.0, 25.0, 25.0)


def _store(path, compressor):
    rng = np.random.default_rng(0)
    write_points(
        str(path), rng.uniform(0, 100, (700, 3)).astype("float32"),
        chunk_shape=_CHUNK, bounds=[[0, 0, 0], [100, 100, 100]],
        compressor=compressor, shard_shape=(2, 2, 2),
        vertex_attributes={"rgb": rng.integers(0, 255, (700, 3)).astype("uint8")},
    )
    return get_resolution_level(open_store(str(path)), 0)


def _plan(lg):
    keys = [".".join(map(str, k)) for k in list_chunk_keys(lg)]
    extra = ["0.3.3", "3.3.0", "9.9.9", "not-a-key"]
    return [(name, keys + extra) for name in ("vertices", "vertex_attributes/rgb")]


def _zarr_cells(lg, plan):
    """What zarr's own sharded reads give, cell by cell."""
    return _batch_reader.sync(_batch_reader._gather_plan(lg.zarr_group._async_group, plan))


@pytest.mark.parametrize("compressor", [None, "zstd", "blosc"])
def test_sharded_prefetch_reads_what_zarr_reads(tmp_path, compressor):
    lg = _store(tmp_path / "s.zarrvectors", compressor)
    plan = _plan(lg)
    want = _zarr_cells(lg, plan)
    got = _batch_reader.flush_prefetch(lg.zarr_group, plan)
    assert got == want
    assert any(v for v in got.values())


def test_it_does_not_go_through_zarr(tmp_path, monkeypatch):
    lg = _store(tmp_path / "s.zarrvectors", "zstd")

    async def _no(*a, **k):
        raise AssertionError("a sharded cell went through zarr's per-cell read")

    monkeypatch.setattr(_batch_reader, "_async_get_sharded_cell", _no)
    _batch_reader.flush_prefetch(lg.zarr_group, _plan(lg))


def _shard_files(lg):
    root = lg._sharded_chunk_array("vertices").store_path
    base = os.path.join(str(root.store.root), root.path, "c")
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(base) for f in fs)


def test_a_missing_shard_reads_as_empty_cells(tmp_path):
    lg = _store(tmp_path / "s.zarrvectors", "zstd")
    os.remove(_shard_files(lg)[0])
    plan = [p for p in _plan(lg) if p[0] == "vertices"]
    got = _batch_reader.flush_prefetch(lg.zarr_group, plan)
    assert got == _zarr_cells(lg, plan)
    assert b"" in got.values()


def test_a_shard_whose_index_fails_its_checksum(tmp_path):
    lg = _store(tmp_path / "s.zarrvectors", "zstd")
    victim = _shard_files(lg)[0]
    with open(victim, "r+b") as fh:
        fh.seek(-2, os.SEEK_END)
        fh.write(b"\xff\xff")  # the index's crc32c no longer matches
    plan = [p for p in _plan(lg) if p[0] == "vertices"]
    good = _batch_reader.flush_prefetch(lg.zarr_group, plan, tolerant=True)
    assert "0.0.0" not in good and good  # that shard's cells, only, are left out
    with pytest.raises(ValueError, match="crc32c"):
        _batch_reader.flush_prefetch(lg.zarr_group, plan)
    batch = read_cells(lg, np.asarray(list_chunk_keys(lg)), ["vertices"])
    assert {e.chunk_key for e in batch.errors} >= {"0.0.0"}
