"""``durable=True`` writes, and ``cell_objects``.

A durable block leaves everything it wrote on disk when it returns: each
cell object (a shard, when sharded) fsynced before its rename, then the
metadata it wrote and the directories it changed. ``cell_objects`` names
the object holding each cell, as zarr addresses it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors import building as zb

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads fsynced paths from /proc",
)

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (100.0, 100.0, 100.0)
CELLS = [(0, 0, 0), (1, 0, 0), (3, 2, 1)]


@pytest.fixture
def fsynced(monkeypatch) -> list[str]:
    """The path of every file or directory fsynced.

    A cell is fsynced as its temporary, ``<object>.<hex>.partial``, before
    the rename that makes it the object; it is recorded as the object.
    """
    paths: list[str] = []
    real = os.fsync

    def recording(fd):
        path = os.path.realpath(os.readlink(f"/proc/self/fd/{fd}"))
        if path.endswith(".partial"):
            path = path.rsplit(".", 2)[0]
        paths.append(path)
        return real(fd)

    monkeypatch.setattr(os, "fsync", recording)
    return paths


def _level(path, **kw):
    root = zb.create_store(str(path), bounds=BOUNDS, chunk_shape=CHUNK, **kw)
    return zb.get_resolution_level(root, 0)


def _write(level, *, durable, bounds=BOUNDS, cells=CELLS):
    with zb.open_write_session(level, bounds=bounds, chunk_shape=CHUNK, durable=durable):
        zb.create_vertices_array(level, dtype="float32")
        for cell in cells:
            pts = np.full((2, 3), 50.0, dtype=np.float32) + np.asarray(cell) * 100.0
            zb.write_chunk_vertices(level, cell, [pts])


def _root(level) -> Path:
    return Path(level.zarr_group.store.root)


def _objects(level, name, cells=CELLS) -> set[str]:
    return {
        os.path.realpath(_root(level) / key)
        for key in zb.cell_objects(level, name, cells)
    }


@pytest.mark.parametrize("shard_shape", [None, 2], ids=["cells", "shards"])
def test_every_object_written_is_fsynced(tmp_path, fsynced, shard_shape):
    level = _level(tmp_path / "s.zv", shard_shape=shard_shape)
    _write(level, durable=True)
    synced = set(fsynced)
    for name in ("vertices", "vertex_fragments"):
        objects = _objects(level, name)
        assert all(os.path.isfile(p) for p in objects)
        assert objects <= synced, name
        # ...and the directories holding them.
        assert {os.path.dirname(p) for p in objects} <= synced, name
    assert not list(tmp_path.rglob("*.partial"))
    got = zb.read_chunk_vertices(level, (3, 2, 1))
    assert np.asarray(got[0]).shape == (2, 3)


def test_the_default_fsyncs_nothing(tmp_path, fsynced):
    _write(_level(tmp_path / "s.zv"), durable=False)
    assert fsynced == []


def test_a_memory_store_needs_nothing(fsynced):
    import zarr

    root = zb.create_store(zarr.storage.MemoryStore(), bounds=BOUNDS, chunk_shape=CHUNK)
    level = zb.get_resolution_level(root, 0)
    _write(level, durable=True)
    assert fsynced == []
    assert np.asarray(zb.read_chunk_vertices(level, (1, 0, 0))[0]).shape == (2, 3)


def test_cell_objects_follow_the_shard_and_the_origin(tmp_path):
    # A grid starting below zero: cell (-1, 0, 0) is array index 0.
    bounds = ([-100.0, 0.0, 0.0], [300.0, 400.0, 400.0])
    root = zb.create_store(str(tmp_path / "s.zv"), bounds=bounds, chunk_shape=CHUNK, shard_shape=2)
    level = zb.get_resolution_level(root, 0)
    cells = [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 3, 3)]
    _write(level, durable=False, bounds=bounds, cells=cells)
    keys = zb.cell_objects(level, "vertices", cells)
    # (-1,0,0) and (0,0,0) are indices 0 and 1: the same shard.
    assert keys[0] == keys[1] != keys[2]
    assert all((_root(level) / k).is_file() for k in keys)
    assert keys[0].endswith("vertices/c/0/0/0")
    with pytest.raises(zb.ArrayError, match="not a cell"):
        zb.cell_objects(level, "vertices", [(9, 0, 0)])
    with pytest.raises(zb.StoreError, match="not a per-chunk array"):
        zb.cell_objects(level, "object_index", [(0, 0, 0)])


def test_a_durable_write_into_a_shard_keeps_what_is_there(tmp_path, fsynced):
    level = _level(tmp_path / "s.zv", shard_shape=2)
    _write(level, durable=True, cells=[(0, 0, 0)])
    # The same shard (indices 0-1 on every axis), another cell.
    _write(level, durable=True, cells=[(1, 1, 0)])
    for cell in ((0, 0, 0), (1, 1, 0)):
        assert np.asarray(zb.read_chunk_vertices(level, cell)[0]).shape == (2, 3)
    assert len(set(zb.cell_objects(level, "vertices", [(0, 0, 0), (1, 1, 0)]))) == 1


def test_an_emptied_cell_is_removed_and_its_directory_fsynced(tmp_path, fsynced):
    level = _level(tmp_path / "s.zv")
    _write(level, durable=True, cells=[(0, 0, 0)])
    (path,) = _objects(level, "vertices", [(0, 0, 0)])
    fsynced.clear()
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK, durable=True):
        level.write_bytes("vertices", "0.0.0", b"")
    assert not os.path.exists(path)
    assert os.path.dirname(path) in fsynced
