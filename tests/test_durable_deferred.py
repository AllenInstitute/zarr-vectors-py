"""``durable="deferred"``: atomic writes, made durable later by ``sync_store``.

A caller that writes many objects from many tasks, and needs them durable
only at a checkpoint of its own (BRIDGE: the end of a stage, before it
trusts any task's "done"), passes ``durable="deferred"``: the writes land
exactly as ``durable=False`` would (atomically, by rename, no fsync) and
one :func:`building.sync_store` later makes everything on the store's
filesystem durable, from every process on the host.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors._api_version import FEATURES, __api_version__

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (100.0, 100.0, 100.0)
SHARD0 = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]


@pytest.fixture
def fsynced(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def recording(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", recording)
    return calls


def _write(level, cells, tag=0.0):
    for cell in cells:
        pts = np.full((3, 3), 50.0, np.float32) + np.asarray(cell) * 100.0 + tag
        zb.write_chunk_vertices(level, cell, [pts])


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("c/*/*/*"))
        if p.is_file() and not p.name.endswith(".partial")
    }


def _sharded(path):
    root = zb.create_store(str(path), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=2)
    level = zb.get_resolution_level(root, 0)
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        zb.create_vertices_array(level, dtype="float32")
    zb.defer_presence(level)
    return level


def test_the_feature_and_api_version():
    assert "durable-deferred" in FEATURES
    assert __api_version__ >= (1, 6)
    assert "sync_store" in zb.__all__


def test_a_deferred_transaction_fsyncs_nothing_and_publishes_the_same_bytes(
    tmp_path, fsynced,
):
    full = _sharded(tmp_path / "full" / "s.zv")
    deferred = _sharded(tmp_path / "deferred" / "s.zv")
    with zb.shard_transaction(full, (0, 0, 0), durable=True):
        _write(full, SHARD0)
    assert fsynced, "durable=True should fsync"
    fsynced.clear()
    with zb.shard_transaction(deferred, (0, 0, 0), durable="deferred", io_threads=4) as tx:
        _write(deferred, SHARD0)
    assert fsynced == []
    assert tx.published
    assert _files(tmp_path / "deferred" / "s.zv") == _files(tmp_path / "full" / "s.zv")
    assert not list(tmp_path.rglob("*.partial"))


def test_a_deferred_write_session_fsyncs_nothing(tmp_path, fsynced):
    root = zb.create_store(str(tmp_path / "s.zv"), bounds=BOUNDS, chunk_shape=CHUNK)
    level = zb.get_resolution_level(root, 0)
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK, durable="deferred"):
        zb.create_vertices_array(level, dtype="float32")
        _write(level, SHARD0)
    assert fsynced == []
    got = zb.read_chunk_vertices(level, (1, 1, 1))
    assert np.asarray(got[0]).shape == (3, 3)


@pytest.mark.parametrize("bad", ["later", 1, "True"])
def test_an_unknown_durability_is_refused(tmp_path, bad):
    level = _sharded(tmp_path / "s.zv")
    with pytest.raises(ValueError, match="durable="):
        with zb.shard_transaction(level, (0, 0, 0), durable=bad):
            pass
    root = zb.create_store(str(tmp_path / "b.zv"), bounds=BOUNDS, chunk_shape=CHUNK)
    lg = zb.get_resolution_level(root, 0)
    with pytest.raises(ValueError, match="durable="):
        with lg.batched_writes(durable=bad):
            pass


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="syncfs is Linux-only")
def test_sync_store_syncs_the_filesystem_of_a_path_a_root_or_a_level(tmp_path):
    root = zb.create_store(str(tmp_path / "s.zv"), bounds=BOUNDS, chunk_shape=CHUNK)
    level = zb.get_resolution_level(root, 0)
    assert zb.sync_store(str(tmp_path / "s.zv")) == "syncfs"
    assert zb.sync_store(tmp_path / "s.zv") == "syncfs"
    assert zb.sync_store(root) == "syncfs"
    assert zb.sync_store(level) == "syncfs"


def test_sync_store_needs_nothing_off_a_local_filesystem(tmp_path):
    assert zb.sync_store("s3://bucket/store.zv") == "none"
    assert zb.sync_store(str(tmp_path / "missing")) == "none"
    from zarr.storage import MemoryStore

    assert zb.sync_store(MemoryStore()) == "none"
