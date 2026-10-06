"""How a shard transaction publishes: ``io_threads``, ``sweep`` and its opens.

Publishing a transaction is, per object, an encode, a create, a write, an
fsync and a rename -- about 110 objects per BRIDGE graph task, each step
a round trip on a network filesystem. ``io_threads`` runs everything but
the renames from a pool and must change nothing else: the same bytes, the
same ``published``, each object durable before its rename, and nothing
left behind when a thread fails. ``sweep=False`` skips the listing of
every shard directory on entry. And the publish opens no array by path:
it holds the metadata already.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from zarr.storage import LocalStore

from zarr_vectors import building as zb
from zarr_vectors.core import shard_txn
from zarr_vectors.exceptions import ArrayError

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (100.0, 100.0, 100.0)
SHARD0 = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]
ATTRS = [f"a{i}" for i in range(12)]


def _level(path):
    root = zb.create_store(str(path), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=2)
    level = zb.get_resolution_level(root, 0)
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        zb.create_vertices_array(level, dtype="float32")
        for name in ATTRS:
            zb.create_attribute_array(level, name, dtype="float32")
    zb.defer_presence(level)
    return level


def _write(level, cells, tag=0.0, attrs=ATTRS, n=3):
    for cell in cells:
        pts = np.full((n, 3), 50.0, np.float32) + np.asarray(cell) * 100.0 + tag
        zb.write_chunk_vertices(level, cell, [pts])
        for i, name in enumerate(attrs):
            zb.write_chunk_attributes(level, name, cell, [np.full(n, tag + i, np.float32)])


def _files(level) -> dict[str, bytes]:
    root = Path(level.zarr_group.store.root)
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("c/*/*/*"))
        if p.is_file() and not p.name.endswith(".partial")
    }


def _partials(level) -> list[Path]:
    return list(Path(level.zarr_group.store.root).rglob("*.partial"))


def _run(path, *, io_threads, mode="replace"):
    level = _level(path)
    # An earlier attempt's shards, so replace removes one and merge keeps it.
    with zb.shard_transaction(level, (0, 0, 0)):
        _write(level, SHARD0, tag=1.0)
    with zb.shard_transaction(level, (0, 0, 0), mode=mode, io_threads=io_threads) as tx:
        _write(level, SHARD0[:2], tag=4.0, attrs=ATTRS[:-1])
    return level, tx


@pytest.mark.parametrize("mode", ["replace", "merge"])
def test_threads_publish_the_same_bytes(tmp_path, mode):
    one, tx1 = _run(tmp_path / "one" / "s.zv", io_threads=None, mode=mode)
    many, tx16 = _run(tmp_path / "many" / "s.zv", io_threads=16, mode=mode)
    assert _files(many) == _files(one)
    assert tx16.published == tx1.published
    assert tx16.written == tx1.written
    # vertices, vertex_fragments and the attributes; the one the retry did
    # not write is removed under replace, and left alone under merge.
    if mode == "replace":
        assert len(tx16.published) == 2 + len(ATTRS)
        assert not any(k.startswith(f"0/vertex_attributes/{ATTRS[-1]}/") for k in _files(many))
    else:
        assert len(tx16.published) == 2 + len(ATTRS) - 1
        assert any(k.startswith(f"0/vertex_attributes/{ATTRS[-1]}/") for k in _files(many))
    assert not _partials(many)


def test_each_object_is_fsynced_before_its_rename(tmp_path, monkeypatch):
    level = _level(tmp_path / "s.zv")
    events: list[tuple[str, str]] = []
    lock = threading.Lock()
    real_fsync, real_replace = os.fsync, os.replace

    threads: set[str] = set()

    def fsync(fd):
        real_fsync(fd)
        with lock:
            events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
            threads.add(threading.current_thread().name)

    def replace(src, dst):
        with lock:
            events.append(("rename", str(src)))
        return real_replace(src, dst)

    if not Path("/proc/self/fd").is_dir():
        pytest.skip("needs /proc to name an fsynced descriptor")
    with zb.shard_transaction(level, (0, 0, 0), io_threads=16) as tx:
        _write(level, SHARD0)
        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(os, "replace", replace)
    renames = [i for i, (kind, _) in enumerate(events) if kind == "rename"]
    assert len(renames) == len(tx.published)
    assert len({t for t in threads if t.startswith("zv-write")}) > 1
    synced = {}
    for i, (kind, path) in enumerate(events):
        if kind == "fsync":
            synced.setdefault(path, i)
    for i in renames:
        src = events[i][1]
        assert synced.get(src, len(events)) < i, f"{src} renamed before its fsync"
    # Every file was synced before the first rename; the directories after
    # the last one, once each.
    files = [p for p in synced if p.endswith(".partial")]
    assert all(synced[p] < renames[0] for p in files)
    dirs = [path for kind, path in events[renames[-1] + 1:] if kind == "fsync"]
    assert dirs and len(dirs) == len(set(dirs))
    assert all(Path(d).is_dir() for d in dirs)


@pytest.mark.parametrize("where", ["encode", "write"])
def test_a_failing_thread_publishes_nothing_and_leaves_no_partials(
    tmp_path, monkeypatch, where,
):
    level = _level(tmp_path / "s.zv")
    with zb.shard_transaction(level, (0, 0, 0)):
        _write(level, SHARD0, tag=1.0)
    before = _files(level)

    if where == "encode":
        real = shard_txn.ShardTransaction._encode_one

        def encode_one(self, name):
            if name == f"vertex_attributes/{ATTRS[5]}":
                raise OSError("encoder lost")
            time.sleep(0.01)  # the other lanes are still writing
            return real(self, name)

        monkeypatch.setattr(shard_txn.ShardTransaction, "_encode_one", encode_one)
    else:
        real = shard_txn._write_partial

        def write_partial(path, value, **kw):
            if f"/{ATTRS[5]}/" in path:
                kw["partials"].append(path + ".x.partial")
                Path(path + ".x.partial").write_bytes(value[:3])
                raise OSError("encoder lost")
            time.sleep(0.01)
            return real(path, value, **kw)

        monkeypatch.setattr(shard_txn, "_write_partial", write_partial)

    with pytest.raises(OSError, match="encoder lost"):
        with zb.shard_transaction(level, (0, 0, 0), io_threads=16):
            _write(level, SHARD0, tag=9.0)
    assert _files(level) == before
    assert not _partials(level)


def test_the_publish_opens_no_array_by_path(tmp_path, monkeypatch):
    level = _level(tmp_path / "s.zv")
    gets: list[str] = []
    real = LocalStore.get

    async def counting(self, key, *a, **kw):
        gets.append(key)
        return await real(self, key, *a, **kw)

    with zb.shard_transaction(level, (0, 0, 0), io_threads=4) as tx:
        _write(level, SHARD0)
        monkeypatch.setattr(LocalStore, "get", counting)
    assert len(tx.published) > 1
    meta = [k for k in gets if k.endswith(("zarr.json", ".zarray", ".zattrs", ".zgroup"))]
    assert meta == []


def test_opening_a_store_probes_no_v2_metadata(tmp_path, monkeypatch):
    _level(tmp_path / "s.zv")
    gets: list[str] = []
    real = LocalStore.get

    async def counting(self, key, *a, **kw):
        gets.append(key)
        return await real(self, key, *a, **kw)

    monkeypatch.setattr(LocalStore, "get", counting)
    level = zb.get_resolution_level(zb.open_store(str(tmp_path / "s.zv"), mode="r+"), 0)
    with zb.shard_transaction(level, (0, 0, 0)):
        _write(level, SHARD0)
    assert gets
    assert [k for k in gets if k.endswith((".zarray", ".zattrs", ".zgroup", ".zmetadata"))] == []


def test_sweep_false_leaves_a_stale_partial_alone(tmp_path):
    level = _level(tmp_path / "s.zv")
    with zb.shard_transaction(level, (0, 0, 0)):
        _write(level, SHARD0, tag=1.0)
    want = _files(level)
    shard = next(p for p in Path(level.zarr_group.store.root).rglob("c/0/0/0") if p.is_file())
    stale = shard.with_name(f"{shard.name}.deadbeef.partial")
    stale.write_bytes(b"left by a crash")

    with zb.shard_transaction(level, (0, 0, 0), sweep=False):
        _write(level, SHARD0, tag=1.0)
    assert stale.exists()
    assert _files(level) == want  # never read, and the publish is unchanged

    with zb.shard_transaction(level, (0, 0, 0), io_threads=8):
        _write(level, SHARD0, tag=1.0)
    assert not stale.exists()
    assert _files(level) == want


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_io_threads_must_be_a_positive_int(tmp_path, bad):
    level = _level(tmp_path / "s.zv")
    with pytest.raises(ArrayError, match="io_threads"):
        with zb.shard_transaction(level, (0, 0, 0), io_threads=bad):
            pass


def test_a_memory_store_publishes_the_same_with_threads():
    import zarr

    def run(io_threads):
        root = zb.create_store(
            zarr.storage.MemoryStore(), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=2,
        )
        level = zb.get_resolution_level(root, 0)
        with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
            zb.create_vertices_array(level, dtype="float32")
            for name in ATTRS:
                zb.create_attribute_array(level, name, dtype="float32")
        zb.defer_presence(level)
        with zb.shard_transaction(level, (0, 0, 0), io_threads=io_threads) as tx:
            _write(level, SHARD0)
        store = level.zarr_group.store
        shards = {
            k: bytes(v.to_bytes()) for k, v in store._store_dict.items() if "/c/" in k
        }
        return tx.published, shards

    assert run(None) == run(8)
