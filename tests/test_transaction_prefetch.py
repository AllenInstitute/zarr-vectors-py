"""``shard_transaction(prefetch=True)``: owned shards read once, off the caller's path.

In merge mode a transaction reads what it merges into: each owned cell the
caller reads but has not staged, and every touched shard again at publish.
On a filesystem where a read is a round trip (NFS, FUSE, a spinning disk)
those small reads sit on the caller's path. With ``prefetch=True`` every
owned shard object is read whole in the background on entry and both are
served from it. Nothing about the result may change: the same cells read,
the same bytes published. And a shard whose every cell is rewritten is not
read at all, prefetch or not.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr.storage._local as zlocal

from zarr_vectors import building as zb
from zarr_vectors._api_version import FEATURES
from zarr_vectors.core import shard_txn

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (100.0, 100.0, 100.0)
SHARD0 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]   # 4 of shard (0,0,0)'s 8 cells
FULL0 = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
ATTRS = ["a0", "a1", "a2"]


def _level(path):
    root = zb.create_store(str(path), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=2)
    level = zb.get_resolution_level(root, 0)
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        zb.create_vertices_array(level, dtype="float32")
        for name in ATTRS:
            zb.create_attribute_array(level, name, dtype="float32")
    zb.defer_presence(level)
    return level


def _write(level, cells, tag):
    for cell in cells:
        pts = np.full((3, 3), 50.0, np.float32) + np.asarray(cell) * 100.0 + tag
        zb.write_chunk_vertices(level, cell, [pts])
        for i, name in enumerate(ATTRS):
            zb.write_chunk_attributes(level, name, cell, [np.full(3, tag + i, np.float32)])


def _files(level) -> dict[str, bytes]:
    root = Path(level.zarr_group.store.root)
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("c/*/*/*"))
            if p.is_file() and not p.name.endswith(".partial")}


def _seeded(path):
    """A level whose shard (0,0,0) already holds every cell (an earlier attempt)."""
    level = _level(path)
    with zb.shard_transaction(level, (0, 0, 0)):
        _write(level, FULL0, tag=1.0)
    return level


def _read_all(level, cells):
    return [np.asarray(zb.read_chunk_vertices(level, c)[0]).tobytes() for c in cells]


def test_the_feature():
    assert "transaction-prefetch" in FEATURES


@pytest.mark.parametrize("io_threads", [None, 8])
def test_prefetch_reads_and_publishes_the_same(tmp_path, io_threads):
    plain = _seeded(tmp_path / "plain" / "s.zv")
    pre = _seeded(tmp_path / "pre" / "s.zv")
    unstaged = [c for c in FULL0 if c not in SHARD0]

    with zb.shard_transaction(plain, (0, 0, 0), mode="merge", io_threads=io_threads):
        _write(plain, SHARD0, tag=4.0)
        seen_plain = _read_all(plain, FULL0)
    with zb.shard_transaction(pre, (0, 0, 0), mode="merge", io_threads=io_threads,
                              prefetch=True):
        _write(pre, SHARD0, tag=4.0)
        seen_pre = _read_all(pre, FULL0)

    assert seen_pre == seen_plain
    # The unstaged cells kept the earlier attempt's values.
    assert _read_all(pre, unstaged) == _read_all(plain, unstaged)
    assert _files(pre) == _files(plain)
    assert not list(tmp_path.rglob("*.partial"))


def test_after_the_prefetch_the_owned_shards_are_not_read_again(tmp_path):
    """Move the shard objects away once prefetched: reads inside the block and
    the merge at publish must still see the earlier attempt's cells."""
    level = _seeded(tmp_path / "s.zv")
    root = Path(level.zarr_group.store.root)
    expected = _read_all(level, FULL0)
    aside = tmp_path / "aside"
    with zb.shard_transaction(level, (0, 0, 0), mode="merge", prefetch=True) as tx:
        for name in tx._owned:
            tx._shard_bytes(name)                  # wait for the prefetch
        for name, owned in tx._owned.items():
            src = root / owned.key
            if src.exists():
                (aside / owned.key).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(aside / owned.key))
        assert _read_all(level, FULL0) == expected
        _write(level, [(0, 0, 0)], tag=9.0)
    got = _read_all(level, FULL0)
    assert got[1:] == expected[1:]                 # merged from memory, not lost
    assert got[0] != expected[0]


def test_a_fully_rewritten_shard_is_not_read(tmp_path, monkeypatch):
    level = _seeded(tmp_path / "s.zv")
    owned_keys: set[str] = set()
    reads: list[str] = []
    real = zlocal.LocalStore.get

    async def get(self, key, prototype=None, byte_range=None):
        if key in owned_keys:
            reads.append(key)
        return await real(self, key, prototype, byte_range)

    with zb.shard_transaction(level, (0, 0, 0), mode="merge") as tx:
        owned_keys.update(o.key for o in tx._owned.values())
        _write(level, FULL0, tag=7.0)
        monkeypatch.setattr(zlocal.LocalStore, "get", get)
    assert reads == []
    want = (np.full((3, 3), 50.0) + np.asarray((1, 1, 1)) * 100.0 + 7.0).astype(np.float32)
    assert _read_all(level, [(1, 1, 1)])[0] == want.tobytes()


def test_replace_mode_prefetches_nothing(tmp_path, monkeypatch):
    level = _seeded(tmp_path / "s.zv")
    calls = []
    real = shard_txn._read_object
    monkeypatch.setattr(shard_txn, "_read_object",
                        lambda store, key: calls.append(key) or real(store, key))
    with zb.shard_transaction(level, (0, 0, 0), mode="replace", prefetch=True):
        _write(level, SHARD0, tag=2.0)
    assert calls == []


def test_an_aborted_transaction_stops_its_prefetch(tmp_path):
    level = _seeded(tmp_path / "s.zv")
    with pytest.raises(RuntimeError, match="boom"):
        with zb.shard_transaction(level, (0, 0, 0), mode="merge", prefetch=True) as tx:
            raise RuntimeError("boom")
    assert tx._prefetch_pool is None
    assert not list(tmp_path.rglob("*.partial"))
