"""``shard_transaction``: one shard written privately, published on exit (A2).

A task that owns a shard used to write straight into the live store, so a
retry started from its failed attempt's cells, appends doubled, and the
caller built resets, quarantines and locks around every task. Inside a
transaction the task's writes are staged, reads of its cells see the
staging, and the shard objects are renamed into place at the end.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.exceptions import ShardOwnershipError, StoreError

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (100.0, 100.0, 100.0)
SHARD0 = [(0, 0, 0), (1, 0, 0), (1, 1, 1)]   # cells of shard (0, 0, 0)
SHARD1 = [(2, 0, 0), (3, 1, 0)]              # cells of shard (1, 0, 0)


def _level(path, *, defer=True, shard_shape=2):
    root = zb.create_store(str(path), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=shard_shape)
    level = zb.get_resolution_level(root, 0)
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        zb.create_vertices_array(level, dtype="float32")
        zb.create_attribute_array(level, "w", dtype="float32")
    zb.create_links_array(level, 2, delta=0, sid_ndim=3)
    if defer:
        zb.defer_presence(level)
    return level


def _points(cell, n, tag):
    return np.full((n, 3), 50.0, np.float32) + np.asarray(cell) * 100.0 + tag


def _write(level, cells, tag=0.0, n=3):
    for cell in cells:
        zb.write_chunk_vertices(level, cell, [_points(cell, n, tag)])
        zb.write_chunk_attributes(level, "w", cell, [np.full(n, tag, np.float32)])


def _vertices(level, cell):
    if not level.chunk_exists("vertices", ".".join(map(str, cell))):
        return np.empty((0, 3), np.float32)
    got = zb.read_chunk_vertices(level, cell)
    return np.concatenate(got) if got else np.empty((0, 3), np.float32)


def _shard_files(level) -> dict[str, bytes]:
    root = Path(level.zarr_group.store.root)
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("c/*/*/*")) if p.is_file()
    }


def test_replace_rebuilds_the_shard_from_its_cells_alone(tmp_path):
    level = _level(tmp_path / "s.zv")
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        _write(level, SHARD0 + SHARD1, tag=1.0)

    with zb.shard_transaction(level, (0, 0, 0)) as tx:
        _write(level, SHARD0[:1], tag=2.0)
    assert tx.written["vertices"] == ["0.0.0"]
    np.testing.assert_array_equal(_vertices(level, SHARD0[0]), _points(SHARD0[0], 3, 2.0))
    for cell in SHARD0[1:]:
        assert _vertices(level, cell).size == 0  # cleared: this shard is the task's
    for cell in SHARD1:
        np.testing.assert_array_equal(_vertices(level, cell), _points(cell, 3, 1.0))


def test_merge_keeps_what_is_there(tmp_path):
    level = _level(tmp_path / "s.zv")
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        _write(level, SHARD0[1:], tag=1.0)
    with zb.shard_transaction(level, (0, 0, 0), mode="merge"):
        _write(level, SHARD0[:1], tag=2.0)
    np.testing.assert_array_equal(_vertices(level, SHARD0[0]), _points(SHARD0[0], 3, 2.0))
    np.testing.assert_array_equal(_vertices(level, SHARD0[1]), _points(SHARD0[1], 3, 1.0))


def test_a_cell_of_another_shard_is_refused_and_nothing_lands(tmp_path):
    level = _level(tmp_path / "s.zv")
    before = _shard_files(level)
    with pytest.raises(ShardOwnershipError, match="outside shard"):
        with zb.shard_transaction(level, (0, 0, 0)):
            _write(level, SHARD0[:1] + SHARD1[:1])
    assert _shard_files(level) == before
    with pytest.raises(ShardOwnershipError, match="not one of this transaction's arrays"):
        with zb.shard_transaction(level, (0, 0, 0), arrays=["vertices"]):
            zb.write_chunk_attributes(level, "w", SHARD0[0], [np.zeros(3, np.float32)])


def test_an_exception_inside_publishes_nothing(tmp_path):
    level = _level(tmp_path / "s.zv")
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        _write(level, SHARD0, tag=1.0)
    before = _shard_files(level)
    with pytest.raises(RuntimeError, match="task failed"):
        with zb.shard_transaction(level, (0, 0, 0)):
            _write(level, SHARD0, tag=9.0)
            raise RuntimeError("task failed")
    assert _shard_files(level) == before
    assert not list(Path(level.zarr_group.store.root).rglob("*.partial"))


def test_a_retry_after_a_crash_converges_to_the_same_bytes(tmp_path, monkeypatch):
    clean = _level(tmp_path / "clean" / "s.zv")
    with zb.shard_transaction(clean, (0, 0, 0)):
        _write(clean, SHARD0, tag=4.0)
    want = _shard_files(clean)
    # Idempotent: a second run of the same task changes no byte.
    with zb.shard_transaction(clean, (0, 0, 0)):
        _write(clean, SHARD0, tag=4.0)
    assert _shard_files(clean) == want

    crashed = _level(tmp_path / "crashed" / "s.zv")
    with zb.open_write_session(crashed, bounds=BOUNDS, chunk_shape=CHUNK):
        _write(crashed, SHARD0, tag=1.0)  # an earlier attempt's cells
    real = os.replace
    calls = []

    def dies_after_one(src, dst):
        calls.append(dst)
        if len(calls) > 1:
            raise OSError("node lost")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", dies_after_one)
    with pytest.raises(OSError, match="node lost"):
        with zb.shard_transaction(crashed, (0, 0, 0)):
            _write(crashed, SHARD0, tag=4.0)
    monkeypatch.setattr(os, "replace", real)
    # Half published: the retry starts from that and converges anyway.
    with zb.shard_transaction(crashed, (0, 0, 0)):
        _write(crashed, SHARD0, tag=4.0)
    assert _shard_files(crashed) == want
    assert not list((tmp_path / "crashed").rglob("*.partial"))


def test_an_append_inside_starts_from_the_transaction_not_the_last_attempt(tmp_path):
    level = _level(tmp_path / "s.zv")
    old = [[((0, 0, 0), 7), ((0, 0, 0), 8)]]
    zb.write_link_cells(level, old, sid_ndim=3)  # a failed attempt's record
    with zb.shard_transaction(level, (0, 0, 0)):
        zb.write_link_cells(level, [[((0, 0, 0), 1), ((0, 0, 0), 2)]], sid_ndim=3)
        zb.write_link_cells(level, [[((1, 0, 0), 3), ((1, 0, 0), 4)]], sid_ndim=3)
        zb.write_link_cells(level, [[((0, 0, 0), 5), ((0, 0, 0), 6)]], sid_ndim=3)
    zb.rebuild_presence(level)
    zb.finalize_links(level, delta=0)
    got = sorted(tuple(r) for r in zb.read_links(level, delta=0))
    assert got == sorted([
        (((0, 0, 0), 1), ((0, 0, 0), 2)),
        (((1, 0, 0), 3), ((1, 0, 0), 4)),
        (((0, 0, 0), 5), ((0, 0, 0), 6)),
    ])


@pytest.mark.parametrize("mode", ["replace", "merge"])
def test_an_array_form_link_append_reads_the_transaction(tmp_path, mode):
    """``write_link_cells(chunks=, vids=)`` prefetches the cells it appends to.

    That prefetch reads the store; inside a transaction the cells the
    transaction holds must win, or a cell cleared in the transaction (a
    retried task's reset, under ``merge``) keeps the store's old rows.
    """
    level = _level(tmp_path / "s.zv")
    zb.create_links_array(level, 2, delta=0, sid_ndim=3, offsets=((1, 0, 0),))
    chunks = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.int64)
    zb.write_link_cells(level, chunks=chunks, vids=np.array([[7, 8]]), sid_ndim=3)
    seam = zb.links_path(0, ((1, 0, 0),))
    with zb.shard_transaction(level, (0, 0, 0), mode=mode):
        if mode == "merge":
            level.write_bytes(seam, "0.0.0", b"", record_presence=False)
        zb.write_link_cells(level, chunks=chunks, vids=np.array([[1, 2]]), sid_ndim=3)
        zb.write_link_cells(level, chunks=chunks, vids=np.array([[3, 4]]), sid_ndim=3)
    zb.rebuild_presence(level)
    zb.finalize_links(level, delta=0)
    got = sorted((int(a[1]), int(b[1])) for a, b in zb.read_links(level, delta=0))
    assert got == [(1, 2), (3, 4)]


def test_what_was_written_is_the_presence(tmp_path):
    level = _level(tmp_path / "s.zv")
    written: dict[str, set[str]] = {}
    for shard, cells in (((0, 0, 0), SHARD0), ((1, 0, 0), SHARD1)):
        assert zb.shard_of(level, cells[0]) == shard
        with zb.shard_transaction(level, shard) as tx:
            _write(level, cells)
        for name, keys in tx.written.items():
            written.setdefault(name, set()).update(keys)
    names = zb.per_chunk_array_paths(level)
    zb.set_presence(level, {n: written.get(n, ()) for n in names}, end_deferral=True)
    expected = sorted(".".join(map(str, c)) for c in SHARD0 + SHARD1)
    assert level.list_chunks("vertices") == expected
    assert level.list_chunks("vertex_attributes/w") == expected


def test_the_preconditions(tmp_path):
    with pytest.raises(StoreError, match="presence deferred"):
        with zb.shard_transaction(_level(tmp_path / "a.zv", defer=False), (0, 0, 0)):
            pass
    with pytest.raises(zb.ArrayError, match="not sharded"):
        with zb.shard_transaction(_level(tmp_path / "b.zv", shard_shape=None), (0, 0, 0)):
            pass
    with pytest.raises(zb.ArrayError, match="outside"):
        with zb.shard_transaction(_level(tmp_path / "c.zv"), (9, 0, 0)):
            pass


@pytest.mark.parametrize("durable", [True, False])
def test_durability_is_what_was_asked(tmp_path, monkeypatch, durable):
    level = _level(tmp_path / "s.zv")
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    with zb.shard_transaction(level, (0, 0, 0), durable=durable):
        _write(level, SHARD0)
    assert bool(calls) is durable


def test_a_memory_store_publishes_by_set():
    import zarr

    root = zb.create_store(
        zarr.storage.MemoryStore(), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=2,
    )
    level = zb.get_resolution_level(root, 0)
    with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
        zb.create_vertices_array(level, dtype="float32")
    zb.defer_presence(level)
    with zb.shard_transaction(level, (0, 0, 0)) as tx:
        zb.write_chunk_vertices(level, (1, 0, 0), [_points((1, 0, 0), 2, 0.0)])
    assert tx.written["vertices"] == ["1.0.0"]
    assert _vertices(level, (1, 0, 0)).shape == (2, 3)
