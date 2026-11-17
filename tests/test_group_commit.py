"""Group commit: durable writes fsync their objects together, after writing them all.

A durable write must fsync each object before the rename that makes it
visible. Interleaving that fsync with the writes (write, fsync, write,
fsync, ...) costs one flush per object; on slow-fsync filesystems (a
rotational disk, FUSE such as ntfs-3g) that measured ~46 ms per object
against ~5 ms when every object is written first and the fsyncs follow
together. Both durable paths -- a shard transaction's publish and a
durable ``batched_writes`` flush -- now write every partial, fsync them
all, then rename. The crash guarantee is unchanged: each object is
fsynced before its rename, and its directory after.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.core._durable import Touched, write_file

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads fsynced paths from /proc",
)

BOUNDS = ([0.0, 0.0, 0.0], [400.0, 400.0, 400.0])
CHUNK = (100.0, 100.0, 100.0)
CELLS = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (3, 2, 1)]
ATTRS = [f"a{i}" for i in range(6)]


@pytest.fixture
def events(monkeypatch, tmp_path):
    """``(kind, path, partials_on_disk)`` for every fsync and rename under ``tmp_path``.

    ``partials_on_disk`` is the set of ``.partial`` files present at the
    moment of the event, so a test can see what was already written.
    """
    log: list[tuple[str, str, frozenset[str]]] = []
    lock = threading.Lock()
    real_fsync, real_replace = os.fsync, os.replace
    root = str(tmp_path)

    def partials() -> frozenset[str]:
        return frozenset(str(p) for p in Path(root).rglob("*.partial"))

    def fsync(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        if path.startswith(root):
            with lock:  # snapshot before the fsync: what was written by then
                log.append(("fsync", path, partials()))
        real_fsync(fd)

    def replace(src, dst):
        if str(src).startswith(root):
            with lock:
                log.append(("rename", str(src), partials()))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    return log


def _is_cell(path: str) -> bool:
    # zarr's own metadata writes (``zarr.json``) land through zarr's
    # rename and are fsynced in place afterwards (``fsync_files``).
    return path.endswith(".partial") and not os.path.basename(path).startswith("zarr.")


def _assert_group_committed(log, *, max_commits: int) -> None:
    """Each cell partial fsynced before its rename; every partial a commit
    renames already written when that commit's first fsync ran; and at most
    ``max_commits`` commits (runs of fsyncs then renames), so the objects
    really were fsynced together rather than one write-fsync-rename each."""
    log = [e for e in log if e[0] != "rename" or _is_cell(e[1])]
    renamed = [path for kind, path, _ in log if kind == "rename" and path.endswith(".partial")]
    assert renamed, "nothing was published"
    synced_at: dict[str, int] = {}
    for i, (kind, path, _) in enumerate(log):
        if kind == "fsync" and path.endswith(".partial"):
            synced_at.setdefault(path, i)
    # Split the log into commits: a run of partial fsyncs, then its renames.
    commits = 0
    i = 0
    while i < len(log):
        if not (log[i][0] == "fsync" and log[i][1].endswith(".partial")):
            i += 1
            continue
        on_disk_at_first_fsync = log[i][2]
        j = i
        while j < len(log) and not (log[j][0] == "rename" and log[j][1].endswith(".partial")):
            j += 1
        k = j
        group: list[str] = []
        while k < len(log) and log[k][0] == "rename":
            group.append(log[k][1])
            k += 1
        for src in group:
            assert synced_at.get(src, len(log)) < log.index(next(e for e in log if e[0] == "rename" and e[1] == src)), \
                f"{src} renamed before its fsync"
        # The heart of group commit: nothing in this commit was written
        # after its fsyncs began.
        assert set(group) <= on_disk_at_first_fsync, "a partial was written after the fsyncs began"
        commits += 1
        i = k
    assert commits <= max_commits < len(renamed), (commits, len(renamed))


def _level(path, *, shard_shape=None, deferred=False):
    root = zb.create_store(str(path), bounds=BOUNDS, chunk_shape=CHUNK, shard_shape=shard_shape)
    level = zb.get_resolution_level(root, 0)
    if deferred:
        with zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK):
            zb.create_vertices_array(level, dtype="float32")
            for name in ATTRS:
                zb.create_attribute_array(level, name, dtype="float32")
        zb.defer_presence(level)
    return level


def _write(level, cells, *, attrs=()):
    for cell in cells:
        pts = np.full((3, 3), 50.0, np.float32) + np.asarray(cell) * 100.0
        zb.write_chunk_vertices(level, cell, [pts])
        for i, name in enumerate(attrs):
            zb.write_chunk_attributes(level, name, cell, [np.full(3, float(i), np.float32)])


def test_a_shard_transaction_writes_every_object_before_any_fsync(tmp_path, events):
    level = _level(tmp_path / "s.zv", shard_shape=2, deferred=True)
    events.clear()
    # One thread, so the order is deterministic: the old publish wrote and
    # fsynced each object in turn, and this must see every write first.
    with zb.shard_transaction(level, (0, 0, 0)) as tx:
        _write(level, [c for c in CELLS if max(c) < 2], attrs=ATTRS)
    _assert_group_committed(events, max_commits=1)
    assert len([e for e in events if e[0] == "rename"]) == len(tx.published)
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize("shard_shape", [None, 2], ids=["cells", "shards"])
def test_a_durable_flush_writes_every_object_before_any_fsync(
    tmp_path, events, shard_shape, monkeypatch,
):
    import zarr

    from zarr_vectors.core import _batch_writer

    # Serial, so the order is deterministic: arrays one at a time, and
    # zarr's writes within an array one at a time.
    monkeypatch.setattr(_batch_writer, "_NO_THREADS", True)
    level = _level(tmp_path / "s.zv", shard_shape=shard_shape)
    with zarr.config.set({"async.concurrency": 1}), \
            zb.open_write_session(level, bounds=BOUNDS, chunk_shape=CHUNK, durable=True):
        zb.create_vertices_array(level, dtype="float32")
        _write(level, CELLS)
    # Two land points per flush: the direct path's cells, then zarr's.
    _assert_group_committed(events, max_commits=2)
    assert not list(tmp_path.rglob("*.partial"))
    got = zb.read_chunk_vertices(level, (3, 2, 1))
    assert np.asarray(got[0]).shape == (3, 3)


def test_a_deferring_touched_publishes_only_on_land(tmp_path):
    t = Touched(defer=True)
    target = tmp_path / "d" / "x"
    write_file(str(target), b"new", t)
    assert not target.exists()
    assert len(list(tmp_path.rglob("*.partial"))) == 1
    t.land()
    assert target.read_bytes() == b"new"
    assert not list(tmp_path.rglob("*.partial"))


def test_land_renames_in_write_order(tmp_path):
    t = Touched(defer=True)
    target = tmp_path / "x"
    write_file(str(target), b"first", t)
    write_file(str(target), b"second", t)
    t.land()
    assert target.read_bytes() == b"second"


def test_discard_and_a_failed_land_leave_no_partials(tmp_path, monkeypatch):
    t = Touched(defer=True)
    write_file(str(tmp_path / "a"), b"a", t)
    t.discard()
    assert not list(tmp_path.iterdir())

    t = Touched(defer=True)
    write_file(str(tmp_path / "b"), b"b", t)
    write_file(str(tmp_path / "c"), b"c", t)

    def boom(fd):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="disk gone"):
        t.land()
    assert not list(tmp_path.rglob("*.partial"))
    assert not (tmp_path / "b").exists() and not (tmp_path / "c").exists()


def test_an_exclusive_write_is_not_deferred(tmp_path):
    t = Touched(defer=True)
    write_file(str(tmp_path / "x"), b"x", t, exclusive=True)
    assert (tmp_path / "x").read_bytes() == b"x"
    with pytest.raises(FileExistsError):
        write_file(str(tmp_path / "x"), b"y", t, exclusive=True)
