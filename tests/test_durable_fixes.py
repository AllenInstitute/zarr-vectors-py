"""Fixes to group commit and durable="deferred": the fsync helper's access
mode, sync_store's refusals, and a deferring store reading its own writes."""

from __future__ import annotations

import asyncio
import os

import pytest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore

from zarr_vectors.core import _durable
from zarr_vectors.core._durable import DurableLocalStore, Touched, _fs_type, sync_store

PROTO = default_buffer_prototype()


def test_fsync_opens_files_for_writing(tmp_path, monkeypatch):
    """Windows' FlushFileBuffers needs write access: a read-only fd fails."""
    f = tmp_path / "x"
    f.write_bytes(b"1")
    flags = []
    real_open = os.open

    def spy(p, fl, *a):
        flags.append(fl)
        return real_open(p, fl, *a)

    monkeypatch.setattr(_durable.os, "open", spy)
    _durable._fsync_path(str(f))
    _durable.fsync_files([str(f), str(tmp_path / "missing")])
    assert flags and all(fl & os.O_RDWR for fl in flags)


MOUNTINFO = "\n".join([
    "22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw",
    "40 22 8:1 / /hdd rw,relatime shared:2 - fuseblk /dev/sda1 rw,user_id=0",
    "41 22 0:50 / /mnt/my\\040disk rw - fuse.sshfs host: rw",
    "42 40 8:3 / /hdd/inner rw - ext4 /dev/sdb1 rw",
])


@pytest.mark.parametrize("path,want", [
    ("/home/me", "ext4"),
    ("/hdd", "fuseblk"),
    ("/hdd/data/store.zv", "fuseblk"),
    ("/hddx", "ext4"),  # a prefix of the name, not the mount
    ("/mnt/my disk/s", "fuse.sshfs"),
    ("/hdd/inner/s", "ext4"),  # the deepest mount wins
])
def test_fs_type_finds_the_deepest_mount(path, want, monkeypatch):
    monkeypatch.setattr(_durable.os.path, "realpath", lambda p: p)
    assert _fs_type(path, MOUNTINFO) == want


@pytest.mark.skipif(not os.sys.platform.startswith("linux"), reason="syncfs is Linux-only")
@pytest.mark.parametrize("fstype", ["fuseblk", "fuse", "fuse.sshfs"])
def test_sync_store_refuses_fuse(tmp_path, monkeypatch, fstype):
    monkeypatch.setattr(_durable, "_fs_type", lambda p: fstype)
    with pytest.raises(OSError, match="FUSE"):
        sync_store(str(tmp_path))


@pytest.mark.skipif(not os.sys.platform.startswith("linux"), reason="syncfs is Linux-only")
def test_sync_store_syncs_a_local_store(tmp_path):
    if (_fs_type(str(tmp_path)) or "").startswith("fuse"):
        pytest.skip("tmp_path is on FUSE")
    assert sync_store(str(tmp_path)) == "syncfs"


def test_sync_store_refuses_a_missing_local_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        sync_store(str(tmp_path / "no-such-store.zv"))
    assert sync_store("s3://bucket/store.zv") == "none"


def _store(tmp_path):
    touched = Touched(defer=True)
    return DurableLocalStore(LocalStore(str(tmp_path)), touched), touched


def _run(coro):
    return asyncio.run(coro)


def test_a_deferring_store_reads_its_own_writes(tmp_path):
    store, touched = _store(tmp_path)
    _run(store.set("a/k", PROTO.buffer.from_bytes(b"new")))
    assert touched.is_pending(str(tmp_path / "a" / "k"))
    assert _run(store.exists("a/k"))
    assert _run(store.get("a/k", PROTO)).to_bytes() == b"new"
    assert not touched.has_pending()  # the read landed it


def test_a_read_of_another_key_keeps_the_batch(tmp_path):
    store, touched = _store(tmp_path)
    _run(store.set("a/k", PROTO.buffer.from_bytes(b"x")))
    assert _run(store.get("a/other", PROTO)) is None
    assert touched.is_pending(str(tmp_path / "a" / "k"))


def test_delete_after_a_deferred_write_wins(tmp_path):
    store, touched = _store(tmp_path)
    _run(store.set("a/k", PROTO.buffer.from_bytes(b"x")))
    _run(store.delete("a/k"))
    touched.land()
    assert not (tmp_path / "a" / "k").exists()
    assert not [p for p in (tmp_path / "a").iterdir() if p.name.endswith(".partial")]


def test_a_path_written_twice_keeps_one_partial_and_the_later_value(tmp_path):
    store, touched = _store(tmp_path)
    _run(store.set("k", PROTO.buffer.from_bytes(b"1")))
    _run(store.set("k", PROTO.buffer.from_bytes(b"2")))
    assert len([p for p in tmp_path.iterdir() if p.name.endswith(".partial")]) == 1
    touched.land()
    assert (tmp_path / "k").read_bytes() == b"2"


def test_listing_lands_held_writes(tmp_path):
    store, _ = _store(tmp_path)
    _run(store.set("p/k", PROTO.buffer.from_bytes(b"x")))

    async def keys():
        return [k async for k in store.list_prefix("p/")]

    assert _run(keys()) == ["p/k"]
