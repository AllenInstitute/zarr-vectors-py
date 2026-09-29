"""Writes that are on disk when the block returns (``durable=True``).

A local write lands the way zarr lands it: a temporary file, renamed over
the object. That is atomic -- a reader never sees half a cell -- but not
durable: neither the file nor its directory is fsynced, so after a power
loss or a node crash a cell whose write "returned" can be missing, or
empty. A caller that records "done" in a journal it does fsync needs the
cells to be at least as durable as the journal line.

So a durable write fsyncs each file before its rename (a rename can reach
the disk before the data it points at, which fsyncing afterwards does not
prevent), and then each directory it created an entry in, once, before
the block returns. Two paths write cells: zarr-vectors' own direct writer
(a local, unsharded array) and zarr's store (everything else, sharded
objects included); :class:`DurableLocalStore` is the second one's hook.

Only local filesystems are covered. On an object store a PUT that has
returned is the durability point, and there is nothing to add; on
icechunk it is the session's commit.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zarr.storage import LocalStore, WrapperStore


class Touched:
    """Directories whose entries a block changed, to fsync at the end."""

    def __init__(self) -> None:
        self._dirs: set[str] = set()
        self._lock = threading.Lock()

    def add(self, directory: str) -> None:
        with self._lock:
            self._dirs.add(directory)

    def makedirs(self, directory: str) -> None:
        """``os.makedirs``, recording each directory it had to create.

        A new directory is an entry in its parent, so the parent (the
        deepest one that already existed) is recorded too.
        """
        missing = []
        d = directory
        while d and not os.path.isdir(d):
            missing.append(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            self._dirs.update(missing)
            self._dirs.add(directory)
            if d:
                self._dirs.add(d)

    def sync(self) -> None:
        """Fsync every recorded directory, deepest first, then forget them."""
        with self._lock:
            dirs, self._dirs = sorted(self._dirs, key=len, reverse=True), set()
        fsync_dirs(dirs)


def fsync_dirs(dirs: Iterable[str]) -> None:
    """Fsync directories, so the entries in them survive a crash.

    Skipped on Windows, which cannot open a directory for fsync (its
    renames are journalled by NTFS instead).
    """
    if os.name == "nt":
        return
    for d in dirs:
        try:
            fd = os.open(d, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def fsync_files(paths: Iterable[str]) -> None:
    """Fsync files already written in place (metadata documents)."""
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_file(path: str, data: bytes, touched: Touched, *, exclusive: bool = False) -> None:
    """Write ``data`` to ``path`` durably: temporary, fsync, rename.

    The directory entry the rename makes is recorded in ``touched``, to be
    fsynced once for the whole block.
    """
    directory = os.path.dirname(path)
    touched.makedirs(directory)
    tmp = f"{path}.{uuid.uuid4().hex}.partial"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if exclusive:
            os.link(tmp, path)  # raises FileExistsError, as zarr's does
            os.unlink(tmp)
        else:
            os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    touched.add(directory)


class DurableLocalStore(WrapperStore[LocalStore]):
    """A :class:`LocalStore` whose writes are fsynced before their rename.

    Reads, listings and everything else pass through to the wrapped
    store. Opened only for the duration of one durable flush.
    """

    def __init__(self, store: LocalStore, touched: Touched) -> None:
        super().__init__(store)
        self._touched = touched

    def _path(self, key: str) -> str:
        return str(Path(self._store.root) / key)

    async def set(self, key: str, value: Any) -> None:
        self._store._check_writable()
        await asyncio.to_thread(
            write_file, self._path(key), bytes(value.to_bytes()), self._touched,
        )

    async def set_if_not_exists(self, key: str, value: Any) -> None:
        self._store._check_writable()
        try:
            await asyncio.to_thread(
                write_file, self._path(key), bytes(value.to_bytes()),
                self._touched, exclusive=True,
            )
        except FileExistsError:
            pass

    async def delete(self, key: str) -> None:
        await self._store.delete(key)
        self._touched.add(os.path.dirname(self._path(key)))
