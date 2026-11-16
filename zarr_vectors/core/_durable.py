"""Writes that are on disk when the block returns (``durable=True``).

zarr's temp-file-and-rename is atomic but not durable. A durable write
fsyncs each file *before* its rename (a rename can reach disk before its
data), then each directory it added an entry to, once, before the block
returns. zarr-vectors' direct writer uses :func:`write_file`; zarr's store
path is hooked through :class:`DurableLocalStore`. Local filesystems only:
on an object store a returned PUT is already durable, on icechunk the commit.
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
        """``os.makedirs``, recording each directory it created and the deepest existing parent."""
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
    """Fsync directories, so the entries in them survive a crash (no-op on Windows)."""
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

    The directory is recorded in ``touched``, to be fsynced once per block.
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

    Everything else passes through. Opened for one durable flush.
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
