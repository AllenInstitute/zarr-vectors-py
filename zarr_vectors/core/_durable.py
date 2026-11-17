"""Writes that are on disk when the block returns (``durable=True``).

zarr's temp-file-and-rename is atomic but not durable. A durable write
fsyncs each file *before* its rename (a rename can reach disk before its
data), then each directory it added an entry to, once, before the block
returns. zarr-vectors' direct writer uses :func:`write_file`; zarr's store
path is hooked through :class:`DurableLocalStore`.

Group commit: a :class:`Touched` made with ``defer=True`` holds each written
``.partial`` instead of fsyncing and renaming it at once; :meth:`Touched.land`
then fsyncs them all together and renames them. Each file is still fsynced
before its rename, so the crash guarantee is unchanged; what changes is that
the fsyncs no longer interleave with the writes, which on slow-fsync
filesystems (rotational disks, FUSE such as ntfs-3g) is several times faster. Local filesystems only:
on an object store a returned PUT is already durable, on icechunk the commit.

Three durability levels (``durable=``): ``False`` (atomic, never fsynced),
``True`` (every object fsynced before its rename; the block returns with
everything on disk) and ``"deferred"`` (atomic like ``False``, with the
caller taking responsibility for durability later, once, through
:func:`sync_store`: one ``syncfs`` of the store's filesystem covering every
write of every process since). ``"deferred"`` is for callers that write
many objects from many tasks and only need them durable at a checkpoint of
their own, such as the end of a stage.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zarr.storage import LocalStore, WrapperStore


#: ``durable=`` value: atomic writes, made durable later by :func:`sync_store`.
DEFERRED = "deferred"


def fsyncs(durable: Any) -> bool:
    """Whether a ``durable=`` value fsyncs as it writes; validates it.

    ``True`` does; ``False``/``None`` and ``"deferred"`` do not (``"deferred"``
    leaves durability to a later :func:`sync_store`).
    """
    if durable is True or durable is False or durable is None:
        return bool(durable)
    if durable == DEFERRED:
        return False
    raise ValueError(f"durable={durable!r}; expected True, False or {DEFERRED!r}")


def sync_store(target: Any) -> str:
    """Make every write already made to a local store durable, by any process.

    ``target`` is a store path, a zarr-vectors root or level :class:`Group`,
    or a zarr ``LocalStore``. On Linux this is one ``syncfs(2)`` of the
    filesystem holding the store: file data, metadata and directory entries
    written before the call, by this or any other process on this host, are
    on stable storage when it returns. Elsewhere it falls back to
    ``os.sync()``. A store that is not on a local filesystem needs nothing
    (an object store's returned PUT already is durable).

    Returns what it did: ``"syncfs"``, ``"sync"`` or ``"none"``.

    Note that a write made on another host (a network filesystem) is only
    covered once that host has closed the file; NFS clients flush on close.
    """
    root = _local_root(target)
    if root is None:
        return "none"
    if sys.platform.startswith("linux"):
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fd = os.open(root, os.O_RDONLY)
        try:
            if libc.syncfs(fd) != 0:
                err = ctypes.get_errno()
                raise OSError(err, f"syncfs({root!r}) failed: {os.strerror(err)}")
        finally:
            os.close(fd)
        return "syncfs"
    os.sync()
    return "sync"


def _local_root(target: Any) -> str | None:
    """The local directory behind ``target``, or None when it is not local."""
    if isinstance(target, (str, os.PathLike)):
        p = os.fspath(target)
        if "://" in p and not p.startswith("file://"):
            return None
        p = p[len("file://"):] if p.startswith("file://") else p
        return p if os.path.isdir(p) else None
    store = getattr(target, "store", None)          # a zarr Group
    zgroup = getattr(target, "zarr_group", None)   # a zarr-vectors Group
    if zgroup is not None:
        store = zgroup.store
    if store is None:
        store = target
    while isinstance(store, WrapperStore):
        store = store._store
    if isinstance(store, LocalStore):
        return str(store.root)
    return None


class Touched:
    """Directories whose entries a block changed, to fsync at the end.

    With ``defer=True`` it also holds the block's written-but-unrenamed
    partials (:meth:`defer`) until :meth:`land` (or :meth:`discard`).
    """

    def __init__(self, *, defer: bool = False) -> None:
        self._dirs: set[str] = set()
        self._lock = threading.Lock()
        self.deferring = bool(defer)
        self._pending: list[tuple[str, str]] = []

    def defer(self, tmp: str, path: str) -> None:
        """Hold ``tmp`` (written, not fsynced) to be renamed onto ``path`` by :meth:`land`."""
        with self._lock:
            self._pending.append((tmp, path))

    def land(self, lanes: int = 16) -> None:
        """Fsync every held partial (concurrently), then rename each into place.

        Renames run in the order the partials were deferred, so a path
        written twice ends with its later value. A failure removes the
        partials not yet renamed and re-raises.
        """
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return
        from zarr_vectors.core.shard_txn import _map_lanes

        try:
            _map_lanes(_fsync_path, [tmp for tmp, _ in pending], lanes)
        except BaseException:
            _remove_quietly(tmp for tmp, _ in pending)
            raise
        for i, (tmp, path) in enumerate(pending):
            try:
                os.replace(tmp, path)
            except BaseException:
                _remove_quietly(t for t, _ in pending[i:])
                raise
            self.add(os.path.dirname(path))

    def discard(self) -> None:
        """Remove every held partial (a failed block)."""
        with self._lock:
            pending, self._pending = self._pending, []
        _remove_quietly(tmp for tmp, _ in pending)

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


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_quietly(paths: Iterable[str]) -> None:
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


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
    When ``touched`` defers (and the write is not ``exclusive``), the
    temporary is written and handed to it; :meth:`Touched.land` fsyncs and
    renames it.
    """
    directory = os.path.dirname(path)
    touched.makedirs(directory)
    tmp = f"{path}.{uuid.uuid4().hex}.partial"
    defer = touched.deferring and not exclusive
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            if not defer:
                fh.flush()
                os.fsync(fh.fileno())
        if defer:
            touched.defer(tmp, path)
            return
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
