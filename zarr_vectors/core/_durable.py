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
filesystems (rotational disks, FUSE such as ntfs-3g) is several times
faster. Local filesystems only: on an object store a returned PUT is already
durable, on icechunk the commit.

Three durability levels (``durable=``): ``False`` (atomic, never fsynced),
``True`` (every object fsynced before its rename; the block returns with
everything on disk) and ``"deferred"`` (atomic like ``False``, with the
caller taking responsibility for durability later, once, through
:func:`sync_store`: one ``syncfs`` of the store's filesystem covering every
write of every process since). ``"deferred"`` is for callers that write
many objects from many tasks and only need them durable at a checkpoint of
their own, such as the end of a stage. It is refused on FUSE filesystems
(ntfs-3g, sshfs, ...): Linux's ``syncfs`` does not ask a FUSE filesystem to
flush, so it would return without the writes being durable.
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

    Raises:
        FileNotFoundError: ``target`` names a local path that does not exist
            (a typo would otherwise sync nothing and report success).
        OSError: the store is on a FUSE filesystem (ntfs-3g, sshfs, ...).
            Linux's ``syncfs`` does not ask a FUSE filesystem to flush, so
            nothing would be made durable; write with ``durable=True``
            there (group-committed, so the fsyncs are batched).

    Note that a write made on another host (a network filesystem) is only
    covered once that host has closed the file; NFS clients flush on close.
    """
    root = _local_root(target)
    if root is None:
        return "none"
    if sys.platform.startswith("linux"):
        fstype = _fs_type(root)
        if fstype is not None and (fstype == "fuse" or fstype.startswith(("fuse.", "fuseblk"))):
            raise OSError(
                f"sync_store: {root!r} is on a FUSE filesystem ({fstype}); syncfs does not "
                "flush FUSE filesystems, so durable='deferred' cannot be made durable there. "
                "Write with durable=True instead."
            )
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
        if not os.path.isdir(p):
            raise FileNotFoundError(f"sync_store: no store directory at {p!r}")
        return p
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


def _fs_type(path: str, mountinfo: str | None = None) -> str | None:
    """The filesystem type of the mount holding ``path`` (Linux), from
    ``/proc/self/mountinfo``; None when it cannot be read."""
    if mountinfo is None:
        try:
            with open("/proc/self/mountinfo", encoding="utf-8") as fh:
                mountinfo = fh.read()
        except OSError:
            return None
    real = os.path.realpath(path)
    best, best_type = "", None
    for line in mountinfo.splitlines():
        fields = line.split(" ")
        try:
            sep = fields.index("-")
        except ValueError:
            continue
        if len(fields) <= sep + 1 or len(fields) < 5:
            continue
        # Mount points escape spaces etc. as octal (\040).
        mount = fields[4].encode().decode("unicode_escape")
        inside = real == mount or real.startswith(mount.rstrip("/") + "/")
        if inside and len(mount) >= len(best):
            best, best_type = mount, fields[sep + 1]
    return best_type


class Touched:
    """Directories whose entries a block changed, to fsync at the end.

    With ``defer=True`` it also holds the block's written-but-unrenamed
    partials (:meth:`defer`) until :meth:`land` (or :meth:`discard`).
    """

    def __init__(self, *, defer: bool = False) -> None:
        self._dirs: set[str] = set()
        self._lock = threading.Lock()
        self.deferring = bool(defer)
        # path -> its latest partial, in the order first deferred.
        self._pending: dict[str, str] = {}

    def defer(self, tmp: str, path: str) -> None:
        """Hold ``tmp`` (written, not fsynced) to be renamed onto ``path`` by :meth:`land`.

        A path deferred again replaces its earlier partial, which is removed.
        """
        with self._lock:
            old = self._pending.pop(path, None)
            self._pending[path] = tmp
        if old is not None:
            _remove_quietly([old])

    def is_pending(self, path: str) -> bool:
        """Whether ``path`` has a held partial (written, not yet in place)."""
        with self._lock:
            return path in self._pending

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def drop(self, path: str) -> None:
        """Forget ``path``'s held partial and remove it (the path is being deleted)."""
        with self._lock:
            tmp = self._pending.pop(path, None)
        if tmp is not None:
            _remove_quietly([tmp])

    def land(self, lanes: int = 16) -> None:
        """Fsync every held partial (concurrently), then rename each into place.

        A path written twice holds only its later value. A failure removes
        the partials not yet renamed and re-raises.
        """
        with self._lock:
            pending, self._pending = [(t, p) for p, t in self._pending.items()], {}
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
            pending, self._pending = list(self._pending.values()), {}
        _remove_quietly(pending)

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
    """Fsync a regular file by path.

    Opened for writing: on Windows ``os.fsync`` (``FlushFileBuffers``) needs
    write access and fails on a read-only descriptor.
    """
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
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
            _fsync_path(p)
        except FileNotFoundError:
            continue


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
        # A write deferred in this block would otherwise land after the delete.
        self._touched.drop(self._path(key))
        await self._store.delete(key)
        self._touched.add(os.path.dirname(self._path(key)))

    # With a deferring Touched a written key is not in place until land().
    # A read of such a key (or a listing, which cannot tell) lands the held
    # partials first, so the block reads its own writes.  Reads of other keys
    # (zarr reads every cell before writing it) do not, which keeps the batch.

    async def _land_if(self, pending: bool) -> None:
        if pending:
            await asyncio.to_thread(self._touched.land)

    async def get(self, key: str, *args: Any, **kwargs: Any) -> Any:
        await self._land_if(self._touched.is_pending(self._path(key)))
        return await self._store.get(key, *args, **kwargs)

    async def get_partial_values(self, prototype: Any, key_ranges: Any) -> Any:
        key_ranges = list(key_ranges)
        await self._land_if(any(self._touched.is_pending(self._path(k)) for k, _ in key_ranges))
        return await self._store.get_partial_values(prototype, key_ranges)

    async def exists(self, key: str) -> bool:
        await self._land_if(self._touched.is_pending(self._path(key)))
        return await self._store.exists(key)

    async def list(self) -> Any:
        await self._land_if(self._touched.has_pending())
        async for k in self._store.list():
            yield k

    async def list_prefix(self, prefix: str) -> Any:
        await self._land_if(self._touched.has_pending())
        async for k in self._store.list_prefix(prefix):
            yield k

    async def list_dir(self, prefix: str) -> Any:
        await self._land_if(self._touched.has_pending())
        async for k in self._store.list_dir(prefix):
            yield k
