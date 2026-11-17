"""Shard transactions: stage one shard's cell writes, publish them on exit.

Inside the block every cell written to the level, by any writer, is held in
memory and must lie in the owned shard; reads of owned cells see the staging.
On exit each touched shard is encoded by zarr's sharding codec, written as
``<object>.<token>.partial``, fsynced and renamed into place. Atomic per
object, not per set: after a crash mid-publish a re-run converges (same cells,
same bytes). Writes no metadata; needs presence deferred and arrays allocated
beforehand. See :func:`zarr_vectors.building.shard_transaction`.

``prefetch=True`` (merge mode): on entry every owned shard object is read
whole, in the background, while the caller computes. Reads of owned cells
the transaction has not staged are then served from that copy, decoded once
per array, and so is the merge at publish; a shard every cell of which the
transaction rewrites is encoded without reading what it replaces. On a
filesystem where each read is a round trip (NFS, a FUSE mount, a rotational
disk) this turns many small reads on the caller's path into one parallel
read per shard off it. Same bytes either way.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from zarr.storage import LocalStore, WrapperStore

from zarr_vectors.exceptions import ArrayError, ShardOwnershipError, StoreError

if TYPE_CHECKING:  # pragma: no cover
    from zarr_vectors.core.group import Group

_T = TypeVar("_T")
_R = TypeVar("_R")

# What an owned shard publishes when zarr found nothing in it to change
# (merge mode): nothing.
_UNCHANGED: Any = object()


@dataclass
class _Owned:
    """One array's owned shard."""

    name: str
    arr: Any                    # the zarr array (live store)
    origin: tuple[int, ...] | None
    lo: tuple[int, ...]         # first owned index, per axis
    hi: tuple[int, ...]         # one past the last (clipped to the shape)
    key: str                    # store key of the shard object


class _StagingStore(WrapperStore):
    """The live store, except that writes are captured, not made.

    ``hidden`` keys read as absent, so zarr encodes a replaced shard from
    the transaction's cells alone; under ``merge`` nothing is hidden and
    zarr merges into what the store holds.
    """

    def __init__(
        self, store: Any, hidden: Iterable[str],
        preloaded: dict[str, bytes | None] | None = None,
    ) -> None:
        super().__init__(store)
        self.hidden = set(hidden)
        self.captured: dict[str, bytes | None] = {}
        # Objects already read (None: absent), served instead of the store.
        self.preloaded = dict(preloaded or {})

    def _staged(self, key: str) -> tuple[bool, bytes | None]:
        if key in self.captured:
            return True, self.captured[key]
        if key in self.hidden:
            return True, None
        if key in self.preloaded:
            return True, self.preloaded[key]
        return False, None

    async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
        staged, value = self._staged(key)
        if not staged:
            return await self._store.get(key, prototype, byte_range)
        if value is None:
            return None
        return prototype.buffer.from_bytes(_slice(value, byte_range))

    async def get_partial_values(self, prototype: Any, key_ranges: Any) -> list[Any]:
        return [await self.get(key, prototype, rng) for key, rng in key_ranges]

    async def exists(self, key: str) -> bool:
        staged, value = self._staged(key)
        if staged:
            return value is not None
        return await self._store.exists(key)

    async def set(self, key: str, value: Any) -> None:
        self.captured[key] = bytes(value.to_bytes())

    async def set_if_not_exists(self, key: str, value: Any) -> None:
        if not await self.exists(key):
            await self.set(key, value)

    async def delete(self, key: str) -> None:
        self.captured[key] = None


def _slice(value: bytes, byte_range: Any) -> bytes:
    if byte_range is None:
        return value
    from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

    if isinstance(byte_range, RangeByteRequest):
        return value[byte_range.start:byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return value[byte_range.offset:]
    if isinstance(byte_range, SuffixByteRequest):
        return value[-byte_range.suffix:] if byte_range.suffix else b""
    raise TypeError(f"unsupported byte range {byte_range!r}")


class ShardTransaction:
    """One shard of a level, written in memory and published on exit.

    Use through :func:`zarr_vectors.building.shard_transaction`.

    Attributes:
        written: ``{array_name: [chunk_key, ...]}``, sorted: the non-empty
            cells this transaction wrote, filled in as it publishes.
            Under ``replace`` it is exactly each touched array's shard
            content -- what :func:`~zarr_vectors.building.set_presence`
            wants.
        published: The store keys published (written or removed).
    """

    def __init__(
        self,
        level_group: Group,
        shard_coords: Sequence[int],
        *,
        arrays: Iterable[str] | None,
        mode: Literal["replace", "merge"],
        durable: bool | str,
        io_threads: int | None = None,
        prefetch: bool = False,
    ) -> None:
        from zarr_vectors.core.group import _grid_origin

        if mode not in ("replace", "merge"):
            raise ArrayError(f"mode={mode!r}; expected 'replace' or 'merge'")
        if io_threads is not None and (
            isinstance(io_threads, bool) or not isinstance(io_threads, int)
            or io_threads < 1
        ):
            raise ArrayError(f"io_threads={io_threads!r}; expected None or an int >= 1")
        self.level_group = level_group
        self.shard_coords = tuple(int(s) for s in shard_coords)
        self.mode = mode
        from zarr_vectors.core._durable import fsyncs

        # True fsyncs each object and directory on publish; False and
        # "deferred" publish atomically without (see _durable.sync_store).
        self.durable = fsyncs(durable)
        self.io_threads = io_threads
        # Merge mode only: replace never reads what it replaces.
        self.prefetch = bool(prefetch) and mode == "merge"
        self._loaded: dict[str, Any] = {}          # name -> Future[bytes | None]
        self._decoded: dict[str, dict[tuple[int, ...], bytes]] = {}
        self._decode_lock = threading.Lock()
        self._prefetch_pool: Any = None
        self.written: dict[str, list[str]] = {}
        self.published: list[str] = []
        self._cells: dict[str, dict[str, bytes]] = {}
        if arrays is None:
            from zarr_vectors.building import per_chunk_array_paths

            arrays = per_chunk_array_paths(level_group)
        self._owned: dict[str, _Owned] = {}
        for name in arrays:
            arr = level_group._sharded_chunk_array(name)
            if arr is None:
                raise StoreError(f"shard_transaction: {name!r} is not a per-chunk array")
            if arr.shards is None:
                raise ArrayError(
                    f"shard_transaction: {name!r} is not sharded; a transaction "
                    f"publishes whole shard objects"
                )
            shard = tuple(int(s) for s in arr.shards)
            if len(shard) != len(self.shard_coords):
                raise ArrayError(
                    f"shard_transaction: shard_coords {self.shard_coords} has rank "
                    f"{len(self.shard_coords)}; {name!r} has rank {len(shard)}"
                )
            shape = tuple(int(n) for n in arr.shape)
            lo = tuple(s * k for s, k in zip(self.shard_coords, shard))
            hi = tuple(min(a + k, n) for a, k, n in zip(lo, shard, shape))
            if any(a < 0 or a >= b for a, b in zip(lo, hi)):
                raise ArrayError(
                    f"shard_transaction: shard {self.shard_coords} is outside "
                    f"{name!r}'s grid of {shape} cells"
                )
            key = arr.metadata.encode_chunk_key(self.shard_coords)
            self._owned[name] = _Owned(
                name, arr, _grid_origin(arr), lo, hi,
                f"{arr.path}/{key}" if arr.path else key,
            )

    # ---------------------------------------------------------------
    # Inside the block

    def _index(self, owned: _Owned, chunk_key: str) -> tuple[int, ...] | None:
        from zarr_vectors.core.group import _coord_to_index, _parse_chunk_coords

        coords = _parse_chunk_coords(chunk_key)
        if coords is None:
            return None
        index = _coord_to_index(coords, owned.origin)
        if len(index) != len(owned.lo):
            return None
        return index

    def owns(self, array_name: str, chunk_key: str) -> bool:
        """Whether ``array_name``'s cell ``chunk_key`` is in this shard."""
        owned = self._owned.get(array_name)
        if owned is None:
            return False
        index = self._index(owned, chunk_key)
        return index is not None and all(
            a <= i < b for i, a, b in zip(index, owned.lo, owned.hi)
        )

    def write_cells(self, array_name: str, cells: Iterable[tuple[str, bytes]]) -> int:
        """Stage cells; each must lie in the owned shard. Last write wins."""
        staged = self._cells.setdefault(array_name, {})
        n = 0
        for chunk_key, data in cells:
            if not self.owns(array_name, chunk_key):
                where = (
                    "not one of this transaction's arrays"
                    if array_name not in self._owned
                    else f"outside shard {self.shard_coords}"
                )
                raise ShardOwnershipError(
                    f"{array_name!r} cell {chunk_key!r} is {where}"
                )
            staged[chunk_key] = bytes(data)
            n += 1
        return n

    def lookup(self, array_name: str, chunk_key: str) -> bytes | None:
        """The transaction's view of a cell, or None to read the store."""
        staged = self._cells.get(array_name)
        if staged is not None and chunk_key in staged:
            return staged[chunk_key]
        if self.mode == "replace" and self.owns(array_name, chunk_key):
            return b""
        if array_name in self._loaded and self.owns(array_name, chunk_key):
            owned = self._owned[array_name]
            return self._prefetched_cells(array_name).get(
                self._index(owned, chunk_key), b"")
        return None

    def serves(self, array_name: str, chunk_key: str) -> bool:
        """Whether :meth:`lookup` answers for this cell without the store."""
        staged = self._cells.get(array_name)
        if staged is not None and chunk_key in staged:
            return True
        return (self.mode == "replace" or array_name in self._loaded) and self.owns(
            array_name, chunk_key)

    # ---------------------------------------------------------------
    # Prefetch (merge mode)

    def start_prefetch(self) -> None:
        """Read every owned shard object in the background (``prefetch=True``)."""
        if not self.prefetch or not self._owned or self._prefetch_pool is not None:
            return
        from concurrent.futures import ThreadPoolExecutor

        store = self.level_group.zarr_group.store
        lanes = max(4, int(self.io_threads or 0))
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=min(lanes, len(self._owned)), thread_name_prefix="zv-txn-prefetch",
        )
        for name, owned in self._owned.items():
            self._loaded[name] = self._prefetch_pool.submit(_read_object, store, owned.key)

    def _shard_bytes(self, name: str) -> bytes | None:
        return self._loaded[name].result()

    def _prefetched_cells(self, name: str) -> dict[tuple[int, ...], bytes]:
        """Every non-empty cell of ``name``'s owned shard as prefetched, by index; decoded once."""
        with self._decode_lock:
            cells = self._decoded.get(name)
            if cells is None:
                cells = self._decode_shard(name)
                self._decoded[name] = cells
            return cells

    def _decode_shard(self, name: str) -> dict[tuple[int, ...], bytes]:
        import warnings

        from zarr.errors import UnstableSpecificationWarning

        from zarr_vectors.core._batch_writer import _array_on

        raw = self._shard_bytes(name)
        if raw is None:
            return {}
        owned = self._owned[name]
        store = _StagingStore(self.level_group.zarr_group.store, [], {owned.key: raw})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            region = _array_on(owned.arr, store)[
                tuple(slice(a, b) for a, b in zip(owned.lo, owned.hi))]
        out: dict[tuple[int, ...], bytes] = {}
        for idx in np.ndindex(region.shape):
            value = region[idx]
            if isinstance(value, (bytes, bytearray, memoryview)) and len(value):
                out[tuple(a + i for a, i in zip(owned.lo, idx))] = bytes(value)
        return out

    def close(self) -> None:
        """Stop the prefetch (on publish or abort)."""
        pool, self._prefetch_pool = self._prefetch_pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    # ---------------------------------------------------------------
    # On exit

    def _encode_one(self, name: str) -> tuple[str, Any, list[str] | None]:
        """``(key, value, written)`` for one owned shard: new bytes, ``None`` to
        remove, or :data:`_UNCHANGED`; ``written`` is ``None`` if no cells were
        staged. Touches no shared state, so arrays encode concurrently."""
        import warnings

        from zarr.errors import UnstableSpecificationWarning

        from zarr_vectors.core._batch_writer import _array_on

        owned = self._owned[name]
        cells = self._cells.get(name, {})
        if not cells:
            return owned.key, (None if self.mode == "replace" else _UNCHANGED), None
        store = self.level_group.zarr_group.store
        region = int(np.prod([b - a for a, b in zip(owned.lo, owned.hi)]))
        # Every cell rewritten: nothing of the old shard survives, so do
        # not read it (the same bytes as merging over it).
        hide = self.mode == "replace" or len(cells) >= region
        preloaded = (
            {owned.key: self._shard_bytes(name)}
            if not hide and name in self._loaded else None
        )
        staging = _StagingStore(store, [owned.key] if hide else [], preloaded)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            # From the handle's metadata: opening by path re-reads zarr.json.
            arr = _array_on(owned.arr, staging)
            selection = tuple(
                np.asarray(axis, dtype=np.intp)
                for axis in zip(*(self._index(owned, k) for k in cells))
            )
            values = np.empty(len(cells), dtype=object)
            values[:] = list(cells.values())
            arr.set_coordinate_selection(selection, values)
        stray = set(staging.captured) - {owned.key}
        if stray:
            raise StoreError(
                f"shard_transaction: writing {name!r} touched {sorted(stray)} "
                f"besides its shard {owned.key!r}"
            )
        # Nothing captured means zarr found nothing to change.
        if owned.key in staging.captured:
            value = staging.captured[owned.key]
        elif self.mode == "replace":
            value = None
        else:
            value = _UNCHANGED
        return owned.key, value, sorted(k for k, v in cells.items() if v)

    def _encode(self) -> dict[str, bytes | None]:
        """Every owned shard object this transaction changes: bytes, or None to remove."""
        return self._collect(
            _map_lanes(self._encode_one, list(self._owned), self._threads()),
        )

    def _collect(
        self, encoded: Iterable[tuple[str, Any, list[str] | None]],
    ) -> dict[str, bytes | None]:
        """``{key: bytes | None}`` from :meth:`_encode_one`'s results, in
        array order, recording :attr:`written` as it goes."""
        out: dict[str, bytes | None] = {}
        for name, (key, value, written) in zip(self._owned, encoded):
            if value is not _UNCHANGED:
                out[key] = value
            if written is not None:
                self.written[name] = written
        return out

    def _threads(self) -> int:
        """Lanes to publish on: 1 unless asked, and always 1 on icechunk,
        whose session must see the writes made through it one at a time."""
        from zarr_vectors.core._batch_writer import _NO_THREADS, _is_icechunk_store

        if (
            self.io_threads is None or self.io_threads <= 1 or _NO_THREADS
            or _is_icechunk_store(self.level_group.zarr_group.store)
            # A job on the shared pool that waits on the pool can deadlock it.
            or threading.current_thread().name.startswith(_POOL_PREFIX)
        ):
            return 1
        return int(self.io_threads)

    def publish(self) -> None:
        store = self.level_group.zarr_group.store
        if isinstance(store, LocalStore):
            self._publish_local(Path(store.root))
            return
        objects = self._encode()
        from zarr.core.buffer import default_buffer_prototype
        from zarr.core.sync import sync

        together = self._threads() > 1

        async def _put() -> None:
            proto = default_buffer_prototype()

            async def one(key: str, value: bytes | None) -> None:
                if value is None:
                    await store.delete(key)
                else:
                    await store.set(key, proto.buffer.from_bytes(value))

            if together:
                await asyncio.gather(*(one(k, v) for k, v in objects.items()))
            else:
                for key, value in objects.items():
                    await one(key, value)

        sync(_put())
        self.published = sorted(objects)

    def _publish_local(self, root: Path) -> None:
        """Encode, write and fsync each object beside its target (on the
        pool when asked), then rename them all into place."""
        from zarr_vectors.core._durable import Touched

        touched = Touched()
        partials: list[str] = []      # appended to from the lanes

        def stage(name: str) -> tuple[tuple[str, Any, list[str] | None], str | None]:
            key, value, written = self._encode_one(name)
            tmp = None
            if isinstance(value, bytes):
                # Not fsynced here: every partial is written first, then all
                # are fsynced together below, before any rename (group commit).
                tmp = _write_partial(
                    str(root / key), value,
                    durable=False, touched=touched, partials=partials,
                )
            return (key, value, written), tmp

        try:
            staged = _map_lanes(stage, list(self._owned), self._threads())
            if self.durable:
                _map_lanes(_fsync_path, [tmp for _e, tmp in staged if tmp], self._threads())
        except BaseException:
            _remove_all(partials)
            raise
        objects = self._collect(encoded for encoded, _tmp in staged)
        renames = [
            (tmp, str(root / key)) for (key, _value, _written), tmp in staged if tmp
        ]
        _publish_renames(root, objects, renames, touched, durable=self.durable)
        self.published = sorted(objects)

    def sweep_partials(self) -> None:
        """Remove ``.partial`` files an interrupted attempt left (local stores)."""
        store = self.level_group.zarr_group.store
        if not isinstance(store, LocalStore):
            return
        root = Path(store.root)

        def sweep(target: Path) -> None:
            if target.parent.is_dir():
                for stale in target.parent.glob(f"{target.name}.*.partial"):
                    stale.unlink(missing_ok=True)

        _map_lanes(sweep, [root / o.key for o in self._owned.values()], self._threads())


#: Thread-name prefix of zarr-vectors' shared writer pool.
_POOL_PREFIX = "zv-write"


def _map_lanes(fn: Callable[[_T], _R], items: list[_T], lanes: int) -> list[_R]:
    """``[fn(x) for x in items]`` on up to ``lanes`` shared-pool threads. Every
    lane finishes before this returns or raises; after a failure lanes stop,
    and the earliest failing item's exception is raised."""
    lanes = min(int(lanes), len(items))
    if lanes <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import wait

    from zarr_vectors.core._batch_writer import _write_pool

    results: list[Any] = [None] * len(items)
    errors: dict[int, BaseException] = {}
    stop = threading.Event()

    def lane(first: int) -> None:
        for i in range(first, len(items), lanes):
            if stop.is_set():
                return
            try:
                results[i] = fn(items[i])
            except BaseException as exc:  # noqa: BLE001 - raised below
                errors[i] = exc
                stop.set()
                return

    pool = _write_pool()
    futures = [pool.submit(lane, k) for k in range(lanes)]
    try:
        wait(futures)
    except BaseException:
        stop.set()
        wait(futures)
        raise
    if errors:
        raise errors[min(errors)]
    return results


def _write_partial(
    path: str, value: bytes, *, durable: bool, touched: Any, partials: list[str],
) -> str:
    """Write ``value`` to ``<path>.<token>.partial`` (fsynced if ``durable``) and
    return its path; recorded in ``partials`` before opening, removed on failure."""
    touched.makedirs(os.path.dirname(path))
    tmp = f"{path}.{uuid.uuid4().hex}.partial"
    partials.append(tmp)
    try:
        with open(tmp, "wb") as fh:
            fh.write(value)
            if durable:
                fh.flush()
                os.fsync(fh.fileno())
    except BaseException:
        _remove_all([tmp])
        raise
    return tmp


def _read_object(store: Any, key: str) -> bytes | None:
    """One whole store object, or None when absent (the prefetch's read)."""
    if isinstance(store, LocalStore):
        try:
            with open(Path(store.root) / key, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.sync import sync

    buf = sync(store.get(key, default_buffer_prototype()))
    return None if buf is None else bytes(buf.to_bytes())


def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_all(paths: Iterable[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _publish_renames(
    root: Path,
    objects: dict[str, bytes | None],
    renames: list[tuple[str, str]],
    touched: Any,
    *,
    durable: bool,
) -> None:
    """Rename each ``(partial, target)`` into place, remove objects mapped to
    None, then fsync touched directories. A failure part-way leaves some objects
    new (a re-run converges) and removes the partials not yet renamed."""
    for i, (tmp, path) in enumerate(renames):
        try:
            os.replace(tmp, path)
        except BaseException:
            _remove_all(left for left, _ in renames[i:])
            raise
        touched.add(os.path.dirname(path))
    for key, value in objects.items():
        if value is None:
            path = root / key
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            touched.add(str(path.parent))
    if durable:
        touched.sync()
