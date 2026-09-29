"""Write one shard of every per-chunk array, then publish it all at once.

A task that owns a shard of a sharded store writes its cells straight
into the live store: each write is a read-modify-write of the shard
object, published as it happens. A retried task therefore starts from its
failed attempt's half-written shards -- and a link cell appends a row
group per call, so a retry doubles its records unless the task first
resets every cell it owns. A straggling first attempt and its retry can
both write. The caller has had to build resets, quarantines, per-shard
locks and its own fsync around every task.

A shard transaction replaces all of that with staging:

- inside the block, every cell written to this level -- through
  :meth:`Group.write_bytes` / :meth:`Group.write_cells`, so through every
  zarr-vectors writer -- is held in memory, and must lie in the owned
  shard (:class:`~zarr_vectors.exceptions.ShardOwnershipError`
  otherwise);
- reads of owned cells see the transaction: a cell written reads back
  its new bytes and, under ``mode="replace"``, a cell not yet written
  reads empty -- so an append starts from nothing, not from a failed
  attempt's rows;
- on exit each touched shard object is encoded by zarr's own sharding
  codec against a staging store, written beside its target as
  ``<object>.<token>.partial`` (fsynced when ``durable``), then renamed
  into place, and the directories fsynced. An exception before that
  publishes nothing.

Atomic per object, not per set: a crash while renaming leaves some
arrays' shards new and others old. Re-running the task converges: under
``replace`` its shards are rebuilt from its cells alone, and zarr encodes a
shard from its cells in a fixed order, so the same cells give the same
bytes.

The transaction writes no metadata. It requires the level's presence to
be deferred (``nonempty_chunks`` is shared by every cell) and every array
to be allocated beforehand: an array created by several tasks at once is
a race this cannot fix.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from zarr.storage import LocalStore, WrapperStore

from zarr_vectors.exceptions import ArrayError, ShardOwnershipError, StoreError

if TYPE_CHECKING:  # pragma: no cover
    from zarr_vectors.core.group import Group


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

    def __init__(self, store: Any, hidden: Iterable[str]) -> None:
        super().__init__(store)
        self.hidden = set(hidden)
        self.captured: dict[str, bytes | None] = {}

    def _staged(self, key: str) -> tuple[bool, bytes | None]:
        if key in self.captured:
            return True, self.captured[key]
        if key in self.hidden:
            return True, None
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
        durable: bool,
    ) -> None:
        from zarr_vectors.core.group import _grid_origin

        if mode not in ("replace", "merge"):
            raise ArrayError(f"mode={mode!r}; expected 'replace' or 'merge'")
        self.level_group = level_group
        self.shard_coords = tuple(int(s) for s in shard_coords)
        self.mode = mode
        self.durable = bool(durable)
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
        return None

    # ---------------------------------------------------------------
    # On exit

    def _encode(self) -> dict[str, bytes | None]:
        """Every owned shard object this transaction changes: bytes, or None to remove."""
        import warnings

        import zarr
        from zarr.errors import UnstableSpecificationWarning

        out: dict[str, bytes | None] = {}
        store = self.level_group.zarr_group.store
        for name, owned in self._owned.items():
            cells = self._cells.get(name, {})
            if not cells:
                if self.mode == "replace":
                    out[owned.key] = None
                continue
            staging = _StagingStore(store, [owned.key] if self.mode == "replace" else [])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnstableSpecificationWarning)
                arr = zarr.open_array(store=staging, path=owned.arr.path, mode="r+")
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
                out[owned.key] = staging.captured[owned.key]
            elif self.mode == "replace":
                out[owned.key] = None
            self.written[name] = sorted(k for k, v in cells.items() if v)
        return out

    def publish(self) -> None:
        objects = self._encode()
        store = self.level_group.zarr_group.store
        if isinstance(store, LocalStore):
            _publish_local(Path(store.root), objects, durable=self.durable)
        else:
            from zarr.core.buffer import default_buffer_prototype
            from zarr.core.sync import sync

            async def _put() -> None:
                proto = default_buffer_prototype()
                for key, value in objects.items():
                    if value is None:
                        await store.delete(key)
                    else:
                        await store.set(key, proto.buffer.from_bytes(value))

            sync(_put())
        self.published = sorted(objects)

    def sweep_partials(self) -> None:
        """Remove ``.partial`` files an interrupted attempt left (local stores)."""
        store = self.level_group.zarr_group.store
        if not isinstance(store, LocalStore):
            return
        root = Path(store.root)
        for owned in self._owned.values():
            target = root / owned.key
            if target.parent.is_dir():
                for stale in target.parent.glob(f"{target.name}.*.partial"):
                    stale.unlink(missing_ok=True)


def _publish_local(root: Path, objects: dict[str, bytes | None], *, durable: bool) -> None:
    """Write every object beside its target, then rename them all into place."""
    from zarr_vectors.core._durable import Touched

    touched = Touched()
    staged: list[tuple[str, str]] = []
    try:
        for key, value in objects.items():
            if value is None:
                continue
            path = str(root / key)
            touched.makedirs(os.path.dirname(path))
            tmp = f"{path}.{uuid.uuid4().hex}.partial"
            with open(tmp, "wb") as fh:
                fh.write(value)
                if durable:
                    fh.flush()
                    os.fsync(fh.fileno())
            staged.append((tmp, path))
    except BaseException:
        for tmp, _ in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    # The publish: renames and removals only, each atomic on its own. A
    # failure part-way leaves some objects new and some old (a re-run
    # converges); the temporaries not yet renamed go with it, and a crash
    # that leaves them is swept by the next attempt.
    for i, (tmp, path) in enumerate(staged):
        try:
            os.replace(tmp, path)
        except BaseException:
            for left, _ in staged[i:]:
                try:
                    os.remove(left)
                except OSError:
                    pass
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
