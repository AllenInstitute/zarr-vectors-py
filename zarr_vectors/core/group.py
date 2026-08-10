"""Backend-agnostic group abstraction wrapping a :class:`zarr.Group`.

This is the format seam: all ZV array I/O routes through this class.
The underlying Zarr store can be any :class:`zarr.abc.store.Store` —
``LocalStore``, ``MemoryStore``, ``FsspecStore``, ``ObjectStore``,
``IcechunkStore``.

Every per-spatial-chunk array is ONE multidim vlen-bytes Zarr array whose
shape is the level's chunk grid; a cell holds that chunk's byte blob:

    level/vertices                → vlen-bytes array, shape = chunk grid
    level/vertex_fragments        → likewise
    level/links/<delta>/<offsets> → likewise (cell = the record's SOURCE chunk)

Per-array metadata (the ``zv_array`` discriminator and friends) lives on
the array node's own ``attributes`` block.  Group nodes remain for the
containers above those arrays — ``links/<delta>``, attribute namespaces,
``object_index`` — and carry family-wide metadata.

Public surface mirrors the legacy :class:`FsGroup` for back-compat:

* ``attrs`` — dict-like access to this group's attributes
* ``create_group`` / ``require_group`` / ``__getitem__`` / ``__contains__``
  / ``__iter__`` — hierarchy navigation (sub-groups only)
* ``write_bytes`` / ``read_bytes`` / ``chunk_exists`` / ``list_chunks`` —
  per-chunk byte I/O
* ``write_array_meta`` / ``read_array_meta`` / ``array_exists`` —
  per-array metadata
* ``path`` — :class:`pathlib.Path` when the backing store is a Zarr
  ``LocalStore``, raises otherwise (use ``url`` instead)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import VLenBytesCodec
from zarr.errors import UnstableSpecificationWarning
from zarr.storage import LocalStore

from zarr_vectors.core._vlen import (
    cell_region as _vlen_cell_region,
)
from zarr_vectors.core._vlen import (
    region_to_bytes as _vlen_region_to_bytes,
)
from zarr_vectors.exceptions import StoreError

# Node-cache sentinel for "this path was probed and is genuinely absent",
# as distinct from "this path was never prefetched".  Only the async
# primer in :mod:`zarr_vectors.core.aio` stores it; without it an offline
# lookup could not tell a legitimately missing array (``array_exists`` →
# False) from a hole in the prefetch plan, and would have to answer one
# of them wrongly.
_ABSENT = object()


class _OfflineSession:
    """The snapshot an offline read is served from, shared by a Group and
    every Group derived from it.

    Sub-groups are built fresh by :meth:`Group._from_zarr` (a level group
    is not the root Group), so per-instance caches would not reach the
    handle that actually does the reading.  One session object, threaded
    down through :meth:`Group.__getitem__` and friends, keeps the whole
    tree served from the same snapshot.

    Keys are **root-relative** paths, so they mean the same thing from
    any Group in the tree.

    ``misses`` is the reason this is a mutable object rather than two
    plain dicts: a lookup that cannot be served records itself here
    *before* raising.  Several ``read_*`` paths wrap optional metadata
    reads in ``except Exception: pass``, which would swallow the raise
    and silently degrade the result; the recorded miss survives that, and
    is what lets :func:`zarr_vectors.core.aio.read_async` discover what
    to fetch next instead of trusting the exception to propagate.

    Four kinds of read are covered, because the readers make four kinds:
    ``nodes`` resolves a path, ``chunks`` holds one vlen *cell* of a
    chunk-grid array, ``arrays`` holds a whole standalone array read
    end-to-end (the ``object_index`` manifests, object attributes), and
    ``listings`` holds a group's child names.

    ``listings`` is the one a browser cannot always fill: a plain
    fetch-backed Store has no listing operation.  Readers that enumerate
    a family — the ``links/<delta>`` offsets segments — need it, so on
    such a store those reads stay out of reach and say so, rather than
    hanging.  Stores with a real listing API (S3, GCS, local) fill it
    like any other dimension.
    """

    __slots__ = ("nodes", "chunks", "arrays", "listings", "misses")

    def __init__(
        self,
        nodes: dict[str, Any] | None = None,
        chunks: dict[tuple[str, str], bytes] | None = None,
        arrays: dict[str, Any] | None = None,
        listings: dict[str, list[str]] | None = None,
    ) -> None:
        self.nodes = nodes if nodes is not None else {}
        self.chunks = chunks if chunks is not None else {}
        self.arrays = arrays if arrays is not None else {}
        self.listings = listings if listings is not None else {}
        self.misses: set[Any] = set()


class Group:
    """A ZV group wrapping an underlying :class:`zarr.Group`."""

    # Class-level defaults so callers that build a Group via ``__new__``
    # (see ``create_store`` / ``open_store``) start with batching off
    # without needing to remember to set the attributes.
    # Queued cell writes: (array_name, chunk_key, data, record_presence).
    _pending_writes: list[tuple[str, str, bytes, bool]] | None = None
    _pending_array_metas: dict[str, dict[str, Any]] | None = None
    _prefetch_cache: dict[tuple[str, str], bytes] | None = None
    # Active codec spec for chunk-array writes, set by
    # :meth:`batched_writes(compressor=...)`.  ``None`` means callers fall
    # back to zarr v3's default codec pipeline (``bytes`` + ``zstd``).
    # Consumed by :meth:`write_bytes` and the batched flush in
    # :mod:`zarr_vectors.core._batch_writer`.
    _active_codecs: list[dict[str, Any]] | None = None
    # Explicit grid config for per-chunk-array creation, overriding what
    # ``arrays._derive_native_config`` would read off the store's
    # metadata.  Set by
    # :meth:`native_sharded_arrays`; the dict holds ``grid_shape`` and
    # ``shard_shape`` (``None`` = unsharded, one storage object per chunk
    # — the default layout).  Used by ``arrays._ensure_array_dir`` and the
    # ``write_bytes`` / ``write_array_meta`` dispatch on this class.
    _native_sharded_config: dict[str, tuple[int, ...] | None] | None = None
    # Node-lookup cache.  Maps a path → the resolved node.  Active for
    # the duration of a :meth:`batched_writes` block (Array nodes only,
    # populated lazily) or of an :meth:`offline_reads` block (Arrays,
    # Groups and ``_ABSENT`` markers, populated up-front by the async
    # primer).  See :meth:`_lookup_node`.
    _node_cache: dict[str, Any] | None = None
    # Active offline-read snapshot, or None for normal store-backed
    # reads.  When set, reads must not touch the store: a miss records
    # itself and raises rather than falling through to a synchronous GET.
    # Propagates to derived Groups — see :class:`_OfflineSession` and
    # :meth:`offline_reads`.  This is what makes a prefetch gap a loud
    # failure rather than a silent sync round-trip, which under Pyodide
    # is not merely slow but deadlocks the event loop.
    _offline: _OfflineSession | None = None

    def __init__(self, zarr_group: zarr.Group) -> None:
        self._zarr = zarr_group
        # Deferred-write queues activated by :meth:`batched_writes`.
        # When set, :meth:`write_bytes` appends to ``_pending_writes``
        # and :meth:`write_array_meta` appends to
        # ``_pending_array_metas``; both flush in one ``asyncio.gather``
        # against the underlying Store on context exit.
        self._pending_writes = None
        self._pending_array_metas = None
        # Prefetch cache activated by :meth:`batched_reads`.  When set,
        # :meth:`read_bytes` looks here first before hitting the store.
        self._prefetch_cache = None
        self._active_codecs = None
        self._native_sharded_config = None
        self._node_cache = None

    @classmethod
    def _from_zarr(
        cls, zarr_group: zarr.Group, _parent: Group | None = None,
    ) -> Group:
        instance = cls.__new__(cls)
        instance._zarr = zarr_group
        instance._pending_writes = None
        instance._pending_array_metas = None
        instance._prefetch_cache = None
        instance._active_codecs = None
        instance._node_cache = None
        # An offline snapshot covers the whole tree, so a Group derived
        # from one stays offline; without this a level group would fall
        # back to the store and issue the very sync read the snapshot
        # exists to avoid.  Everything else stays per-instance.
        instance._offline = _parent._offline if _parent is not None else None
        return instance

    @classmethod
    def _from_backend(cls, store_or_shim: Any, prefix: str = "") -> Group:
        """Build a Group from a Zarr store (or a legacy ``_BackendShim``).

        Kept for back-compat with callers in :mod:`zarr_vectors.lazy`
        that resurrect a root-level Group from a stored backend handle.
        Always returns a write-capable handle — a read-only store
        reference (left over from an ``open_store(mode='r')`` flow) is
        unwrapped via ``store.with_read_only(False)``.
        """
        store = store_or_shim._store if isinstance(store_or_shim, _BackendShim) else store_or_shim
        if getattr(store, "read_only", False) and hasattr(store, "with_read_only"):
            store = store.with_read_only(False)
        path = "/" + prefix.strip("/") if prefix else "/"
        zg = zarr.open_group(store, path=path, mode="r+")
        return cls._from_zarr(zg)

    # ---------------- attributes ----------------

    @property
    def attrs(self) -> _Attrs:
        return _Attrs(self._zarr.attrs)

    # ---------------- sub-groups ----------------

    def create_group(self, name: str, **_kwargs: Any) -> Group:
        zg = self._zarr.require_group(name)
        return type(self)._from_zarr(zg, self)

    def require_group(self, name: str) -> Group:
        zg = self._zarr.require_group(name)
        return type(self)._from_zarr(zg, self)

    def _full_path(self, path: str) -> str:
        """``path`` made root-relative, the key form an offline snapshot
        uses so a lookup means the same thing from any Group in the tree.
        """
        base = self._zarr.path.strip("/")
        return f"{base}/{path}" if base else path

    def __getitem__(self, key: str) -> Group:
        # Via ``_lookup_node`` rather than ``self._zarr[key]`` so this
        # shares the node cache — which both collapses the repeated
        # ``zarr.json`` GETs a level lookup would otherwise pay and lets
        # the call be served offline.
        node = self._lookup_node(key)
        if node is None:
            raise StoreError(
                f"Group {key!r} not found under {self._zarr.path or '<root>'}"
            )
        if not isinstance(node, zarr.Group):
            raise StoreError(
                f"{key!r} under {self._zarr.path or '<root>'} is a "
                f"{type(node).__name__}, not a Group"
            )
        return type(self)._from_zarr(node, self)

    def __contains__(self, key: str) -> bool:
        return self._lookup_node(key) is not None

    def __iter__(self) -> Iterator[str]:
        yield from sorted(self._zarr.group_keys())

    def children(self) -> list[str]:
        """Return every immediate child name — both arrays and groups.

        Use this wherever a parent path's children may be arrays rather
        than groups: ``links/<delta>`` and ``link_attributes/<name>/
        <delta>`` hold one array per offsets segment, and object / group
        attributes are flat arrays.  ``__iter__`` yields sub-*groups*
        only, so it sees none of those.
        """
        offline = self._offline
        if offline is not None:
            own = self._zarr.path.strip("/")
            hit = offline.listings.get(own)
            if hit is not None:
                return list(hit)
            # Speculative empty, same contract as an unresolved node in
            # :meth:`_lookup_node`: the miss is recorded and only a
            # zero-miss pass is accepted, so an enumeration that came up
            # short is re-run once the listing is in the snapshot.
            offline.misses.add(("list", own))
            return []
        return sorted(
            set(self._zarr.group_keys()) | set(self._zarr.array_keys())
        )

    # ---------------- chunk I/O ------------------------------------------
    #
    # ``<array_name>`` is always a single multidim vlen-bytes Zarr array
    # whose shape is the level's chunk grid.  Unsharded, each cell is its
    # own object at ``<array>/c/i/j/k``; under the Zarr v3
    # ``sharding_indexed`` codec many cells pack into one object.  Created
    # via :meth:`create_sharded_chunk_array` (directly, or through
    # ``arrays._ensure_array_dir``).
    #
    # Every ZV per-chunk family maps onto this — including both link
    # families, whose cell is the record's source chunk and whose
    # endpoint relationship is factored into the ``<offsets>`` path
    # segment rather than the cell key.  So ``read_bytes`` /
    # ``write_bytes`` / ``chunk_exists`` / ``list_chunks`` have a single
    # layout to serve.

    def write_bytes(
        self,
        array_name: str,
        chunk_key: str,
        data: bytes,
        *,
        record_presence: bool = True,
    ) -> None:
        """Write one spatial chunk's payload into ``array_name``'s cell.

        ``array_name`` must already hold the multidim vlen-bytes Zarr
        array whose shape is the level's chunk grid — every ZV
        per-spatial-chunk array (vertices, fragments, ``links/<delta>/
        <offsets>``, attributes) is one, allocated by
        :func:`zarr_vectors.core.arrays._ensure_array_dir`.

        Args:
            record_presence: When False, skip the ``nonempty_chunks``
                update.  That manifest is ONE attribute shared by every
                cell of the array, so stamping it is a read-modify-write
                of state outside the cell being written: two workers
                writing *disjoint* cells concurrently still race, and the
                loser's key vanishes from the manifest even though its
                payload is on disk.  Decentralized writers therefore pass
                False and leave the manifest to a coordinator's
                :meth:`derive_nonempty_chunks` (see
                :func:`zarr_vectors.core.arrays.finalize_links`), which
                rebuilds it from the store listing after all workers
                finish.  Cell payloads themselves are independent objects
                and never race.
        """
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is None:
            raise StoreError(
                f"Cannot write to {array_name!r} in "
                f"{self._zarr.path or '<root>'}: no chunk array at that "
                f"path. Per-chunk arrays must be allocated first (see "
                f"arrays._ensure_array_dir / create_sharded_chunk_array)."
            )
        coords = _parse_chunk_coords(chunk_key)
        if coords is None:
            raise StoreError(
                f"Cannot write to array {array_name!r}: "
                f"chunk_key {chunk_key!r} is not a coord tuple"
            )
        index = _coord_to_index(coords, _grid_origin(sharded_arr))
        _check_coords_in_bounds(index, sharded_arr.shape, array_name)
        # Batched mode: queue the cell write; the batch flush writes
        # every cell of an array in one concurrent
        # ``set_coordinate_selection`` and stamps ``nonempty_chunks``
        # once (see :mod:`zarr_vectors.core._batch_writer`).
        if self._pending_writes is not None:
            self._pending_writes.append(
                (array_name, chunk_key, bytes(data), record_presence)
            )
            return
        _vlen_set_cell(sharded_arr, index, bytes(data))
        if record_presence:
            _record_nonempty_chunk(sharded_arr, chunk_key, present=bool(data))

    @contextmanager
    def batched_reads(
        self,
        plan: list[tuple[str, list[str]]],
    ) -> Iterator[None]:
        """Prefetch every chunk in ``plan`` via one
        :func:`asyncio.gather` and serve subsequent :meth:`read_bytes`
        calls from the resulting in-memory cache.

        ``plan`` is a list of ``(array_name, [chunk_keys, ...])`` pairs
        — typically ``(VERTICES, list_chunk_keys(group, VERTICES))``
        plus the parallel ``vertex_fragments`` and per-attribute
        arrays.  On entry every (array_name, chunk_key) pair is fetched
        in a single async gather; on exit the cache is dropped.

        Reads for a key NOT in the plan fall through to the sync
        :meth:`read_bytes` path, so under-specifying the plan
        degrades performance gracefully (still correct).

        Use for chunk-heavy read loops against high-latency object
        stores (GCS / S3 / Azure).  Each per-chunk GET becomes one async
        task instead of one serial sync call, so the total wall time
        approaches one round-trip rather than ``N`` round-trips.

        Nesting is not supported and raises :class:`StoreError`.
        Writes inside the block are unaffected.

        Example::

            chunk_keys = list_chunk_keys(level_group, VERTICES)
            with level_group.batched_reads([
                (VERTICES, chunk_keys),
                (VERTEX_FRAGMENTS, chunk_keys),
                *((f"{VERTEX_ATTRIBUTES}/{a}", chunk_keys) for a in attrs),
            ]):
                for cc in chunk_keys:
                    fragments = read_chunk_vertices(level_group, cc, ...)
        """
        if self._prefetch_cache is not None:
            raise StoreError("batched_reads() does not support nesting")
        if self._offline is not None:
            # Under offline_reads (the async prime-and-replay path used by
            # aio.read_async), the chunks this plan would prefetch are already in
            # the session, and read_bytes serves them from there -- so the sync
            # prefetch is redundant. It is also unavailable: flush_prefetch calls
            # sync(), which under Pyodide needs WebAssembly stack switching
            # (JSPI). Skip it; any genuine miss is recorded by the offline
            # session and fetched by the next aio round.
            yield
            return
        from zarr_vectors.core._batch_reader import flush_prefetch

        self._prefetch_cache = flush_prefetch(self._zarr, plan)
        # No node cache here, deliberately: :meth:`read_bytes` answers from
        # ``_prefetch_cache`` before it resolves a node, so inside this block
        # there is nothing left for one to cache.  (Measured: adding one
        # changes the store round-trip count not at all.)
        try:
            yield
        finally:
            self._prefetch_cache = None

    @contextmanager
    def offline_reads(self, session: _OfflineSession) -> Iterator[None]:
        """Serve every read inside the block from ``session``, touching
        the store not at all.

        This is the replay half of the async read path.  The snapshot is
        built by :mod:`zarr_vectors.core.aio`, which resolves the nodes
        and fetches the chunk cells with ``await``; inside this block the
        ordinary synchronous ``read_*`` functions then run to completion
        against it.  The decode, filtering and assembly work they do is
        pure computation, so it needs no async treatment — it only ever
        needed its I/O supplied up front.

        That is why there is no async mirror of each ``read_*``: given a
        primed handle, the sync ones *are* the async ones.

        Any lookup or chunk read the snapshot does not cover records
        itself in ``session.misses`` and raises :class:`StoreError`
        rather than falling back to the store: a gap must surface, since
        the silent alternative is the synchronous round-trip this path
        exists to avoid.  The caller inspects ``misses`` to decide what
        to fetch next — see :func:`~zarr_vectors.core.aio.read_async`.

        Nesting is not supported and raises :class:`StoreError`.  The
        block is read-only by construction; writes inside it are not
        supported.
        """
        if self._offline is not None:
            raise StoreError("offline_reads() does not support nesting")
        if self._node_cache is not None or self._prefetch_cache is not None:
            raise StoreError(
                "offline_reads() cannot be combined with batched_reads() "
                "or batched_writes()"
            )
        self._offline = session
        try:
            yield
        finally:
            self._offline = None

    @contextmanager
    def batched_writes(self, compressor: Any = None) -> Iterator[None]:
        """Defer every :meth:`write_bytes` and :meth:`write_array_meta`
        call inside the block and flush them in a single
        :func:`asyncio.gather` on exit.

        Use for chunk-heavy write loops against high-latency object
        stores (GCS / S3 / Azure).  Each per-chunk PUT and each per-array
        ``zarr.json`` PUT becomes one async task instead of one serial
        sync call, so the total wall time approaches one round-trip
        rather than ``N`` round-trips.

        Args:
            compressor: Codec selection applied to every chunk array
                written inside the block.  See
                :func:`zarr_vectors.encoding.compression.resolve_compressor`
                for accepted values; the default ``None`` resolves to
                zarr v3's default (``bytes`` + ``zstd``).
                # TODO: per-array-type codec dict (vertices vs fragments
                # vs links) — future work; today every chunk gets the
                # same codec.

        Nesting is not supported and raises :class:`StoreError`.  Reads
        inside the block are unaffected and execute synchronously.

        Example::

            with level_group.batched_writes():
                create_vertices_array(level_group, dtype="float32")
                create_attribute_array(level_group, "intensity")
                for cc in chunk_coords:
                    write_chunk_vertices(level_group, cc, ...)
                    write_chunk_attributes(level_group, "intensity", cc, ...)
            # exit point: every PUT scheduled above flushes in parallel
        """
        if self._pending_writes is not None:
            raise StoreError("batched_writes() does not support nesting")
        from zarr_vectors.encoding.compression import resolve_compressor

        codecs = resolve_compressor(compressor)
        self._pending_writes = []
        self._pending_array_metas = {}
        self._active_codecs = codecs
        # Node lookups are cached for the life of the block only: inside
        # it every node we resolve is one we created, and on exit the
        # handles are dropped rather than left to go stale.  See
        # :meth:`_lookup_node`.
        self._node_cache = {}
        try:
            yield
            pending_writes = self._pending_writes
            pending_metas = self._pending_array_metas
            self._pending_writes = None
            self._pending_array_metas = None
            if pending_writes or pending_metas:
                # Lazy import to avoid pulling the asyncio/zarr-sync
                # machinery into the import path of every Group caller.
                from zarr_vectors.core._batch_writer import flush_batch

                flush_batch(
                    self._zarr,
                    pending_writes,
                    array_metas=pending_metas,
                    codecs=codecs,
                )
        finally:
            # On normal exit the queues are already None.  On an
            # exception, drop them so the Group stays usable.
            self._pending_writes = None
            self._pending_array_metas = None
            self._active_codecs = None
            # Drop the cached handles with the block that made them: the
            # flush above stamps ``nonempty_chunks`` through its own
            # freshly-resolved nodes, so anything held here is stale the
            # moment the session ends.
            self._node_cache = None

    @contextmanager
    def native_sharded_arrays(
        self,
        shard_shape: tuple[int, ...] | None,
        grid_shape: tuple[int, ...],
        *,
        origin: tuple[int, ...] | None = None,
    ) -> Iterator[None]:
        """Pin the chunk grid subsequent per-chunk-array creations use.

        Inside this block, every :func:`zarr_vectors.core.arrays._ensure_array_dir`
        call for a per-spatial-chunk array (vertices, vertex_fragments,
        links/<delta>/<offsets>, link_fragments, vertex_attributes/<name>,
        fragment_attributes/<name>) allocates its vlen-bytes array against
        THIS grid and sharding, rather than the one derived from the
        store's metadata.  Opened by
        :func:`zarr_vectors.core.arrays.open_write_session`; without it
        the derived config applies, so the layout is the same either way.

        Args:
            shard_shape: Outer-chunk shape in *inner-chunk* units, e.g.
                ``(8, 8, 8)``.  ``None`` (the default) means unsharded —
                one storage object per spatial chunk.  When set, its
                length must match ``grid_shape`` and the Zarr v3
                ``sharding_indexed`` codec packs many cells per shard.
            grid_shape: Number of spatial chunks along each axis at
                this level.  Compute via
                :func:`zarr_vectors.spatial.chunking.compute_grid_shape`.

        Nesting is not supported; an inner call raises :class:`StoreError`.
        The ``batched_writes`` context can be active concurrently — cell
        writes are queued and flushed together (see
        :mod:`zarr_vectors.core._batch_writer`).
        """
        if self._native_sharded_config is not None:
            raise StoreError(
                "native_sharded_arrays() does not support nesting"
            )
        if shard_shape is not None and len(shard_shape) != len(grid_shape):
            raise StoreError(
                f"shard_shape rank {len(shard_shape)} != grid_shape "
                f"rank {len(grid_shape)}"
            )
        self._native_sharded_config = {
            "shard_shape": (
                None if shard_shape is None
                else tuple(int(s) for s in shard_shape)
            ),
            "grid_shape": tuple(int(g) for g in grid_shape),
            "origin": (
                None if origin is None
                else tuple(int(o) for o in origin)
            ),
        }
        try:
            yield
        finally:
            self._native_sharded_config = None

    def read_bytes(self, array_name: str, chunk_key: str) -> bytes:
        # Batched-read mode (see :meth:`batched_reads`): serve from the
        # prefetch cache when possible.  Cache misses fall through to
        # the sync path below — useful when a caller under-specifies
        # the plan or hits an array the prefetch skipped.
        offline = self._offline
        if offline is not None:
            key = (self._full_path(array_name), chunk_key)
            cached = offline.chunks.get(key)
            if cached is not None:
                return cached
            # The node itself may well be in the snapshot, so the lookup
            # below would succeed and then read the cell straight off the
            # store.  Stop here instead: offline means offline.
            offline.misses.add(key)
            raise StoreError(
                f"Offline read of chunk {key[0]!r}/{chunk_key!r}: not in "
                f"the prefetched chunk snapshot."
            )

        if self._prefetch_cache is not None:
            cached = self._prefetch_cache.get((array_name, chunk_key))
            if cached is not None:
                return cached

        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is None:
            raise StoreError(
                f"Chunk {array_name!r}/{chunk_key!r} not found in "
                f"{self._zarr.path or '<root>'}"
            )
        coords = _parse_chunk_coords(chunk_key)
        index = (
            None if coords is None
            else _coord_to_index(coords, _grid_origin(sharded_arr))
        )
        if index is None or not _coords_in_bounds(index, sharded_arr.shape):
            raise StoreError(
                f"Chunk {array_name!r}/{chunk_key!r} not found in "
                f"{self._zarr.path or '<root>'}"
            )
        return _vlen_get_cell(sharded_arr, index)

    def chunk_exists(self, array_name: str, chunk_key: str) -> bool:
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is None:
            return False
        present = sharded_arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
        if present is not None:
            return chunk_key in present
        # Fall back to inspecting the cell — slow path used when the
        # presence manifest is missing (e.g. mid-migration, or before a
        # coordinator's ``derive_nonempty_chunks``).  That inspection is
        # a store read, so offline it can only be served from the
        # snapshot — via ``read_bytes``, which records the miss.
        if self._offline is not None:
            try:
                return self.read_bytes(array_name, chunk_key) != b""
            except StoreError:
                return False
        coords = _parse_chunk_coords(chunk_key)
        index = (
            None if coords is None
            else _coord_to_index(coords, _grid_origin(sharded_arr))
        )
        if index is None or not _coords_in_bounds(index, sharded_arr.shape):
            return False
        return _vlen_get_cell(sharded_arr, index) != b""

    def list_chunks(self, array_name: str) -> list[str]:
        # Trust the per-array presence manifest written by
        # ``write_bytes``; without it we'd have to fetch every shard
        # index to find non-empty cells.  A family group (``links/<delta>``)
        # is not a chunk array and holds no cells of its own.
        arr = self._sharded_chunk_array(array_name)
        if arr is None:
            return []
        present = arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
        return sorted(present) if present else []

    # ---------------- array metadata ----------------

    def write_array_meta(self, array_name: str, meta: dict[str, Any]) -> None:
        # An existing chunk array takes its metadata directly on its own
        # ``attrs`` (the Zarr-spec location for per-array user metadata).
        # Checked BEFORE the batched queue, so a chunk array never routes
        # through the group-meta flush.
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is not None:
            sharded_arr.attrs.update(_json_safe(meta))
            return
        # Otherwise ``array_name`` is a group — a ``links/<delta>``
        # family, an attribute namespace, ``object_index``.  Batched-write
        # mode (see :meth:`batched_writes`): queue the full ``zarr.json``
        # content so it flushes in the gather instead of paying a sync
        # ``require_group + attrs.update`` (2-3 round-trips each on
        # cloud).  Merge with anything already queued for the same name
        # so successive ``write_array_meta`` calls within the block
        # compose, matching the existing ``attrs.update`` semantics.
        if self._pending_array_metas is not None:
            existing = self._pending_array_metas.get(array_name, {})
            merged = {**existing, **_json_safe(meta)}
            self._pending_array_metas[array_name] = merged
            return
        arr_group = self._zarr.require_group(array_name)
        arr_group.attrs.update(_json_safe(meta))

    def read_array_meta(self, array_name: str) -> dict[str, Any]:
        node = self._lookup_node(array_name)
        if not isinstance(node, (zarr.Group, zarr.Array)):
            return {}
        return dict(node.attrs)

    def array_exists(self, array_name: str) -> bool:
        return isinstance(
            self._lookup_node(array_name), (zarr.Group, zarr.Array)
        )

    # ---------------- standard Zarr v3 arrays (single array per path) ----------------
    #
    # The methods above address one CELL of a chunk-grid-shaped vlen
    # array.  The methods below write a whole typed Zarr v3 array at a
    # path — the shape used by object-level data (``object_index``,
    # ``object_attributes/<name>``), which has no spatial chunk grid.
    #
    # The batched-writes / batched-reads queues only cover the per-cell
    # path; these methods always execute synchronously.

    def write_array(
        self,
        path: str,
        data: Any,
        *,
        chunks: tuple[int, ...] | None = None,
        fill_value: Any = None,
        attributes: dict[str, Any] | None = None,
        compressors: Any = None,
    ) -> None:
        """Write a chunked Zarr v3 array at ``path``.

        Args:
            path: Logical path of the array within this group, e.g.
                ``"object_attributes/intensity"``.  Intermediate path
                segments become Zarr groups if absent.
            data: Numpy-coercible array.  ``dtype`` and ``shape`` set
                the array's dtype and shape.
            chunks: Chunk shape.  Defaults to ``data.shape`` (single
                chunk).
            fill_value: Value Zarr returns for unwritten chunk positions.
                Special floats are JSON-string-encoded per spec
                (``"NaN"``, ``"Infinity"``, ``"-Infinity"``).
            attributes: Dict applied to the array's ``attributes`` block
                in its own ``zarr.json`` — the spec-blessed location for
                per-array user metadata.
            compressors: Override the codec pipeline.  ``None`` uses the
                active session codec (from :meth:`batched_writes`) or
                zarr v3's default (``bytes`` + ``zstd``).
        """
        arr_data = np.asarray(data)
        if chunks is None:
            chunks = arr_data.shape if arr_data.shape else (1,)

        self._invalidate_node(path)
        parent_path, _, leaf = path.rpartition("/")
        parent = self._zarr.require_group(parent_path) if parent_path else self._zarr
        if leaf in parent:
            del parent[leaf]

        create_kwargs: dict[str, Any] = {
            "name": leaf,
            "shape": arr_data.shape,
            "chunks": chunks,
            "dtype": arr_data.dtype,
        }
        if fill_value is not None:
            create_kwargs["fill_value"] = fill_value

        resolved = self._resolve_codecs(compressors)
        if resolved is not None:
            from zarr_vectors.encoding.compression import codecs_for_create_array
            create_kwargs["compressors"] = codecs_for_create_array(resolved)

        arr = parent.create_array(**create_kwargs)
        arr[:] = arr_data

        if attributes:
            arr.attrs.update(_json_safe(attributes))

    def read_array(self, path: str) -> np.ndarray:
        """Read a chunked Zarr array at ``path`` as a numpy array."""
        cached = self._offline_array(path)
        if cached is not None:
            return np.asarray(cached)
        node = self._require_array_node(path)
        return np.asarray(node[:])

    def _offline_array(self, path: str) -> Any | None:
        """Whole-array value for ``path`` from the offline snapshot.

        Returns ``None`` when not offline, so callers fall through to
        their normal store read.  Offline, a value absent from the
        snapshot records a miss and raises — the same contract as
        :meth:`_lookup_node` and :meth:`read_bytes`.
        """
        offline = self._offline
        if offline is None:
            return None
        full = self._full_path(path)
        hit = offline.arrays.get(full)
        if hit is not None:
            return hit
        offline.misses.add(("array", full))
        raise StoreError(
            f"Offline read of array {full!r}: not in the prefetched "
            f"array snapshot."
        )

    def write_vlen_array(
        self,
        path: str,
        blobs: Sequence[bytes],
        *,
        chunks: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Write a 1D variable-length-bytes Zarr v3 array at ``path``.

        Each element of ``blobs`` becomes one row of the resulting
        array.  Mirrors the layout used by ``object_index/manifests``.

        Args:
            path: Logical path of the array.
            blobs: Iterable of bytes blobs.  Empty input writes no
                array (caller's responsibility to check).
            chunks: Elements per chunk.  Defaults to ``len(blobs)``
                (single chunk).
            attributes: Dict applied to the array's ``attributes`` block.
        """
        blob_list = list(blobs)
        n = len(blob_list)
        if n == 0:
            return

        self._invalidate_node(path)
        parent_path, _, leaf = path.rpartition("/")
        parent = self._zarr.require_group(parent_path) if parent_path else self._zarr
        if leaf in parent:
            del parent[leaf]

        chunk_size = chunks if chunks is not None else n

        with warnings.catch_warnings():
            # vlen-bytes lacks a finalised V3 spec — see the matching
            # suppression in `_write_object_index_manifests`.
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            arr = parent.create_array(
                leaf,
                shape=(n,),
                chunks=(chunk_size,),
                dtype="bytes",
                serializer=VLenBytesCodec(),
            )
            obj = np.empty(n, dtype=object)
            for i, blob in enumerate(blob_list):
                obj[i] = blob
            arr[:] = obj

        if attributes:
            arr.attrs.update(_json_safe(attributes))

    def read_vlen_array(self, path: str) -> list[bytes]:
        """Read a vlen-bytes Zarr array at ``path`` as a list of bytes."""
        cached = self._offline_array(path)
        if cached is not None:
            return [bytes(b) for b in cached]
        node = self._require_array_node(path)
        return [bytes(b) for b in node[:]]

    def read_vlen_element(self, path: str, index: int) -> bytes:
        """Read ONE element of the vlen-bytes array at ``path``.

        Kept distinct from :meth:`read_vlen_array` so the online path
        still fetches a single element rather than the whole array —
        ``object_index/manifests`` has one row per object, so reading it
        whole to answer a by-id lookup would not scale.  Offline the
        distinction is moot: the snapshot holds the array entire, and
        this just indexes into it.
        """
        cached = self._offline_array(path)
        if cached is not None:
            return _vlen_region_to_bytes(cached[index:index + 1])
        node = self._require_array_node(path)
        # Slice-then-extract, never scalar-index: see core._vlen.
        return _vlen_region_to_bytes(node[index:index + 1])

    def read_vlen_elements(self, path: str, indices: Sequence[int]) -> list[bytes]:
        """Read MANY elements of the vlen-bytes array at ``path``, at once.

        The plural form exists because the singular one, called in a
        loop, is one request per element — which is exactly the shape a
        selective read is trying to avoid.  One coordinate selection
        fetches only the zarr chunks the wanted rows actually fall in,
        so reading a hundred of twenty-one million object manifests costs
        a handful of reads rather than a hundred, or than all of them.

        Results are returned in the order of ``indices``, so a caller can
        zip them against the ids that produced them without re-sorting.
        """
        if not len(indices):
            return []
        cached = self._offline_array(path)
        if cached is not None:
            return [_vlen_region_to_bytes(cached[i:i + 1]) for i in indices]
        node = self._require_array_node(path)
        idx = np.asarray(indices, dtype=np.int64)
        try:
            selected = node.get_coordinate_selection((idx,))
        except Exception:
            # Not every store supports coordinate selection; one read per
            # element is slower but identical in result.
            return [
                _vlen_region_to_bytes(node[int(i):int(i) + 1]) for i in indices
            ]
        return [bytes(b) for b in np.asarray(selected).ravel()]

    # ---------------- native-sharded chunk array (sharding_indexed) -------

    def create_sharded_chunk_array(
        self,
        array_name: str,
        grid_shape: tuple[int, ...],
        *,
        shard_shape: tuple[int, ...] | None = None,
        origin: tuple[int, ...] | None = None,
        compressors: list[dict[str, Any]] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Allocate a vlen-bytes Zarr array for per-chunk blobs.

        A single Zarr v3 array whose shape equals the level's chunk
        grid; each cell holds one ZVF spatial chunk's payload.  When
        ``shard_shape`` is provided the ``sharding_indexed`` codec packs
        many cells into a single storage object — the standard cloud
        layout described in
        :doc:`/spec/chunking/sharding`.

        Args:
            array_name: Logical path of the array, e.g. ``"vertices"``
                or ``"links/0/0.0.+1"``.  Intermediate path segments become
                Zarr sub-groups if absent.
            grid_shape: Number of chunks along each spatial axis at
                this level.  Computed via
                :func:`zarr_vectors.spatial.chunking.compute_grid_shape`.
            shard_shape: Outer-chunk shape in *inner-chunk* units.
                ``None`` (default) creates an unsharded array — one
                storage object per ZVF chunk, same I/O cost as the
                legacy layout but already in the new structural form.
                A typical cloud workload uses ``(8, 8, 8)``: 512 ZVF
                chunks per storage object.
            compressors: BytesBytes compressor list (already stripped of
                the ``bytes`` serializer, i.e. the ``create_array``
                ``compressors=`` form).  ``None`` leaves the codec
                pipeline at vlen-bytes only (no compression) — matching
                the historical no-compression default of the earlier
                per-chunk-array writer.
                Pass ``[]`` for the same effect explicitly, or a
                ``[{"name": "zstd", ...}]``-style list to compress.
            attributes: Per-array metadata merged into the array's
                ``zarr.json`` ``attributes`` block (the standard Zarr
                v3 location for user metadata).
        """
        ndim = len(grid_shape)
        if shard_shape is not None and len(shard_shape) != ndim:
            raise StoreError(
                f"shard_shape rank {len(shard_shape)} != grid_shape "
                f"rank {ndim} for array {array_name!r}"
            )

        parent_path, _, leaf = array_name.rpartition("/")
        parent = (
            self._zarr.require_group(parent_path) if parent_path else self._zarr
        )
        # Replace whatever sat at this path (legacy group, prior array)
        # before creating the new sharded array.
        if leaf in parent:
            del parent[leaf]

        create_kwargs: dict[str, Any] = {
            "shape": grid_shape,
            "chunks": (1,) * ndim,
            "dtype": "bytes",
            "serializer": VLenBytesCodec(),
            # Default to no compression (vlen-bytes only).  Omitting the
            # kwarg would let zarr add its default zstd; we instead honor
            # the session's resolved codecs so the layout matches the
            # historical uncompressed default and only compresses when
            # the caller asked for it.
            "compressors": list(compressors) if compressors else [],
        }
        if shard_shape is not None:
            # Zarr 3.2 high-level ``shards=`` kwarg: wraps the inner
            # ``chunks=`` codec in ``sharding_indexed`` automatically.
            create_kwargs["shards"] = tuple(shard_shape)

        # Every attribute is passed to ``create_array`` rather than
        # assigned afterwards: each ``arr.attrs[...] = ...`` rewrites the
        # whole ``zarr.json``, so assigning them post-hoc cost one store
        # write per attribute on top of the creation itself — three writes
        # of the same object where one does.  Writers that allocate an
        # array per offsets segment pay that per array.
        initial_attrs: dict[str, Any] = {_NONEMPTY_CHUNKS_ATTR: []}
        # Store the grid origin so cell ``index = coord - origin``.
        # Only when non-trivial — a zero origin is the common case
        # and its absence means "coords are array indices".
        if origin is not None and any(int(o) != 0 for o in origin):
            initial_attrs[_CHUNK_GRID_ORIGIN_ATTR] = [int(o) for o in origin]
        if attributes:
            initial_attrs.update(_json_safe(attributes))
        create_kwargs["attributes"] = initial_attrs

        self._invalidate_node(array_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            arr = parent.create_array(leaf, **create_kwargs)
        # Seed the cache with the handle we just built: the writes that
        # follow (one per cell of this array) would otherwise each re-read
        # the ``zarr.json`` we have in hand.
        if self._node_cache is not None:
            self._node_cache[array_name] = arr

    def _lookup_node(self, path: str) -> zarr.Array | zarr.Group | None:
        """Return the Zarr node at ``path``, or ``None`` if absent.

        The single node-resolution path for this class.  Two properties
        matter for throughput, because every per-cell write resolves a
        node first:

        - **One store read, not two.**  ``path in self._zarr`` and
          ``self._zarr[path]`` each cost a full ``zarr.json`` GET, and
          ``__getitem__`` already raises ``KeyError`` for an absent path
          — so the membership test ahead of it doubled the reads for no
          extra information.
        - **Cached inside a batched-writes block.**  A write session
          resolves the same handful of arrays thousands of times (once
          per queued cell), and within the block those nodes are created
          by us and cannot change underneath us.  Only ``zarr.Array``
          nodes are cached: a *negative* result would go stale the moment
          an array is created at the path, and a group node is cheap to
          re-resolve.  The cache lives exactly as long as the block (see
          :meth:`batched_writes`), so no handle outlives the session that
          made it.  Mutators invalidate it — see :meth:`_invalidate_node`.
        - **Authoritative inside an offline-reads block.**  When
          ``_offline_reads`` is set the cache is the whole world: it was
          filled up-front by the async primer, so a hit for an absent
          path is the ``_ABSENT`` marker rather than nothing at all, and
          a genuine miss is a bug in the prefetch plan.  Falling through
          to the store there would issue exactly the synchronous GET the
          async path exists to avoid, so it raises instead.
        """
        offline = self._offline
        if offline is not None:
            full = self._full_path(path)
            hit = offline.nodes.get(full)
            if hit is not None:
                return None if hit is _ABSENT else hit
            # Record the miss, then answer "absent" rather than raising.
            #
            # Unlike a chunk read, an unresolved node can be answered
            # speculatively without risking a wrong result: only a pass
            # that records *no* misses is ever accepted (see
            # :func:`zarr_vectors.core.aio.read_async`), so a branch
            # taken on a wrong "absent" is discarded and re-run once the
            # node is in the snapshot.
            #
            # It also matters for speed.  Readers probe families of
            # candidate paths — the 26 neighbour offsets under
            # ``links/<delta>/`` among them — and raising at the first
            # would reveal one path per round. Continuing surfaces the
            # whole family in a single pass instead.
            offline.misses.add(full)
            return None
        cache = self._node_cache
        if cache is not None:
            hit = cache.get(path)
            if hit is not None:
                return None if hit is _ABSENT else hit
        try:
            node = self._zarr[path]
        except KeyError:
            return None
        if cache is not None and isinstance(node, zarr.Array):
            cache[path] = node
        return node

    def _invalidate_node(self, path: str) -> None:
        """Drop ``path`` and everything beneath it from the node cache.

        Called by every method that creates, replaces, or deletes a node,
        so a cached handle can never outlive the node it points at.
        """
        cache = self._node_cache
        if not cache:
            return
        prefix = f"{path}/"
        for key in [
            k for k in cache if k == path or k.startswith(prefix)
        ]:
            del cache[key]

    def _sharded_chunk_array(self, array_name: str) -> zarr.Array | None:
        """Return the multidim vlen-bytes Zarr array at ``array_name``
        (the per-spatial-chunk layout), else ``None``.

        ``None`` means the path is absent or holds a group — a family
        group such as ``links/<delta>``, never a chunk array.
        """
        node = self._lookup_node(array_name)
        return node if isinstance(node, zarr.Array) else None

    def derive_nonempty_chunks(self, array_name: str) -> list[str]:
        """Rebuild ``array_name``'s ``nonempty_chunks`` from the store listing.

        The coordinator half of ``write_bytes(..., record_presence=False)``:
        workers write cell payloads without touching the shared manifest
        attribute, then one caller runs this once to restore it.  The
        listing bounds *which* cells to consider; the cells themselves
        still have to be read, because a cell written with an empty
        payload is an object on disk that must not be recorded as
        present.  Those reads go out as one :func:`asyncio.gather` — the
        same prefetch :meth:`batched_reads` uses — so the cost is one
        listing plus one round-trip per array rather than a GET per cell.

        Only meaningful for an **unsharded** array, where each cell is its
        own object at ``<array>/c/i/j/k`` and therefore visible in the
        listing.  A sharded array packs many cells into one object whose
        inner index is not derivable from key names — so sharding stays a
        coordinator pass that runs *after* this, which is the same
        ordering :func:`zarr_vectors.sharding.shard_store` already
        requires.  Returns the sorted keys now recorded.
        """
        arr = self._sharded_chunk_array(array_name)
        if arr is None:
            raise StoreError(
                f"{array_name!r} in {self._zarr.path or '<root>'} is not a "
                f"chunk array; nothing to derive presence for"
            )
        base = self._zarr.path.strip("/")
        prefix = f"{base}/{array_name}/c/" if base else f"{array_name}/c/"
        origin = _grid_origin(arr)
        candidates: list[str] = []
        for stored in _list_store_prefix(self._zarr.store, prefix):
            # ``c/i/j/k`` → cell index (i, j, k) → absolute coord.
            parts = stored[len(prefix):].split("/")
            if len(parts) != arr.ndim:
                continue
            try:
                index = tuple(int(p) for p in parts)
            except ValueError:
                continue
            coords = (
                index if origin is None
                else tuple(i + o for i, o in zip(index, origin))
            )
            candidates.append(_format_chunk_key(coords))

        keys: set[str] = set()
        if candidates:
            from zarr_vectors.core._batch_reader import flush_prefetch

            # flush_prefetch applies the grid origin itself and omits any
            # cell that reads back empty or missing, so a key surviving in
            # the cache is exactly the ``if _vlen_get_cell(...)`` this
            # replaced.  It also carries the icechunk serial fallback.
            cells = flush_prefetch(self._zarr, [(array_name, candidates)])
            keys = {
                chunk_key for (_name, chunk_key), data in cells.items() if data
            }
        arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(keys)
        return sorted(keys)

    def read_array_attrs(self, path: str) -> dict[str, Any]:
        """Read the ``attributes`` block of a Zarr array at ``path``.

        Returns ``{}`` for missing paths or paths that point at a group
        rather than an array.
        """
        node = self._lookup_node(path)
        if not isinstance(node, zarr.Array):
            return {}
        return dict(node.attrs)

    def read_array_fill_value(self, path: str) -> Any:
        """Return the ``fill_value`` of the standard Zarr array at ``path``.

        ``fill_value`` lives at the top level of the array's ``zarr.json``
        (not in the user ``attributes`` block), so this helper exists
        alongside :meth:`read_array_attrs` rather than being folded into
        it.  Raises :class:`StoreError` if ``path`` is missing or points
        at a group.
        """
        return self._require_array_node(path).fill_value

    def standalone_array_exists(self, path: str) -> bool:
        """``True`` when ``path`` points at a Zarr array (not a group).

        Counterpart to :meth:`array_exists`, which reports ``True`` for
        either node type — so it also answers "does this family group
        exist" for a path like ``links/<delta>``.
        """
        return isinstance(self._lookup_node(path), zarr.Array)

    def _require_array_node(self, path: str) -> zarr.Array:
        node = self._lookup_node(path)
        if node is None:
            raise StoreError(
                f"Array {path!r} not found in {self._zarr.path or '<root>'}"
            )
        if not isinstance(node, zarr.Array):
            raise StoreError(
                f"{path!r} is a {type(node).__name__}, not an Array"
            )
        return node

    def _resolve_codecs(
        self, compressors: Any,
    ) -> list[dict[str, Any]] | None:
        """Pick effective codecs: explicit override > session codec > None."""
        if compressors is not None:
            from zarr_vectors.encoding.compression import resolve_compressor
            return resolve_compressor(compressors)
        return self._active_codecs

    # ---------------- delete ----------------

    def delete_subtree(self, name: str) -> None:
        self._invalidate_node(name)
        if name in self._zarr:
            del self._zarr[name]

    # ---------------- path / url ----------------

    @property
    def path(self) -> Path:
        store = self._zarr.store
        if not isinstance(store, LocalStore):
            raise StoreError(
                f"Group.path is only available for LocalStore; got "
                f"{type(store).__name__}. Use Group.url instead."
            )
        root = _local_root(store)
        if self._zarr.path:
            return root.joinpath(*self._zarr.path.strip("/").split("/"))
        return root

    @property
    def url(self) -> str:
        store = self._zarr.store
        if isinstance(store, LocalStore):
            base = _local_root(store).absolute().as_uri()
        else:
            base = repr(store)
        if self._zarr.path:
            return f"{base.rstrip('/')}/{self._zarr.path.strip('/')}"
        return base

    @property
    def prefix(self) -> str:
        return self._zarr.path

    # ---------------- back-compat shims (used by lazy/) ----------------

    @property
    def backend(self) -> _BackendShim:
        return _BackendShim(self._zarr.store)

    @property
    def _backend(self) -> _BackendShim:  # noqa: D401  (legacy callers)
        return _BackendShim(self._zarr.store)

    @property
    def zarr_group(self) -> zarr.Group:
        """The underlying :class:`zarr.Group`."""
        return self._zarr

    # ---------------- repr ----------------

    def __repr__(self) -> str:
        return f"Group({self.url!r})"


# ===================================================================
# .attrs dict-like wrapper
# ===================================================================


class _Attrs:
    """Dict-like wrapper around :attr:`zarr.Group.attrs`.

    The wrapper exists for API parity with the legacy on-disk-JSON
    ``_Attrs`` — callers use ``attrs.to_dict()``, ``attrs.update(d)``,
    ``attrs[k]``, ``attrs.get(k, default)``, ``k in attrs``.
    """

    def __init__(self, zarr_attrs: Any) -> None:
        self._attrs = zarr_attrs

    def __getitem__(self, key: str) -> Any:
        return self._attrs[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._attrs[key] = _json_safe_value(value)

    def __contains__(self, key: str) -> bool:
        return key in self._attrs

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._attrs[key]
        except KeyError:
            return default

    def update(self, other: dict[str, Any]) -> None:
        self._attrs.update(_json_safe(other))

    def to_dict(self) -> dict[str, Any]:
        return dict(self._attrs)

    def __repr__(self) -> str:
        return f"_Attrs({dict(self._attrs)!r})"


# ===================================================================
# Back-compat shim for `group._backend.url` / `Group._from_backend`
# ===================================================================


class _BackendShim:
    """Minimal compat shim for callers that reach for ``group._backend``.

    Provides the ``url`` accessor and identity needed by
    :class:`Group._from_backend`.  Anything else raises ``AttributeError``
    so we notice if some caller depends on the deeper legacy surface.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    @property
    def url(self) -> str:
        if isinstance(self._store, LocalStore):
            return _local_root(self._store).absolute().as_uri()
        return repr(self._store)

    def close(self) -> None:
        try:
            close = getattr(self._store, "close", None)
            if close is not None:
                close()
        except Exception:  # pragma: no cover  (defensive)
            pass


# ===================================================================
# Helpers
# ===================================================================


def _local_root(store: LocalStore) -> Path:
    raw = store.root
    return raw if isinstance(raw, Path) else Path(raw)


# Per-array attribute name listing which chunk keys hold non-empty
# payload in a native-sharded array.  Vlen-bytes fill_value is ``b""``
# (indistinguishable from an explicitly-empty write), so we keep a
# tiny sidecar so ``list_chunks`` / ``chunk_exists`` stay O(1) without
# scanning every shard tail.
_NONEMPTY_CHUNKS_ATTR = "nonempty_chunks"

# Per-array attribute holding the chunk-grid origin (``floor(min_corner
# / chunk_shape)`` per axis).  A single vlen array is 0-indexed, so a
# spatial chunk at coord ``c`` lands in cell ``c - origin``.  Absent (the
# common case) means a zero origin — chunk coords are cell indices.
_CHUNK_GRID_ORIGIN_ATTR = "chunk_grid_origin"


def _grid_origin(arr: zarr.Array) -> tuple[int, ...] | None:
    """Return the stored chunk-grid origin for a vlen array, or ``None``."""
    raw = arr.attrs.get(_CHUNK_GRID_ORIGIN_ATTR)
    if not raw:
        return None
    return tuple(int(x) for x in raw)


async def _collect_store_prefix(store: Any, prefix: str) -> list[str]:
    return [key async for key in store.list_prefix(prefix)]


def _list_store_prefix(store: Any, prefix: str) -> list[str]:
    """List every stored key under ``prefix`` (a zarr Store is async-only)."""
    from zarr.core.sync import sync

    return sync(_collect_store_prefix(store, prefix))


def _coord_to_index(
    coords: tuple[int, ...], origin: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Translate an absolute chunk coord to a 0-based array index."""
    if origin is None:
        return coords
    return tuple(c - o for c, o in zip(coords, origin))


def _parse_chunk_coords(key: str) -> tuple[int, ...] | None:
    """Parse a dot-separated chunk key into integer coords.

    ``"0.1.2"`` → ``(0, 1, 2)``.  Returns ``None`` if any segment is
    not an integer.
    """
    parts = key.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def _format_chunk_key(coords: tuple[int, ...]) -> str:
    return ".".join(str(int(c)) for c in coords)


def _coords_in_bounds(coords: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    if len(coords) != len(shape):
        return False
    return all(0 <= c < s for c, s in zip(coords, shape))


def _check_coords_in_bounds(
    coords: tuple[int, ...], shape: tuple[int, ...], array_name: str,
) -> None:
    if not _coords_in_bounds(coords, shape):
        raise StoreError(
            f"Chunk coords {coords} out of grid {shape} for "
            f"array {array_name!r}"
        )


def _vlen_get_cell(
    arr: zarr.Array, coords: tuple[int, ...],
) -> bytes:
    """Read one vlen-bytes cell as ``bytes``.

    Slice-then-extract via the shared :mod:`zarr_vectors.core._vlen`
    helpers, so this, the batched async/sync readers, and the manifest
    reader all decode a vlen cell the same way.
    """
    slices = _vlen_cell_region(coords)
    return _vlen_region_to_bytes(arr[slices])


def _vlen_set_cell(
    arr: zarr.Array, coords: tuple[int, ...], data: bytes,
) -> None:
    obj = np.empty((1,) * len(coords), dtype=object)
    obj.flat[0] = bytes(data)
    slices = tuple(slice(c, c + 1) for c in coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnstableSpecificationWarning)
        arr[slices] = obj


def _record_nonempty_chunk(
    arr: zarr.Array, chunk_key: str, *, present: bool,
) -> None:
    """Update the per-array ``nonempty_chunks`` manifest.

    The manifest is a sorted list of dot-separated chunk keys —
    consulted by :meth:`Group.list_chunks` and :meth:`chunk_exists` so
    those calls don't have to scan every shard tail just to enumerate
    non-empty cells.
    """
    current = arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
    keys = set(current) if current else set()
    if present:
        keys.add(chunk_key)
    else:
        keys.discard(chunk_key)
    arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(keys)


def _json_safe(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _json_safe_value(v) for k, v in d.items()}


def _json_safe_value(v: Any) -> Any:
    """Coerce numpy scalars / arrays to JSON-native types for zarr attrs."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_json_safe_value(x) for x in v]
    if isinstance(v, dict):
        return _json_safe(v)
    return v
