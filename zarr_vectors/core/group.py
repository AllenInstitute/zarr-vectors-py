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
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import zarr
from zarr.codecs import VLenBytesCodec
from zarr.errors import UnstableSpecificationWarning
from zarr.storage import LocalStore

from zarr_vectors.constants import (
    VERTEX_FRAGMENTS as _VERTEX_FRAGMENTS_ARRAY,
)
from zarr_vectors.constants import (
    VERTICES as _VERTICES_ARRAY,
)
from zarr_vectors.core._vlen import (
    cell_region as _vlen_cell_region,
)
from zarr_vectors.core._vlen import (
    region_to_bytes as _vlen_region_to_bytes,
)
from zarr_vectors.exceptions import ShardedPresenceError, StoreError

# Where ``LevelMetadata`` lives on a level group's attrs.  Duplicated
# rather than imported because ``core.metadata`` imports this module.
_LEVEL_META_KEY = "zarr_vectors_level"

# Node-cache sentinel for "this path was probed and is genuinely absent",
# as distinct from "this path was never prefetched".  Only the async
# primer in :mod:`zarr_vectors.core.aio` stores it; without it an offline
# lookup could not tell a legitimately missing array (``array_exists`` →
# False) from a hole in the prefetch plan, and would have to answer one
# of them wrongly.
_ABSENT = object()

# Miss tags the offline session writes for a *planned* cell and for one
# row of a standalone array.  ``_engine.plan.from_misses`` reads them
# back (it keeps its own copies of the literals: importing it here would
# be a cycle).  A planned cell is one a reader's prefetch plan named --
# it is fetched exactly, with no fan-out to the rest of its array,
# which is what lets an object read cost the object rather than the
# level.  A bare ``(array, key)`` pair stays the discovery form.
_MISS_CELL = "cell"
_MISS_ROW = "row"


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
    # Collected presence stamps: (array_name, chunk_key, present).  Set by
    # :meth:`collect_presence`; while it is set the two stamp sites append
    # here instead of writing ``nonempty_chunks``, and the PAYLOADS still
    # go out inline -- which is the whole difference from
    # ``_pending_writes``.  Deliberately NOT propagated to derived Groups:
    # a collecting token is per-worker state, and sharing it across
    # threads would reintroduce the interleaving this exists to remove.
    _pending_presence: list[tuple[str, str, bool]] | None = None
    _pending_array_metas: dict[str, dict[str, Any]] | None = None
    _prefetch_cache: dict[tuple[str, str], bytes] | None = None
    # Active codec spec for chunk-array writes, set by
    # :meth:`batched_writes(compressor=...)`.  ``None`` means callers fall
    # back to zarr v3's default codec pipeline (``bytes`` + ``zstd``).
    # Consumed by :meth:`write_bytes` and the batched flush in
    # :mod:`zarr_vectors.core._batch_writer`.
    _active_codecs: list[dict[str, Any]] | None = None
    # Rows written into each ``vertices`` cell this session, keyed by
    # chunk key.  Written by ``write_chunk_vertices``, consumed by
    # ``stamp_fragments_tile``; see :meth:`note_vertex_rows`.
    _vertex_rows_written: dict[str, int] | None = None
    # object_index path -> (sorted ids, rows), built on first lookup.
    # Resolving an id to a row otherwise re-reads and re-sorts the id
    # table on every single-object read, which turns a point lookup into
    # a whole-index read.
    _object_id_lookup_cache: dict[str, Any] | None = None
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
    # True when ``_node_cache`` was opened by :meth:`cached_nodes` — a
    # read-only block, so Groups may be cached alongside Arrays and the
    # cache may be shared with derived handles.  False for the
    # :meth:`batched_writes` cache, which is Arrays-only and stays put.
    _node_cache_readonly: bool = False
    # Set once this handle has dealt with the level's
    # ``fragments_tile`` claim, so a bulk write of N chunks checks it
    # once rather than N times.  See :meth:`_clear_fragments_tile`.
    _tiling_claim_settled: bool = False
    # Per-array presence manifests, cached for the same read-only session
    # as ``_node_cache`` and keyed the same way.  See
    # :meth:`_chunk_listing` and :class:`_ChunkListing`.
    _listing_cache: dict[str, _ChunkListing] | None = None
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
        self._pending_presence = None
        # Prefetch cache activated by :meth:`batched_reads`.  When set,
        # :meth:`read_bytes` looks here first before hitting the store.
        self._prefetch_cache = None
        self._active_codecs = None
        self._native_sharded_config = None
        self._node_cache = None
        self._node_cache_readonly = False
        self._listing_cache = None
        self._tiling_claim_settled = False
        self._vertex_rows_written = None
        self._object_id_lookup_cache = None

    @classmethod
    def _from_zarr(
        cls, zarr_group: zarr.Group, _parent: Group | None = None,
    ) -> Group:
        instance = cls.__new__(cls)
        instance._zarr = zarr_group
        instance._pending_writes = None
        instance._pending_array_metas = None
        instance._pending_presence = None
        instance._prefetch_cache = None
        instance._active_codecs = None
        # An offline snapshot covers the whole tree, so a Group derived
        # from one stays offline; without this a level group would fall
        # back to the store and issue the very sync read the snapshot
        # exists to avoid.  Everything else stays per-instance.
        instance._offline = _parent._offline if _parent is not None else None
        # A read-only node cache covers the whole tree for the same
        # reason: a reader resolves the root, then the level, then each
        # array beneath it, and the lookups worth collapsing are spread
        # across all three handles.  Only the read-only variant
        # propagates -- a :meth:`batched_writes` cache stays on the Group
        # that opened it, because that block creates nodes as it goes and
        # a derived handle has no way to learn about an invalidation.
        share = (
            _parent is not None
            and _parent._node_cache is not None
            and _parent._node_cache_readonly
        )
        instance._node_cache = _parent._node_cache if share else None
        instance._node_cache_readonly = share
        instance._listing_cache = _parent._listing_cache if share else None
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
        if array_name in (_VERTICES_ARRAY, _VERTEX_FRAGMENTS_ARRAY):
            # Either write can break the tiling a level may be claiming:
            # the index directly, the buffer by changing its row count.
            self._clear_fragments_tile()
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
            if self._pending_presence is not None:
                self._pending_presence.append((array_name, chunk_key, bool(data)))
            else:
                _record_nonempty_chunk(
                    sharded_arr, array_name, chunk_key, present=bool(data),
                )

    def write_cells(
        self,
        array_name: str,
        cells: Iterable[tuple[str, bytes]],
        *,
        record_presence: bool = True,
    ) -> int:
        """:meth:`write_bytes` for many cells of one array.

        The node is resolved and the grid read once for the whole batch;
        each cell then costs its coordinate check and, inside a
        :meth:`batched_writes` block, one append to the queue.  A links
        family writes a hundred thousand cells of a few rows each, and
        the per-call overhead of the singular form was most of that
        write.  Returns how many cells were written.
        """
        if array_name in (_VERTICES_ARRAY, _VERTEX_FRAGMENTS_ARRAY):
            self._clear_fragments_tile()
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is None:
            raise StoreError(
                f"Cannot write to {array_name!r} in "
                f"{self._zarr.path or '<root>'}: no chunk array at that "
                f"path. Per-chunk arrays must be allocated first (see "
                f"arrays._ensure_array_dir / create_sharded_chunk_array)."
            )
        origin = _grid_origin(sharded_arr)
        shape = sharded_arr.shape
        pending = self._pending_writes
        n = 0
        for chunk_key, data in cells:
            coords = _parse_chunk_coords(chunk_key)
            if coords is None:
                raise StoreError(
                    f"Cannot write to array {array_name!r}: "
                    f"chunk_key {chunk_key!r} is not a coord tuple"
                )
            index = _coord_to_index(coords, origin)
            _check_coords_in_bounds(index, shape, array_name)
            n += 1
            if pending is not None:
                pending.append((array_name, chunk_key, bytes(data), record_presence))
                continue
            _vlen_set_cell(sharded_arr, index, bytes(data))
            if record_presence:
                if self._pending_presence is not None:
                    self._pending_presence.append(
                        (array_name, chunk_key, bool(data))
                    )
                else:
                    _record_nonempty_chunk(
                        sharded_arr, array_name, chunk_key, present=bool(data),
                    )
        return n

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
            # aio.read_async and the engine), the chunks this plan would
            # prefetch are served from the session -- so the sync
            # prefetch is redundant. It is also unavailable: flush_prefetch
            # calls sync(), which under Pyodide needs WebAssembly stack
            # switching (JSPI).
            #
            # What a plan IS good for offline is saying, all at once, what
            # the snapshot still lacks.  Left to ``read_bytes`` the reader
            # would surface one missing cell per round -- the first one it
            # touched -- and a level with more cells than the round limit
            # never converged.  So every planned cell the session cannot
            # serve is recorded here as a *planned* miss, tagged so the
            # next round fetches exactly those cells and does not fan out
            # to every cell of their arrays, and the pass is abandoned
            # immediately rather than decoding what it already knows it
            # cannot finish.  Cells the snapshot has confirmed absent, or
            # whose array is known to be a group or missing, are not
            # misses: asking for those again would learn nothing.
            self._record_planned_misses(plan)
            yield
            return
        from zarr_vectors.core._batch_reader import flush_prefetch

        # Hand over the array handles we already hold, and the direct-read
        # specs derived from them.  Inside a :meth:`cached_nodes` block
        # that is all of them, so the prefetch resolves nothing and
        # derives nothing -- it goes straight to the cells.
        names = {name for name, _ in plan}
        self._prefetch_cache = flush_prefetch(
            self._zarr, plan,
            {name: self._sharded_chunk_array(name) for name in names},
            specs={name: self._direct_spec_cached(name) for name in names},
        )
        # No node cache opened here, deliberately: :meth:`read_bytes`
        # answers from ``_prefetch_cache`` before it resolves a node, so
        # inside this block there is nothing left for one to cache.
        # (Measured: adding one changes the store round-trip count not at
        # all.)  The lookups a read *does* repeat happen on either side of
        # this block -- ``read_array_meta``, ``list_chunks``, the level
        # handle itself -- which is what :meth:`cached_nodes` covers.
        try:
            yield
        finally:
            self._prefetch_cache = None

    def _record_planned_misses(self, plan: list[tuple[str, list[str]]]) -> None:
        """Record every planned cell the offline session cannot serve.

        See :meth:`batched_reads`.  Raises :class:`StoreError` once the
        misses are recorded, which is the signal the replay loop acts
        on; a plan the session fully covers returns normally.
        """
        offline = self._offline
        if offline is None:
            return
        absent = getattr(offline, "absent", ())
        chunks = offline.chunks
        missing: list[tuple[str, str]] = []
        unresolved: list[str] = []
        for name, keys in plan:
            full = self._full_path(name)
            node = offline.nodes.get(full)
            if node is _ABSENT or (node is not None and not isinstance(node, zarr.Array)):
                # Not a chunk array: nothing there can ever be a cell.
                continue
            if node is None:
                unresolved.append(full)
            for key in keys:
                cell = (full, key)
                if cell not in chunks and cell not in absent:
                    missing.append(cell)
        if not missing:
            return
        offline.misses.update(unresolved)
        offline.misses.update((_MISS_CELL, a, k) for a, k in missing)
        raise StoreError(
            f"Offline read: {len(missing)} planned cell(s) are not in the "
            f"prefetched snapshot (first: {missing[0]!r})."
        )

    @contextmanager
    def cached_nodes(self) -> Iterator[None]:
        """Resolve each node at most once for the duration of one read.

        A reader resolves the same handful of nodes over and over: the
        level group, ``vertices``, ``vertex_fragments``, one array per
        requested attribute -- once to read its metadata, again to list
        its chunk keys, again per chunk read that misses the prefetch.
        Every one of those is a ``zarr.json`` GET, and against a store
        holding a single point that was the whole cost of the query: a
        one-point ``read_points`` issued nine metadata reads to fetch two
        chunk objects.

        Read-only by contract, which is what lets it cache more than
        :meth:`batched_writes` can.  Nothing inside the block creates or
        deletes a node, so a *negative* result cannot go stale and Groups
        are as cacheable as Arrays; and because the cache is keyed
        root-relative it is shared with every Group derived inside the
        block, so the level handle a reader opens on entry hits the same
        cache the root does.

        Writes inside the block are not supported -- a mutation would
        leave a handle cached past the node it points at.  Nesting is a
        no-op: the outer block already covers the inner one.

        Example::

            root = open_store(store_path)
            with root.cached_nodes():
                level_group = get_resolution_level(root, level)
                ...
        """
        if self._node_cache is not None:
            # Already inside a cache block (or a batched-writes one, whose
            # cache is narrower but still correct for reads).  Re-entering
            # would swap the dict out from under the outer block on exit.
            yield
            return
        self._node_cache = {}
        self._node_cache_readonly = True
        self._listing_cache = {}
        try:
            yield
        finally:
            self._node_cache = None
            self._node_cache_readonly = False
            self._listing_cache = None

    def prime_nodes(self, paths: Sequence[str]) -> None:
        """Resolve several nodes in one round-trip, into the active cache.

        :meth:`cached_nodes` stops a node being resolved *twice*; this
        stops the first resolutions being paid one after another.  A
        point read needs three nodes — the level group, ``vertices``,
        ``vertex_fragments`` — and resolving them as they come up is
        three serial ``zarr.json`` reads before any data moves, which on
        a single-chunk query was most of the query.  Asked for together
        they cost one gather.

        A no-op outside a :meth:`cached_nodes` block (there is nowhere to
        put the result), under :meth:`offline_reads` (the snapshot is
        already primed, and its misses are the caller's signal), and for
        any path already cached.  Speculative by design: a path that does
        not exist caches as absent, which is an answer the reader wants
        just as much.

        Args:
            paths: Node paths relative to this Group.  Order is
                irrelevant; duplicates are collapsed.
        """
        cache = self._node_cache
        if (
            cache is None
            or not self._node_cache_readonly
            or self._offline is not None
        ):
            return
        missing = {p for p in paths if self._full_path(p) not in cache}
        if not missing:
            return
        from zarr.core.sync import sync

        from zarr_vectors.core.aio import _resolve_nodes

        try:
            resolved = sync(_resolve_nodes(self._zarr._async_group, missing))
        except Exception:
            # Priming is an optimisation; a failure here must not fail
            # the read.  Every path falls back to its own lookup.
            return
        for path, node in resolved.items():
            cache[self._full_path(path)] = node

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
    def chunk_array_codecs(self, compressor: Any) -> Iterator[None]:
        """Set the codec pipeline for chunk arrays *created* inside the block.

        The creation half of :meth:`batched_writes` on its own: it selects the
        codecs new per-chunk arrays are stamped with and changes nothing else.
        Writes inside the block stay synchronous and immediate.

        That separation is the point. ``batched_writes`` is the only other way
        to reach the session codec, and it also defers every write to a single
        flush on exit — which a caller that read-modify-writes a cell cannot
        use, because the read would not see what the same block just wrote
        (``write_chunk_fragments(mode="append")`` and every
        ``nonempty_chunks`` stamp do exactly that). A writer that wants
        compression but not deferral had no way to ask.

        Only *creation* is affected, and only for arrays that do not exist
        yet: an array already on disk keeps the pipeline it was stamped with,
        since its written cells are encoded under it. To change an existing
        store's codecs, delete the arrays and rewrite them.

        Args:
            compressor: See
                :func:`zarr_vectors.encoding.compression.resolve_compressor`.
                ``None``/``"none"``/``False`` means no compression,
                ``"zstd"`` is zarr v3's default (level 0), ``"blosc"`` is
                Blosc(Zstd, BitShuffle, l5), or pass an explicit codec list
                such as ``[{"name": "zstd", "configuration": {"level": 5}}]``.

        Example::

            with level_group.chunk_array_codecs("zstd"):
                create_vertices_array(level_group, dtype="float32")
                write_chunk_vertices(level_group, cc, [positions])
        """
        if self._pending_writes is not None:
            raise StoreError(
                "chunk_array_codecs() cannot be nested inside batched_writes() "
                "— that block already carries its own compressor="
            )
        from zarr_vectors.encoding.compression import resolve_compressor

        previous = self._active_codecs
        self._active_codecs = resolve_compressor(compressor)
        try:
            yield
        finally:
            self._active_codecs = previous

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
        if self._pending_presence is not None:
            raise StoreError(
                "batched_writes() cannot run inside collect_presence(): this "
                "block stamps nonempty_chunks itself when it flushes, which "
                "would land the very stamps the collecting block is holding "
                "back -- and land them on the flush's thread, outside "
                "whatever lock the caller meant to apply them under."
            )
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
    def collect_presence(self) -> Iterator[list[tuple[str, str, bool]]]:
        """Defer the ``nonempty_chunks`` stamps inside the block, so the
        caller decides when — and under what lock — they are applied.

        The third presence mode, for a writer that can serialise a short
        section but not the whole flush.  The other two each give up one
        of the properties such a writer needs:

        * ``record_presence=True`` (the default) stamps inline.  Correct
          serially, but the manifest is ONE attribute shared by every
          cell, so it is a read-modify-write of state outside the cell
          being written: two workers writing DISJOINT cells still race,
          and the loser's key vanishes while its payload sits on disk.
        * ``record_presence=False`` plus a coordinator's
          :func:`zarr_vectors.building.rebuild_presence` has no race, but
          the cell is invisible to :meth:`list_chunks` until the rebuild
          runs — and consumers legitimately run before it.

        Inside the block, any write that would stamp instead records
        ``(array_name, chunk_key, present)`` into the yielded list.  The
        PAYLOADS ARE NOT DEFERRED: they go out inline, exactly as they
        would otherwise, which is the essential difference from
        :meth:`batched_writes`.  So the bulk of the I/O proceeds
        unlocked and only :meth:`apply_presence` needs serialising::

            with level_group.collect_presence() as pending:
                flush(level_group)          # payloads, no lock held
            with store_write_lock(path):    # short, lock held
                level_group.apply_presence(pending)

        Leaving the block restores normal stamping but DOES NOT apply —
        the caller choosing the moment is the entire point.  ``pending``
        is the list itself, so if the body raises it is still fully
        populated and still applicable; the payloads that did land are
        recorded, and applying is how they become visible.

        ``record_presence=False`` is still honoured inside the block: an
        opt-out is a decision not to stamp at all, not a request to stamp
        later.

        ``pending`` belongs to one thread.  This Group must not be shared
        across workers while collecting, and the token must not be handed
        to another thread to apply.

        Nesting is not supported and raises :class:`StoreError`, as is
        collecting inside :meth:`batched_writes` — that block defers the
        payloads too and stamps its own manifest at flush time, so the
        two cannot both own the stamp.
        """
        if self._pending_presence is not None:
            raise StoreError("collect_presence() does not support nesting")
        if self._pending_writes is not None:
            raise StoreError(
                "collect_presence() cannot run inside batched_writes(): that "
                "block defers the cell payloads and stamps nonempty_chunks "
                "itself when it flushes, so the stamps are not this block's "
                "to collect. Use chunk_array_codecs() if what you wanted was "
                "the codec selection without the deferred writes."
            )
        pending: list[tuple[str, str, bool]] = []
        self._pending_presence = pending
        try:
            yield pending
        finally:
            self._pending_presence = None

    def apply_presence(self, pending: list[tuple[str, str, bool]]) -> int:
        """Apply stamps collected by :meth:`collect_presence`.

        Coalesced to ONE read-modify-write of ``nonempty_chunks`` per
        array, not one per cell.  That is what makes the call short
        enough to hold a lock across, and it is fewer attribute writes
        than stamping each in turn would have been.

        Entries are folded in order, so the last write to a chunk key
        wins: present-then-absent within a block leaves the key absent,
        matching the ``present=bool(data)`` semantics of an inline stamp.

        Each array is re-resolved BY NAME here rather than through a
        handle held from collection time — a sharded array resolves via
        ``_sharded_chunk_array``, and the level may have been re-opened
        in between.

        Applying an empty ``pending`` is a no-op, not an error, so a
        caller need not test before calling.

        Returns:
            How many arrays were stamped.

        Raises:
            StoreError: If an array named in ``pending`` is gone.
        """
        if not pending:
            return 0

        # dict preserves insertion order and last-write-wins, which IS
        # the ordering rule -- no explicit sort or dedupe needed.
        by_array: dict[str, dict[str, bool]] = {}
        for array_name, chunk_key, present in pending:
            by_array.setdefault(array_name, {})[chunk_key] = present

        for array_name, stamps in by_array.items():
            arr = self._sharded_chunk_array(array_name)
            if arr is None:
                raise StoreError(
                    f"Cannot apply presence to {array_name!r} in "
                    f"{self._zarr.path or '<root>'}: no chunk array at that "
                    f"path. The cells were written, so the array existed "
                    f"when they landed; it has been deleted or replaced "
                    f"since. Rebuild with rebuild_presence() instead."
                )
            current = arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
            keys = set(current) if current else set()
            for chunk_key, present in stamps.items():
                if present:
                    keys.add(chunk_key)
                else:
                    keys.discard(chunk_key)
            # Same mechanism as ``_record_nonempty_chunk``, so the handle
            # we just resolved keeps serving the value we just wrote.
            arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(keys)
            # ``_chunk_listing`` memoizes the manifest and has no refresh
            # path, so a warm entry would keep answering with the
            # pre-stamp list and defeat the visibility this call exists
            # to provide.  Targeted rather than ``_invalidate_node``,
            # which would also drop the node handle and the unrelated
            # object-id lookup.  Belt-and-braces: ``cached_nodes`` is
            # documented read-only, so a caller mixing the two is
            # already outside the contract.
            if self._listing_cache:
                self._listing_cache.pop(self._full_path(array_name), None)
            for chunk_key, present in stamps.items():
                _emit_presence(array_name, chunk_key, present)

        return len(by_array)

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

    def _chunk_listing(self, array_name: str) -> _ChunkListing:
        """The presence manifest for ``array_name``, session-cached.

        Trusts the per-array manifest written by :meth:`write_bytes`;
        without it we would have to fetch every shard index to find
        non-empty cells.  A family group (``links/<delta>``) is not a
        chunk array and holds no cells of its own, so it lists empty.

        Inside a :meth:`cached_nodes` block the manifest is read, sorted
        and parsed once and every reader shares the result.  Outside one,
        a fresh listing is built per call — the same work the reader did
        before this cache existed, so nothing regresses.
        """
        cache = self._listing_cache
        key = self._full_path(array_name)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        arr = self._sharded_chunk_array(array_name)
        present = (
            arr.attrs.get(_NONEMPTY_CHUNKS_ATTR) if arr is not None else None
        )
        listing = _ChunkListing(sorted(present) if present else [])
        if cache is not None:
            cache[key] = listing
        return listing

    def note_vertex_rows(self, chunk_key: str, n_rows: int) -> None:
        """Record how many vertex rows a cell was just given.

        :func:`~zarr_vectors.core.arrays.stamp_fragments_tile` has to
        know each cell's row count to check the fragment index against
        it, and it used to get that by re-reading every ``vertices``
        cell it had just written -- measured at 1.00x the bytes written,
        so a bulk write downloaded its own output in full before
        returning.  The writer already knows the number, so it says so.

        Only a hint: a key with no recorded count is read back as
        before, which is what keeps the stamp correct for a level whose
        cells this session did not write.
        """
        if self._vertex_rows_written is None:
            self._vertex_rows_written = {}
        self._vertex_rows_written[chunk_key] = int(n_rows)

    def take_vertex_rows(self) -> dict[str, int]:
        """Consume and clear the recorded counts.

        Cleared on read so a later stamp cannot trust a count from
        before an intervening edit.
        """
        recorded = self._vertex_rows_written or {}
        self._vertex_rows_written = None
        return recorded

    def list_chunks(self, array_name: str) -> list[str]:
        """The dotted chunk keys this array holds data for, sorted."""
        return self._chunk_listing(array_name).keys

    def list_chunk_coords(self, array_name: str) -> list[tuple[int, ...]]:
        """:meth:`list_chunks`, parsed to coordinate tuples and sorted.

        The form every reader actually wants, and the one worth caching:
        the strings come off the presence manifest already, but turning
        125 of them into tuples is 125 splits and 375 ``int`` calls, paid
        on every read no matter how few chunks the query goes on to
        touch.  Inside a :meth:`cached_nodes` block the parse happens
        once; outside it, every call parses, exactly as before.
        """
        return self._chunk_listing(array_name).coords()

    def chunk_present_set(self, array_name: str) -> frozenset[tuple[int, ...]]:
        """:meth:`list_chunk_coords` as a set, for membership tests.

        See :meth:`_ChunkListing.present` for why a reader wants it.
        """
        return self._chunk_listing(array_name).present()

    def chunk_index_by_spatial(
        self, array_name: str, nd: int,
    ) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """Present coordinates grouped by their trailing ``nd`` axes.

        See :meth:`_ChunkListing.by_spatial`.
        """
        return self._chunk_listing(array_name).by_spatial(nd)

    def _direct_spec_cached(self, array_name: str) -> Any:
        """The local-filesystem direct-read spec for ``array_name``.

        Rebuilding it means re-reading the array's metadata and
        reconstructing an ``ArraySpec`` — 13.6 us of a 285 us warm
        one-point read, paid again on every read of a session even though
        the node it describes was resolved once.

        Outside a :meth:`cached_nodes` block it is derived per call, as
        before; caching there would mean building a throwaway listing
        record to hold it.
        """
        from zarr_vectors.core._batch_reader import _direct_spec

        cache = self._listing_cache
        if cache is None:
            return _direct_spec(
                self._zarr, array_name, self._sharded_chunk_array(array_name),
            )
        listing = self._chunk_listing(array_name)
        if not listing.spec_known:
            listing.set_spec(_direct_spec(
                self._zarr, array_name,
                self._sharded_chunk_array(array_name),
            ))
        return listing.spec

    def chunk_grid_bounds(
        self, array_name: str,
    ) -> tuple[tuple[int, ...] | None, tuple[int, ...]] | None:
        """``(origin, shape)`` of ``array_name``'s cell grid, or None.

        The grid the chunk keys live in: cell ``i`` of the array holds
        absolute chunk coord ``i + origin`` (``origin`` is ``None`` for a
        zero origin, which is how it is stored).  A box resolver needs
        this to clamp — ``chunks_intersecting_bbox`` speaks absolute
        coords and knows nothing of the store's extent, so an unclamped
        box can name vastly more cells than the grid could ever hold.

        ``None`` when the path holds no chunk array.
        """
        arr = self._sharded_chunk_array(array_name)
        if arr is None:
            return None
        return _grid_origin(arr), tuple(int(s) for s in arr.shape)

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

    def extend_array(
        self,
        path: str,
        rows: Any,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> int:
        """Append ``rows`` along axis 0, leaving existing rows untouched.

        The cheap half of an append.  :meth:`write_array` recreates the
        array from a full in-memory copy, so growing one by read →
        concatenate → write costs the whole array on every call; a writer
        that appends once per spatial chunk therefore pays
        O(chunks x rows) over a build, and each call is slower than the
        last.  This resizes in place and writes only the new region, so
        the cost is the rows actually added — provided the array is
        chunked along axis 0, which is why :func:`write_object_attributes`
        gives its arrays a bounded row chunk when it creates them.

        Args:
            path: Logical path of an existing array.
            rows: Rows to append.  Tail dimensions must match the array's.
            attributes: Merged into the array's attributes block after the
                write.

        Returns:
            The array's new length along axis 0.

        Notes:
            An array whose attributes block records its own ``shape`` (what
            :meth:`write_array` stamps, and what the O(1) length readers key
            on) has it restamped from the resized array — so the recorded
            shape cannot drift from the real one across an append.

        Raises:
            StoreError: If ``path`` does not exist or is not an array.
            ValueError: If the tail dimensions do not match.
        """
        arr = self._require_array_node(path)
        row_data = np.asarray(rows)
        if tuple(row_data.shape[1:]) != tuple(arr.shape[1:]):
            raise ValueError(
                f"extend_array shape mismatch at {path!r}: existing "
                f"{arr.shape} vs new {row_data.shape} — tail dimensions "
                f"must match"
            )
        n0 = int(arr.shape[0])
        total = n0 + int(row_data.shape[0])
        if total != n0:
            arr.resize((total,) + tuple(arr.shape[1:]))
            arr[n0:total] = row_data.astype(arr.dtype, copy=False)
        stamped = dict(attributes) if attributes else {}
        if "shape" in arr.attrs:
            stamped["shape"] = [total, *arr.shape[1:]]
        if stamped:
            arr.attrs.update(_json_safe(stamped))
        self._invalidate_node(path)
        return total

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
        offline = self._offline_rows(path, [int(index)])
        if offline is not None:
            return offline[0]
        node = self._require_array_node(path)
        # Slice-then-extract, never scalar-index: see core._vlen.
        return _vlen_region_to_bytes(node[index:index + 1])

    def _offline_rows(self, path: str, indices: Sequence[int]) -> list[bytes] | None:
        """``indices`` of the vlen array at ``path`` from the offline
        snapshot, or ``None`` when not offline.

        Served from the whole array when the snapshot holds it, else from
        the rows a fetch selected -- which is what keeps a by-id read of
        twenty-one million manifests from reading twenty-one million
        manifests.  A row the snapshot lacks records a *row* miss, so the
        next round fetches exactly those rows by coordinate selection,
        and then raises: same contract as :meth:`_offline_array`.
        """
        offline = self._offline
        if offline is None:
            return None
        full = self._full_path(path)
        whole = offline.arrays.get(full)
        if whole is not None:
            return [_vlen_region_to_bytes(whole[i:i + 1]) for i in indices]
        held = getattr(offline, "rows", None)
        known = held.get(full, {}) if held else {}
        missing = [int(i) for i in indices if int(i) not in known]
        if not missing:
            return [bytes(known[int(i)]) for i in indices]
        offline.misses.update((_MISS_ROW, full, i) for i in missing)
        raise StoreError(
            f"Offline read of {len(missing)} row(s) of {full!r}: not in the "
            f"prefetched snapshot."
        )

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
        offline = self._offline_rows(path, indices)
        if offline is not None:
            return offline
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
        grid; each cell holds one Zarr Vectors spatial chunk's payload.  When
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
                storage object per Zarr Vectors chunk, same I/O cost as the
                legacy layout but already in the new structural form.
                A typical cloud workload uses ``(8, 8, 8)``: 512 Zarr Vectors
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
            self._node_cache[self._full_path(array_name)] = arr

    def create_sharded_chunk_arrays(
        self,
        arrays: Sequence[tuple[str, dict[str, Any] | None]],
        grid_shape: tuple[int, ...],
        *,
        shard_shape: tuple[int, ...] | None = None,
        origin: tuple[int, ...] | None = None,
        compressors: list[dict[str, Any]] | None = None,
        replace: Sequence[str] | set[str] = (),
    ) -> None:
        """:meth:`create_sharded_chunk_array` for several arrays at once.

        One ``zarr.json`` per array still has to be written, but they are
        independent, so they go out in one gather rather than through a
        blocking ``sync()`` each -- which is the difference between
        allocating a links family of 1,700 offsets arrays in a second and
        in a minute.

        ``arrays`` is ``[(array_name, attributes), ...]``; ``replace``
        names those whose existing node must be dropped first (the
        caller has looked, so this does not look again).
        """
        import asyncio

        from zarr.core.sync import sync

        if not arrays:
            return
        ndim = len(grid_shape)
        if shard_shape is not None and len(shard_shape) != ndim:
            raise StoreError(
                f"shard_shape rank {len(shard_shape)} != grid_shape "
                f"rank {ndim}"
            )
        # Every parent group first, once each: the arrays are created by
        # path below, and a node created under a group that does not
        # exist has no hierarchy to be found in.
        for parent_path in sorted({name.rpartition("/")[0] for name, _ in arrays}):
            if parent_path:
                self._zarr.require_group(parent_path)
        for array_name in replace:
            parent_path, _, leaf = array_name.rpartition("/")
            parent = self._zarr[parent_path] if parent_path else self._zarr
            if leaf in parent:
                del parent[leaf]

        base_kwargs: dict[str, Any] = {
            "shape": grid_shape,
            "chunks": (1,) * ndim,
            "dtype": "bytes",
            "serializer": VLenBytesCodec(),
            "compressors": list(compressors) if compressors else [],
        }
        if shard_shape is not None:
            base_kwargs["shards"] = tuple(shard_shape)
        base_attrs: dict[str, Any] = {_NONEMPTY_CHUNKS_ATTR: []}
        if origin is not None and any(int(o) != 0 for o in origin):
            base_attrs[_CHUNK_GRID_ORIGIN_ATTR] = [int(o) for o in origin]

        for array_name, _ in arrays:
            self._invalidate_node(array_name)

        # The first array goes through zarr, which settles everything
        # about the layout -- codecs, dtype, chunk grid -- into one
        # metadata object.  Every other array in the batch IS that
        # metadata with its own attributes, so the rest are written as
        # their ``zarr.json`` directly, all in one gather, and wrapped in
        # handles without a round-trip: what zarr's ``create_array`` does
        # per array is parse the same arguments, probe the store for an
        # existing node, and write the same document, which for a links
        # family of 1,700 arrays was twelve seconds.
        import dataclasses

        from zarr.core.array import AsyncArray
        from zarr.core.buffer import default_buffer_prototype

        first_name, first_attributes = arrays[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            first = self._zarr.create_array(
                first_name,
                **base_kwargs,
                attributes={**base_attrs, **_json_safe(first_attributes or {})},
            )
        handles: list[tuple[str, zarr.Array]] = [(first_name, first)]
        rest = arrays[1:]
        if rest:
            template = first.metadata
            prototype = default_buffer_prototype()
            store = self._zarr.store
            puts: list[tuple[str, Any]] = []
            for array_name, attributes in rest:
                meta = dataclasses.replace(
                    template,
                    attributes={**base_attrs, **_json_safe(attributes or {})},
                )
                store_path = self._zarr.store_path / array_name
                key = f"{store_path.path}/zarr.json" if store_path.path else "zarr.json"
                puts.append((key, meta.to_buffer_dict(prototype)["zarr.json"]))
                handles.append((
                    array_name,
                    zarr.Array(AsyncArray(metadata=meta, store_path=store_path)),
                ))

            async def _put_all() -> None:
                await asyncio.gather(*(store.set(key, buf) for key, buf in puts))

            sync(_put_all())
        if self._node_cache is not None:
            for array_name, handle in handles:
                self._node_cache[self._full_path(array_name)] = handle

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
        # Keyed root-relative, like the offline snapshot: a read-only
        # cache is shared with every Group derived from this one, and a
        # bare relative path means different nodes from different points
        # in the tree (root's "vertices" is not level 0's "vertices").
        cache_key = self._full_path(path) if cache is not None else path
        if cache is not None:
            hit = cache.get(cache_key)
            if hit is not None:
                return None if hit is _ABSENT else hit
        try:
            node = self._zarr[path]
        except KeyError:
            return None
        if cache is not None and (
            isinstance(node, zarr.Array) or self._node_cache_readonly
        ):
            cache[cache_key] = node
        return node

    def _clear_fragments_tile(self) -> None:
        """Drop this level's ``fragments_tile`` claim, if it holds one.

        Called from :meth:`write_bytes` for the two arrays the claim is
        about.  That is the total chokepoint — ``write_chunk_vertices``
        and ``write_chunk_fragments`` both persist through it, and a
        batched write queues through it too — so a claim cannot outlive
        the layout it describes.

        Cheap by construction: the level group's attributes are already
        in memory, and the store write happens at most once per handle
        and only for a level that actually carries the flag.  A writer
        stamping the flag *after* its chunk writes therefore trips this
        once, harmlessly, while the flag is still absent.
        """
        if self._tiling_claim_settled:
            return
        self._tiling_claim_settled = True
        try:
            level = self._zarr.attrs.get(_LEVEL_META_KEY)
        except Exception:
            return
        if not isinstance(level, dict) or not level.get("fragments_tile"):
            return
        self._zarr.attrs.update({
            _LEVEL_META_KEY: {
                k: v for k, v in level.items() if k != "fragments_tile"
            },
        })

    def _invalidate_node(self, path: str) -> None:
        """Drop ``path`` and everything beneath it from the node cache.

        Called by every method that creates, replaces, or deletes a node,
        so a cached handle can never outlive the node it points at.
        """
        full = self._full_path(path)
        prefix = f"{full}/"
        # The id table is a node like any other, so a write that
        # replaces it must drop the lookup built from it.
        self._object_id_lookup_cache = None
        for cache in (self._node_cache, self._listing_cache):
            if not cache:
                continue
            for key in [
                k for k in cache if k == full or k.startswith(prefix)
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

    def derive_nonempty_chunks(
        self,
        array_name: str,
        *,
        on_sharded: Literal["derive", "skip", "raise"] = "derive",
    ) -> list[str]:
        """Rebuild ``array_name``'s ``nonempty_chunks`` from what is on disk.

        The coordinator half of ``write_bytes(..., record_presence=False)``:
        workers write cell payloads without touching the shared manifest
        attribute, then one caller runs this once to restore it.  The
        listing bounds *which* cells to consider; the cells themselves
        still have to be read, because a cell written with an empty
        payload is an object on disk that must not be recorded as
        present.  Those reads go out as one :func:`asyncio.gather` — the
        same prefetch :meth:`batched_reads` uses — so the cost is one
        listing plus one round-trip per array rather than a GET per cell.

        Works on a **sharded** array too, by a different route.  There the
        listing enumerates shard objects rather than cells — zarr builds a
        sharded array's chunk grid from the SHARD shape, so a key under
        ``<array>/c/`` is a shard index — so each listed key is expanded
        to the cell region it covers (``index * shards`` up to
        ``+shards``, clipped to the array) and that region is read in one
        slice.  One read per shard that exists, which is strictly fewer
        than the unsharded branch pays, and only shards that were written
        are visited.  Returns the sorted keys now recorded.

        This is why sharding no longer has to be the last coordinator
        pass.  It used to: presence could only be rebuilt from a per-cell
        listing, so an array had to stay flat until every worker had
        finished.  A store may now be born sharded.

        Not reported to :func:`observe_presence_writes`.  That instrument
        watches the INCREMENTAL stamps, where a read-modify-write of
        shared state can lose a key; this rebuild derives the whole
        manifest from what is on disk, so it is idempotent, has no
        per-cell semantics, and there is nothing for an observer to
        police.

        Args:
            on_sharded: What to do when ``array_name`` *is* natively
                sharded.  ``"derive"`` (the default) rebuilds it, as
                above; the name is a little odd, since deriving is also
                what happens when the array is flat, but it is kept
                because the parameter is on a supported surface.
                ``"skip"`` returns the manifest already recorded without
                rewriting it, for a caller that wants the cheap answer.
                ``"raise"`` raises
                :class:`~zarr_vectors.exceptions.ShardedPresenceError`,
                for a caller asserting an array is not sharded.

        ``"raise"`` was the default while the sharded case could not be
        derived at all, and it guarded a real failure: rebuilding from a
        listing that resolves nothing took a store with 1605 recorded
        cells down to 2, with no exception and no warning, and the rest
        became unreachable through ``list_chunks``.  The guard is kept as
        an assertion rather than removed, but it is no longer what a
        caller wants by default.
        """
        arr = self._sharded_chunk_array(array_name)
        if arr is None:
            raise StoreError(
                f"{array_name!r} in {self._zarr.path or '<root>'} is not a "
                f"chunk array; nothing to derive presence for"
            )
        shards = getattr(arr, "shards", None)
        if shards is not None:
            if on_sharded == "skip":
                return sorted(arr.attrs.get(_NONEMPTY_CHUNKS_ATTR, []) or [])
            if on_sharded == "raise":
                raise ShardedPresenceError(
                    f"Cannot derive presence for {array_name!r} in "
                    f"{self._zarr.path or '<root>'}: it is natively sharded "
                    f"and on_sharded='raise' was requested. Pass "
                    f"on_sharded='derive' (the default) to rebuild it, or "
                    f"'skip' to leave this array's manifest alone."
                )
            keys = self._derive_presence_sharded(arr, array_name, tuple(shards))
            arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(keys)
            return sorted(keys)
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

    def _array_has_stored_data(self, array_name: str) -> bool:
        """Whether ``array_name`` has any chunk object on disk.

        The question ``nonempty_chunks`` is *supposed* to answer, asked of
        the store instead — because the cases where it matters are exactly
        the cases where the manifest is not to be trusted.  A decentralised
        writer passing ``record_presence=False`` leaves it empty while the
        payloads are on disk, and an array whose manifest lost an update
        under-reports.  Believing it there is how a populated array gets
        deleted as "empty".

        Short-circuits on the first key, so it costs one listing request
        rather than a full enumeration, and it is only asked when the
        cheap signal already said "empty".
        """
        from zarr.core.sync import sync

        base = self._zarr.path.strip("/")
        prefix = f"{base}/{array_name}/c/" if base else f"{array_name}/c/"
        return bool(sync(_any_key_under(self._zarr.store, prefix)))

    def _derive_presence_sharded(
        self, arr: zarr.Array, array_name: str, shards: tuple[int, ...],
    ) -> set[str]:
        """The sharded half of :meth:`derive_nonempty_chunks`.

        Zarr builds a sharded array's chunk grid from the SHARD shape
        (``chunks_out = shard_shape`` when ``shards=`` is given), so a
        stored key ``c/i/j/k`` names a shard, not a cell.  Each one is
        expanded to the cell region it covers and read whole: unwritten
        cells come back as ``b""``, which is exactly the emptiness test
        the unsharded branch applies to its per-cell reads.

        Deliberately not routed through ``flush_prefetch``: its direct
        path already declines sharded arrays, and its gather path would
        issue one request per CELL — a thousand reads for an 8³ shard
        that one slice satisfies.  Reading region by region also keeps
        peak memory at one shard, where the unsharded branch holds every
        cell payload at once.
        """
        base = self._zarr.path.strip("/")
        prefix = f"{base}/{array_name}/c/" if base else f"{array_name}/c/"
        origin = _grid_origin(arr)
        shape = arr.shape
        ndim = arr.ndim

        keys: set[str] = set()
        for stored in _list_store_prefix(self._zarr.store, prefix):
            parts = stored[len(prefix):].split("/")
            if len(parts) != ndim:
                continue
            try:
                shard_index = tuple(int(p) for p in parts)
            except ValueError:
                continue
            # The cell region this shard covers, clipped to the array: a
            # shard at the edge of the grid is only partly in bounds, and
            # a shard may legitimately be larger than the whole array.
            starts = tuple(i * s for i, s in zip(shard_index, shards))
            stops = tuple(
                min(start + s, dim)
                for start, s, dim in zip(starts, shards, shape)
            )
            if any(stop <= start for start, stop in zip(starts, stops)):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UnstableSpecificationWarning)
                block = np.asarray(
                    arr[tuple(slice(a, b) for a, b in zip(starts, stops))]
                )
            for offset in np.ndindex(block.shape):
                if not block[offset]:
                    continue
                index = tuple(a + o for a, o in zip(starts, offset))
                coords = (
                    index if origin is None
                    else tuple(i + o for i, o in zip(index, origin))
                )
                keys.add(_format_chunk_key(coords))
            # Dropped before the next shard: only truthiness was wanted.
            del block
        return keys

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


class PresenceEvent(NamedTuple):
    """One stamp applied to an array's ``nonempty_chunks`` manifest.

    Delivered to whatever :func:`observe_presence_writes` has installed.
    ``present`` mirrors ``_record_nonempty_chunk``'s own argument: True
    added the key, False discarded it.
    """

    array_name: str
    chunk_key: str
    present: bool


# Installed by :func:`observe_presence_writes`.  A list rather than a
# single slot so the block nests, and module-level rather than per-Group
# because the writers that stamp are spread across Group, the batch
# flush and apply_presence.
_presence_observers: list[Callable[[PresenceEvent], None]] = []


@contextmanager
def observe_presence_writes(
    callback: Callable[[PresenceEvent], None],
) -> Iterator[None]:
    """Call ``callback`` for every incremental ``nonempty_chunks`` stamp.

    An INSTRUMENT, not a data API: it exists so a test can assert *where*
    a stamp happened — inside a lock, on which thread — which nothing
    observable from the store can answer after the fact.  Reach for
    :meth:`Group.list_chunks` to ask what the manifest says.

    ``callback`` runs SYNCHRONOUSLY, on the thread performing the write,
    at the moment the stamp lands.  That is the whole point: a caller
    checking its own lock depth can only do so while the write is
    happening, so an async or batched-at-exit notification would answer a
    different question.  Keep the callback cheap and non-raising — it
    runs in the ``write_bytes`` hot path, and an exception propagates
    into the writer.

    Covers the INCREMENTAL stamp sites: :meth:`Group.write_bytes`,
    :meth:`Group.write_cells`, :meth:`Group.apply_presence` and the
    :meth:`Group.batched_writes` flush.  NOT
    :meth:`Group.derive_nonempty_chunks`, which rebuilds the whole
    manifest from the store listing: it is idempotent and has no
    per-cell semantics, so it cannot lose a key the way a
    read-modify-write can, and there is nothing there to police.

    Example::

        seen = []
        with observe_presence_writes(seen.append):
            write_chunk_vertices(level_group, (0, 0, 0), positions)
        assert [e.chunk_key for e in seen] == ["0.0.0"]
    """
    _presence_observers.append(callback)
    try:
        yield
    finally:
        _presence_observers.remove(callback)


def _emit_presence(array_name: str, chunk_key: str, present: bool) -> None:
    """Notify observers that ``chunk_key``'s stamp just landed.

    The empty-list check is why this is cheap enough to sit in the write
    path: with nothing installed it costs one truth test and no
    allocation.
    """
    if not _presence_observers:
        return
    event = PresenceEvent(array_name, chunk_key, present)
    # Copy: a callback is allowed to unwind its own block.
    for callback in tuple(_presence_observers):
        callback(event)

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


def _listable_prefix(cells_prefix: str) -> str:
    """The array node a ``<array>/c/`` prefix lives under.

    Listing is asked of the ARRAY'S prefix and filtered to its cells,
    never of ``c/`` directly: icechunk accepts only a prefix that names a
    group or an array, and ``c/`` is neither, so asking for it raised on
    every icechunk store -- which took ``write_points`` down with it once
    the layout check started asking the store whether an array is empty.
    The cost is the array's own ``zarr.json`` in the listing, dropped by
    the filter.
    """
    if not cells_prefix.endswith("c/"):
        raise ValueError(f"not a cell prefix: {cells_prefix!r}")
    return cells_prefix[: -len("c/")]


async def _collect_store_prefix(store: Any, prefix: str) -> list[str]:
    return [
        key async for key in store.list_prefix(_listable_prefix(prefix))
        if key.startswith(prefix)
    ]


def _list_store_prefix(store: Any, prefix: str) -> list[str]:
    """List every stored key under the cell prefix ``<array>/c/``.

    A zarr Store is async-only, hence the ``sync``.  See
    :func:`_listable_prefix` for why the listing starts one level up.
    """
    from zarr.core.sync import sync

    return sync(_collect_store_prefix(store, prefix))


async def _any_key_under(store: Any, prefix: str) -> bool:
    async for key in store.list_prefix(_listable_prefix(prefix)):
        if key.startswith(prefix):
            return True
    return False


def _coord_to_index(
    coords: tuple[int, ...], origin: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Translate an absolute chunk coord to a 0-based array index.

    A coord whose rank does not match the origin's is returned UNCHANGED
    rather than zipped.  ``zip`` truncates to the shorter sequence, which
    turned a rank disagreement into a silently wrong cell: a rank-4 key
    against a rank-3 origin produced a rank-3 index that dropped the last
    spatial axis and offset the leading one by the wrong origin
    component, then passed the length check in
    :func:`_coords_in_bounds` and aliased every key differing only in the
    dropped axis onto one cell.  Returning the coords lets that same
    length check reject them, which is what every caller here already
    handles -- :meth:`Group.read_bytes` treats it as absent,
    :meth:`Group.write_bytes` raises.
    """
    if origin is None or len(coords) != len(origin):
        return coords
    return tuple(c - o for c, o in zip(coords, origin))


class _ChunkListing:
    """One chunk array's presence manifest, in the forms readers ask for.

    Built once per :meth:`Group.cached_nodes` session from the array's
    ``nonempty_chunks`` attribute.  Every derived form is lazy: an array a
    read only lists never pays the coordinate parse, and a read that never
    resolves a bounding box never builds the spatial index.
    """

    __slots__ = (
        "keys", "_coords", "_present", "_by_spatial", "_by_spatial_nd",
        "_spec", "_spec_known",
    )

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self._coords: list[tuple[int, ...]] | None = None
        self._present: frozenset[tuple[int, ...]] | None = None
        self._by_spatial: dict[tuple[int, ...], list[tuple[int, ...]]] | None = None
        self._by_spatial_nd: int | None = None
        # The direct-read spec rides along: it is per-array, has the same
        # lifetime, and is dropped by the same invalidation.  Tracked with
        # a separate flag because ``None`` is a real answer -- it means
        # "this array is not direct-readable", which is worth caching too.
        self._spec: Any = None
        self._spec_known = False

    @property
    def spec_known(self) -> bool:
        return self._spec_known

    @property
    def spec(self) -> Any:
        return self._spec

    def set_spec(self, spec: Any) -> None:
        self._spec = spec
        self._spec_known = True

    def coords(self) -> list[tuple[int, ...]]:
        """The keys parsed to coordinate tuples, in numeric order.

        Keys that do not parse are skipped — the manifest should hold
        nothing else, but a stray entry is not worth failing a read over.
        """
        if self._coords is None:
            self._coords = sorted(
                c for c in (_parse_chunk_coords(k) for k in self.keys)
                if c is not None
            )
        return self._coords

    def present(self) -> frozenset[tuple[int, ...]]:
        """The same coordinates as a set, for membership tests.

        What lets a bounding box be resolved from the box side: the box
        names a handful of cells and asks which of them exist, instead of
        asking every cell in the level whether it is in the box.
        """
        if self._present is None:
            self._present = frozenset(self.coords())
        return self._present

    def by_spatial(self, nd: int) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """Coordinates grouped by their trailing ``nd`` (spatial) axes.

        The index a box-side lookup needs on a store whose keys carry a
        leading non-spatial axis — ``chunk_by_attribute``, or a rechunk by
        a computed dimension — where one spatial cell maps to one key per
        bin.  On an ordinary store this is an identity grouping and pure
        overhead, which is why callers check arity first and use
        :meth:`present` instead.
        """
        if self._by_spatial is None or self._by_spatial_nd != nd:
            index: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
            for c in self.coords():
                index.setdefault(c[-nd:], []).append(c)
            self._by_spatial = index
            self._by_spatial_nd = nd
        return self._by_spatial


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
    if len(coords) != len(shape):
        # Named separately from an out-of-range coord because the cause
        # is different and the fix is elsewhere: the array was allocated
        # at the wrong rank, not addressed at the wrong place.  The usual
        # reason is a level chunked by an attribute -- its keys carry a
        # leading bin axis -- whose array was allocated spatial-only.
        raise StoreError(
            f"Chunk coords {coords} have rank {len(coords)} but array "
            f"{array_name!r} has grid {shape}, rank {len(shape)}. The "
            f"array was allocated at the wrong rank for this level; if "
            f"the level is chunked by an attribute its keys carry a "
            f"leading bin axis that the array does not have."
        )
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
    arr: zarr.Array, array_name: str, chunk_key: str, *, present: bool,
) -> None:
    """Update the per-array ``nonempty_chunks`` manifest.

    The manifest is a sorted list of dot-separated chunk keys —
    consulted by :meth:`Group.list_chunks` and :meth:`chunk_exists` so
    those calls don't have to scan every shard tail just to enumerate
    non-empty cells.

    ``array_name`` is carried only to name the array in the
    :func:`observe_presence_writes` event; ``arr`` is what gets written.
    """
    current = arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
    keys = set(current) if current else set()
    if present:
        keys.add(chunk_key)
    else:
        keys.discard(chunk_key)
    arr.attrs[_NONEMPTY_CHUNKS_ATTR] = sorted(keys)
    _emit_presence(array_name, chunk_key, present)


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
