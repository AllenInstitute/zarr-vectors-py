"""Backend-agnostic group abstraction wrapping a :class:`zarr.Group`.

This is the format seam: all ZV array I/O routes through this class.
The underlying Zarr store can be any :class:`zarr.abc.store.Store` —
``LocalStore``, ``MemoryStore``, ``FsspecStore``, ``ObjectStore``,
``IcechunkStore``.

Per-chunk byte blobs are stored as tiny single-chunk 1D ``uint8`` Zarr
arrays under a per-array Zarr group (Option G in the design doc):

    level/vertices/0.1.2          → Zarr 1D uint8 array, shape=(N,), chunks=(N,)
    level/vertex_fragments/0.1.2  → likewise
    level/object_index/data       → likewise (one blob per slot)

Per-array metadata (the ``zv_array`` discriminator and friends) lives on
the *group* node (``vertices/.attrs`` in v3 maps to ``zarr.json``
``attributes``).

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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import zarr
from zarr.codecs import VLenBytesCodec
from zarr.errors import UnstableSpecificationWarning
from zarr.storage import LocalStore

from zarr_vectors.exceptions import StoreError


class Group:
    """A ZV group wrapping an underlying :class:`zarr.Group`."""

    # Class-level defaults so callers that build a Group via ``__new__``
    # (see ``create_store`` / ``open_store``) start with batching off
    # without needing to remember to set the attributes.
    _pending_writes: list[tuple[str, str, bytes]] | None = None
    _pending_array_metas: dict[str, dict[str, Any]] | None = None
    _prefetch_cache: dict[tuple[str, str], bytes] | None = None
    # Active codec spec for chunk-array writes, set by
    # :meth:`batched_writes(compressor=...)`.  ``None`` means callers fall
    # back to zarr v3's default codec pipeline (``bytes`` + ``zstd``).
    # Consumed by :meth:`write_bytes` and the batched flush in
    # :mod:`zarr_vectors.core._batch_writer`.
    _active_codecs: list[dict[str, Any]] | None = None
    # When set, per-chunk-array creations produce a single multidim
    # vlen-bytes Zarr array (one cell per spatial chunk) instead of the
    # legacy Option-G group-of-tiny-arrays.  Set by
    # :meth:`native_sharded_arrays`; the dict holds ``grid_shape`` and
    # ``shard_shape`` (``None`` = unsharded, one storage object per chunk
    # — the default layout).  Used by ``arrays._ensure_array_dir`` and the
    # ``write_bytes`` / ``write_array_meta`` dispatch on this class.
    _native_sharded_config: dict[str, tuple[int, ...] | None] | None = None

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

    @classmethod
    def _from_zarr(cls, zarr_group: zarr.Group) -> Group:
        instance = cls.__new__(cls)
        instance._zarr = zarr_group
        instance._pending_writes = None
        instance._pending_array_metas = None
        instance._prefetch_cache = None
        instance._active_codecs = None
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
        return type(self)._from_zarr(zg)

    def require_group(self, name: str) -> Group:
        zg = self._zarr.require_group(name)
        return type(self)._from_zarr(zg)

    def __getitem__(self, key: str) -> Group:
        try:
            node = self._zarr[key]
        except KeyError:
            raise StoreError(
                f"Group {key!r} not found under {self._zarr.path or '<root>'}"
            ) from None
        if not isinstance(node, zarr.Group):
            raise StoreError(
                f"{key!r} under {self._zarr.path or '<root>'} is a "
                f"{type(node).__name__}, not a Group"
            )
        return type(self)._from_zarr(node)

    def __contains__(self, key: str) -> bool:
        return key in self._zarr

    def __iter__(self) -> Iterator[str]:
        yield from sorted(self._zarr.group_keys())

    def children(self) -> list[str]:
        """Return every immediate child name — both arrays and groups.

        Use this when iterating a parent path that may contain a mix of
        Option-G chunk-group children (vertex / link attributes) and
        flat-array children (object / group attributes after the 0.8.1
        migration).  ``__iter__`` deliberately yields groups only to
        preserve back-compat for callers that predate the migration.
        """
        return sorted(
            set(self._zarr.group_keys()) | set(self._zarr.array_keys())
        )

    # ---------------- chunk I/O ------------------------------------------
    #
    # Two physical layouts share this surface:
    #
    # * **Legacy "Option G"** — ``<array_name>`` is a Zarr Group whose
    #   children are single-chunk ``uint8`` arrays, one per chunk key.
    # * **Native sharded** — ``<array_name>`` is a single multidim Zarr
    #   array (vlen-bytes) whose shape is the level's chunk grid; the
    #   Zarr v3 ``sharding_indexed`` codec packs many cells into one
    #   storage object.  Created via :meth:`create_sharded_chunk_array`
    #   or by the migration tool in :mod:`zarr_vectors.sharding.io`.
    #
    # ``read_bytes`` / ``write_bytes`` / ``chunk_exists`` / ``list_chunks``
    # dispatch on the node type at runtime so callers don't care which
    # layout they're talking to.

    def write_bytes(self, array_name: str, chunk_key: str, data: bytes) -> None:
        # Single-array layout: a Zarr Array at ``array_name`` is the
        # multidim vlen-bytes array whose cells are spatial chunks.  This
        # is the layout produced under :meth:`native_sharded_arrays` for
        # every per-spatial-chunk array (vertices, fragments, links, …).
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is not None:
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
                self._pending_writes.append((array_name, chunk_key, bytes(data)))
                return
            _vlen_set_cell(sharded_arr, index, bytes(data))
            _record_nonempty_chunk(sharded_arr, chunk_key, present=bool(data))
            return

        # Per-cell primitive: ``array_name`` is a Zarr Group whose
        # children are single-chunk ``uint8`` arrays, one per key.  This
        # is retained ONLY for ``cross_chunk_links`` / ``…_attributes``,
        # whose cell keys are canonical-sorted endpoint tuples (not
        # spatial grid coords) and so cannot map onto a grid-shaped
        # array.  Spatial per-chunk arrays never reach here — they are
        # pre-created as single vlen arrays by ``_ensure_array_dir``.
        if self._pending_writes is not None:
            self._pending_writes.append((array_name, chunk_key, bytes(data)))
            return

        arr_group = self._zarr.require_group(array_name)
        if chunk_key in arr_group:
            del arr_group[chunk_key]
        # When a session compressor is active we pass an explicit codec
        # list to ``create_array``; otherwise zarr 3.x's default applies
        # (which is ``bytes`` + ``zstd``).  See
        # :func:`zarr_vectors.encoding.compression.resolve_compressor`.
        from zarr_vectors.encoding.compression import codecs_for_create_array
        extra_kwargs: dict[str, Any] = {}
        if self._active_codecs is not None:
            extra_kwargs["compressors"] = codecs_for_create_array(
                self._active_codecs
            )
        n = len(data)
        if n == 0:
            arr_group.create_array(
                chunk_key, shape=(0,), chunks=(1,), dtype="uint8",
                **extra_kwargs,
            )
            return
        a = arr_group.create_array(
            chunk_key, shape=(n,), chunks=(n,), dtype="uint8",
            **extra_kwargs,
        )
        a[:] = np.frombuffer(data, dtype="uint8")

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
        from zarr_vectors.core._batch_reader import flush_prefetch

        self._prefetch_cache = flush_prefetch(self._zarr, plan)
        try:
            yield
        finally:
            self._prefetch_cache = None

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

    @contextmanager
    def native_sharded_arrays(
        self,
        shard_shape: tuple[int, ...] | None,
        grid_shape: tuple[int, ...],
        *,
        origin: tuple[int, ...] | None = None,
    ) -> Iterator[None]:
        """Make subsequent per-chunk-array creations single-array.

        Inside this block, every :func:`zarr_vectors.core.arrays._ensure_array_dir`
        call for a per-spatial-chunk array (vertices, vertex_fragments,
        links/<delta>, link_fragments, vertex_attributes/<name>,
        fragment_attributes/<name>) allocates a single multidim
        vlen-bytes Zarr array at that path — one cell per spatial chunk,
        chunk files at ``<array>/c/i/j/k`` — instead of the legacy
        Option-G group-of-tiny-arrays.  Subsequent :meth:`write_bytes`
        calls go through the grid-coord write dispatch on this class, so
        callers don't need to change.  This is the default layout opened
        by :func:`zarr_vectors.core.arrays.open_write_session`.

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
        if self._prefetch_cache is not None:
            cached = self._prefetch_cache.get((array_name, chunk_key))
            if cached is not None:
                return cached

        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is not None:
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

        path = f"{array_name}/{chunk_key}"
        try:
            arr = self._zarr[path]
        except KeyError:
            raise StoreError(
                f"Chunk {array_name!r}/{chunk_key!r} not found in "
                f"{self._zarr.path or '<root>'}"
            ) from None
        if not isinstance(arr, zarr.Array):
            raise StoreError(
                f"{path!r} is a {type(arr).__name__}, not an Array"
            )
        if arr.shape[0] == 0:
            return b""
        return bytes(np.asarray(arr[:]).tobytes())

    def chunk_exists(self, array_name: str, chunk_key: str) -> bool:
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is not None:
            present = sharded_arr.attrs.get(_NONEMPTY_CHUNKS_ATTR)
            if present is not None:
                return chunk_key in present
            # Fall back to inspecting the cell — slow path used when the
            # presence manifest is missing (e.g. mid-migration).
            coords = _parse_chunk_coords(chunk_key)
            index = (
                None if coords is None
                else _coord_to_index(coords, _grid_origin(sharded_arr))
            )
            if index is None or not _coords_in_bounds(index, sharded_arr.shape):
                return False
            return _vlen_get_cell(sharded_arr, index) != b""

        return f"{array_name}/{chunk_key}" in self._zarr

    def list_chunks(self, array_name: str) -> list[str]:
        if array_name not in self._zarr:
            return []
        try:
            node = self._zarr[array_name]
        except KeyError:
            return []
        if isinstance(node, zarr.Array):
            # Native-sharded layout.  Trust the per-array presence
            # manifest written by ``write_bytes``; without it we'd have
            # to fetch every shard index to find non-empty cells.
            present = node.attrs.get(_NONEMPTY_CHUNKS_ATTR)
            return sorted(present) if present else []
        if not isinstance(node, zarr.Group):
            return []
        return sorted(node.array_keys())

    # ---------------- array metadata ----------------

    def write_array_meta(self, array_name: str, meta: dict[str, Any]) -> None:
        # Native-sharded fast path: write directly to the multidim
        # array's ``attrs`` (the Zarr-spec location for per-array user
        # metadata).  Checked BEFORE the batched queue so that writers
        # under :meth:`native_sharded_arrays` always land on the array.
        sharded_arr = self._sharded_chunk_array(array_name)
        if sharded_arr is not None:
            sharded_arr.attrs.update(_json_safe(meta))
            return
        # Batched-write mode (see :meth:`batched_writes`): queue the
        # full parent-group ``zarr.json`` content so it flushes in the
        # gather instead of paying a per-array sync ``require_group +
        # attrs.update`` (which costs 2-3 round-trips each on cloud).
        # Merge with anything already queued for the same name so
        # successive ``write_array_meta`` calls within the block
        # compose, matching the existing ``attrs.update`` semantics.
        if self._pending_array_metas is not None:
            existing = self._pending_array_metas.get(array_name, {})
            merged = {**existing, **_json_safe(meta)}
            self._pending_array_metas[array_name] = merged
            return
        arr_group = self._zarr.require_group(array_name)
        arr_group.attrs.update(_json_safe(meta))

    def read_array_meta(self, array_name: str) -> dict[str, Any]:
        if array_name not in self._zarr:
            return {}
        try:
            node = self._zarr[array_name]
        except KeyError:
            return {}
        if not isinstance(node, (zarr.Group, zarr.Array)):
            return {}
        return dict(node.attrs)

    def array_exists(self, array_name: str) -> bool:
        if array_name not in self._zarr:
            return False
        try:
            node = self._zarr[array_name]
        except KeyError:
            return False
        return isinstance(node, (zarr.Group, zarr.Array))

    # ---------------- standard Zarr v3 arrays (single array per path) ----------------
    #
    # The methods above (``write_bytes`` / ``read_bytes``) implement the
    # legacy "Option G" layout where every logical array is a Zarr group
    # holding single-chunk ``uint8`` arrays.  The methods below write a
    # single standard Zarr v3 array at the given path — the target of
    # the v0.7 format migration (see plan
    # ``can-you-do-an-compressed-wilkes.md``).  Both layouts coexist
    # during the transition; new sites should prefer these.
    #
    # Batched-writes / batched-reads queues only cover the Option-G
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
        node = self._require_array_node(path)
        return np.asarray(node[:])

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
        node = self._require_array_node(path)
        return [bytes(b) for b in node[:]]

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
        """Allocate a sharded vlen-bytes Zarr array for per-chunk blobs.

        Replaces the legacy "Option G" group-of-tiny-arrays layout with
        a single Zarr v3 array whose shape equals the level's chunk
        grid.  Each cell holds one ZVF spatial chunk's payload.  When
        ``shard_shape`` is provided the ``sharding_indexed`` codec packs
        many cells into a single storage object — the standard cloud
        layout described in
        :doc:`/spec/chunking/sharding`.

        Args:
            array_name: Logical path of the array, e.g. ``"vertices"``
                or ``"links/0"``.  Intermediate path segments become
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
                the historical default of the batched Option-G writer.
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            arr = parent.create_array(leaf, **create_kwargs)
            arr.attrs[_NONEMPTY_CHUNKS_ATTR] = []
            # Store the grid origin so cell ``index = coord - origin``.
            # Only when non-trivial — a zero origin is the common case
            # and its absence means "coords are array indices".
            if origin is not None and any(int(o) != 0 for o in origin):
                arr.attrs[_CHUNK_GRID_ORIGIN_ATTR] = [int(o) for o in origin]
            if attributes:
                arr.attrs.update(_json_safe(attributes))

    def _sharded_chunk_array(self, array_name: str) -> zarr.Array | None:
        """Return the single multidim vlen-bytes Zarr array at
        ``array_name`` (the per-spatial-chunk layout), else ``None``.

        Returns ``None`` when a Zarr Group sits at the path instead — the
        per-cell primitive used by ``cross_chunk_links`` — so callers
        fall back to the per-cell logic.
        """
        if array_name not in self._zarr:
            return None
        try:
            node = self._zarr[array_name]
        except KeyError:
            return None
        return node if isinstance(node, zarr.Array) else None

    def read_array_attrs(self, path: str) -> dict[str, Any]:
        """Read the ``attributes`` block of a Zarr array at ``path``.

        Returns ``{}`` for missing paths or paths that point at a group
        rather than an array.
        """
        if path not in self._zarr:
            return {}
        try:
            node = self._zarr[path]
        except KeyError:
            return {}
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
        the legacy Option-G layout (a group containing chunk arrays).
        """
        if path not in self._zarr:
            return False
        try:
            node = self._zarr[path]
        except KeyError:
            return False
        return isinstance(node, zarr.Array)

    def _require_array_node(self, path: str) -> zarr.Array:
        if path not in self._zarr:
            raise StoreError(
                f"Array {path!r} not found in {self._zarr.path or '<root>'}"
            )
        node = self._zarr[path]
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

    Zarr 3.x vlen-bytes scalar indexing returns a 0-d object array; we
    slice-then-extract so the result is always a ``bytes`` instance.
    """
    slices = tuple(slice(c, c + 1) for c in coords)
    region = np.asarray(arr[slices])
    if region.size == 0:
        return b""
    val = region.flat[0]
    if val is None:
        return b""
    return bytes(val)


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
