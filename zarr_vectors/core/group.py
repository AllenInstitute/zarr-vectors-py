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

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
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

    # ---------------- chunk I/O (Option G: 1 tiny array per chunk) ----------------

    def write_bytes(self, array_name: str, chunk_key: str, data: bytes) -> None:
        # Batched-write mode (see :meth:`batched_writes`): defer until
        # the context manager flushes all queued PUTs concurrently.
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

    def read_bytes(self, array_name: str, chunk_key: str) -> bytes:
        # Batched-read mode (see :meth:`batched_reads`): serve from the
        # prefetch cache when possible.  Cache misses fall through to
        # the sync path below — useful when a caller under-specifies
        # the plan or hits an array the prefetch skipped.
        if self._prefetch_cache is not None:
            cached = self._prefetch_cache.get((array_name, chunk_key))
            if cached is not None:
                return cached

        path = f"{array_name}/{chunk_key}"
        try:
            arr = self._zarr[path]
        except KeyError:
            shard_bytes = _read_from_shard(self._zarr, array_name, chunk_key)
            if shard_bytes is not None:
                return shard_bytes
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
        if f"{array_name}/{chunk_key}" in self._zarr:
            return True
        return _shard_index_contains(self._zarr, array_name, chunk_key)

    def list_chunks(self, array_name: str) -> list[str]:
        if array_name not in self._zarr:
            return []
        try:
            node = self._zarr[array_name]
        except KeyError:
            return []
        if not isinstance(node, zarr.Group):
            return []
        keys: set[str] = set()
        for k in node.array_keys():
            if k.startswith("__shard_"):
                shard_arr = node[k]
                index = shard_arr.attrs.get("shard_index", None)
                if index:
                    keys.update(index.keys())
            else:
                keys.add(k)
        return sorted(keys)

    def list_subgroups(self, array_name: str) -> list[str]:
        """Return immediate subgroup names under ``array_name`` (sorted).

        Used by the v0.8 partitioned cross-chunk-link layout, which
        nests K-deep groups (one per sorted-unique-chunk segment) under
        ``cross_chunk_links/<delta>/``.  ``array_name`` is the parent
        path; subgroup names are returned without that prefix.

        Returns ``[]`` when ``array_name`` doesn't exist or isn't a
        Group.
        """
        if array_name not in self._zarr:
            return []
        try:
            node = self._zarr[array_name]
        except KeyError:
            return []
        if not isinstance(node, zarr.Group):
            return []
        return sorted(node.group_keys())

    # ---------------- array metadata ----------------

    def write_array_meta(self, array_name: str, meta: dict[str, Any]) -> None:
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
        if not isinstance(node, zarr.Group):
            return {}
        return dict(node.attrs)

    def array_exists(self, array_name: str) -> bool:
        if array_name not in self._zarr:
            return False
        try:
            node = self._zarr[array_name]
        except KeyError:
            return False
        return isinstance(node, zarr.Group)

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


# ---------------- shard fallback (transparent read of packed chunks) ----------

def _read_from_shard(
    zarr_group: zarr.Group, array_name: str, chunk_key: str,
) -> bytes | None:
    """Locate ``chunk_key`` inside any ``__shard_<id>`` array under
    ``array_name`` and return its bytes.  Returns ``None`` when no
    shard contains the key.
    """
    if array_name not in zarr_group:
        return None
    try:
        arr_group = zarr_group[array_name]
    except KeyError:
        return None
    if not isinstance(arr_group, zarr.Group):
        return None
    for name in arr_group.array_keys():
        if not name.startswith("__shard_"):
            continue
        shard_arr = arr_group[name]
        index = shard_arr.attrs.get("shard_index", None)
        if not index or chunk_key not in index:
            continue
        offset, nbytes = index[chunk_key]
        packed = np.asarray(shard_arr[offset:offset + nbytes])
        return bytes(packed.tobytes())
    return None


def _shard_index_contains(
    zarr_group: zarr.Group, array_name: str, chunk_key: str,
) -> bool:
    if array_name not in zarr_group:
        return False
    try:
        arr_group = zarr_group[array_name]
    except KeyError:
        return False
    if not isinstance(arr_group, zarr.Group):
        return False
    for name in arr_group.array_keys():
        if not name.startswith("__shard_"):
            continue
        index = arr_group[name].attrs.get("shard_index", None)
        if index and chunk_key in index:
            return True
    return False


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
