"""Chunk sources: a *composite* level whose objects reference fragments of other stores.

A level may declare a leading ``"source"`` chunk axis, exactly as an
attribute-chunked level declares its bin axis (``chunk_dims`` names it first;
cell keys become ``s.z.y.x``; object manifests carry one more coordinate).
:attr:`~zarr_vectors.core.metadata.LevelMetadata.chunk_sources` says what each
value of that axis is:

* ``chunk_sources[0] == {"self": true}`` -- this store's own cells, written
  and read as any level's;
* ``chunk_sources[s]`` for ``s >= 1`` -- another store::

      {"ref": {"path": "../member.zarrvectors", "id": "world"},
       "level": 0,
       "transform": [[...], ...],        # (D+1) x (D+1): source -> this store
       "fingerprint": {"object_count": 1234, ...}}

  ``ref`` is an OME-Zarr RFC-8 ``Reference``: ``path`` relative to this
  store's root (or absolute), ``id`` the coordinate system it binds to (a
  zarr-vectors store's ``world``). A cell ``(s, *spatial)`` is that source
  level's cell ``spatial``; its ``vertices`` come back through ``transform``
  (in this store's frame), and every other per-chunk array -- fragment
  indexes, vertex attributes, links -- comes back as stored, because their row
  numbers refer to the source's own vertex buffer, which is what is read.

So no manifest block changes: an object's fragment ``(s, z, y, x, f)`` is
fragment ``f`` of the source's cell ``(z, y, x)``, resolved where every read
already passes (:meth:`zarr_vectors.core.group.Group.read_bytes`). A source is
never written through a composite, and a source whose recorded fingerprint no
longer matches (it was rebuilt) is an error on first use, not a silent misread.
Spatial box queries over a composite level are refused (a source's cells are
in the source's grid); objects are read through their manifests.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.exceptions import ArrayError, MetadataError, StoreError

__all__ = [
    "SOURCE_AXIS",
    "chunk_sources",
    "is_composite",
    "set_chunk_sources",
    "source_fingerprint",
    "validate_chunk_sources",
    "verify_chunk_sources",
]

#: The name of a composite level's leading chunk axis.
SOURCE_AXIS = "source"
_LEVEL_KEY = "zarr_vectors_level"
#: Fingerprint keys zarr-vectors computes and checks; any other key a writer
#: records is carried, not checked.
_CHECKED = ("object_count",)


def validate_chunk_sources(sources: Any, chunk_dims: Any) -> None:
    """Raise :class:`MetadataError` unless *sources* is a well-formed source table."""
    if not isinstance(sources, (list, tuple)) or not sources:
        raise MetadataError("chunk_sources must be a non-empty list")
    if not chunk_dims or chunk_dims[0] != SOURCE_AXIS:
        raise MetadataError(
            f"a composite level's chunk_dims must lead with {SOURCE_AXIS!r}, "
            f"got {chunk_dims!r}")
    first = sources[0]
    if not (isinstance(first, dict) and first.get("self") is True):
        raise MetadataError('chunk_sources[0] must be {"self": true} (this store)')
    for i, s in enumerate(sources[1:], start=1):
        if not isinstance(s, dict) or s.get("self"):
            raise MetadataError(f"chunk_sources[{i}] must name another store")
        ref = s.get("ref")
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]:
            raise MetadataError(f"chunk_sources[{i}].ref must be {{'path': ..., 'id': ...}}")
        if int(s.get("level", 0)) < 0:
            raise MetadataError(f"chunk_sources[{i}].level must be >= 0")
        M = np.asarray(s.get("transform"), dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] < 2:
            raise MetadataError(
                f"chunk_sources[{i}].transform must be a square (D+1)x(D+1) matrix")
        if not isinstance(s.get("fingerprint", {}), dict):
            raise MetadataError(f"chunk_sources[{i}].fingerprint must be a dict")


def chunk_sources(level_group) -> "list[dict] | None":
    """The level's source table, or ``None`` for an ordinary level."""
    lv = level_group.attrs.get(_LEVEL_KEY) or {}
    got = lv.get("chunk_sources") if isinstance(lv, dict) else None
    return list(got) if got is not None else None


def is_composite(level_group) -> bool:
    return chunk_sources(level_group) is not None


def verify_chunk_sources(level_group) -> "list[dict] | None":
    """Open every source of a composite level now, raising if one cannot be read as recorded.

    The check a read makes on first use of each source, made for all of them
    at once: a reader that tolerates unreadable cells (and so would see a
    rebuilt source's cells as missing) calls this first. Raises
    :class:`~zarr_vectors.exceptions.StoreError` for a source that cannot be
    opened and :class:`~zarr_vectors.exceptions.ArrayError` for one whose
    fingerprint no longer matches. Returns the table (``None`` for an
    ordinary level, which has nothing to check).
    """
    res = resolver_for(level_group)
    if res is None:
        return None
    for s in range(1, len(res._table)):
        res.open(s)
    return list(res._table)


def source_fingerprint(level_group) -> dict:
    """What identifies a source level's content for a composite that references it."""
    from zarr_vectors.core.arrays import object_count

    return {"object_count": int(object_count(level_group))}


def set_chunk_sources(level_group, sources: list, *, fingerprint: bool = True) -> list:
    """Make *level_group* a composite level of *sources*; returns the stored table.

    Call before the level's arrays are allocated, and allocate them with the
    leading axis (``open_write_session(..., bin_count=len(sources))``). Each
    source ``ref.path`` is resolved against this store's root; with
    *fingerprint* (default) every source is opened once now and its
    :func:`source_fingerprint` merged into the entry (keys the caller gave
    win), so a later read can tell when it was rebuilt.
    """
    from zarr_vectors.core.metadata import LevelMetadata

    table = [dict(s) for s in sources]
    lv = dict(level_group.attrs.get(_LEVEL_KEY) or {})
    meta = LevelMetadata.from_dict({_LEVEL_KEY: lv})
    spatial = list(meta.chunk_dims or [])
    if spatial and spatial[0] == SOURCE_AXIS:
        spatial = spatial[1:]
    if not spatial:
        from zarr_vectors.core.store import read_root_metadata

        root = _store_root_group(level_group)
        spatial = [f"dim{i}" for i in range(int(read_root_metadata(root).sid_ndim))]
    meta.chunk_dims = [SOURCE_AXIS] + spatial
    if fingerprint:
        res = _Resolver(level_group, table)
        for s in range(1, len(table)):
            src = res.open(s, check=False)
            table[s]["fingerprint"] = {**source_fingerprint(src.level),
                                       **dict(table[s].get("fingerprint") or {})}
    meta.chunk_sources = table
    meta.validate()
    level_group.attrs.update(meta.to_dict())
    level_group._chunk_sources_state = None          # re-read on next use
    return table


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _store_root_dir(level_group) -> Path:
    from zarr.storage import LocalStore

    store = level_group._zarr.store
    if not isinstance(store, LocalStore):
        raise StoreError(
            f"chunk sources resolve relative paths on a local store; got "
            f"{type(store).__name__}")
    raw = store.root
    return raw if isinstance(raw, Path) else Path(raw)


def _store_root_group(level_group):
    from zarr_vectors.core.store import open_store

    return open_store(str(_store_root_dir(level_group)), mode="r")


class _Source:
    __slots__ = ("level", "transform", "dtype", "ndim")

    def __init__(self, level, transform, dtype, ndim):
        self.level = level
        self.transform = transform
        self.dtype = dtype
        self.ndim = ndim


class _Resolver:
    """Opens a composite level's sources on first use and reads their cells."""

    def __init__(self, level_group, table: list):
        self._table = table
        self._level = level_group
        self._root = _store_root_dir(level_group)
        self._open: dict = {}
        self._lock = threading.Lock()
        self._dtype = None

    def _out_dtype(self, fallback) -> np.dtype:
        """The composite level's declared vertex dtype (the source's when it declares none)."""
        if self._dtype is None:
            from zarr_vectors.core.arrays import vertices_dtype

            try:
                self._dtype = np.dtype(vertices_dtype(self._level))
            except Exception:                           # noqa: BLE001 - not allocated yet
                return np.dtype(fallback)
        return self._dtype

    def open(self, s: int, *, check: bool = True) -> _Source:
        got = self._open.get(s)
        if got is not None:
            return got
        with self._lock:
            got = self._open.get(s)
            if got is not None:
                return got
            if not 0 < s < len(self._table):
                raise ArrayError(f"chunk source {s} is not in this level's table "
                                 f"({len(self._table)} entries)")
            from zarr_vectors.core.arrays import vertices_dtype
            from zarr_vectors.core.store import get_resolution_level, open_store, read_root_metadata

            entry = self._table[s]
            raw = entry["ref"]["path"]
            path = Path(raw) if Path(raw).is_absolute() else (self._root / raw)
            try:
                root = open_store(str(path.resolve()), mode="r")
                level = get_resolution_level(root, int(entry.get("level", 0)))
            except Exception as exc:
                raise StoreError(f"chunk source {s} ({raw}) cannot be opened: {exc}") from exc
            if check:
                want = dict(entry.get("fingerprint") or {})
                have = source_fingerprint(level) if any(k in want for k in _CHECKED) else {}
                for k in _CHECKED:
                    if k in want and have.get(k) != want[k]:
                        raise ArrayError(
                            f"chunk source {s} ({raw}) has changed since this level was "
                            f"written: {k} is {have.get(k)!r}, recorded {want[k]!r}. "
                            f"Rewrite the composite level against the source as it is.")
            try:
                dtype = vertices_dtype(level)
            except Exception:                           # noqa: BLE001 - no vertices array
                dtype = np.dtype("float32")
            ndim = int(read_root_metadata(root).sid_ndim)
            got = _Source(level, np.asarray(entry["transform"], dtype=np.float64), dtype, ndim)
            self._open[s] = got
            return got

    def read_bytes(self, s: int, array_name: str, spatial_key: str) -> bytes:
        from zarr_vectors.constants import VERTICES

        src = self.open(s)
        raw = src.level.read_bytes(array_name, spatial_key)
        if array_name != VERTICES or not raw:
            return raw
        M = src.transform
        D = src.ndim
        out_dtype = self._out_dtype(src.dtype)
        if np.array_equal(M, np.eye(D + 1)) and out_dtype == src.dtype:
            return raw
        pts = np.frombuffer(raw, dtype=src.dtype).reshape(-1, D).astype(np.float64)
        out = pts @ M[:D, :D].T + M[:D, D]
        return np.ascontiguousarray(out.astype(out_dtype)).tobytes()


def resolver_for(group) -> "_Resolver | None":
    """The group's resolver when it is a composite level, cached on the group; else ``None``."""
    state = getattr(group, "_chunk_sources_state", None)
    if state is None:
        table = chunk_sources(group) if hasattr(group, "attrs") else None
        state = _Resolver(group, table) if table is not None else False
        group._chunk_sources_state = state
    return state or None


def split_source_key(chunk_key: str) -> "tuple[int, str] | None":
    """``(source, spatial key)`` of a composite cell key ``s.z.y.x``; ``None`` if malformed."""
    head, sep, rest = chunk_key.partition(".")
    if not sep:
        return None
    try:
        return int(head), rest
    except ValueError:
        return None
