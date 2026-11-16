"""Synchronous ``zarr.json`` resolution for nodes on a local filesystem.

:func:`resolve_local_node` reads a node's ``zarr.json`` with a plain
``open()`` on the calling thread instead of zarr's event-loop round trip,
and caches the parsed metadata process-wide.  A cache hit is validated
with one ``os.stat`` (inode, size, mtime, ctime); a document modified
within :data:`_RACY_NS` of being read is never cached (git's racy-clean
rule).  Each hit returns a fresh node with its own ``attributes`` copy.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

import zarr
from zarr.core.array import AsyncArray
from zarr.core.group import AsyncGroup, _build_metadata_v3
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import LocalStore

# A document modified this recently when it was read is not cached:
# well above the filesystem timestamp granularity (a jiffy on Linux).
_RACY_NS = 50_000_000

# Entries kept before the oldest is evicted.
_MAX_ENTRIES = 4096

# ``zarr.json`` path -> (stat signature, metadata, attributes as JSON).
_CACHE: dict[str, tuple[tuple[int, int, int, int], Any, str]] = {}

_MISSING = (FileNotFoundError, IsADirectoryError, NotADirectoryError)


def _signature(st: os.stat_result) -> tuple[int, int, int, int]:
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _with_fresh_attributes(metadata: Any, attributes_json: str) -> Any:
    """``metadata`` with its own copy of ``attributes``, without re-running
    the validating ``__init__`` that ``dataclasses.replace`` would."""
    fresh = copy.copy(metadata)
    object.__setattr__(fresh, "attributes", json.loads(attributes_json))
    return fresh


def _read_metadata(fn: str) -> Any:
    """Parse ``fn``, caching the result when it is safe to; ``None`` if
    absent."""
    started = time.time_ns()
    try:
        with open(fn, "rb") as fh:
            st = os.fstat(fh.fileno())
            raw = fh.read()
    except _MISSING:
        return None
    metadata = _build_metadata_v3(json.loads(raw))
    if st.st_mtime_ns < started - _RACY_NS:
        if len(_CACHE) >= _MAX_ENTRIES:
            try:
                del _CACHE[next(iter(_CACHE))]
            except (KeyError, RuntimeError, StopIteration):
                pass
        _CACHE[fn] = (
            _signature(st), metadata, json.dumps(metadata.attributes),
        )
        metadata = _with_fresh_attributes(metadata, _CACHE[fn][2])
    return metadata


def resolve_local_node(
    group: zarr.Group, path: str,
) -> zarr.Array | zarr.Group | None:
    """Resolve ``path`` under ``group`` the way ``group[path]`` would.

    Returns the node, or ``None`` when no ``zarr.json`` is there.
    Raises :class:`NotImplementedError` when the group is not a plain
    Zarr v3 ``LocalStore`` group without consolidated metadata -- the
    caller then takes zarr's own path, which is the authority for every
    other store (wrappers, icechunk, object stores, Pyodide).
    """
    store = group.store
    if type(store) is not LocalStore:
        raise NotImplementedError
    group_metadata = group.metadata
    if (
        group_metadata.zarr_format != 3
        or group_metadata.consolidated_metadata is not None
    ):
        raise NotImplementedError
    store_path = group.store_path / path
    fn = os.path.join(str(store.root), store_path.path, "zarr.json")
    try:
        st = os.stat(fn)
    except _MISSING:
        return None
    entry = _CACHE.get(fn)
    if entry is not None and entry[0] == _signature(st):
        metadata = _with_fresh_attributes(entry[1], entry[2])
    else:
        metadata = _read_metadata(fn)
        if metadata is None:
            return None
    if isinstance(metadata, ArrayV3Metadata):
        return zarr.Array(AsyncArray(metadata, store_path=store_path))
    return zarr.Group(AsyncGroup(metadata, store_path=store_path))
