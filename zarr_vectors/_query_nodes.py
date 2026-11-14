"""Nodes resolved together at the start of a query (experiment ``prime_nodes``).

A read resolves the level group and its arrays one at a time as it
discovers them -- the level, its attribute names, the presence listing,
then the engine's own resolution of the same arrays -- and behind an
object store each is a sequential round trip.  :func:`primed` resolves
the level and every array a read of it can touch in one gather, and for
the rest of the query :meth:`Group._lookup_node` and the engine's
:func:`~zarr_vectors.core.aio._resolve_nodes` answer from that set.

Read-only: nothing inside the block may create or remove a node.  The set
is keyed by store, so a query on one store never sees another's nodes.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

from zarr_vectors.constants import (
    LINK_FRAGMENTS,
    LINKS,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)

_lock = threading.Lock()
# (id(store), root-relative path) -> node, or _ABSENT for a path that is not there.
_nodes: dict[tuple[int, str], Any] = {}
_active: dict[int, int] = {}  # id(store) -> open blocks

#: What a read of a level can touch, below the level group.
_LEVEL_PATHS = (
    "", VERTICES, VERTEX_FRAGMENTS, VERTEX_ATTRIBUTES, OBJECT_INDEX,
    f"{OBJECT_INDEX}/manifests", OBJECT_ATTRIBUTES, LINKS, f"{LINKS}/0", LINK_FRAGMENTS,
)


def lookup(store: Any, path: str) -> Any:
    """The primed node at ``path`` (``_ABSENT`` when primed absent), or
    ``None`` when ``path`` was not primed."""
    if not _active:
        return None
    return _nodes.get((id(store), path.strip("/")))


@contextlib.contextmanager
def primed(root: Any, level: int) -> Iterator[None]:
    """Resolve ``level`` and its arrays in one gather, for this block.

    ``root`` is the store-root :class:`~zarr_vectors.core.group.Group`.
    Nested blocks on the same store reuse the outer one's nodes.
    """
    from zarr.core.sync import sync

    from zarr_vectors.core.aio import _resolve_nodes

    store = root._zarr.store
    sid = id(store)
    with _lock:
        outer = _active.get(sid, 0)
        _active[sid] = outer + 1
    try:
        if not outer:
            paths = {f"{level}/{p}".rstrip("/") for p in _LEVEL_PATHS}
            found = sync(_resolve_nodes(root._zarr._async_group, paths))
            with _lock:
                for path, node in found.items():
                    _nodes[(sid, path)] = node
        yield
    finally:
        with _lock:
            _active[sid] -= 1
            if not _active[sid]:
                del _active[sid]
                for key in [k for k in _nodes if k[0] == sid]:
                    del _nodes[key]
