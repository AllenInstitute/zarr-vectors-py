"""Nodes resolved together at the start of a query.

A read resolves the level group and its arrays one at a time as it
discovers them -- the level, its attribute names, the presence listing,
then the engine's own resolution of the same arrays -- and behind an
object store each is a sequential round trip.  :func:`primed` resolves
the level and every array a read of it can touch in one gather, and for
the rest of the query :meth:`Group._lookup_node` and the engine's
:func:`~zarr_vectors.core.aio._resolve_nodes` answer from that set.  A
node either of them resolves later in the query (a link family's offset
arrays, found by listing) joins the set, so it too is fetched once.

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
# (id(store), root-relative group path) -> its children's names, sorted.
_listings: dict[tuple[int, str], list[str]] = {}
_active: dict[int, int] = {}  # id(store) -> open blocks

#: What a read of a level can touch, below the level group.
_LEVEL_PATHS = (
    "", VERTICES, VERTEX_FRAGMENTS, VERTEX_ATTRIBUTES, OBJECT_INDEX,
    f"{OBJECT_INDEX}/manifests", OBJECT_ATTRIBUTES, LINKS, f"{LINKS}/0", LINK_FRAGMENTS,
)
#: Groups below the level whose children a read lists: the link family's
#: offset arrays.  Listed in the same gather, children kept.
_LEVEL_LISTINGS = (f"{LINKS}/0",)


def lookup(store: Any, path: str) -> Any:
    """The primed node at ``path`` (``_ABSENT`` when primed absent), or
    ``None`` when ``path`` was not primed."""
    if not _active:
        return None
    return _nodes.get((id(store), path.strip("/")))


def listing(store: Any, path: str) -> list[str] | None:
    """The children primed for group ``path``, or ``None`` when not primed."""
    if not _active:
        return None
    return _listings.get((id(store), path.strip("/")))


def remember(store: Any, path: str, node: Any) -> None:
    """Keep ``node`` (or ``_ABSENT``) for the rest of a query on ``store``."""
    if _active.get(id(store)):
        with _lock:
            _nodes[(id(store), path.strip("/"))] = node


@contextlib.contextmanager
def primed(root: Any, level: int) -> Iterator[None]:
    """Resolve ``level`` and its arrays in one gather, for this block.

    ``root`` is the store-root :class:`~zarr_vectors.core.group.Group`.
    Nested blocks on the same store reuse the outer one's nodes.
    """
    import asyncio

    from zarr.core.sync import sync

    from zarr_vectors.core.aio import _fetch_listings, _resolve_nodes

    store = root._zarr.store
    sid = id(store)
    with _lock:
        outer = _active.get(sid, 0)
        _active[sid] = outer + 1
    try:
        if not outer:
            paths = {f"{level}/{p}".rstrip("/") for p in _LEVEL_PATHS}
            groups = {f"{level}/{p}" for p in _LEVEL_LISTINGS}
            aroot = root._zarr._async_group

            async def both() -> tuple[dict[str, Any], dict[str, list[str]]]:
                return await asyncio.gather(
                    _resolve_nodes(aroot, paths), _fetch_listings(aroot, groups),
                )

            found, listed = sync(both())
            with _lock:
                for path, node in found.items():
                    _nodes.setdefault((sid, path), node)
                for path, names in listed.items():
                    _listings[(sid, path)] = names
        yield
    finally:
        with _lock:
            _active[sid] -= 1
            if not _active[sid]:
                del _active[sid]
                for key in [k for k in _nodes if k[0] == sid]:
                    del _nodes[key]
                for key in [k for k in _listings if k[0] == sid]:
                    del _listings[key]
