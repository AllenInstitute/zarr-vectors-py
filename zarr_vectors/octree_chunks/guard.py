"""Whether a level has octree chunks, and whether this package is converting it.

An octree-chunked level stores each leaf at its anchor cell, so its
``vertices`` cells are not the uniform cells every ordinary writer
addresses: a writer that appends to, edits or rebuilds such a level
writes rows the leaf and cell tables do not account for.  Only the
conversions in :mod:`zarr_vectors.octree_chunks.convert` write it (and
the shard repackers, which move its cells unchanged), and they mark
every group handle they write through with :func:`converting` for as
long as they do, so a writer can tell their writes from anyone else's::

    if is_octree_level(level_group) and not is_converting(level_group):
        raise ...

The mark lives on the handle (a :class:`~zarr_vectors.core.group.Group`
instance), not in the store: it says "this call stack is the
conversion", which no other process or handle should inherit.

:func:`refuse_octree_write` is that check with the error every writer
raises; :class:`~zarr_vectors.core.group.Group` runs it on each cell,
array and node write to a level, and the writers, editors, pyramid
builders and rechunkers run it (or :func:`refuse_octree_levels`) before
they start, so nothing is changed before the refusal.
:func:`refuse_octree_read` is the same for the readers and level APIs
that only understand uniform cells.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LEVEL_META_KEY = "zarr_vectors_level"   # the level group's metadata attribute
LEVEL_KEY = "octree_chunks"             # the octree block inside it
_FLAG = "_octree_chunks_converting"     # per-handle nesting count


def is_octree_level(level_group: Any) -> bool:
    """Whether the level carries an octree block -- octree chunks, or a
    conversion to or from them that has not finished."""
    try:
        level = level_group.attrs.get(LEVEL_META_KEY)
    except Exception:  # noqa: BLE001 - a node with no readable attrs is no octree level
        return False
    return isinstance(level, dict) and isinstance(level.get(LEVEL_KEY), dict)


@contextmanager
def converting(level_group: Any, *more: Any) -> Iterator[None]:
    """Mark ``level_group`` (and any further handles) as written by an
    octree conversion for the duration of the block.  Nests.

    The shard repackers (:func:`zarr_vectors.sharding.io.shard_store`,
    ``unshard_store``, ``set_octree``) use it too: they put every cell back
    unchanged, so an octree level stays exactly what it was."""
    groups = (level_group, *more)
    for g in groups:
        g.__dict__[_FLAG] = g.__dict__.get(_FLAG, 0) + 1
    try:
        yield
    finally:
        for g in groups:
            n = g.__dict__.get(_FLAG, 0) - 1
            if n > 0:
                g.__dict__[_FLAG] = n
            else:
                g.__dict__.pop(_FLAG, None)


def is_converting(level_group: Any) -> bool:
    """Whether this handle is inside a :func:`converting` block."""
    return bool(getattr(level_group, "__dict__", {}).get(_FLAG, 0))


def _level_block(level_group: Any) -> tuple[Any, dict[str, Any]]:
    level = level_group.attrs.get(LEVEL_META_KEY)
    return level.get("level", "?"), dict(level.get(LEVEL_KEY))


def _refusal(level_group: Any, what: str, instead: str) -> Exception:
    from zarr_vectors.octree_chunks.table import OctreeError

    n, block = _level_block(level_group)
    if "converting" in block:
        return OctreeError(
            f"level {n} is part way through {block['converting']} (interrupted?), so "
            f"{what} cannot run on it; run zarr_vectors.octree_chunks.{block['converting']}"
            f"(store, level={n}) again to finish it first"
        )
    return OctreeError(
        f"level {n} has octree chunks: its chunks are leaves stored at their anchor "
        f"cells, not uniform cells, and {what} {instead}. Convert it back first with "
        f"zarr_vectors.octree_chunks.to_uniform(store, level={n}) (and to_octree again after)"
    )


def refuse_octree_write(level_group: Any, what: str = "this write") -> None:
    """Raise :class:`~zarr_vectors.octree_chunks.table.OctreeError` when
    ``level_group`` is an octree level (or one mid-conversion) and the
    handle is not a conversion's own (:func:`converting`).

    ``what`` names the operation in the message.
    """
    if is_octree_level(level_group) and not is_converting(level_group):
        raise _refusal(level_group, what,
                       "addresses uniform cells, so it would corrupt the level")


def refuse_octree_read(level_group: Any, what: str = "this read") -> None:
    """:func:`refuse_octree_write` for a reader or level API that only
    understands uniform cells (and would take each leaf for one cell)."""
    if is_octree_level(level_group) and not is_converting(level_group):
        raise _refusal(level_group, what,
                       "reads uniform cells, so it would take each leaf for one cell "
                       "(read_points / read_graph / read_octree read octree levels)")


def refuse_unfinished_conversions(root: Any, what: str) -> None:
    """Raise for the first level of the store at ``root`` that is part way
    through ``to_octree`` / ``to_uniform``: the shard repackers would repack
    some of its arrays (the live ones) and not the staged ones swapped in
    later, leaving the level's arrays packed two ways."""
    from zarr_vectors.core.store import get_resolution_level, list_resolution_levels

    for m in list_resolution_levels(root):
        level_group = get_resolution_level(root, m)
        if is_octree_level(level_group) and "converting" in _level_block(level_group)[1]:
            raise _refusal(level_group, what, "")


def octree_levels(root: Any) -> list[int]:
    """The levels of the store at ``root`` that have octree chunks (or are
    mid-conversion)."""
    from zarr_vectors.core.store import get_resolution_level, list_resolution_levels

    return [m for m in list_resolution_levels(root)
            if is_octree_level(get_resolution_level(root, m))]


def refuse_octree_levels(root: Any, levels: Any = None, what: str = "this write",
                         *, read: bool = False) -> None:
    """:func:`refuse_octree_write` (or, with ``read``, :func:`refuse_octree_read`)
    for each of ``levels`` of the store at ``root`` -- every level when ``None``;
    levels that do not exist are skipped."""
    from zarr_vectors.core.store import get_resolution_level, list_resolution_levels

    have = list_resolution_levels(root)
    check = refuse_octree_read if read else refuse_octree_write
    for m in (have if levels is None else levels):
        if m is not None and int(m) in have:
            check(get_resolution_level(root, int(m)), what)


__all__ = [
    "converting", "is_converting", "is_octree_level", "octree_levels",
    "refuse_octree_levels", "refuse_octree_read", "refuse_octree_write",
    "refuse_unfinished_conversions",
]
