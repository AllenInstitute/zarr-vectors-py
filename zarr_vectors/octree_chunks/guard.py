"""Whether a level has octree chunks, whether a conversion is writing it, and the refusals.

Only the conversions (and the shard repackers, which move cells unchanged)
may write an octree level; they mark the group handles they write through
with :func:`converting`.  The mark is per handle, not in the store, so no
other process or handle inherits it.  See ``docs/spec/chunking/octree_chunks.md``
(Compatibility, Converting safely).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LEVEL_META_KEY = "zarr_vectors_level"   # the level group's metadata attribute
LEVEL_KEY = "octree_chunks"             # the octree block inside it
_FLAG = "_octree_chunks_converting"     # per-handle nesting count


def octree_meta(level_group: Any) -> dict[str, Any] | None:
    """The level's octree block (``{"converting": ...}`` mid-swap), ``None`` if uniform."""
    level = level_group.attrs.get(LEVEL_META_KEY)
    block = level.get(LEVEL_KEY) if isinstance(level, dict) else None
    return dict(block) if isinstance(block, dict) else None


def is_octree_level(level_group: Any) -> bool:
    """Whether the level has octree chunks, or a conversion to or from them not finished."""
    try:
        return octree_meta(level_group) is not None
    except Exception:  # noqa: BLE001 - a node with no readable attrs is no octree level
        return False


def unfinished(block: dict[str, Any]) -> str:
    """The advice for a level whose octree ``block`` is mid-conversion."""
    c = block["converting"]
    return (f"part way through {c} (interrupted?); "
            f"run zarr_vectors.octree_chunks.{c} on it again to finish")


@contextmanager
def converting(level_group: Any, *more: Any) -> Iterator[None]:
    """Mark ``level_group`` (and ``more``) as written by an octree conversion
    or shard repacker for the duration of the block.  Nests."""
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


def _refusal(level_group: Any, what: str, instead: str) -> Exception:
    from zarr_vectors.octree_chunks.table import OctreeError

    n, block = level_group.attrs.get(LEVEL_META_KEY).get("level", "?"), octree_meta(level_group)
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
    """Raise :class:`~zarr_vectors.octree_chunks.table.OctreeError` for an octree
    level (or one mid-conversion) unless the handle is :func:`converting`.
    ``what`` names the operation in the message."""
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
    """Raise for the first level of the store at ``root`` part way through
    ``to_octree`` / ``to_uniform`` (a shard repacker would miss its staged arrays)."""
    from zarr_vectors.core.store import get_resolution_level, list_resolution_levels

    for m in list_resolution_levels(root):
        level_group = get_resolution_level(root, m)
        if is_octree_level(level_group) and "converting" in octree_meta(level_group):
            raise _refusal(level_group, what, "")


def octree_levels(root: Any) -> list[int]:
    """The levels of the store at ``root`` that have octree chunks (or are mid-conversion)."""
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
    "converting", "is_converting", "is_octree_level", "octree_levels", "octree_meta",
    "refuse_octree_levels", "refuse_octree_read", "refuse_octree_write",
    "refuse_unfinished_conversions", "unfinished",
]
