"""Additive pyramid refinement (``refinement: "add"`` levels).

In an ordinary pyramid every level holds its complete content: level 1 is
a coarser or sparser *replacement* for level 0, and a viewer zooming in
throws level 1's data away to draw level 0's.  A level marked
``refinement: "add"`` instead holds only what the next coarser level does
not: its complete content is its own data together with the complete
content of level ``L + 1``::

    chain(L) = [L] + (chain(L + 1) if level L is "add" else [])

so a pyramid built by object sparsity stores every object once (storage
about 1x rather than 1.14-1.33x), and a viewer zooming in keeps what it
has already drawn and fetches only the difference.

This changes what a level's data *means*, so a store that uses it lists
``CAP_ADDITIVE_LEVELS`` in the root's ``required_capabilities``, and a
reader that does not implement it must refuse the store
(:func:`zarr_vectors.core.store.check_required_capabilities`).

The public readers in :mod:`zarr_vectors.types` return a level's complete
content by default -- the union over :func:`level_chain` -- and take
``own_level_only=True`` for the stored data of one level.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zarr_vectors.constants import (
    CAP_ADDITIVE_LEVELS,
    REFINEMENT_ADD,
    REFINEMENT_REPLACE,
    REFINEMENT_VALUES,
)
from zarr_vectors.core.group import Group
from zarr_vectors.exceptions import MetadataError, StoreError

_LEVEL_META_KEY = "zarr_vectors_level"

#: Reader result keys that are counts, summed across a chain.
_COUNT_KEYS = frozenset({
    "vertex_count", "node_count", "edge_count", "face_count",
    "polyline_count", "line_count", "fragment_count",
})
#: Index tables, offset by the rows that went before them.
_INDEX_KEYS = {"edges": ("positions",), "faces": ("vertices",)}


def _root(root_or_path: Any) -> Group:
    if isinstance(root_or_path, Group):
        return root_or_path
    from zarr_vectors.core.store import open_store

    return open_store(str(root_or_path) if isinstance(root_or_path, Path) else root_or_path)


def level_refinement(level_group: Group) -> str:
    """A level's ``refinement``: ``"replace"`` (the default) or ``"add"``.

    Raises:
        MetadataError: On any other value.
    """
    try:
        block = level_group.attrs.get(_LEVEL_META_KEY) or {}
    except Exception:  # noqa: BLE001 - a node with no readable attrs
        block = {}
    value = block.get("refinement") if isinstance(block, dict) else None
    if value is None:
        return REFINEMENT_REPLACE
    if value not in REFINEMENT_VALUES:
        raise MetadataError(
            f"level refinement must be one of {list(REFINEMENT_VALUES)}, "
            f"got {value!r}"
        )
    return str(value)


def level_chain(root_or_path: Any, level: int) -> list[int]:
    """The levels whose own data together make up level ``level``'s
    complete content, finest first: ``[level]`` for an ordinary level, and
    ``[level] + level_chain(level + 1)`` for an additive one.

    Only a store whose root declares ``CAP_ADDITIVE_LEVELS`` (as every
    store with an additive level must) is looked into; any other store
    answers ``[level]`` from the root's attributes alone, so an ordinary
    read pays nothing for the check.

    Raises:
        StoreError: If ``level`` does not exist.
        MetadataError: If an additive level has no next coarser level.
    """
    from zarr_vectors.core.store import get_resolution_level

    root = _root(root_or_path)
    current = int(level)
    chain = [current]
    if not root_declares_additive(root):
        return chain
    group = get_resolution_level(root, current)
    while level_refinement(group) == REFINEMENT_ADD:
        nxt = current + 1
        try:
            group = get_resolution_level(root, nxt)
        except StoreError:
            raise MetadataError(
                f"level {current} is additive (refinement: \"add\") but the "
                f"store has no level {nxt} to complete it"
            ) from None
        chain.append(nxt)
        current = nxt
    return chain


def root_declares_additive(root: Group) -> bool:
    """Whether the root lists ``CAP_ADDITIVE_LEVELS`` -- read from the
    attributes the root handle already holds, so it costs no I/O."""
    try:
        block = root.attrs.get("zarr_vectors") or {}
    except Exception:  # noqa: BLE001 - a node with no readable attrs
        return False
    caps = set(block.get("required_capabilities") or ())
    caps |= set(block.get("format_capabilities") or ())
    return CAP_ADDITIVE_LEVELS in caps


def additive_levels(root_or_path: Any) -> list[int]:
    """Every level marked ``refinement: "add"``."""
    from zarr_vectors.core.store import get_resolution_level, list_resolution_levels

    root = _root(root_or_path)
    return [
        lv for lv in list_resolution_levels(root)
        if level_refinement(get_resolution_level(root, lv)) == REFINEMENT_ADD
    ]


def stored_object_ids(level_group: Group) -> np.ndarray | None:
    """Sorted ids of the objects a level stores (a non-empty manifest), or
    ``None`` for a level that tracks no objects."""
    from zarr_vectors.constants import OBJECT_INDEX
    from zarr_vectors.core.arrays import read_object_manifest_rows

    if not level_group.array_exists(OBJECT_INDEX):
        return None
    meta = level_group.read_array_meta(OBJECT_INDEX) or {}
    if not meta or int(meta.get("num_objects", 0) or 0) == 0:
        return np.zeros(0, dtype=np.int64)
    ids, manifests = read_object_manifest_rows(level_group)
    keep = np.fromiter((bool(m) for m in manifests), dtype=bool, count=len(manifests))
    return np.sort(np.asarray(ids, dtype=np.int64)[keep])


def is_additive(root_or_path: Any) -> bool:
    """Whether any level of the store is additive."""
    return bool(additive_levels(root_or_path))


def refuse_additive(root_or_path: Any, action: str) -> None:
    """Raise for an operation that rewrites levels from one another and
    cannot keep an additive pyramid's split between them.

    A store that cannot be opened has no additive levels to protect; the
    operation itself reports why it cannot proceed.

    Raises:
        StoreError: If the store has additive levels.
    """
    try:
        root = _root(root_or_path)
    except StoreError:
        return
    levels = additive_levels(root)
    if levels:
        raise StoreError(
            f"{action} is not supported on a store with additive levels "
            f"{levels}: those levels hold only what the next coarser level "
            f"does not, so rebuilding one from another would lose or "
            f"duplicate data. Rebuild the pyramid from a store whose levels "
            f"are complete (refinement \"replace\") instead."
        )


def declare_required_capability(root: Group, capability: str) -> None:
    """Add ``capability`` to the root's ``format_capabilities`` and
    ``required_capabilities``."""
    from zarr_vectors.core.store import read_root_metadata, update_root_metadata

    required = list(read_root_metadata(root).required_capabilities)
    if capability not in required:
        required.append(capability)
    update_root_metadata(
        root, add_capabilities=[capability], required_capabilities=required,
    )


def declare_additive_levels(root: Group, levels: Sequence[int]) -> None:
    """Mark ``levels`` ``refinement: "add"``.

    The root's ``required_capabilities`` gains ``CAP_ADDITIVE_LEVELS``
    *first*, so a store never has an additive level that an older reader
    of the list could open as complete.  Marking a level before removing
    the data its coarser level duplicates is also the safe order for a
    writer: an interrupted conversion then reads back with some objects
    twice rather than with some missing.

    Raises:
        MetadataError: If a level is the coarsest, or has no next level.
    """
    from zarr_vectors.core.store import (
        get_resolution_level,
        list_resolution_levels,
        update_level_metadata,
    )

    present = set(list_resolution_levels(root))
    for lv in levels:
        if lv not in present:
            raise MetadataError(f"level {lv} does not exist")
        if lv + 1 not in present:
            raise MetadataError(
                f"level {lv} cannot be additive: there is no level {lv + 1} "
                f"to complete it (the coarsest level is always complete)"
            )
    if not levels:
        return
    declare_required_capability(root, CAP_ADDITIVE_LEVELS)
    for lv in levels:
        update_level_metadata(get_resolution_level(root, lv), refinement=REFINEMENT_ADD)


# ------------------------------------------------------------- reading


def level_cell_shapes(root: Group, levels: Sequence[int]) -> dict[int, tuple[float, ...]]:
    """Each level's physical chunk (cell) shape."""
    from zarr_vectors.core.metadata import get_level_chunk_shape
    from zarr_vectors.core.store import read_level_metadata, read_root_metadata

    root_meta = read_root_metadata(root)
    return {
        lv: tuple(float(c) for c in get_level_chunk_shape(
            root_meta, read_level_metadata(root, lv),
        ))
        for lv in levels
    }


def map_cells(
    cells: Sequence[Sequence[int]],
    source_shape: Sequence[float],
    target_shape: Sequence[float],
) -> list[tuple[int, ...]]:
    """Cells of a grid of ``target_shape`` that overlap the given cells
    of a grid of ``source_shape`` (both anchored at the origin).

    Leading coordinates beyond the spatial rank -- an attribute bin, on a
    level chunked by attribute -- are carried unchanged.  A coarser target
    therefore returns the cells *containing* the requested ones, which
    cover more than was asked for: a coarser level's data is read for
    whole cells.
    """
    src = np.asarray(source_shape, dtype=np.float64)
    dst = np.asarray(target_shape, dtype=np.float64)
    rank = src.size
    if np.array_equal(src, dst):
        return sorted({tuple(int(c) for c in cell) for cell in cells})
    out: set[tuple[int, ...]] = set()
    for cell in cells:
        coords = [int(c) for c in cell]
        lead, spatial = coords[:-rank], np.asarray(coords[-rank:], dtype=np.float64)
        lo = spatial * src
        hi = (spatial + 1.0) * src
        first = np.floor(lo / dst + 1e-9).astype(np.int64)
        last = np.ceil(hi / dst - 1e-9).astype(np.int64) - 1
        last = np.maximum(last, first)
        for combo in itertools.product(*(
            range(int(a), int(b) + 1) for a, b in zip(first, last)
        )):
            out.add((*lead, *combo))
    return sorted(out)


def read_level_chain(
    reader: Callable[..., Any],
    root: Group,
    chain: Sequence[int],
    *,
    chunks: Sequence[Sequence[int]] | None = None,
    **kwargs: Any,
) -> Any:
    """Read each level of ``chain`` with ``reader`` (its own data only) and
    concatenate the results into level ``chain[0]``'s complete content.

    ``chunks``, when given, are cells of ``chain[0]``'s grid; each coarser
    level reads the cells of its own grid that overlap them
    (:func:`map_cells`).
    """
    shapes = level_cell_shapes(root, chain) if chunks is not None else {}
    parts = []
    for lv in chain:
        part_kwargs = dict(kwargs)
        if chunks is not None:
            part_kwargs["chunks"] = map_cells(chunks, shapes[chain[0]], shapes[lv])
        parts.append(reader(root, level=lv, own_level_only=True, **part_kwargs))
    return merge_level_reads(parts)


def merge_level_reads(parts: Sequence[Any]) -> Any:
    """Concatenate per-level reader results, finest level first.

    Arrays and lists are concatenated; ``edges`` and ``faces`` are offset
    by the vertices that precede them; counts are summed; per-vertex
    attributes are kept where every level that returned vertices has them
    (a level without one cannot be given a value for it).  ``None`` parts
    (a reader that found nothing) are skipped, and all-``None`` is
    ``None``.
    """
    present = [p for p in parts if p is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    first = present[0]
    out: dict[str, Any] = {}
    for key in first:
        values = [p.get(key) for p in present]
        if key in _COUNT_KEYS:
            out[key] = int(sum(int(v or 0) for v in values))
        elif key in _INDEX_KEYS:
            out[key] = _concat_index(present, key, _INDEX_KEYS[key])
        elif isinstance(first[key], Mapping):
            out[key] = _merge_columns(present, key)
        elif isinstance(first[key], np.ndarray):
            arrays = [np.asarray(v) for v in values if v is not None]
            out[key] = np.concatenate(arrays, axis=0) if arrays else first[key]
        elif isinstance(first[key], list):
            merged: list[Any] = []
            for v in values:
                merged.extend(v or [])
            out[key] = merged
        else:
            out[key] = first[key]
    return out


def _rows(part: Mapping[str, Any]) -> int:
    for key in ("positions", "vertices", "endpoints"):
        if key in part and part[key] is not None:
            return len(part[key])
    if "polylines" in part:
        return int(part.get("vertex_count") or 0)
    return int(part.get("vertex_count") or 0)


def _concat_index(
    parts: Sequence[Mapping[str, Any]], key: str, bases: Sequence[str],
) -> np.ndarray:
    tables = []
    offset = 0
    for part in parts:
        table = np.asarray(part.get(key))
        if table.size:
            tables.append(table + offset)
        base = next((part[b] for b in bases if b in part), None)
        offset += len(base) if base is not None else 0
    if not tables:
        return np.asarray(parts[0].get(key))
    return np.concatenate(tables, axis=0)


def _merge_columns(parts: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    with_rows = [p for p in parts if _rows(p) > 0]
    if not with_rows:
        return dict(parts[0].get(key) or {})
    names = set(with_rows[0].get(key) or {})
    for p in with_rows[1:]:
        names &= set(p.get(key) or {})
    out: dict[str, Any] = {}
    for name in sorted(names):
        columns = [
            np.asarray(p[key][name]) for p in parts
            if name in (p.get(key) or {}) and _rows(p) > 0
        ]
        out[name] = np.concatenate(columns, axis=0) if columns else None
    return out


__all__ = [
    "REFINEMENT_ADD",
    "REFINEMENT_REPLACE",
    "additive_levels",
    "declare_additive_levels",
    "declare_required_capability",
    "is_additive",
    "level_cell_shapes",
    "level_chain",
    "level_refinement",
    "map_cells",
    "merge_level_reads",
    "read_level_chain",
    "refuse_additive",
    "root_declares_additive",
    "stored_object_ids",
]
