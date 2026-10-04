"""Link groups that follow vertex fragments (``fragment_link_groups``).

A level stores each chunk's intra-chunk links as flat rows plus a
``link_fragments/`` index of row ranges, one per link group.  The format
lets groups be anything.  When every chunk of a level has exactly one group
per vertex fragment, in fragment order, and group ``k`` holds exactly the
links whose endpoints all lie in fragment ``k``, an object's manifest -- which
names vertex fragments -- also names its link groups, so a reader can fetch
one object's links by byte range instead of the whole cell.

A level advertises that with ``LevelMetadata.fragment_link_groups``.  The
claim follows the ``fragments_tile`` pattern:

* **Stamped after the writes, verified, not asserted** --
  :func:`stamp_fragment_link_groups` checks every chunk against the store,
  using bounds :func:`~zarr_vectors.core.arrays.write_chunk_links` recorded
  as it wrote so it need not re-read the rows it has just written.
* **Cleared by any later write** to ``vertex_fragments``, ``link_fragments``
  or the intra-chunk link array (:meth:`Group._clear_fragment_link_groups`),
  so a writer that does not know about it withdraws it rather than leaving
  it stale.
* **Checked by** ``validate_consistency``.

The link rows are the same whichever way they are grouped, and readers that
concatenate the groups see no difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_FRAGMENT_LINK_GROUPS,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    VERTEX_FRAGMENTS,
)
from zarr_vectors.core.group import Group
from zarr_vectors.core.paths import intra_offsets, links_group_path, links_path
from zarr_vectors.core.store import update_root_metadata
from zarr_vectors.encoding.fragments import ChunkFragmentIndex, encode_fragments

_LEVEL_META_KEY = "zarr_vectors_level"
FRAGMENT_LINK_GROUPS = "fragment_link_groups"


def fragment_of_rows(
    fragments: ChunkFragmentIndex, n_rows: int | None = None,
) -> npt.NDArray[np.int64]:
    """Fragment index of each vertex row of a chunk, ``-1`` where none."""
    spans: list[npt.NDArray[np.int64]] = []
    top = 0
    for f in range(fragments.num_fragments):
        if fragments.is_range(f):
            start, count = fragments.range(f)
            rows = np.arange(start, start + count, dtype=np.int64)
        else:
            rows = np.asarray(fragments.indices(f), dtype=np.int64)
        spans.append(rows)
        if rows.size:
            top = max(top, int(rows.max()) + 1)
    out = np.full(max(top, n_rows or 0), -1, dtype=np.int64)
    for f, rows in enumerate(spans):
        out[rows] = f
    return out


def split_links_by_fragment(
    rows: npt.NDArray[np.integer],
    fragment_of: npt.NDArray[np.int64],
    num_fragments: int,
) -> list[npt.NDArray[np.integer]] | None:
    """Intra-chunk link rows as one group per fragment, in fragment order
    (empty where a fragment has none), or ``None`` when some row's endpoints
    are not all in one fragment -- then no such grouping exists.

    The split is stable: rows keep their order within a group, so rows that
    are already in fragment order come back unchanged.
    """
    rows = np.asarray(rows)
    width = rows.shape[1] if rows.ndim == 2 else 1
    rows = rows.reshape(-1, width)
    empty = np.zeros((0, width), dtype=rows.dtype)
    if len(rows) == 0:
        return [empty] * num_fragments
    if rows.min() < 0 or rows.max() >= len(fragment_of):
        return None
    owner = fragment_of[rows]
    if (owner < 0).any() or (owner != owner[:, :1]).any():
        return None
    order = np.argsort(owner[:, 0], kind="stable")
    ordered = rows[order]
    bounds = np.searchsorted(owner[order, 0], np.arange(num_fragments + 1))
    return [ordered[bounds[f]:bounds[f + 1]] for f in range(num_fragments)]


def group_bounds(
    groups: Sequence[npt.NDArray[np.integer]],
) -> list[tuple[int, int] | None]:
    """Smallest and largest endpoint of each group (``None`` when empty):
    what :func:`stamp_fragment_link_groups` checks a group against."""
    out: list[tuple[int, int] | None] = []
    for g in groups:
        g = np.asarray(g)
        out.append(None if g.size == 0 else (int(g.min()), int(g.max())))
    return out


def intra_links_name(level_group: Group) -> tuple[str, int] | None:
    """The level's intra-chunk link array and its link width, if it has a
    links family at delta 0."""
    family = level_group.read_array_meta(links_group_path(0)) or {}
    if not family:
        return None
    link_width = int(family.get("link_width", 2))
    sid_ndim = int(family.get("sid_ndim", 3))
    return links_path(0, intra_offsets(sid_ndim, link_width)), link_width


def write_link_groups(
    level_group: Group,
    chunk_coords: Sequence[int],
    group_sizes: Sequence[int],
    *,
    bounds: Sequence[tuple[int, int] | None] | None = None,
) -> None:
    """Re-cut a chunk's intra-chunk link groups without rewriting its rows.

    For rows that are already in the order the new groups want: only
    ``link_fragments/<chunk>`` is rewritten, so a store that is copied or
    synced elsewhere changes by one small cell per chunk.  ``bounds``, when
    the caller knows them (:func:`group_bounds`), saves
    :func:`stamp_fragment_link_groups` re-reading the rows.
    """
    key = ".".join(str(int(c)) for c in chunk_coords)
    ranges: list[tuple[int, int]] = []
    start = 0
    for n in group_sizes:
        ranges.append((start, int(n)))
        start += int(n)
    level_group.write_bytes(LINK_FRAGMENTS, key, encode_fragments(ranges))
    if bounds is not None:
        level_group.note_link_group_bounds(key, list(bounds))


def _chunk_violation(
    level_group: Group,
    chunk_coords: tuple[int, ...],
    link_width: int,
    bounds: list[tuple[int, int] | None] | None,
) -> str | None:
    """Why one chunk breaks the rule, or ``None`` if it keeps it."""
    from zarr_vectors.core.arrays import (
        read_chunk_links,
        read_link_fragment_index,
        read_vertex_fragment_index,
    )

    vertex_fragments = read_vertex_fragment_index(level_group, chunk_coords)
    link_fragments = read_link_fragment_index(level_group, chunk_coords)
    if link_fragments.num_fragments != vertex_fragments.num_fragments:
        return (
            f"chunk {chunk_coords} has {link_fragments.num_fragments} link "
            f"groups for {vertex_fragments.num_fragments} vertex fragments"
        )
    usable = (
        bounds is not None
        and len(bounds) == vertex_fragments.num_fragments
        and all(
            b is None or vertex_fragments.is_range(f)
            for f, b in enumerate(bounds)
        )
    )
    if usable:
        for f, b in enumerate(bounds):  # type: ignore[arg-type]
            if b is None:
                continue
            start, count = vertex_fragments.range(f)
            if b[0] < start or b[1] >= start + count:
                return f"chunk {chunk_coords}: link group {f} leaves its fragment"
        return None
    groups = read_chunk_links(level_group, chunk_coords, link_width=link_width)
    if len(groups) != vertex_fragments.num_fragments:
        return f"chunk {chunk_coords}: link groups do not match its fragments"
    owner = fragment_of_rows(vertex_fragments)
    for f, g in enumerate(groups):
        g = np.asarray(g)
        if g.size == 0:
            continue
        if g.min() < 0 or g.max() >= len(owner) or (owner[g] != f).any():
            return f"chunk {chunk_coords}: link group {f} leaves its fragment"
    return None


def verify_fragment_link_groups(
    level_group: Group,
    bounds: dict[str, list[tuple[int, int] | None]] | None = None,
) -> str | None:
    """Why the level does not keep the ``fragment_link_groups`` rule, or
    ``None`` if every chunk does.  ``bounds`` are per-chunk hints recorded by
    the writer; chunks without one are checked against their stored rows."""
    from zarr_vectors.core.arrays import _maybe_batched_reads, list_chunk_keys

    found = intra_links_name(level_group)
    if found is None:
        return "the level has no links family"
    name, link_width = found
    if not level_group.array_exists(name):
        return f"the level has no {name} array"
    keys = list_chunk_keys(level_group, name)
    if not keys:
        return "the level has no intra-chunk links"
    key_strs = [".".join(str(c) for c in cc) for cc in keys]
    bounds = bounds or {}
    with _maybe_batched_reads(level_group, [
        (VERTEX_FRAGMENTS, key_strs),
        (LINK_FRAGMENTS, key_strs),
    ]):
        for cc, key in zip(keys, key_strs):
            try:
                reason = _chunk_violation(
                    level_group, tuple(cc), link_width, bounds.get(key),
                )
            except Exception as e:  # noqa: BLE001
                return f"chunk {cc} could not be checked ({e})"
            if reason is not None:
                return reason
    return None


def stamp_fragment_link_groups(level_group: Group, root: Any = None) -> bool:
    """Record that the level's link groups follow its vertex fragments, if
    they do.  Called by a writer once its chunk writes are done; returns what
    it stamped.

    Verified against the store, not asserted, exactly as
    :func:`~zarr_vectors.core.arrays.stamp_fragments_tile` is: a wrong claim
    hands readers incomplete objects.  Must run after the writes -- a write
    clears the claim, which is the mechanism that keeps edits honest.
    ``root``, when given, also gets the ``CAP_FRAGMENT_LINK_GROUPS`` token.
    """
    hints = level_group.take_link_group_bounds()
    if verify_fragment_link_groups(level_group, hints) is not None:
        return False
    level = level_group.attrs.get(_LEVEL_META_KEY)
    if not isinstance(level, dict):
        return False
    level_group.attrs.update({_LEVEL_META_KEY: {**level, FRAGMENT_LINK_GROUPS: True}})
    # The stamp is this handle's last word; a later write must be able to
    # clear it again.
    level_group._link_groups_claim_settled = False
    if root is not None:
        update_root_metadata(root, add_capabilities=[CAP_FRAGMENT_LINK_GROUPS])
    return True


def index_fragment_link_groups(
    store_path: Any,
    levels: Sequence[int] | None = None,
    *,
    dry_run: bool = False,
    verify: bool = False,
) -> list[dict[str, Any]]:
    """Give an existing store's levels link groups that follow their vertex
    fragments, and stamp the levels where every chunk allows it.

    Per chunk, the cheapest change that works: nothing when the groups
    already follow the fragments; a new ``link_fragments`` cell when the
    rows are already in fragment order (the common case -- writers emit one
    object's links together); otherwise the rows are regrouped, unless the
    level has intra-chunk link attributes, which are row-aligned and would
    have to move with them.  A level where some link joins two fragments
    cannot be stamped and is left as it is.

    Levels already stamped are skipped unless ``verify`` is set, which
    re-checks them (and repairs them where possible).  Every write is valid
    on its own, so an interrupted run leaves a correct, partly regrouped
    store; it just stays unstamped until a run completes.
    """
    from zarr_vectors.core.arrays import (
        list_chunk_keys,
        list_link_attribute_offsets,
        read_chunk_links,
        read_vertex_fragment_index,
        write_chunk_links,
    )
    from zarr_vectors.core.store import (
        get_resolution_level,
        list_resolution_levels,
        open_store,
    )

    root = open_store(str(store_path), mode="r" if dry_run else "r+")
    if levels is None:
        levels = list_resolution_levels(root)
    reports: list[dict[str, Any]] = []
    for level in levels:
        report: dict[str, Any] = {
            "level": int(level), "chunks": 0, "regrouped": 0, "reordered": 0,
        }
        reports.append(report)
        lg = get_resolution_level(root, int(level))
        found = intra_links_name(lg)
        if found is None or not lg.array_exists(found[0]):
            report["skipped"] = "no intra-chunk links"
            continue
        name, link_width = found
        stamped = bool((lg.attrs.get(_LEVEL_META_KEY) or {}).get(FRAGMENT_LINK_GROUPS))
        if stamped and not verify:
            report["skipped"] = "already stamped"
            continue
        meta = lg.read_array_meta(name) or {}
        dtype = np.dtype(meta.get("dtype", "int64"))
        offsets_segment = name.split("/")[-1]
        row_aligned_attributes = any(
            offsets_segment in list_link_attribute_offsets(lg, attr, 0)
            for attr in _link_attribute_names(lg)
        )
        reason = None
        for cc in list_chunk_keys(lg, name):
            groups = read_chunk_links(lg, cc, dtype=dtype, link_width=link_width)
            if not groups:
                continue
            report["chunks"] += 1
            fragments = read_vertex_fragment_index(lg, cc)
            rows = np.concatenate(
                [np.asarray(g).reshape(-1, link_width) for g in groups],
            )
            split = split_links_by_fragment(
                rows, fragment_of_rows(fragments), fragments.num_fragments,
            )
            key = ".".join(str(c) for c in cc)
            if split is None:
                reason = f"chunk {key}: a link joins two vertex fragments"
                break
            bounds = group_bounds(split)
            if len(groups) == len(split) and all(
                np.array_equal(np.asarray(a).reshape(-1, link_width), b)
                for a, b in zip(groups, split)
            ):
                lg.note_link_group_bounds(key, bounds)
                continue
            if np.array_equal(rows, np.concatenate(split)):
                report["regrouped"] += 1
                if not dry_run:
                    write_link_groups(
                        lg, cc, [len(g) for g in split], bounds=bounds,
                    )
                continue
            if row_aligned_attributes:
                reason = (
                    f"chunk {key}: links must be reordered, but the level "
                    "has row-aligned intra-chunk link attributes"
                )
                break
            report["reordered"] += 1
            if not dry_run:
                write_chunk_links(
                    lg, cc, split, dtype=dtype, delta=0, link_width=link_width,
                )
        if reason is not None:
            report["skipped"] = reason
            lg.take_link_group_bounds()
            continue
        if dry_run:
            report["would_stamp"] = True
            continue
        report["stamped"] = stamp_fragment_link_groups(lg, root)
        if not report["stamped"]:
            report["skipped"] = "verification failed after regrouping"
    return reports


def _link_attribute_names(level_group: Group) -> list[str]:
    if not level_group.array_exists(LINK_ATTRIBUTES):
        return []
    try:
        return sorted(level_group[LINK_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001
        return []
