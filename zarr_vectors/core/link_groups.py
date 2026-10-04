"""Link groups that follow vertex fragments (``fragment_link_groups``).

A level stores each chunk's intra-chunk links as flat rows plus a
``link_fragments/`` index of row ranges, one per link group.  The format
lets groups be anything.  When every chunk of a level has exactly one group
per vertex fragment, in fragment order, and group ``k`` holds exactly the
links whose endpoints all lie in fragment ``k``, an object's manifest -- which
names vertex fragments -- also names its link groups, so a reader can fetch
one object's links by byte range instead of the whole cell.

That needs the chunk's vertex fragments to be disjoint (a row in two
fragments would put a link in two groups) and its link groups to hold every
row of the cell exactly once (a row in no group is a link a range-reading
reader never sees).  Both are part of the rule.

A level advertises that with ``LevelMetadata.fragment_link_groups``.  The
claim follows the ``fragments_tile`` pattern:

* **Stamped after the writes, verified, not asserted** --
  :func:`stamp_fragment_link_groups` checks every chunk against the store.
  What :func:`~zarr_vectors.core.arrays.write_chunk_links` recorded as it
  wrote lets it skip re-reading rows it has just written, but only while
  the stored ``link_fragments`` cell still says exactly what was recorded.
* **Cleared by any later write** to ``vertex_fragments``, ``link_fragments``
  or the intra-chunk link array (:meth:`Group._clear_fragment_link_groups`),
  so a writer that does not know about it withdraws it rather than leaving
  it stale.
* **Checked by** ``validate_consistency``.

The link rows are the same whichever way they are grouped, and readers that
concatenate the groups see no difference.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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
from zarr_vectors.exceptions import StoreError

_LEVEL_META_KEY = "zarr_vectors_level"
FRAGMENT_LINK_GROUPS = "fragment_link_groups"

#: :func:`fragment_of_rows` marks a row that more than one fragment
#: claims with this value (a row in no fragment is ``-1``).
SHARED_ROW = -2

# Chunks whose rows are read back per prefetch, when a check has to.
_ROW_READ_WINDOW = 64


def _expand_rows(
    fragments: ChunkFragmentIndex,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Every row each fragment names, in fragment order, and the fragment
    each came from: ``(rows, owners)``, both 1-D and the same length."""
    n = fragments.num_fragments
    table = fragments.ranges()
    if table is not None:
        starts = np.asarray(table[:, 0], dtype=np.int64)
        counts = np.asarray(table[:, 1], dtype=np.int64)
        total = int(counts.sum()) if n else 0
        if total == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        owners = np.repeat(np.arange(n, dtype=np.int64), counts)
        firsts = np.cumsum(counts) - counts
        rows = np.arange(total, dtype=np.int64) - firsts[owners] + starts[owners]
        return rows, owners
    spans: list[npt.NDArray[np.int64]] = []
    for f in range(n):
        if fragments.is_range(f):
            start, count = fragments.range(f)
            spans.append(np.arange(start, start + count, dtype=np.int64))
        else:
            spans.append(np.asarray(fragments.indices(f), dtype=np.int64))
    if not spans:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    lengths = np.fromiter((len(s) for s in spans), dtype=np.int64, count=n)
    return (
        np.concatenate(spans),
        np.repeat(np.arange(n, dtype=np.int64), lengths),
    )


def fragment_of_rows(
    fragments: ChunkFragmentIndex, n_rows: int | None = None,
) -> npt.NDArray[np.int64]:
    """Fragment index of each vertex row of a chunk: ``-1`` where no
    fragment names the row, :data:`SHARED_ROW` where more than one does.

    Raises:
        ValueError: If a fragment names a negative row.
    """
    rows, owners = _expand_rows(fragments)
    if rows.size and int(rows.min()) < 0:
        raise ValueError("a vertex fragment names a negative row")
    top = int(rows.max()) + 1 if rows.size else 0
    out = np.full(max(top, n_rows or 0), -1, dtype=np.int64)
    out[rows] = owners
    if rows.size:
        hits = np.bincount(rows, minlength=out.size)
        out[hits > 1] = SHARED_ROW
    return out


def group_of_rows(
    link_fragments: ChunkFragmentIndex, n_rows: int,
) -> npt.NDArray[np.int64] | None:
    """Link group of each of a cell's ``n_rows`` rows, or ``None`` unless
    the groups hold every row exactly once (none twice, none left out, none
    past the end)."""
    rows, owners = _expand_rows(link_fragments)
    if rows.size != n_rows:
        return None
    if n_rows == 0:
        return owners
    if int(rows.min()) < 0 or int(rows.max()) >= n_rows:
        return None
    if not (np.bincount(rows, minlength=n_rows) == 1).all():
        return None
    out = np.empty(n_rows, dtype=np.int64)
    out[rows] = owners
    return out


def contiguous_group_sizes(
    link_fragments: ChunkFragmentIndex,
) -> npt.NDArray[np.int64] | None:
    """The group sizes when the groups are ranges that follow each other
    from row 0 in group order -- the layout
    :func:`~zarr_vectors.core.arrays.write_chunk_links` writes, where the
    groups' concatenation *is* the cell -- else ``None``."""
    table = link_fragments.ranges()
    if table is None:
        return None
    counts = np.asarray(table[:, 1], dtype=np.int64)
    starts = np.cumsum(counts) - counts
    if not np.array_equal(np.asarray(table[:, 0], dtype=np.int64), starts):
        return None
    return counts


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


def link_group_hint(
    groups: Sequence[npt.NDArray[np.integer]],
) -> npt.NDArray[np.int64]:
    """``(G, 3)`` int64: each group's row count and smallest and largest
    endpoint (``-1, -1`` when empty) -- what
    :meth:`Group.note_link_group_bounds` records for a cell just written."""
    out = np.full((len(groups), 3), -1, dtype=np.int64)
    for i, g in enumerate(groups):
        g = np.asarray(g)
        n = int(g.shape[0]) if g.ndim >= 1 else 0
        out[i, 0] = n
        if g.size:
            out[i, 1] = int(g.min())
            out[i, 2] = int(g.max())
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


def _intra_layout(
    level_group: Group, name: str, link_width: int,
) -> tuple[np.dtype, int]:
    """The intra array's dtype and stored row width."""
    meta = level_group.read_array_meta(name) or {}
    dtype = np.dtype(meta.get("dtype", "int64"))
    ncols = link_width + (1 if meta.get("has_perm") else 0)
    return dtype, max(1, ncols)


def write_link_groups(
    level_group: Group,
    chunk_coords: Sequence[int],
    group_sizes: Sequence[int],
    *,
    rows: npt.NDArray[np.integer] | None = None,
) -> None:
    """Re-cut a chunk's intra-chunk link groups without rewriting its rows.

    The groups become consecutive ranges of the stored rows, in order: only
    ``link_fragments/<chunk>`` is rewritten, so a store that is copied or
    synced elsewhere changes by one small cell per chunk.  ``rows``, when
    the caller has the cell's stored rows in hand, lets
    :func:`stamp_fragment_link_groups` check the new groups without reading
    them again.
    """
    key = ".".join(str(int(c)) for c in chunk_coords)
    sizes = [int(n) for n in group_sizes]
    ranges: list[tuple[int, int]] = []
    start = 0
    for n in sizes:
        ranges.append((start, n))
        start += n
    level_group.write_bytes(LINK_FRAGMENTS, key, encode_fragments(ranges))
    if rows is not None:
        rows = np.asarray(rows)
        if rows.shape[0] != start:
            raise ValueError(
                f"group sizes sum to {start} but the cell has {rows.shape[0]} rows"
            )
        cuts = np.cumsum(sizes)[:-1] if sizes else []
        level_group.note_link_group_bounds(
            key, link_group_hint(np.split(rows, cuts) if sizes else []),
        )


def _ranges_overlap(table: npt.NDArray[np.integer]) -> bool:
    """Whether any two non-empty ``(start, count)`` ranges share a row."""
    table = np.asarray(table, dtype=np.int64)
    live = table[table[:, 1] > 0]
    if len(live) < 2:
        return False
    live = live[np.argsort(live[:, 0], kind="stable")]
    return bool((live[:-1, 0] + live[:-1, 1] > live[1:, 0]).any())


def _check_with_hint(
    vertex_fragments: ChunkFragmentIndex,
    link_fragments: ChunkFragmentIndex,
    hint: npt.NDArray[np.int64] | None,
    chunk: str,
) -> str | None | bool:
    """Decide a chunk from what its writer recorded.  ``True`` when the
    hint cannot decide it (absent, stale, or a fragment is not a range) and
    the rows have to be read; else the violation, or ``None``."""
    if hint is None:
        return True
    sizes = contiguous_group_sizes(link_fragments)
    # The recorded groups are trusted only while the stored index still
    # says exactly them: anything else rewrote the cell since.
    if sizes is None or not np.array_equal(sizes, hint[:, 0]):
        return True
    table = vertex_fragments.ranges()
    if table is None:
        return True
    if _ranges_overlap(table):
        return f"chunk {chunk}: its vertex fragments overlap"
    live = hint[:, 0] > 0
    starts = np.asarray(table[:, 0], dtype=np.int64)
    ends = starts + np.asarray(table[:, 1], dtype=np.int64)
    bad = live & ((hint[:, 1] < starts) | (hint[:, 2] >= ends))
    if bad.any():
        return f"chunk {chunk}: link group {int(np.argmax(bad))} leaves its fragment"
    return None


def _check_rows(
    vertex_fragments: ChunkFragmentIndex,
    link_fragments: ChunkFragmentIndex,
    raw: bytes,
    dtype: np.dtype,
    ncols: int,
    chunk: str,
) -> str | None:
    """Decide a chunk from its stored rows."""
    row_bytes = dtype.itemsize * ncols
    n_rows, rem = divmod(len(raw), row_bytes)
    if rem:
        return f"chunk {chunk}: the link cell is not a whole number of rows"
    group_of = group_of_rows(link_fragments, n_rows)
    if group_of is None:
        return (
            f"chunk {chunk}: its link groups do not hold every link of the "
            f"cell exactly once"
        )
    owner = fragment_of_rows(vertex_fragments)
    if (owner == SHARED_ROW).any():
        return f"chunk {chunk}: its vertex fragments overlap"
    if n_rows == 0:
        return None
    full = np.frombuffer(raw, dtype=dtype).reshape(n_rows, ncols)
    if int(full.min()) < 0 or int(full.max()) >= owner.size:
        return f"chunk {chunk}: a link endpoint lies in no vertex fragment"
    bad = (owner[full] != group_of[:, None]).any(axis=1)
    if bad.any():
        return (
            f"chunk {chunk}: link group {int(group_of[np.argmax(bad)])} "
            f"leaves its fragment"
        )
    return None


def _windows(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def verify_fragment_link_groups(
    level_group: Group,
    bounds: dict[str, npt.NDArray[np.int64]] | None = None,
) -> str | None:
    """Why the level does not keep the ``fragment_link_groups`` rule, or
    ``None`` if every chunk does.

    ``bounds`` are per-chunk hints recorded by the writer
    (:meth:`Group.take_link_group_bounds`).  One is used only while the
    chunk's stored ``link_fragments`` cell is exactly the contiguous groups
    it records; every other chunk is checked against its stored rows.
    """
    from zarr_vectors.core.arrays import (
        _maybe_batched_reads,
        list_chunk_keys,
        read_link_fragment_index,
        read_vertex_fragment_index,
    )

    found = intra_links_name(level_group)
    if found is None:
        return "the level has no links family"
    name, link_width = found
    if not level_group.array_exists(name):
        return f"the level has no {name} array"
    keys = list_chunk_keys(level_group, name)
    if not keys:
        return "the level has no intra-chunk links"
    dtype, ncols = _intra_layout(level_group, name, link_width)
    key_strs = [".".join(str(c) for c in cc) for cc in keys]
    bounds = bounds or {}
    unread: list[tuple[str, ChunkFragmentIndex, ChunkFragmentIndex]] = []
    with _maybe_batched_reads(level_group, [
        (VERTEX_FRAGMENTS, key_strs),
        (LINK_FRAGMENTS, key_strs),
    ]):
        for cc, key in zip(keys, key_strs):
            try:
                vertex_fragments = read_vertex_fragment_index(level_group, cc)
                link_fragments = read_link_fragment_index(level_group, cc)
            except Exception as e:  # noqa: BLE001
                return f"chunk {key} could not be checked ({e})"
            if link_fragments.num_fragments != vertex_fragments.num_fragments:
                return (
                    f"chunk {key} has {link_fragments.num_fragments} link "
                    f"groups for {vertex_fragments.num_fragments} vertex "
                    f"fragments"
                )
            verdict = _check_with_hint(
                vertex_fragments, link_fragments, bounds.get(key), key,
            )
            if verdict is True:
                unread.append((key, vertex_fragments, link_fragments))
            elif verdict is not None:
                return verdict
    # The rows of chunks the hints could not decide, a window at a time:
    # one gather per window, and never the whole level in memory.
    for window in _windows(unread, _ROW_READ_WINDOW):
        with _maybe_batched_reads(level_group, [(name, [k for k, _, _ in window])]):
            for key, vertex_fragments, link_fragments in window:
                try:
                    raw = level_group.read_bytes(name, key)
                    reason = _check_rows(
                        vertex_fragments, link_fragments, raw, dtype, ncols, key,
                    )
                except Exception as e:  # noqa: BLE001
                    return f"chunk {key} could not be checked ({e})"
                if reason is not None:
                    return reason
    return None


def _refuse_pending_writes(level_group: Group, what: str) -> None:
    if level_group._pending_writes is not None:
        raise StoreError(
            f"{what} must run after the writes have reached the store, "
            f"not inside batched_writes() or open_write_session(): it "
            f"checks the store, and writes still queued would land after "
            f"the claim without withdrawing it"
        )


def stamp_fragment_link_groups(level_group: Group, root: Any = None) -> bool:
    """Record that the level's link groups follow its vertex fragments, if
    they do.  Called by a writer once its chunk writes are done; returns what
    it stamped.  A level that does not keep the rule is left -- or made --
    unstamped.

    Verified against the store, not asserted, exactly as
    :func:`~zarr_vectors.core.arrays.stamp_fragments_tile` is: a wrong claim
    hands readers incomplete objects.  Must run after the writes have
    reached the store -- a write clears the claim, which is the mechanism
    that keeps edits honest -- so it refuses to run with writes still
    queued.  ``root``, when given, also gets the
    ``CAP_FRAGMENT_LINK_GROUPS`` token.

    Raises:
        StoreError: Inside a :meth:`Group.batched_writes` block (which
            :func:`~zarr_vectors.core.arrays.open_write_session` opens).
    """
    _refuse_pending_writes(level_group, "stamp_fragment_link_groups")
    hints = level_group.take_link_group_bounds()
    if verify_fragment_link_groups(level_group, hints) is not None:
        withdraw_fragment_link_groups(level_group)
        return False
    if not isinstance(level_group.attrs.get(_LEVEL_META_KEY), dict):
        return False
    level_group._set_level_claims({FRAGMENT_LINK_GROUPS: True})
    # The stamp is this handle's last word; a later write must be able to
    # clear it again.
    level_group._link_groups_claim_settled = False
    if root is not None:
        update_root_metadata(root, add_capabilities=[CAP_FRAGMENT_LINK_GROUPS])
    return True


def withdraw_fragment_link_groups(level_group: Group) -> bool:
    """Remove the level's ``fragment_link_groups`` claim, whatever this
    handle has written.  Returns whether it was set."""
    return level_group._drop_level_claim(FRAGMENT_LINK_GROUPS)


def index_fragment_link_groups(
    store_path: Any,
    levels: Sequence[int] | None = None,
    *,
    dry_run: bool = False,
    verify: bool = False,
) -> list[dict[str, Any]]:
    """Give an existing store's levels link groups that follow their vertex
    fragments, and stamp the levels where every chunk allows it.

    Per chunk, the cheapest change that works:

    * nothing, when the groups already follow the fragments;
    * a new ``link_fragments`` cell, when the stored rows are already in
      fragment order and the old groups are consecutive ranges covering
      them (so every reader sees the same rows in the same order before
      and after -- the common case, since writers emit one object's links
      together);
    * otherwise the rows are rewritten in fragment order (keeping their
      order within a fragment), unless the level has intra-chunk link
      attributes, which are row-aligned and would have to move with them.

    A level is left as it is, unstamped, where some link joins two
    fragments, the vertex fragments overlap, or the old groups do not hold
    every row exactly once (then readers that concatenate the groups and
    readers that read the whole cell already disagree about which links
    exist, and no regrouping can say which is right).

    Levels already stamped are skipped unless ``verify`` is set, which
    re-checks them, repairs them where possible, and withdraws the stamp
    where not.  Every write is valid on its own, so an interrupted run
    leaves a correct, partly regrouped store; it just stays unstamped until
    a run completes.
    """
    from zarr_vectors.core.arrays import (
        _maybe_batched_reads,
        list_chunk_keys,
        list_link_attribute_offsets,
        read_link_fragment_index,
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
        stamped = bool(lg.level_claims().get(FRAGMENT_LINK_GROUPS))
        if stamped and not verify:
            report["skipped"] = "already stamped"
            continue
        dtype, ncols = _intra_layout(lg, name, link_width)
        row_bytes = dtype.itemsize * ncols
        offsets_segment = name.split("/")[-1]
        row_aligned_attributes = any(
            offsets_segment in list_link_attribute_offsets(lg, attr, 0)
            for attr in _link_attribute_names(lg)
        )
        keys = list_chunk_keys(lg, name)
        reason = None
        for window in _windows(keys, _ROW_READ_WINDOW):
            window_keys = [".".join(str(c) for c in cc) for cc in window]
            with _maybe_batched_reads(lg, [
                (name, window_keys),
                (LINK_FRAGMENTS, window_keys),
                (VERTEX_FRAGMENTS, window_keys),
            ]):
                for cc, key in zip(window, window_keys):
                    reason = _index_chunk(
                        lg, cc, key, name, dtype, ncols, row_bytes, link_width,
                        row_aligned_attributes, report, dry_run,
                        read_link_fragment_index, read_vertex_fragment_index,
                        write_chunk_links,
                    )
                    if reason is not None:
                        break
            if reason is not None:
                break
        if reason is not None:
            report["skipped"] = reason
            lg.take_link_group_bounds()
            if stamped:
                # Only reachable with ``verify``: the level claims a rule a
                # chunk breaks, so the claim goes.
                if dry_run:
                    report["would_withdraw"] = True
                else:
                    report["withdrawn"] = withdraw_fragment_link_groups(lg)
            continue
        if dry_run:
            report["would_stamp"] = True
            continue
        report["stamped"] = stamp_fragment_link_groups(lg, root)
        if not report["stamped"]:
            report["skipped"] = "verification failed after regrouping"
    return reports


def _index_chunk(
    lg: Group,
    cc: tuple[int, ...],
    key: str,
    name: str,
    dtype: np.dtype,
    ncols: int,
    row_bytes: int,
    link_width: int,
    row_aligned_attributes: bool,
    report: dict[str, Any],
    dry_run: bool,
    read_link_fragment_index: Any,
    read_vertex_fragment_index: Any,
    write_chunk_links: Any,
) -> str | None:
    """One chunk of :func:`index_fragment_link_groups`: why the level cannot
    be stamped, or ``None`` once the chunk is (or would be) regrouped."""
    raw = lg.read_bytes(name, key)
    if not raw:
        return None
    report["chunks"] += 1
    n_rows, rem = divmod(len(raw), row_bytes)
    if rem:
        return f"chunk {key}: the link cell is not a whole number of rows"
    # The rows as stored, in physical order -- not the concatenation of the
    # old groups, which need not be the same thing.
    full = np.frombuffer(raw, dtype=dtype).reshape(n_rows, ncols)
    try:
        vertex_fragments = read_vertex_fragment_index(lg, cc)
        link_fragments = read_link_fragment_index(lg, cc)
        owner = fragment_of_rows(vertex_fragments)
    except Exception as e:  # noqa: BLE001 - an index missing or malformed
        return f"chunk {key}: its fragment indexes could not be read ({e})"
    if (owner == SHARED_ROW).any():
        return f"chunk {key}: its vertex fragments overlap"
    group_of = group_of_rows(link_fragments, n_rows)
    if group_of is None:
        return (
            f"chunk {key}: its link groups do not hold every link of the "
            f"cell exactly once"
        )
    if int(full.min()) < 0 or int(full.max()) >= owner.size:
        return f"chunk {key}: a link endpoint lies in no vertex fragment"
    own = owner[full]
    first = own[:, 0]
    if (own < 0).any() or (own != first[:, None]).any():
        return f"chunk {key}: a link joins two vertex fragments"
    num_fragments = vertex_fragments.num_fragments
    sizes = contiguous_group_sizes(link_fragments)
    if (
        link_fragments.num_fragments == num_fragments
        and np.array_equal(group_of, first)
    ):
        # Already follows its fragments.  Recorded where the hint's form
        # (consecutive ranges) fits, so the stamp need not read it again.
        if sizes is not None:
            cuts = np.cumsum(sizes)[:-1]
            lg.note_link_group_bounds(
                key, link_group_hint(np.split(full, cuts)),
            )
        return None
    new_sizes = np.bincount(first, minlength=num_fragments)
    if sizes is not None and bool((np.diff(first) >= 0).all()):
        # Same rows in the same order for every reader: re-cut the index.
        report["regrouped"] += 1
        if not dry_run:
            write_link_groups(lg, cc, new_sizes.tolist(), rows=full)
        return None
    if row_aligned_attributes:
        return (
            f"chunk {key}: links must be reordered, but the level has "
            f"row-aligned intra-chunk link attributes"
        )
    # Rewrite the rows in fragment order, keeping the order a reader that
    # concatenates the old groups sees within each fragment.
    rows_in_group_order, _ = _expand_rows(link_fragments)
    logical = full[rows_in_group_order]
    split = split_links_by_fragment(logical, owner, num_fragments)
    if split is None:  # pragma: no cover - ruled out above
        return f"chunk {key}: a link joins two vertex fragments"
    report["reordered"] += 1
    if not dry_run:
        write_chunk_links(
            lg, cc, split, dtype=dtype, delta=0, link_width=link_width,
        )
    return None


def _link_attribute_names(level_group: Group) -> list[str]:
    if not level_group.array_exists(LINK_ATTRIBUTES):
        return []
    try:
        return sorted(level_group[LINK_ATTRIBUTES].children())
    except Exception:  # noqa: BLE001
        return []
