"""Fragment-level edits: add / edit / remove a fragment within a chunk.

A *fragment* is one row group inside a chunk's ``vertices/<chunk>``
array.  These operations are coarse — they touch every row of one
fragment at once — and are usually invoked by higher-level code rather
than end-user UIs.

Semantics:

- ``add_fragment`` always appends a new fragment to the named chunk
  (the new index is ``len(vertex_groups)`` for that chunk).  Returns
  the resulting :class:`FragmentRef`.
- ``edit_fragment(ref, new_vertices=...)`` replaces every row of the
  named fragment.  Under ``atomic=True`` a new fragment is appended
  with ``new_vertices`` and every referring object's manifest is
  rewritten to point at the new fragment under a fresh OID; under
  ``atomic=False`` the rows are replaced in place.
- ``remove_fragment(ref, atomic=True)`` is a **tombstone** — the
  fragment's row list is set to empty but the fragment index stays in
  the chunk's sidecar so downstream fragment indices don't shift.
  Vacuum's tombstone-GC pass (deferred) can later drop empty
  fragments and shift indices.  ``atomic=False`` is rejected for the
  same reason ``remove_vertex(atomic=False)`` is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.refs import FragmentRef
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession, PropagateTo


def add_fragment_in_session(
    session: EditSession,
    *,
    level: int,
    chunk: ChunkCoords,
    vertices: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
) -> FragmentRef:
    """Append a new fragment to ``chunk`` at ``level``.

    Returns the new :class:`FragmentRef`.  The caller is responsible
    for wiring object manifests if the fragment should be reachable
    via an OID — ``add_fragment`` itself only touches chunk-level data.
    """
    rows = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    if rows.size == 0:
        raise EditError("add_fragment: vertices must be non-empty")
    builder = session._builder(level, chunk)
    attrs_dict: dict[str, npt.NDArray] | None = None
    if attrs:
        attrs_dict = {
            k: np.atleast_1d(np.asarray(v)) for k, v in attrs.items()
        }
    new_idx = builder.append_fragment(rows, attrs=attrs_dict)
    session._mark_edit(level)
    return FragmentRef(
        level=level, chunk=tuple(int(c) for c in chunk),
        fragment=int(new_idx),
    )


def edit_fragment_in_session(
    session: EditSession,
    ref: FragmentRef,
    *,
    new_vertices: npt.ArrayLike,
    new_attrs: dict[str, npt.ArrayLike] | None,
    atomic: bool,
    propagate: PropagateTo,
) -> None:
    """Replace every row of ``ref``'s fragment.

    Atomic: append a new fragment with ``new_vertices`` and rewrite
    referring object manifests to point at it (under fresh OIDs).

    Minimal: overwrite the fragment's row group in place (object
    manifests stay because the fragment index is unchanged).
    """
    rows = np.atleast_2d(np.asarray(new_vertices, dtype=np.float64))
    if rows.size == 0:
        raise EditError(
            "edit_fragment: new_vertices is empty; use remove_fragment "
            "to clear a fragment instead."
        )

    builder = session._builder(ref.level, ref.chunk)
    if ref.fragment < 0 or ref.fragment >= len(builder.vertex_groups):
        raise EditError(
            f"edit_fragment: fragment {ref.fragment} out of range "
            f"for chunk {ref.chunk} "
            f"(has {len(builder.vertex_groups)} fragments)"
        )

    attrs_dict: dict[str, npt.NDArray] | None = None
    if new_attrs:
        attrs_dict = {
            k: np.atleast_1d(np.asarray(v)) for k, v in new_attrs.items()
        }

    if not atomic:
        # In-place replace: overwrite the array, refresh attribute
        # alignment.  Manifest references stay intact.
        builder.vertex_groups[ref.fragment] = rows.astype(
            builder.vertex_dtype, copy=True,
        )
        builder.vertices_dirty = True
        if attrs_dict:
            for name, val in attrs_dict.items():
                attr_list = builder.attr_groups.get(name)
                if attr_list is None:
                    continue
                attr_list[ref.fragment] = val.astype(
                    builder.attr_dtype.get(name, np.float32), copy=True,
                )
                builder.attrs_dirty[name] = True
        session._mark_edit(ref.level)
        return

    # Atomic: append a new fragment with the new rows, then redirect
    # every referring object's manifest entry from ref.fragment to
    # new_idx.
    new_idx = builder.append_fragment(rows, attrs=attrs_dict)
    affected = session._oids_referencing(ref.level, ref.chunk, ref.fragment)
    targets = session._select_targets(affected, propagate)
    chunk_t = tuple(ref.chunk)
    for oid in targets:
        manifest = session._get_manifest(ref.level, oid)
        new_manifest = [
            (chunk_t, int(new_idx)) if (
                tuple(cc) == chunk_t and fi == ref.fragment
            ) else (cc, fi)
            for (cc, fi) in manifest
        ]
        session._stage_manifest(ref.level, oid, new_manifest, atomic=True)
    session._mark_edit(ref.level)


def split_fragment_in_session(
    session: EditSession,
    ref: FragmentRef,
    *,
    row_partition: list[npt.NDArray] | npt.NDArray,
    atomic: bool,
    propagate: PropagateTo,
) -> list[FragmentRef]:
    """Physically slice one fragment into N new fragments.

    Delegates the chunk-level rewrite to :func:`partition_fragment_rows`
    in this module, then handles manifest updates for every referring
    OID (filtered by ``propagate``).
    """
    builder = session._builder(ref.level, ref.chunk)
    new_fragment_idxs = partition_fragment_rows(
        builder,
        fragment=ref.fragment,
        row_partition=row_partition,
        atomic=atomic,
        root=session.root,
    )
    new_refs = [
        FragmentRef(
            level=ref.level,
            chunk=tuple(int(c) for c in ref.chunk),
            fragment=int(idx),
        )
        for idx in new_fragment_idxs
    ]
    # If the partition collapsed to a single slice the worker returned
    # ``[ref.fragment]`` — nothing else to do.
    if len(new_fragment_idxs) <= 1:
        session._mark_edit(ref.level)
        return new_refs

    # Update referring object manifests.  For every OID listed by
    # ``propagate`` whose manifest references ``(chunk, ref.fragment)``,
    # replace that single entry with the list of new fragment indices
    # (preserves manifest order).
    affected = session._oids_referencing(ref.level, ref.chunk, ref.fragment)
    targets = session._select_targets(affected, propagate)
    chunk_t = tuple(int(c) for c in ref.chunk)
    for oid in targets:
        manifest = session._get_manifest(ref.level, oid)
        new_manifest: list[tuple[ChunkCoords, int]] = []
        for cc, fi in manifest:
            if tuple(cc) == chunk_t and fi == ref.fragment:
                for new_idx in new_fragment_idxs:
                    new_manifest.append((chunk_t, int(new_idx)))
            else:
                new_manifest.append((cc, fi))
        session._stage_manifest(ref.level, oid, new_manifest, atomic=atomic)
    session._mark_edit(ref.level)
    return new_refs


def remove_fragment_in_session(
    session: EditSession,
    ref: FragmentRef,
    *,
    atomic: bool,
    propagate: PropagateTo,
) -> None:
    """Tombstone a fragment: empty its row list, keep its index.

    Under ``atomic=True``: referring object manifests are rewritten
    under fresh OIDs that drop the ``(chunk, ref.fragment)`` block;
    the original OIDs continue to point at the (now empty) fragment
    so the read path stays consistent for old readers.

    Under ``atomic=False``: refused (consistent with
    ``remove_vertex(atomic=False)``).  Use the atomic path and run
    ``vacuum`` later to physically compact.
    """
    if not atomic:
        raise EditError(
            "remove_fragment(atomic=False) would require physical "
            "fragment-index compaction; not supported in this iteration. "
            "Use atomic=True (tombstone) and run vacuum() later."
        )
    builder = session._builder(ref.level, ref.chunk)
    if ref.fragment < 0 or ref.fragment >= len(builder.vertex_groups):
        raise EditError(
            f"remove_fragment: fragment {ref.fragment} out of range "
            f"for chunk {ref.chunk}"
        )

    # Empty the fragment's row group + its parallel link / attr groups.
    builder.vertex_groups[ref.fragment] = np.empty(
        (0, builder.vertex_ndim), dtype=builder.vertex_dtype,
    )
    builder.vertices_dirty = True
    for name, attr_list in builder.attr_groups.items():
        if ref.fragment < len(attr_list):
            shape = attr_list[ref.fragment].shape
            new_shape = (0,) + shape[1:]
            attr_list[ref.fragment] = np.empty(
                new_shape,
                dtype=builder.attr_dtype.get(name, np.float32),
            )
            builder.attrs_dirty[name] = True
    for delta, link_list in builder.link_groups.items():
        if ref.fragment < len(link_list):
            shape = link_list[ref.fragment].shape
            new_shape = (0,) + shape[1:]
            link_list[ref.fragment] = np.empty(new_shape, dtype=np.int64)
            builder.links_dirty[delta] = True

    # Drop the fragment from every referring manifest (under fresh OIDs).
    affected = session._oids_referencing(ref.level, ref.chunk, ref.fragment)
    targets = session._select_targets(affected, propagate)
    chunk_t = tuple(ref.chunk)
    for oid in targets:
        manifest = session._get_manifest(ref.level, oid)
        new_manifest = [
            (cc, fi) for (cc, fi) in manifest
            if not (tuple(cc) == chunk_t and fi == ref.fragment)
        ]
        session._stage_manifest(ref.level, oid, new_manifest, atomic=True)
    session._mark_edit(ref.level)


# ---------------------------------------------------------------------
# partition_fragment_rows: the chunk-rewrite worker
# ---------------------------------------------------------------------

def partition_fragment_rows(
    builder,
    *,
    fragment: int,
    row_partition: list[npt.NDArray] | npt.NDArray,
    atomic: bool,
    root,
) -> list[int]:
    """Physically slice one fragment of a chunk into N new fragments.

    Returns the list of new fragment indices in slice order (``[f1, f2,
    ..., fN]``).

    Atomic semantics:

    - ``atomic=True``: the original fragment is left intact; N new
      fragments are appended.  Callers decide which OIDs get re-pointed
      at the new fragments and which stay on the original.
    - ``atomic=False``: the original fragment's row group is emptied
      (tombstoned but kept at its index for stability); N new fragments
      are appended.  Callers overwrite referring manifests in place.

    Per-vertex attributes and per-fragment link groups are split in
    lock-step with the vertices so the 1:1 fragment alignment invariant
    holds.  Per-fragment intra-chunk link rows whose endpoints fall in
    a single slice are routed to that slice's new fragment; links
    straddling slices are dropped (caller already decided they go away).
    """
    if fragment < 0 or fragment >= len(builder.vertex_groups):
        raise EditError(
            f"partition_fragment_rows: fragment {fragment} out of range "
            f"for chunk {builder.chunk}"
        )
    rows_old = builder.vertex_groups[fragment]
    n_rows = int(rows_old.shape[0])

    slices = _normalise_row_partition(row_partition, n_rows)
    if len(slices) < 2:
        return [fragment]  # nothing to do — partition has one bucket

    # Eagerly load every attribute on this builder so slicing can
    # extend their parallel lists.
    from zarr_vectors.constants import VERTEX_ATTRIBUTES
    from zarr_vectors.core.store import get_resolution_level
    try:
        level_group = get_resolution_level(root, builder.level)
        if VERTEX_ATTRIBUTES in level_group:
            for name in level_group[VERTEX_ATTRIBUTES]:
                builder.require_attribute(root, name)
    except Exception:
        pass

    # Pre-compute per-row → slice mapping for link partitioning.
    row_to_slice: dict[int, int] = {}
    for si, rows in enumerate(slices):
        for r in rows:
            row_to_slice[int(r)] = si

    # Capture link rows of the original fragment (per delta) before
    # mutating anything — append_fragment extends link_groups with
    # empty arrays.
    original_link_rows: dict[int, npt.NDArray] = {}
    for delta, link_list in builder.link_groups.items():
        if fragment < len(link_list):
            original_link_rows[delta] = link_list[fragment].copy()

    new_fragments: list[int] = []
    for rows in slices:
        slice_rows = rows_old[rows]
        attrs_dict: dict[str, npt.NDArray] = {}
        for name, attr_list in builder.attr_groups.items():
            if fragment < len(attr_list):
                attrs_dict[name] = attr_list[fragment][rows]
        new_idx = builder.append_fragment(slice_rows, attrs=attrs_dict or None)
        new_fragments.append(new_idx)

    # Reroute intra-chunk link rows from the original fragment.
    frag_chunk_start = sum(
        int(g.shape[0]) for g in builder.vertex_groups[:fragment]
    )
    for delta, original_rows in original_link_rows.items():
        per_slice: dict[int, list[npt.NDArray]] = {
            i: [] for i in range(len(slices))
        }
        for row in original_rows:
            src, dst = int(row[0]), int(row[1])
            src_local = src - frag_chunk_start
            dst_local = dst - frag_chunk_start
            si_src = row_to_slice.get(src_local)
            si_dst = row_to_slice.get(dst_local)
            if si_src is None or si_dst is None:
                continue
            if si_src != si_dst:
                continue
            per_slice[si_src].append(row)
        for si, rows_list in per_slice.items():
            if not rows_list:
                continue
            new_frag_idx = new_fragments[si]
            stacked = np.stack(rows_list, axis=0).astype(np.int64, copy=False)
            new_frag_start = sum(
                int(g.shape[0])
                for g in builder.vertex_groups[:new_frag_idx]
            )
            row_remap: dict[int, int] = {}
            for new_local, original_local in enumerate(slices[si]):
                row_remap[
                    int(original_local) + frag_chunk_start
                ] = new_frag_start + new_local
            rebased = stacked.copy()
            for i, (a, b) in enumerate(stacked):
                rebased[i, 0] = row_remap.get(int(a), int(a))
                rebased[i, 1] = row_remap.get(int(b), int(b))
            builder.link_groups[delta][new_frag_idx] = rebased
            builder.links_dirty[delta] = True

    if not atomic:
        empty_rows = np.empty((0, builder.vertex_ndim), dtype=builder.vertex_dtype)
        builder.vertex_groups[fragment] = empty_rows
        for name, attr_list in builder.attr_groups.items():
            if fragment < len(attr_list):
                shape = attr_list[fragment].shape
                attr_list[fragment] = np.empty(
                    (0,) + shape[1:],
                    dtype=builder.attr_dtype.get(name, np.float32),
                )
                builder.attrs_dirty[name] = True
        for delta, link_list in builder.link_groups.items():
            if fragment < len(link_list):
                shape = link_list[fragment].shape
                link_list[fragment] = np.empty((0,) + shape[1:], dtype=np.int64)
                builder.links_dirty[delta] = True

    builder.vertices_dirty = True
    return new_fragments


def _normalise_row_partition(
    row_partition: list[npt.NDArray] | npt.NDArray,
    n_rows: int,
) -> list[npt.NDArray]:
    """Coerce ``row_partition`` to a list of disjoint row-index arrays.

    Accepts either the list-of-arrays form or a single 1-D label array.
    Validates that every row appears in exactly one slice and that no
    index is out of range.
    """
    if isinstance(row_partition, list):
        slices = [np.asarray(s, dtype=np.int64) for s in row_partition]
        seen = np.zeros(n_rows, dtype=bool)
        for k, s in enumerate(slices):
            if s.ndim != 1:
                raise EditError(
                    f"row_partition slice {k} must be 1-D, got shape {s.shape}"
                )
            if s.size == 0:
                continue
            if int(s.min()) < 0 or int(s.max()) >= n_rows:
                raise EditError(
                    f"row_partition slice {k} contains out-of-range row "
                    f"index (n_rows={n_rows})"
                )
            if seen[s].any():
                raise EditError(
                    f"row_partition slice {k} overlaps with an earlier slice"
                )
            seen[s] = True
        if not seen.all():
            missing = np.flatnonzero(~seen)
            raise EditError(
                f"row_partition does not cover every row; missing "
                f"{missing.tolist()}"
            )
        return slices

    labels = np.asarray(row_partition, dtype=np.int64).reshape(-1)
    if labels.shape[0] != n_rows:
        raise EditError(
            f"row_partition label array length {labels.shape[0]} != "
            f"fragment row count {n_rows}"
        )
    if labels.size == 0:
        return [np.empty(0, dtype=np.int64)]
    n_slices = int(labels.max()) + 1
    if int(labels.min()) < 0:
        raise EditError("row_partition label values must be >= 0")
    return [np.flatnonzero(labels == si) for si in range(n_slices)]
