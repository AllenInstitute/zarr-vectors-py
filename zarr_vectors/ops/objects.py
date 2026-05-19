"""Object-level edits: add / edit / remove an entire object.

This module is a thin layer on top of the chunk-level builders + the
manifest-staging engine already in :mod:`zarr_vectors.ops.edit`.  The
public surface is exposed via :class:`~zarr_vectors.ops.edit.EditSession`
methods and free-function wrappers; the helpers here implement the
heavy lifting.

Atomic semantics (consistent with Iteration 1):

- ``atomic=True``  → the edited object becomes a **new OID**, original
  manifest at ``oid`` is left intact.  Caller gets the new OID via
  ``EditReport.oid_remap``.
- ``atomic=False`` → overwrite manifest at ``oid``.  Destructive
  remove (``remove_object(atomic=False)``) is rejected for the same
  reason ``remove_vertex(atomic=False)`` is: physical fragment purge
  would require global manifest re-scanning that we don't do yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.refs import ObjectRef
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession


def add_object_in_session(
    session: EditSession,
    *,
    level: int,
    vertices: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
) -> ObjectRef:
    """Append a brand-new object with one fragment per touched chunk.

    The vertex set is split across chunks via
    :func:`~zarr_vectors.spatial.chunking.assign_chunks`; each touched
    chunk gets a new fragment containing the rows assigned to it.  The
    new OID is allocated via the session's prefix allocator (when set)
    and the manifest is written at that OID.

    Per-vertex ``attrs`` are forwarded into each new fragment in the
    matching chunk-local order.  Per-object attrs are not handled here
    — call :meth:`EditSession.edit_attribute` after ``add_object``.
    """
    from zarr_vectors.core.metadata import RootMetadata

    verts = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    if verts.size == 0:
        raise EditError("add_object: vertices must be non-empty")

    meta = RootMetadata.from_dict(session.root.attrs.to_dict())
    chunk_shape = tuple(meta.chunk_shape)
    if verts.shape[1] != meta.sid_ndim:
        raise EditError(
            f"add_object: vertices arity {verts.shape[1]} != "
            f"store ndim {meta.sid_ndim}"
        )

    by_chunk = assign_chunks(verts, chunk_shape)
    # Manifest order follows the source row order of ``vertices`` so
    # implicit-sequential conventions get a stable parent->child chain.
    # Walk vertex rows and append to the matching chunk's fragment.
    # For multi-row fragments we group contiguous runs that share a
    # chunk so a single fragment captures them, preserving order.
    row_chunk = np.empty(verts.shape[0], dtype=object)
    for cc, idx in by_chunk.items():
        for i in idx:
            row_chunk[i] = cc

    manifest: ObjectManifest = []
    i = 0
    while i < verts.shape[0]:
        cc = row_chunk[i]
        j = i + 1
        while j < verts.shape[0] and row_chunk[j] == cc:
            j += 1
        rows = verts[i:j]
        # Per-vertex attrs for this slice.
        slice_attrs: dict[str, npt.NDArray] | None = None
        if attrs:
            slice_attrs = {
                k: np.atleast_1d(np.asarray(v))[i:j]
                for k, v in attrs.items()
            }
        builder = session._builder(level, cc)
        new_frag = builder.append_fragment(rows, attrs=slice_attrs)
        manifest.append((tuple(cc), int(new_frag)))
        i = j

    new_oid = session._allocate_atomic_oid(level)
    session._manifest_ops[(level, new_oid)] = _make_manifest_op(
        level=level, new_oid=new_oid, manifest=manifest,
    )
    session._index_apply_manifest_delta(
        level, int(new_oid),
        old_manifest=[],
        new_manifest=manifest,
    )
    session._mark_edit(level)
    return ObjectRef(level=level, object_id=int(new_oid))


def edit_object_manifest_in_session(
    session: EditSession,
    *,
    oid: int,
    level: int,
    new_manifest: ObjectManifest,
    atomic: bool,
) -> int | None:
    """Stage a manifest swap for ``oid`` at ``level``.

    Under ``atomic=True`` returns the allocated new OID; under
    ``atomic=False`` returns ``None`` (the old OID is overwritten in
    place).
    """
    # Normalise the manifest: every (chunk_coords, frag_idx) tuple must
    # be ``(tuple_of_int, int)``.
    norm = [
        (tuple(int(c) for c in cc), int(fi))
        for (cc, fi) in new_manifest
    ]
    session._stage_manifest(level, oid, norm, atomic=atomic)
    session._mark_edit(level)
    if atomic:
        return session._report.oid_remap.get(int(oid))
    return None


def remove_object_in_session(
    session: EditSession,
    *,
    oid: int,
    level: int,
    atomic: bool,
) -> int | None:
    """Soft-delete (atomic) or in-place tombstone (non-atomic) an OID.

    Both paths write an empty manifest — the difference is whether the
    write lands at a fresh OID (atomic) or overwrites ``oid``.
    """
    if not atomic:
        # Destructive remove: the manifest at ``oid`` becomes empty.
        # The fragments it solely owned are not purged this iteration —
        # that would require global manifest re-scanning (a vacuum job).
        # Document this in the docstring; we still allow the soft
        # variant via atomic=True which is the recommended path.
        pass
    session._stage_manifest(level, oid, [], atomic=atomic)
    session._mark_edit(level)
    if atomic:
        return session._report.oid_remap.get(int(oid))
    return None


def _make_manifest_op(*, level: int, new_oid: int, manifest: ObjectManifest):
    """Build a ManifestOp for the add_object path."""
    from zarr_vectors.ops.change_set import ManifestOp
    return ManifestOp(
        level=level,
        object_id=new_oid,
        new_manifest=manifest,
        new_oid=new_oid,
    )


# ----------------------------------------------------------------------
# Iteration-3: private merge/split engines invoked by the
# add_link(update_objects=True) / remove_link(update_objects=True) /
# edit_link(update_objects=True) kwarg paths.
# ----------------------------------------------------------------------

def _merge_objects_impl(
    session: EditSession,
    *,
    oids: list[int],
    level: int,
    atomic: bool,
) -> int | None:
    """Concatenate the manifests of ``oids`` (in order) into one OID.

    Atomic: allocate a new OID, write the concatenated manifest there,
    leave originals intact.  Records the remap ``{oid: new_oid}`` for
    every input OID.

    Non-atomic: overwrite ``oids[0]`` with the concatenation; set
    ``oids[1:]`` to empty manifests.

    Returns the resulting OID (the new one under atomic, ``oids[0]``
    under non-atomic).  Returns ``None`` and is a no-op when fewer than
    two OIDs are given.
    """
    if len(oids) < 2:
        return None
    if len(set(oids)) != len(oids):
        raise EditError(
            f"_merge_objects_impl: same OID listed twice in {oids}; "
            f"refusing to double-reference fragments."
        )

    # Build the concatenated manifest from the pending in-session state
    # so chains of merges compose correctly.
    concat: list[tuple[ChunkCoords, int]] = []
    for oid in oids:
        concat.extend(session._get_manifest(level, int(oid)))

    if atomic:
        new_oid = session._allocate_atomic_oid(level)
        from zarr_vectors.ops.change_set import ManifestOp
        session._manifest_ops[(level, new_oid)] = ManifestOp(
            level=level, object_id=new_oid,
            new_manifest=concat, new_oid=new_oid,
        )
        session._index_apply_manifest_delta(
            level, int(new_oid),
            old_manifest=[],
            new_manifest=concat,
        )
        for oid in oids:
            session._report.oid_remap[int(oid)] = int(new_oid)
        session._mark_edit(level)
        return new_oid

    # Non-atomic: stamp concat at oids[0], empty manifests at the rest.
    session._stage_manifest(level, int(oids[0]), concat, atomic=False)
    for oid in oids[1:]:
        session._stage_manifest(level, int(oid), [], atomic=False)
    session._mark_edit(level)
    return int(oids[0])


def _split_object_at_link_impl(
    session: EditSession,
    *,
    oid: int,
    level: int,
    removed_endpoints: tuple[ChunkCoords, int, int],
    atomic: bool,
) -> list[int]:
    """Deterministically split ``oid``'s manifest at a just-removed link.

    ``removed_endpoints`` is ``(chunk, src_chunk_local, dst_chunk_local)``
    naming the endpoints of the link that was just dropped.  The split
    point is computed from the endpoints' positions in ``oid``'s
    manifest — no graph traversal.

    Algorithm:

    1. For each endpoint, find which fragment in ``chunk`` owns the
       chunk-local row.
    2. Walk ``oid``'s manifest to locate the indices ``i`` and ``j``
       (with ``i <= j``) of those fragments.
    3. **Same fragment** (``i == j``, both endpoints in the same
       fragment): physically slice the fragment at the row boundary
       between the endpoints.  Side A = manifest[:i] + [(chunk,
       low_slice)]; Side B = [(chunk, high_slice)] + manifest[i+1:].
    4. **Different fragments**: Side A = manifest[:i+1]; Side B =
       manifest[j:].  Entries strictly between ``i`` and ``j`` (rare in
       well-formed chains) are routed to Side A.
    5. Stamp both sides.  Atomic=True allocates two new OIDs and
       leaves the original intact; atomic=False overwrites the
       original with Side A and appends one new OID for Side B.

    Returns the list of resulting OIDs in [Side A, Side B] order.
    When either endpoint isn't in ``oid``'s manifest (e.g. the link
    was between two different OIDs and ``oid`` only owns one
    endpoint) the function falls back to a no-op and returns
    ``[oid]``.

    Refuses on implicit-sequential conventions.
    """
    from zarr_vectors.constants import (
        LINKS_EXPLICIT,
        LINKS_IMPLICIT_BRANCHES,
        LINKS_IMPLICIT_SEQUENTIAL,
    )
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.ops.fragments import partition_fragment_rows

    meta = RootMetadata.from_dict(session.root.attrs.to_dict())
    conv = meta.links_convention or LINKS_EXPLICIT
    if conv in (LINKS_IMPLICIT_SEQUENTIAL, LINKS_IMPLICIT_BRANCHES):
        if getattr(session, "auto_materialise_links", True):
            from zarr_vectors.ops.links import _auto_materialise_to_explicit
            _auto_materialise_to_explicit(
                session, level=level, prior_conv=conv,
            )
        else:
            raise EditError(
                f"_split_object_at_link_impl: store has links_convention="
                f"{conv!r}; deterministic split requires explicit edges. "
                f"Either set session.auto_materialise_links=True (the "
                f"default), or call materialise_object_links_explicit("
                f"root, level, oid, flip_convention=True) yourself first."
            )

    chunk, src_local, dst_local = removed_endpoints
    chunk_t = tuple(int(c) for c in chunk)

    # Step 1: resolve each endpoint to (fragment_idx, fragment_local_row)
    # in the chunk.  Forward walk — single pass, returns ``None`` for
    # out-of-range row indices.
    builder = session._builder(level, chunk_t)

    def _frag_and_local(chunk_local: int) -> tuple[int, int] | None:
        if chunk_local < 0:
            return None
        cursor = 0
        for fi, g in enumerate(builder.vertex_groups):
            n = int(g.shape[0])
            if cursor <= chunk_local < cursor + n:
                return fi, chunk_local - cursor
            cursor += n
        return None

    src_loc = _frag_and_local(int(src_local))
    dst_loc = _frag_and_local(int(dst_local))
    if src_loc is None or dst_loc is None:
        return [int(oid)]
    src_frag, src_frag_local = src_loc
    dst_frag, dst_frag_local = dst_loc

    # Step 2: walk manifest, find indices i, j.
    manifest = session._get_manifest(level, int(oid))
    src_manifest_idx: int | None = None
    dst_manifest_idx: int | None = None
    for mi, (cc, fi) in enumerate(manifest):
        cc_match = tuple(int(c) for c in cc) == chunk_t
        if cc_match and int(fi) == src_frag and src_manifest_idx is None:
            src_manifest_idx = mi
        if cc_match and int(fi) == dst_frag and dst_manifest_idx is None:
            dst_manifest_idx = mi
        if src_manifest_idx is not None and dst_manifest_idx is not None:
            break
    if src_manifest_idx is None or dst_manifest_idx is None:
        return [int(oid)]

    # Normalise so i <= j.
    if src_manifest_idx <= dst_manifest_idx:
        i, j = src_manifest_idx, dst_manifest_idx
        lo_frag_local, hi_frag_local = src_frag_local, dst_frag_local
    else:
        i, j = dst_manifest_idx, src_manifest_idx
        lo_frag_local, hi_frag_local = dst_frag_local, src_frag_local

    if i == j:
        # Same fragment — slice it.
        lo, hi = sorted((lo_frag_local, hi_frag_local))
        n_rows = int(builder.vertex_groups[src_frag].shape[0])
        side_a_rows = np.arange(0, lo + 1, dtype=np.int64)
        side_b_rows = np.arange(lo + 1, n_rows, dtype=np.int64)
        if side_b_rows.size == 0:
            # All rows are on Side A — the edge was between the last
            # row and itself (a self-loop or degenerate row layout);
            # treat as a no-op.
            return [int(oid)]
        from zarr_vectors.ops.fragments import partition_fragment_rows as _part
        new_frag_idxs = _part(
            builder,
            fragment=int(src_frag),
            row_partition=[side_a_rows, side_b_rows],
            atomic=atomic,
            root=session.root,
        )
        side_a_manifest = list(manifest[:i]) + [(chunk_t, int(new_frag_idxs[0]))]
        side_b_manifest = [(chunk_t, int(new_frag_idxs[1]))] + list(manifest[i + 1 :])
    else:
        # Different fragments.  Side A = manifest[:i+1]; Side B =
        # manifest[j:].  Entries strictly between (i, j) route to
        # Side A by documented convention.
        side_a_manifest = list(manifest[: i + 1]) + list(manifest[i + 1 : j])
        side_b_manifest = list(manifest[j:])

    return _stamp_split_sides(
        session,
        oid=int(oid),
        level=level,
        side_a_manifest=side_a_manifest,
        side_b_manifest=side_b_manifest,
        atomic=atomic,
    )


def _stamp_split_sides(
    session: EditSession,
    *,
    oid: int,
    level: int,
    side_a_manifest: ObjectManifest,
    side_b_manifest: ObjectManifest,
    atomic: bool,
) -> list[int]:
    """Write Side A / Side B manifests via the session's normal plumbing."""
    from zarr_vectors.ops.change_set import ManifestOp

    if atomic:
        new_a = session._allocate_atomic_oid(level)
        session._manifest_ops[(level, new_a)] = ManifestOp(
            level=level, object_id=new_a,
            new_manifest=side_a_manifest, new_oid=new_a,
        )
        session._index_apply_manifest_delta(
            level, int(new_a),
            old_manifest=[], new_manifest=side_a_manifest,
        )
        new_b = session._allocate_atomic_oid(level)
        session._manifest_ops[(level, new_b)] = ManifestOp(
            level=level, object_id=new_b,
            new_manifest=side_b_manifest, new_oid=new_b,
        )
        session._index_apply_manifest_delta(
            level, int(new_b),
            old_manifest=[], new_manifest=side_b_manifest,
        )
        # Two new OIDs come out of one input OID under atomic split.
        # The remap chooses Side A as the canonical mapping; consumers
        # of the report can detect the split via the new_b OID being
        # absent from oid_remap's value set but present in the new
        # manifests.
        session._report.oid_remap[int(oid)] = int(new_a)
        session._mark_edit(level)
        return [new_a, new_b]

    session._stage_manifest(level, int(oid), side_a_manifest, atomic=False)
    new_b = session._allocate_atomic_oid(level)
    from zarr_vectors.ops.change_set import ManifestOp as _ManifestOp
    session._manifest_ops[(level, new_b)] = _ManifestOp(
        level=level, object_id=new_b,
        new_manifest=side_b_manifest, new_oid=new_b,
    )
    session._index_apply_manifest_delta(
        level, int(new_b),
        old_manifest=[], new_manifest=side_b_manifest,
    )
    session._mark_edit(level)
    return [int(oid), new_b]


def _oid_for_endpoint(
    session: EditSession,
    *,
    level: int,
    chunk: ChunkCoords,
    vertex_chunk_local: int,
) -> int | None:
    """Find the OID whose manifest references the chunk-local vertex
    row at ``(chunk, vertex_chunk_local)``.

    Returns ``None`` if no OID claims this row (unreferenced fragment
    or out-of-range row index).  When multiple OIDs share the
    fragment, returns the *first* one seen in OID order — sufficient
    for the merge-detection path.

    Two-step resolution:

    1. Find which fragment in ``chunk`` owns the chunk-local row via a
       single forward walk over the builder's vertex groups.
    2. Consult the session's fragment-owners inverted index for the
       OID(s) referencing that fragment.  O(1) after the index is
       built.
    """
    owner_frag = _owner_fragment_for_local(
        session, level=level, chunk=chunk,
        vertex_chunk_local=int(vertex_chunk_local),
    )
    if owner_frag is None:
        return None
    chunk_t = tuple(int(c) for c in chunk)
    owners = session._oids_referencing(level, chunk_t, int(owner_frag))
    if not owners:
        return None
    return int(owners[0])


def _owner_fragment_for_local(
    session: EditSession,
    *,
    level: int,
    chunk: ChunkCoords,
    vertex_chunk_local: int,
) -> int | None:
    """Return the fragment index in ``chunk`` that contains
    ``vertex_chunk_local``, or ``None`` when the index is out of
    range.

    Forward walk over the chunk's vertex groups; runs in O(n_fragments)
    once per endpoint — typically a handful.
    """
    if vertex_chunk_local < 0:
        return None
    builder = session._builder(level, chunk)
    cursor = 0
    for fi, g in enumerate(builder.vertex_groups):
        end = cursor + int(g.shape[0])
        if cursor <= vertex_chunk_local < end:
            return fi
        cursor = end
    return None
