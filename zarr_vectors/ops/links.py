"""Link edits: cell-addressed + convention-aware.

Connectivity is one family.  ``links/<delta>/<offsets>/`` is a rank-D
vlen array whose cell is the record's **source** chunk; ``offsets`` are
the other endpoints' chunk offsets relative to that source, so endpoint
``k``'s index is local to ``src + offsets[k - 1]``.  An intra-chunk link
is simply one whose offsets are all zero — not a separate shape — and a
cross-chunk link one whose offsets are not.  Every link therefore lives
in a cell addressed by ``(offsets, source_chunk)``, which is exactly what
:class:`~zarr_vectors.ops.refs.LinkRef` names.

``link_attributes/<name>/<delta>/<offsets>/`` mirrors it cell for cell.

Under ``implicit_sequential_with_branches`` the intra cell additionally
holds branch-override rows (child → parent pairs contradicting the
implicit ``parent = i-1`` baseline).

Placement is never decided here: records route through
:func:`zarr_vectors.core.arrays._partition_links`, the same choke point
the whole-family writer uses, so an edit lands in the cell a rewrite
would have chosen.  Because the ref names a cell, a staged append knows
its row index immediately — no post-flush prediction.

Convention contract:

| ``links_convention`` | add / edit / remove behaviour |
|---|---|
| ``"explicit"`` | every link is a stored row; all edits work uniformly |
| ``"implicit_sequential"`` | no rows stored; all link edits raise (caller must promote the store via ``materialise_object_links_explicit``) |
| ``"implicit_sequential_with_branches"`` | branch-override rows only; add/edit/remove operate on the stored branch entries; structural removal of an implicit edge raises |

Atomic semantics for links are weaker than for vertices: links don't
carry OID identity directly.  Under ``atomic=True`` an edit appends a
new row and leaves the old one in place; under ``atomic=False`` the
row is overwritten.  Object manifests are unaffected by link edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
)
from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.change_set import LinkCell, link_cell
from zarr_vectors.ops.refs import LinkRef
from zarr_vectors.typing import ChunkCoords

if TYPE_CHECKING:
    from zarr_vectors.ops.edit import EditSession


# ---------------------------------------------------------------------
# Link edits
# ---------------------------------------------------------------------

def _link_family_settings(
    session: EditSession, level: int, delta: int, link_width: int,
) -> tuple[int, bool, str]:
    """``(link_width, directed, store)`` for ``links/<delta>/``.

    The family group owns the policy every cell under it shares; absent
    means the family does not exist yet and this edit's defaults create
    it.
    """
    from zarr_vectors.core.arrays import link_family_policy
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(session.root, level)
    policy = link_family_policy(level_group, delta)
    if policy is None:
        return (link_width, False, "canonical")
    fam_width, _sid_ndim, directed, store = policy
    return (int(fam_width), bool(directed), str(store))


def _require_ref_cell(
    session: EditSession, ref: LinkRef, op: str,
) -> tuple[object, LinkCell, npt.NDArray[np.integer]]:
    """Load the cell ``ref`` addresses and bounds-check it.

    Returns ``(builder, cell, row_group)`` where ``cell`` is the
    ``(delta, offsets)`` :data:`LinkCell`.  Raises :class:`EditError`
    when the cell, the fragment or the row is absent — a stale ref must
    not silently edit the wrong row.
    """
    from zarr_vectors.core.paths import intra_offsets

    link_width, _directed, _store = _link_family_settings(
        session, ref.level, ref.delta, 2,
    )
    offsets = (
        intra_offsets(len(ref.chunk), link_width)
        if ref.offsets is None else ref.offsets
    )
    builder = session._builder(ref.level, ref.chunk)
    groups = builder.require_links(
        session.root, delta=ref.delta, link_width=link_width, offsets=offsets,
    )
    if ref.fragment >= len(groups):
        raise EditError(
            f"{op}: fragment {ref.fragment} out of range for chunk "
            f"{ref.chunk} cell delta={ref.delta} offsets={ref.offsets} "
            f"({len(groups)} groups)"
        )
    group = groups[ref.fragment]
    if ref.row >= group.shape[0]:
        raise EditError(
            f"{op}: row {ref.row} out of range in fragment {ref.fragment} "
            f"(size {group.shape[0]})"
        )
    return builder, link_cell(ref.delta, offsets), group


def add_link_in_session(
    session: EditSession,
    *,
    level: int,
    src: int | None = None,
    dst: int | None = None,
    chunk: ChunkCoords | None = None,
    endpoints: list[tuple[ChunkCoords, int]] | None = None,
    fragment: int | None = None,
    delta: int = 0,
    attrs: dict[str, npt.ArrayLike] | None = None,
    update_objects: bool = False,
) -> LinkRef:
    """Append a link at any offset and any level delta.

    Two spellings of the same operation:

    - ``endpoints=[(chunk, vi), ...]`` — the general form.  Every
      endpoint names its own chunk, so the record may span chunks or
      levels.
    - ``src=`` / ``dst=`` / ``chunk=`` — shorthand for the common
      two-endpoint case where both indices are local to one chunk; it
      expands to ``endpoints=[(chunk, src), (chunk, dst)]``.

    Placement is delegated to the core partitioner, so the record lands
    in the cell a whole-family rewrite would have chosen — including the
    all-zero (intra) cell when every endpoint shares a chunk.  Nothing
    here special-cases intra vs cross.

    Args:
        endpoints: Full ``(chunk_coords, vertex_index)`` per endpoint.
        src, dst, chunk: Shorthand for a one-chunk two-endpoint record.
        fragment: Which row group of the target cell receives the row.
            Only meaningful for the intra cell, whose groups track vertex
            fragments; defaults to the chunk's first fragment.  Ignored
            for other cells, which have no per-vertex-fragment meaning.
        delta: ``0`` intra-level, non-zero cross-level.
        attrs: per-link attribute values to write alongside the new row.
        update_objects: merge the endpoints' objects when the new edge
            bridges two OIDs.

    Returns:
        A :class:`LinkRef` naming the row just staged.  Under
        ``store="duplicate"`` the record is filed in several cells; the
        ref names the first in sorted cell order.
    """
    _check_link_convention(session, write_branch_entry=True, level=level)

    if endpoints is None:
        if chunk is None or src is None or dst is None:
            raise EditError(
                "add_link requires either endpoints=[(chunk, vi), ...] or "
                "all of src=, dst= and chunk=."
            )
        endpoints = [(chunk, int(src)), (chunk, int(dst))]
    elif src is not None or dst is not None:
        raise EditError(
            "add_link takes endpoints= or src=/dst=, not both."
        )
    record = [
        (tuple(int(c) for c in cc), int(vi)) for cc, vi in endpoints
    ]
    if len(record) < 2:
        raise EditError(
            f"add_link needs at least 2 endpoints, got {len(record)}"
        )

    from zarr_vectors.core.arrays import (
        _pack_link_rows,
        _partition_links,
        links_has_perm,
    )
    from zarr_vectors.core.paths import parse_offsets
    from zarr_vectors.core.store import get_resolution_level

    sid_ndim = len(record[0][0])
    link_width, directed, store = _link_family_settings(
        session, level, delta, len(record),
    )
    if len(record) != link_width:
        raise EditError(
            f"add_link: record has {len(record)} endpoints but "
            f"links/{delta} is a link_width={link_width} family."
        )
    level_group = get_resolution_level(session.root, level)
    buckets = _partition_links(
        level_group, [record], link_width, sid_ndim,
        delta=delta, directed=directed, store=store,
    )

    new_ref: LinkRef | None = None
    for (seg, src_chunk), entries in sorted(buckets.items()):
        offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
        builder = session._builder(level, src_chunk)
        groups = builder.require_links(
            session.root, delta=delta, link_width=link_width, offsets=offsets,
        )
        cell = link_cell(delta, offsets)
        if cell[1] is None:
            # Intra rows are indexed per vertex fragment, so the chunk must
            # already have one to address.
            if not groups:
                raise EditError(
                    f"add_link: chunk {src_chunk} has no vertex fragments "
                    f"yet; add a vertex first or call add_fragment()."
                )
            target_frag = 0 if fragment is None else int(fragment)
            if target_frag < 0 or target_frag >= len(groups):
                raise EditError(
                    f"add_link: fragment {target_frag} out of range for "
                    f"chunk {src_chunk} (has {len(groups)} fragments)"
                )
        else:
            target_frag = 0
        rows = _pack_link_rows(
            entries,
            has_perm=links_has_perm(
                offsets, delta=delta, directed=directed, store=store,
            ),
            link_width=link_width,
            dtype=np.int64,
        )
        for row in rows:
            row_idx = builder.append_link_row(cell, target_frag, row)
        ref = LinkRef(
            level=level, chunk=src_chunk, fragment=target_frag,
            row=row_idx, delta=delta, offsets=offsets,
        )
        if new_ref is None:
            new_ref = ref
            if attrs:
                _write_link_attr_row(session, ref, attrs)

    if new_ref is None:  # unreachable: one record always buckets somewhere
        raise EditError(f"add_link: record {record} produced no placement")

    session._mark_edit(level)

    if update_objects:
        # If the endpoints belong to different OIDs at ``level``, merge
        # them.  Each endpoint resolves against its own chunk, so this
        # works for a cross-chunk edge too.  When the OID lookup is
        # ambiguous (unreferenced fragment, or both endpoints in one OID)
        # the call is a no-op beyond the link row write.
        from zarr_vectors.ops.objects import _merge_objects_impl, _oid_for_endpoint
        oids = {
            _oid_for_endpoint(
                session, level=level, chunk=cc, vertex_chunk_local=vi,
            )
            for cc, vi in record
        }
        oids.discard(None)
        if len(oids) > 1:
            _merge_objects_impl(
                session,
                oids=sorted(int(o) for o in oids),
                level=level,
                atomic=session.atomic,
            )
    return new_ref


def edit_link_in_session(
    session: EditSession,
    ref: LinkRef,
    *,
    new_endpoints: tuple[int, int] | None,
    new_attrs: dict[str, npt.ArrayLike] | None,
    atomic: bool,
    update_objects: bool = False,
) -> None:
    _check_link_convention(session, write_branch_entry=True, level=ref.level)
    builder, cell, group = _require_ref_cell(session, ref, "edit_link")

    if new_endpoints is not None and not ref.is_intra:
        # Endpoints are chunk-local index pairs, which only describes a
        # row of the intra cell.  Re-pointing a cross-cell record can move
        # it to a different cell entirely, so it is a remove + add, not an
        # in-place rewrite.
        raise EditError(
            f"edit_link(new_endpoints=...) needs an intra-chunk link; "
            f"{ref} is in cell offsets={ref.offsets}.  Re-point it with "
            f"remove_link(ref) + add_link(endpoints=[...]) so the record "
            f"is re-routed to the cell its new endpoints imply."
        )

    # Capture the pre-edit endpoints so update_objects can run split
    # against the OIDs that held the OLD edge (before we change it).
    # Intra only: elsewhere column 0 may be perm_idx, and a link_width=1
    # cell has no column 1 at all.
    if ref.is_intra:
        old_src, old_dst = int(group[ref.row, 0]), int(group[ref.row, 1])

    if new_endpoints is not None:
        src, dst = int(new_endpoints[0]), int(new_endpoints[1])
        new_row = np.asarray([src, dst], dtype=group.dtype)
        if atomic:
            new_idx = builder.append_link_row(cell, ref.fragment, new_row)
            target_row = new_idx
        else:
            builder.overwrite_link_row(cell, ref.fragment, ref.row, new_row)
            target_row = ref.row
    else:
        target_row = ref.row

    if new_attrs:
        _write_link_attr_row(
            session,
            LinkRef(
                level=ref.level, chunk=ref.chunk, fragment=ref.fragment,
                row=target_row, delta=ref.delta, offsets=ref.offsets,
            ),
            new_attrs,
        )
    session._mark_edit(ref.level)

    # update_objects path: decompose into a deterministic split on the
    # OIDs that held the old edge + a merge on the OIDs the new edge
    # bridges.  Under atomic=True we leave the old row in place at the
    # link level (the new endpoints are an appended row), but for the
    # object-membership decomposition we still treat the conceptual
    # remove + add as a topology change.
    if update_objects and new_endpoints is not None:
        from zarr_vectors.ops.objects import (
            _merge_objects_impl,
            _oid_for_endpoint,
            _split_object_at_link_impl,
        )
        # Split pass on the old endpoints' OID(s).
        for oid_candidate in {
            _oid_for_endpoint(
                session, level=ref.level,
                chunk=ref.chunk, vertex_chunk_local=old_src,
            ),
            _oid_for_endpoint(
                session, level=ref.level,
                chunk=ref.chunk, vertex_chunk_local=old_dst,
            ),
        }:
            if oid_candidate is None:
                continue
            _split_object_at_link_impl(
                session,
                oid=int(oid_candidate),
                level=ref.level,
                removed_endpoints=(tuple(ref.chunk), int(old_src), int(old_dst)),
                atomic=atomic,
            )
        # Merge pass on the new endpoints' OID(s).
        oid_src_new = _oid_for_endpoint(
            session, level=ref.level, chunk=ref.chunk,
            vertex_chunk_local=int(new_endpoints[0]),
        )
        oid_dst_new = _oid_for_endpoint(
            session, level=ref.level, chunk=ref.chunk,
            vertex_chunk_local=int(new_endpoints[1]),
        )
        if (
            oid_src_new is not None
            and oid_dst_new is not None
            and oid_src_new != oid_dst_new
        ):
            _merge_objects_impl(
                session,
                oids=[oid_src_new, oid_dst_new],
                level=ref.level,
                atomic=atomic,
            )


def remove_link_in_session(
    session: EditSession,
    ref: LinkRef,
    *,
    atomic: bool,
    update_objects: bool = False,
) -> None:
    """Drop the link row.

    Under ``atomic=True`` the row is removed from the per-fragment
    group (links have no OID identity, so atomic == minimal for the
    remove case — the ``atomic`` kwarg only affects the
    ``update_objects=True`` split pass).

    Works on any cell: the ref names the row's address, so an intra and a
    cross-chunk link are dropped the same way.  ``update_objects=True``
    is intra-only — the split pass reads the dropped row as a pair of
    chunk-local indices.
    """
    _check_link_convention(session, write_branch_entry=True, level=ref.level)
    builder, cell, group = _require_ref_cell(session, ref, "remove_link")

    # Capture the endpoints of the row we're about to drop so the
    # update_objects=True split pass can target the right OID(s).
    old_src, old_dst = None, None
    if update_objects:
        if not ref.is_intra:
            raise EditError(
                f"remove_link(update_objects=True) needs an intra-chunk "
                f"link; {ref} is in cell offsets={ref.offsets}.  Drop it "
                f"with update_objects=False and adjust object membership "
                f"explicitly."
            )
        old_src = int(group[ref.row, 0])
        old_dst = int(group[ref.row, 1])

    builder.drop_link_row(cell, ref.fragment, ref.row)
    session._mark_edit(ref.level)

    if update_objects and old_src is not None and old_dst is not None:
        from zarr_vectors.ops.objects import (
            _oid_for_endpoint,
            _split_object_at_link_impl,
        )
        # Deterministic split: the removed edge's endpoints uniquely
        # identify the manifest split point.  Two OIDs can share an
        # edge (e.g. synapse-style cross-object connection); each
        # gets its own split pass.
        candidates: set[int] = set()
        for end in (old_src, old_dst):
            oid_h = _oid_for_endpoint(
                session, level=ref.level,
                chunk=ref.chunk, vertex_chunk_local=end,
            )
            if oid_h is not None:
                candidates.add(int(oid_h))
        for oid in candidates:
            _split_object_at_link_impl(
                session,
                oid=oid,
                level=ref.level,
                removed_endpoints=(tuple(ref.chunk), int(old_src), int(old_dst)),
                atomic=atomic,
            )


# ---------------------------------------------------------------------
# Cross-chunk spellings
#
# Kept as named entry points because "add a link spanning chunks" is a
# distinct intent worth naming, but they are no longer a separate
# mechanism: each is the general path with endpoints that happen to
# straddle chunks.  A record whose endpoints all land in one chunk routes
# to the intra cell here exactly as it would through add_link.
# ---------------------------------------------------------------------

def add_cross_chunk_link_in_session(
    session: EditSession,
    *,
    level: int,
    endpoints: list[tuple[ChunkCoords, int]],
    delta: int = 0,
) -> LinkRef:
    """Append a link given full ``(chunk, vertex_index)`` endpoints."""
    return add_link_in_session(
        session, level=level, endpoints=endpoints, delta=delta,
    )


def edit_cross_chunk_link_in_session(
    session: EditSession,
    ref: LinkRef,
    *,
    new_endpoints: list[tuple[ChunkCoords, int]],
    atomic: bool,
) -> LinkRef:
    """Re-point a link at new ``(chunk, vertex_index)`` endpoints.

    New endpoints may imply a different cell, so this is a re-route, not
    an in-place rewrite: the record is added at its new address and the
    old row dropped unless ``atomic`` keeps it.  Returns the new ref —
    the old one is stale either way, since dropping a row shifts the rows
    after it within its group.
    """
    _check_link_convention(session, write_branch_entry=True, level=ref.level)
    if not atomic:
        remove_link_in_session(session, ref, atomic=False)
    return add_link_in_session(
        session, level=ref.level, endpoints=new_endpoints, delta=ref.delta,
    )


def remove_cross_chunk_link_in_session(
    session: EditSession,
    ref: LinkRef,
) -> None:
    remove_link_in_session(session, ref, atomic=True)


# ---------------------------------------------------------------------
# Materialise implicit-sequential as explicit branch table
# ---------------------------------------------------------------------

def materialise_object_links_explicit(
    root,
    level: int,
    object_id: int,
    *,
    flip_convention: bool = False,
) -> int:
    """Convert one object's implicit-sequential topology into explicit
    branch-table rows so :class:`EditSession` link edits can address
    every edge in the chain.

    Walks the object's manifest, lists every vertex in manifest order,
    and emits a ``(child, parent)`` row for every consecutive pair into
    the chunk containing the child.  The reader behaviour is unchanged
    (the same edges are produced) but downstream ``edit_link`` /
    ``remove_link`` calls can now address each edge by its row index.

    Args:
        root: ZV store group.
        level: resolution level.
        object_id: object whose chain to materialise.
        flip_convention: when True, also flip the store's
            ``links_convention`` to ``"explicit"``.  Off by default
            since the flip is global and affects every other object's
            read path.

    Returns:
        Number of branch-table rows added.
    """
    from zarr_vectors.core.arrays import (
        list_chunk_keys,
        read_chunk_links,
        read_chunk_vertices,
        read_object_manifest,
        write_chunk_links,
    )
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.core.store import get_resolution_level

    meta = RootMetadata.from_dict(root.attrs.to_dict())
    ndim = meta.sid_ndim
    conv = meta.links_convention or LINKS_EXPLICIT
    if conv == LINKS_EXPLICIT:
        return 0  # already explicit; nothing to materialise

    level_group = get_resolution_level(root, level)
    manifest = read_object_manifest(level_group, object_id)
    if not manifest:
        return 0

    # Expand every manifest entry to a full per-vertex sequence of
    # ``(chunk, chunk_local_index)``.  The implicit_sequential reader
    # uses chunk-local row indices over the chunk's *flat* vertex
    # array, so we need that index too.
    chunk_cache: dict[ChunkCoords, list[int]] = {}

    def _fragment_start_in_chunk(cc: ChunkCoords, frag_idx: int) -> tuple[int, int]:
        """Return ``(start, count)`` of fragment ``frag_idx`` in the
        chunk's flat vertex array."""
        if cc not in chunk_cache:
            chunks = read_chunk_vertices(level_group, cc, ndim=ndim)
            chunk_cache[cc] = [int(g.shape[0]) for g in chunks]
        counts = chunk_cache[cc]
        if frag_idx >= len(counts):
            return (0, 0)
        return (sum(counts[:frag_idx]), counts[frag_idx])

    sequence: list[tuple[ChunkCoords, int]] = []  # (chunk, chunk_local_idx)
    for (cc, frag_idx) in manifest:
        start, count = _fragment_start_in_chunk(cc, frag_idx)
        for k in range(count):
            sequence.append((cc, start + k))
    if len(sequence) < 2:
        return 0

    # For each consecutive (parent, child) pair, append
    # ``(child_local, parent_local)`` to the child's chunk's link group.
    # When parent and child sit in different chunks we'd emit a
    # cross-chunk row; that path is deferred for this iteration and the
    # pair is skipped.
    grouped_writes: dict[ChunkCoords, list[npt.NDArray]] = {}
    n_added = 0
    for i in range(1, len(sequence)):
        parent_cc, parent_local = sequence[i - 1]
        child_cc, child_local = sequence[i]
        if parent_cc != child_cc:
            continue
        grouped_writes.setdefault(child_cc, []).append(
            np.array([child_local, parent_local], dtype=np.int64),
        )

    for cc, new_rows in grouped_writes.items():
        try:
            current_groups = read_chunk_links(level_group, cc, delta=0)
        except Exception:
            # No link array yet at this chunk — start with one empty
            # group per fragment.
            current_groups = []
        if not current_groups:
            chunks = read_chunk_vertices(level_group, cc, ndim=ndim)
            current_groups = [
                np.empty((0, 2), dtype=np.int64) for _ in chunks
            ]
        # Append every new row into the first fragment of the chunk
        # (the implicit_sequential reader doesn't care which fragment
        # holds the branch override — it walks all rows).
        appended = np.stack(new_rows, axis=0)
        current_groups[0] = np.concatenate(
            [current_groups[0], appended], axis=0,
        )
        write_chunk_links(level_group, cc, current_groups, delta=0)
        n_added += len(new_rows)

    if flip_convention:
        attrs = root.attrs.to_dict()
        zv = dict(attrs.get("zarr_vectors", {}))
        zv["links_convention"] = LINKS_EXPLICIT
        root.attrs.update({"zarr_vectors": zv})

    del list_chunk_keys  # silence unused-import warning
    return n_added


# ---------------------------------------------------------------------
# Reorder vertices so explicit links collapse to implicit_sequential_with_branches
# ---------------------------------------------------------------------

def reorder_vertices_implicit(
    root,
    level: int,
    *,
    object_ids: "Iterable[int] | None" = None,
    flip_convention: bool = False,
    dtype: "np.dtype | str" = np.float32,
) -> dict:
    """Inverse of :func:`materialise_object_links_explicit`.

    Walks each tree-like object's explicit link graph, computes a
    deterministic DFS pre-order, and physically reorders each chunk's
    vertices so the object's spine becomes a contiguous run of
    chunk-local indices in DFS order.  Edges along the spine then
    collapse under the ``implicit_sequential_with_branches`` baseline
    (``parent[i] = i-1`` over the reader's chunk-sorted concatenation);
    only true branch overrides remain, in ``links/0/<offsets>/<chunk>`` —
    the all-zero offsets cell for a within-chunk override, a non-zero one
    for an override whose endpoints straddle chunks.

    Objects whose link graph is not a tree (cycles, multi-parent,
    disconnected within their manifest) are skipped with a
    ``UserWarning`` — their vertices and link rows are left untouched.

    Args:
        root: ZV store group.
        level: resolution level.
        object_ids: optional whitelist of object ids to process; default
            is every object in the level.
        flip_convention: when True and **every** requested object
            qualified, also flip the store's ``links_convention`` to
            ``"implicit_sequential_with_branches"``.  If any object was
            skipped, the flip is refused (the skipped object's edges
            still depend on the explicit convention being live).
        dtype: vertex dtype for the per-chunk rewrite.

    Returns:
        Report dict with keys ``objects_processed``,
        ``objects_skipped_non_tree``, ``skipped_oids``,
        ``branch_overrides_written``,
        ``cross_chunk_branch_overrides_written``, ``chunks_repermuted``,
        ``fragments_split``, ``convention_flipped``.

    Raises:
        EditError: when invoked on a store whose ``links_convention``
            is already ``implicit_sequential`` (no edge data on disk
            to analyse) — materialise to explicit first, then reorder.
    """
    import warnings

    from zarr_vectors.core.arrays import (
        finalize_links,
        list_link_offsets,
        read_all_object_manifests,
        read_chunk_link_attributes,
        read_chunk_links,
        read_chunk_vertices,
        read_link_attributes,
        read_links,
        read_vertex_fragment_index,
        write_chunk_fragment_attributes,
        write_chunk_links,
        write_chunk_vertices,
        write_link_attribute_cells,
        write_link_attributes,
        write_link_cells,
        write_links,
        write_object_index,
    )
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.core.paths import links_group_path
    from zarr_vectors.core.store import get_resolution_level

    _empty_report = lambda: {
        "objects_processed": 0,
        "objects_skipped_non_tree": 0,
        "skipped_oids": [],
        "branch_overrides_written": 0,
        "cross_chunk_branch_overrides_written": 0,
        "chunks_repermuted": 0,
        "fragments_split": 0,
        "convention_flipped": False,
    }

    meta = RootMetadata.from_dict(root.attrs.to_dict())
    sid_ndim = meta.sid_ndim
    conv = meta.links_convention or LINKS_EXPLICIT
    if conv == LINKS_IMPLICIT_BRANCHES:
        return _empty_report()
    if conv == LINKS_IMPLICIT_SEQUENTIAL:
        raise EditError(
            "reorder_vertices_implicit requires links_convention="
            f"{LINKS_EXPLICIT!r}; got {conv!r}. The implicit_sequential "
            "store has no stored edges to analyse — call "
            "materialise_object_links_explicit first (with flip_convention=True) "
            "and then reorder_vertices_implicit."
        )
    if conv != LINKS_EXPLICIT:
        raise EditError(
            f"reorder_vertices_implicit: unsupported links_convention={conv!r}"
        )

    level_group = get_resolution_level(root, level)
    all_manifests = read_all_object_manifests(level_group)
    n_objects = len(all_manifests)
    if object_ids is None:
        target_oids = list(range(n_objects))
    else:
        target_oids = [int(o) for o in object_ids]
        for o in target_oids:
            if o < 0 or o >= n_objects:
                raise EditError(
                    f"object_id {o} out of range [0, {n_objects})"
                )
    if not target_oids:
        return _empty_report()

    # --- caches -------------------------------------------------------
    fi_cache: dict = {}

    def _get_fi(cc):
        if cc not in fi_cache:
            fi_cache[cc] = read_vertex_fragment_index(level_group, cc)
        return fi_cache[cc]

    chunk_links_cache: dict = {}

    def _get_chunk_links(cc):
        if cc not in chunk_links_cache:
            try:
                chunk_links_cache[cc] = read_chunk_links(
                    level_group, cc, delta=0,
                )
            except Exception:
                chunk_links_cache[cc] = []
        return chunk_links_cache[cc]

    # --- Phase A: per-object analysis --------------------------------
    # vkey = (chunk_coords, chunk_local_idx)
    # vkey_owner[vkey] = (oid, manifest_position_within_oid)
    vkey_owner: dict = {}
    oid_vkeys: dict = {}
    oid_fragment_runs: dict = {}  # oid -> list of (cc, frag_idx, vkey_slice_start, vkey_slice_end)

    for oid in target_oids:
        manifest = all_manifests[oid]
        vkey_list = []
        frag_runs = []
        for (cc, frag_idx) in manifest:
            fi = _get_fi(cc)
            if frag_idx < 0 or frag_idx >= fi.num_fragments:
                raise EditError(
                    f"object {oid}: manifest references missing fragment "
                    f"({cc}, {frag_idx})"
                )
            if not fi.is_range(frag_idx):
                raise EditError(
                    f"object {oid}: fragment ({cc}, {frag_idx}) is explicit; "
                    "reorder_vertices_implicit currently only supports "
                    "range fragments"
                )
            start, count = fi.range(frag_idx)
            run_start = len(vkey_list)
            for k in range(int(count)):
                vkey = (tuple(cc), int(start) + k)
                if vkey in vkey_owner:
                    raise EditError(
                        f"vertex {vkey} is referenced by multiple objects "
                        f"(at least {vkey_owner[vkey][0]} and {oid}); "
                        "reorder_vertices_implicit requires unique vertex "
                        "ownership"
                    )
                vkey_owner[vkey] = (oid, len(vkey_list))
                vkey_list.append(vkey)
            frag_runs.append(
                (tuple(cc), int(frag_idx), run_start, len(vkey_list)),
            )
        oid_vkeys[oid] = vkey_list
        oid_fragment_runs[oid] = frag_runs

    # Collect edges for each target oid.
    oid_edges: dict = {oid: [] for oid in target_oids}
    touched_chunks: set = set()
    for vk in vkey_owner:
        touched_chunks.add(vk[0])

    for cc in touched_chunks:
        for lg in _get_chunk_links(cc):
            for row in np.asarray(lg).reshape(-1, 2):
                child_local = int(row[0])
                parent_local = int(row[1])
                cv = (cc, child_local)
                pv = (cc, parent_local)
                co = vkey_owner.get(cv)
                po = vkey_owner.get(pv)
                if co is None or po is None or co[0] != po[0]:
                    continue
                oid_edges[co[0]].append((cv, pv))

    # read_links returns the whole family, intra cells included; the loop
    # above already collected those from the per-chunk read, so take only
    # the records that actually span chunks.
    #
    # Keep each record's index into the *unfiltered* family: the parallel
    # attribute family is one row per record over that same full set, so
    # Phase F must index it by the original position, not by the position
    # within this filtered list.
    _all_records = read_links(level_group, delta=0)
    cross_src_idx = [
        i for i, record in enumerate(_all_records)
        if len({tuple(cc) for cc, _vi in record}) > 1
    ]
    all_ccls = [_all_records[i] for i in cross_src_idx]
    for record in all_ccls:
        if len(record) != 2:
            continue  # only handle edge-shaped (link_width=2) records
        (cc_a, vi_a), (cc_b, vi_b) = record
        va = (tuple(cc_a), int(vi_a))
        vb = (tuple(cc_b), int(vi_b))
        oa = vkey_owner.get(va)
        ob = vkey_owner.get(vb)
        if oa is None or ob is None or oa[0] != ob[0]:
            continue
        oid_edges[oa[0]].append((va, vb))

    # Classify topology + DFS per oid.
    qualifying: dict = {}  # oid -> dict
    skipped: list = []

    for oid in target_oids:
        result = _classify_and_dfs(
            oid_vkeys[oid], oid_edges[oid], oid,
        )
        if result is None:
            skipped.append(oid)
            warnings.warn(
                f"reorder_vertices_implicit: object {oid} is not a tree "
                "(cycle, multi-parent, or disconnected); skipping.",
                UserWarning, stacklevel=2,
            )
            continue
        dfs_order, dfs_parent = result
        # Build dfs_rank: vkey -> position in dfs_order
        dfs_rank = {vk: i for i, vk in enumerate(dfs_order)}
        qualifying[oid] = {
            "dfs_order": dfs_order,
            "dfs_parent": dfs_parent,
            "dfs_rank": dfs_rank,
        }

    if not qualifying:
        report = _empty_report()
        report["objects_skipped_non_tree"] = len(skipped)
        report["skipped_oids"] = skipped
        return report

    # --- Phase B: verify each fragment is a contiguous DFS run -------
    # For each qualifying oid, every (cc, frag_idx) in its manifest
    # must have its vertices form a contiguous block in dfs_order.
    # If not, raise (MVP doesn't split).
    n_fragments_split = 0  # always 0 in MVP

    for oid in qualifying:
        dfs_rank = qualifying[oid]["dfs_rank"]
        for (cc, frag_idx, run_start, run_end) in oid_fragment_runs[oid]:
            vkeys_in_frag = oid_vkeys[oid][run_start:run_end]
            ranks = sorted(dfs_rank[v] for v in vkeys_in_frag)
            if ranks and ranks[-1] - ranks[0] + 1 != len(ranks):
                raise EditError(
                    f"object {oid}: fragment ({cc}, {frag_idx}) has "
                    f"vertices that don't form a contiguous DFS run "
                    "(pathological writer layout). MVP requires each "
                    "fragment's vertices to be a single contiguous "
                    "subsequence of the object's DFS order."
                )

    # --- Phase C: compute per-chunk permutations ---------------------
    # For each touched chunk: build the new chunk-local order.
    # Strategy: walk existing fragments in fragment-index order.
    #   - non-qualifying-OID fragment: emit vertices in current order.
    #   - qualifying-OID fragment: emit vertices in DFS-rank order.
    # qualifying_frag_oid: (cc, frag_idx) -> oid (only for qualifying)
    qualifying_frag_oid: dict = {}
    for oid in qualifying:
        for (cc, frag_idx, _s, _e) in oid_fragment_runs[oid]:
            qualifying_frag_oid[(cc, frag_idx)] = oid

    # Find every chunk that has any fragment we need to read.
    # We must process every chunk that touches a qualifying OID
    # (vertex/link/CCL update) plus any chunk whose CCLs we'll rewrite.
    chunks_to_rewrite: set = set()
    for oid in qualifying:
        for (cc, _f, _s, _e) in oid_fragment_runs[oid]:
            chunks_to_rewrite.add(cc)

    # Per-chunk: perm (new_local -> old_local), inv_perm, fragment_groups
    chunk_perm: dict = {}  # cc -> np.ndarray perm[new] = old
    chunk_inv_perm: dict = {}  # cc -> np.ndarray inv_perm[old] = new
    chunk_new_groups: dict = {}  # cc -> list[(new_local_start, count, old_frag_idx)]

    for cc in chunks_to_rewrite:
        fi = _get_fi(cc)
        new_order: list = []  # list of old chunk-local indices in new order
        new_groups: list = []  # list of (new_start, count, old_frag_idx)
        for f in range(fi.num_fragments):
            if not fi.is_range(f):
                raise EditError(
                    f"chunk {cc} has an explicit fragment {f}; "
                    "reorder_vertices_implicit currently only supports "
                    "range fragments"
                )
            start, count = fi.range(f)
            new_start = len(new_order)
            oid_for_frag = qualifying_frag_oid.get((cc, f))
            if oid_for_frag is None:
                # Keep original order within this fragment.
                new_order.extend(range(int(start), int(start) + int(count)))
            else:
                dfs_rank = qualifying[oid_for_frag]["dfs_rank"]
                frag_old_locals = list(range(int(start), int(start) + int(count)))
                frag_old_locals.sort(key=lambda lo: dfs_rank[(cc, lo)])
                new_order.extend(frag_old_locals)
            new_groups.append((new_start, int(count), f))
        perm = np.asarray(new_order, dtype=np.int64)
        n_chunk_verts = int(perm.shape[0])
        inv_perm = np.empty(n_chunk_verts, dtype=np.int64)
        inv_perm[perm] = np.arange(n_chunk_verts, dtype=np.int64)
        chunk_perm[cc] = perm
        chunk_inv_perm[cc] = inv_perm
        chunk_new_groups[cc] = new_groups

    # --- Phase D: emit branch overrides ------------------------------
    # For each qualifying oid, walk its DFS sequence. For each
    # non-root vertex v at rank r, p = dfs_parent[v]. Compute (in new
    # chunk-local indices) and decide implicit vs override.
    # We need each vertex's new global index (reader's chunk-sorted
    # concatenation). Compute chunk_global_offsets in that order.
    sorted_touched_chunks = sorted(chunks_to_rewrite)
    chunk_global_offset: dict = {}
    running = 0
    for cc in sorted_touched_chunks:
        chunk_global_offset[cc] = running
        running += int(chunk_perm[cc].shape[0])
    # NB: chunks that don't touch any qualifying object still appear in
    # the reader's enumeration. We only use chunk_global_offset for
    # comparing within touched chunks (where qualifying-object vertices
    # live), so non-touched chunks don't affect correctness here.
    # Strictly: the reader uses ALL chunks. Other chunks' offsets shift
    # things. Build the full picture from the level group's vertex chunks.
    from zarr_vectors.core.arrays import list_chunk_keys as _list_chunk_keys
    all_vertex_chunks = sorted(_list_chunk_keys(level_group))
    chunk_global_offset = {}
    running = 0
    for cc in all_vertex_chunks:
        chunk_global_offset[cc] = running
        if cc in chunks_to_rewrite:
            running += int(chunk_perm[cc].shape[0])
        else:
            fi = _get_fi(cc)
            n = 0
            for f in range(fi.num_fragments):
                if fi.is_range(f):
                    _s, c = fi.range(f)
                    n += int(c)
                else:
                    n += int(fi.indices(f).shape[0])
            running += n

    def _new_local(vk):
        cc, old_local = vk
        return int(chunk_inv_perm[cc][old_local])

    def _new_global(vk):
        cc, old_local = vk
        return chunk_global_offset[cc] + _new_local(vk)

    # Per-chunk branch-override accumulators.
    intra_branch_rows: dict = {cc: [] for cc in chunks_to_rewrite}
    new_cross_branches: list = []  # list of ((parent_cc, parent_new_local), (child_cc, child_new_local))

    n_intra_branches = 0
    n_cross_branches = 0

    for oid, qd in qualifying.items():
        dfs_order = qd["dfs_order"]
        dfs_parent = qd["dfs_parent"]
        for r in range(1, len(dfs_order)):
            v = dfs_order[r]
            p = dfs_parent.get(v)
            if p is None:
                continue
            v_global = _new_global(v)
            p_global = _new_global(p)
            if p_global == v_global - 1:
                continue  # implicit baseline handles it
            # Emit override
            v_cc, _ = v
            p_cc, _ = p
            if v_cc == p_cc:
                intra_branch_rows[v_cc].append(
                    (int(_new_local(v)), int(_new_local(p))),
                )
                n_intra_branches += 1
            else:
                new_cross_branches.append(
                    (
                        (p_cc, int(_new_local(p))),
                        (v_cc, int(_new_local(v))),
                    ),
                )
                n_cross_branches += 1

    # --- Phase E: write per-chunk vertices, fragments, links ---------
    np_dtype = np.dtype(dtype)
    fragment_attribute_names = _list_fragment_attribute_names(level_group)
    link_attribute_names = _list_link_attribute_names(level_group, delta=0)

    for cc in chunks_to_rewrite:
        perm = chunk_perm[cc]
        new_groups = chunk_new_groups[cc]

        # Vertices
        old_fragments = read_chunk_vertices(
            level_group, cc, dtype=np_dtype, ndim=sid_ndim,
        )
        if old_fragments:
            flat = np.concatenate(old_fragments, axis=0)
        else:
            flat = np.empty((0, sid_ndim), dtype=np_dtype)
        new_flat = flat[perm]
        new_fragment_groups = [
            new_flat[ns:ns + nc] for (ns, nc, _ofi) in new_groups
        ]
        write_chunk_vertices(
            level_group, cc, new_fragment_groups, dtype=np_dtype,
        )

        # Per-fragment attributes (1 row per fragment, identity mapping
        # since we don't split fragments in the MVP).
        for fname in fragment_attribute_names:
            try:
                fa = _read_chunk_fragment_attributes_safe(
                    level_group, fname, cc,
                )
            except Exception:
                continue
            if fa is None or fa.shape[0] == 0:
                continue
            # Reorder rows by new_groups[i][2] = old_frag_idx.
            old_order = np.asarray(
                [og for (_ns, _nc, og) in new_groups], dtype=np.int64,
            )
            new_fa = fa[old_order]
            write_chunk_fragment_attributes(
                level_group, fname, cc, new_fa, dtype=fa.dtype,
            )

        # Intra-chunk link rows
        old_link_groups = _get_chunk_links(cc)
        # Build set of qualifying vkeys in this chunk
        qualifying_old_locals_in_cc: set = set()
        for oid in qualifying:
            for (fcc, _f, _s, _e) in oid_fragment_runs[oid]:
                if fcc != cc:
                    continue
            for vk in oid_vkeys[oid]:
                if vk[0] == cc:
                    qualifying_old_locals_in_cc.add(vk[1])

        # Strip rows owned by qualifying objects (both endpoints qualifying).
        # Re-translate kept rows through inv_perm.
        new_kept_rows: list = []
        # Per-fragment link attributes need parallel rebuild.
        per_fragment_link_attr_rows: dict = {}
        for fname in link_attribute_names:
            try:
                per_fragment_link_attr_rows[fname] = (
                    _read_chunk_link_attribute_groups(
                        level_group, fname, cc, len(old_link_groups),
                    )
                )
            except Exception:
                per_fragment_link_attr_rows[fname] = None

        kept_link_attr_rows: dict = {fname: [] for fname in link_attribute_names}

        for g_idx, lg in enumerate(old_link_groups):
            arr = np.asarray(lg).reshape(-1, 2)
            for row_idx in range(arr.shape[0]):
                c_loc = int(arr[row_idx, 0])
                p_loc = int(arr[row_idx, 1])
                if (
                    c_loc in qualifying_old_locals_in_cc
                    and p_loc in qualifying_old_locals_in_cc
                    and vkey_owner.get((cc, c_loc), (None,))[0]
                    == vkey_owner.get((cc, p_loc), (None,))[0]
                    and vkey_owner.get((cc, c_loc), (None,))[0] in qualifying
                ):
                    # Drop — qualifying-object edge replaced by branch logic.
                    continue
                inv = chunk_inv_perm[cc]
                new_row = (int(inv[c_loc]), int(inv[p_loc]))
                new_kept_rows.append(new_row)
                for fname in link_attribute_names:
                    src = per_fragment_link_attr_rows.get(fname)
                    if src is not None and g_idx < len(src) and src[g_idx] is not None:
                        if row_idx < src[g_idx].shape[0]:
                            kept_link_attr_rows[fname].append(src[g_idx][row_idx])

        # Append branch overrides
        for (cl, pl) in intra_branch_rows[cc]:
            new_kept_rows.append((cl, pl))
            # No source attribute data for branch overrides; pad with zero default below.

        # Bucket all new rows into a per-fragment groups list (group 0
        # carries all rows; the reader walks groups uniformly).
        n_frags = len(new_groups)
        new_link_groups = [
            np.empty((0, 2), dtype=np.int64) for _ in range(n_frags)
        ]
        if new_kept_rows:
            new_link_groups[0] = np.asarray(new_kept_rows, dtype=np.int64).reshape(-1, 2)
        write_chunk_links(level_group, cc, new_link_groups, delta=0)

        # Link attributes
        for fname in link_attribute_names:
            kept = kept_link_attr_rows[fname]
            # Pad branch overrides with zeros to keep alignment.
            n_branches_here = len(intra_branch_rows[cc])
            if n_branches_here > 0:
                src = per_fragment_link_attr_rows.get(fname)
                sample = None
                if src:
                    for gp in src:
                        if gp is not None and gp.shape[0] > 0:
                            sample = gp[0]
                            break
                if sample is not None:
                    pad_shape = sample.shape
                    pad_dtype = sample.dtype
                    for _ in range(n_branches_here):
                        kept.append(np.zeros(pad_shape, dtype=pad_dtype))
            if not kept:
                continue
            arr = np.stack(kept, axis=0) if kept[0].ndim > 0 else np.asarray(kept)
            group0 = arr
            group_rest = [np.empty((0,) + arr.shape[1:], dtype=arr.dtype) for _ in range(n_frags - 1)]
            from zarr_vectors.core.arrays import write_chunk_link_attributes
            write_chunk_link_attributes(
                level_group, fname, cc, [group0, *group_rest],
                dtype=arr.dtype, delta=0,
            )

    # --- Phase F: cross-chunk-link rewrite ---------------------------
    new_ccl_records: list = []
    new_ccl_attr_rows: dict = {}  # fname -> list of attribute rows in new record order

    ccl_attr_names = _list_link_attribute_names(level_group, delta=0)
    ccl_attrs: dict = {}
    for fname in ccl_attr_names:
        try:
            ccl_attrs[fname] = read_link_attributes(
                level_group, fname, delta=0,
            )
        except Exception:
            ccl_attrs[fname] = None
        new_ccl_attr_rows[fname] = []

    for i, record in enumerate(all_ccls):
        # ``src_i`` indexes the full family that the attribute rows
        # parallel; ``i`` only indexes the cross-spanning subset.
        src_i = cross_src_idx[i]
        if len(record) != 2:
            # Pass through (e.g. triangle CCLs, link_width != 2).
            new_ccl_records.append(record)
            for fname in ccl_attr_names:
                if ccl_attrs[fname] is not None and src_i < ccl_attrs[fname].shape[0]:
                    new_ccl_attr_rows[fname].append(ccl_attrs[fname][src_i])
            continue
        (cc_a, vi_a), (cc_b, vi_b) = record
        va = (tuple(cc_a), int(vi_a))
        vb = (tuple(cc_b), int(vi_b))
        oa = vkey_owner.get(va)
        ob = vkey_owner.get(vb)
        if (
            oa is not None and ob is not None
            and oa[0] == ob[0] and oa[0] in qualifying
        ):
            # Both endpoints belong to a qualifying object — drop.
            continue
        # Otherwise re-translate endpoints whose chunk got permuted.
        def _maybe_translate(cc, vi):
            if cc in chunk_inv_perm:
                return int(chunk_inv_perm[cc][int(vi)])
            return int(vi)
        new_record = (
            (tuple(cc_a), _maybe_translate(tuple(cc_a), vi_a)),
            (tuple(cc_b), _maybe_translate(tuple(cc_b), vi_b)),
        )
        new_ccl_records.append(new_record)
        for fname in ccl_attr_names:
            if ccl_attrs[fname] is not None and src_i < ccl_attrs[fname].shape[0]:
                new_ccl_attr_rows[fname].append(ccl_attrs[fname][src_i])

    # Append new branch overrides discovered in Phase D
    for (parent_endpoint, child_endpoint) in new_cross_branches:
        # CCL semantics: each record is just a list of endpoints.
        # For branch overrides, the row in the link array is
        # (child_local, parent_local) -- but CCLs encode endpoints
        # without role labels. The reader at line 689-693 of types/
        # graphs.py reads them in the order stored, then column-stacks
        # into (child_global, parent_global) under `(lg + chunk_offset)`.
        # For a branch override, we want (child, parent), so endpoint 0
        # is the child and endpoint 1 is the parent. Match that order.
        new_ccl_records.append((child_endpoint, parent_endpoint))
        for fname in ccl_attr_names:
            src = ccl_attrs[fname]
            if src is not None and src.shape[0] > 0:
                new_ccl_attr_rows[fname].append(np.zeros_like(src[0]))

    # Only the cross-spanning records reach here — Phase E owns the
    # all-zero-offsets (intra) array.  ``write_links(mode="replace")``
    # scopes its delete to the offset arrays these records target, so the
    # intra array survives; do not widen that scope.
    cross_partition = None
    if new_ccl_records:
        cross_partition = write_links(
            level_group, new_ccl_records, sid_ndim=sid_ndim,
            delta=0, link_width=2, mode="replace",
        )
    else:
        # write_links short-circuits on empty input, so drop the stale
        # cross-offset arrays ourselves.  Delete per segment rather than
        # the whole links/<0> group: that group now also holds the intra
        # links Phase E just rewrote, and a family-wide wipe would
        # silently destroy them.
        from zarr_vectors.core.paths import is_intra, parse_offsets

        for seg in list_link_offsets(level_group, 0):
            try:
                offs = parse_offsets(seg, sid_ndim=sid_ndim, link_width=2)
            except ValueError:
                continue
            if is_intra(offs):
                continue
            level_group.delete_subtree(f"{links_group_path(0)}/{seg}")

    for fname in ccl_attr_names:
        rows = new_ccl_attr_rows[fname]
        if rows and ccl_attrs[fname] is not None and cross_partition is not None:
            arr = np.stack(rows, axis=0) if rows[0].ndim > 0 else np.asarray(rows)
            # Pass the partition from the matching write_links: rows are in
            # input order, and the partition is what maps them onto the
            # (offsets, cell) layout that writer chose.
            write_link_attributes(
                level_group, fname, arr,
                num_links=cross_partition.num_links,
                partition=cross_partition,
                delta=0,
            )

    # --- Phase G: manifests + convention flip ------------------------
    # Manifests for qualifying objects stay structurally identical (no
    # fragment-splitting in the MVP) -- only vertex order within each
    # fragment changed, which doesn't affect the manifest. So we skip
    # rewriting object_index entirely.

    convention_flipped = False
    if flip_convention and not skipped:
        attrs = root.attrs.to_dict()
        zv = dict(attrs.get("zarr_vectors", {}))
        zv["links_convention"] = LINKS_IMPLICIT_BRANCHES
        root.attrs.update({"zarr_vectors": zv})
        convention_flipped = True

    del write_object_index, read_chunk_link_attributes  # silence unused-import warnings

    return {
        "objects_processed": len(qualifying),
        "objects_skipped_non_tree": len(skipped),
        "skipped_oids": skipped,
        "branch_overrides_written": n_intra_branches,
        "cross_chunk_branch_overrides_written": n_cross_branches,
        "chunks_repermuted": len(chunks_to_rewrite),
        "fragments_split": n_fragments_split,
        "convention_flipped": convention_flipped,
    }


def _classify_and_dfs(vkeys, edges, oid):
    """Union-find classification + deterministic DFS pre-order.

    Returns ``None`` if the graph is not a tree (cycle, multi-parent,
    disconnected within the manifest, or edge endpoints not in vkeys).
    Otherwise returns ``(dfs_order, dfs_parent)`` where dfs_order is a
    list of vkeys in DFS pre-order and dfs_parent[vkey] is the parent
    vkey in the DFS (None for the root).
    """
    n = len(vkeys)
    if n == 0:
        return ([], {})
    if n == 1:
        return ([vkeys[0]], {vkeys[0]: None})

    vkey_to_idx = {vk: i for i, vk in enumerate(vkeys)}
    manifest_pos = vkey_to_idx  # manifest order is the input order

    # Validate edges & build adjacency.
    adj: list = [[] for _ in range(n)]
    parent_uf = list(range(n))

    def _find(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    seen_edges = set()
    for (a, b) in edges:
        if a not in vkey_to_idx or b not in vkey_to_idx:
            return None
        ia = vkey_to_idx[a]
        ib = vkey_to_idx[b]
        if ia == ib:
            return None  # self-loop
        # Deduplicate undirected edges from possibly-doubled storage.
        ekey = (min(ia, ib), max(ia, ib))
        if ekey in seen_edges:
            continue
        seen_edges.add(ekey)
        ra = _find(ia)
        rb = _find(ib)
        if ra == rb:
            return None  # cycle
        parent_uf[ra] = rb
        adj[ia].append(ib)
        adj[ib].append(ia)

    # Must be connected.
    roots = {_find(i) for i in range(n)}
    if len(roots) != 1:
        return None

    n_unique_edges = len(seen_edges)
    if n_unique_edges != n - 1:
        return None

    # Deterministic root: degree-1 vertex with smallest manifest_pos;
    # if no leaves (single-vertex), pick the manifest's first vertex.
    leaves = [i for i in range(n) if len(adj[i]) <= 1]
    if not leaves:
        root_idx = 0
    else:
        root_idx = min(leaves, key=lambda i: manifest_pos[vkeys[i]])

    # DFS pre-order; sort neighbours by manifest_pos for determinism.
    for i in range(n):
        adj[i].sort(key=lambda j: manifest_pos[vkeys[j]])

    dfs_order: list = []
    dfs_parent: dict = {}
    visited = [False] * n
    # Iterative DFS, push children in reverse so first child is processed first.
    stack = [(root_idx, None)]
    while stack:
        node, par = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        dfs_order.append(vkeys[node])
        dfs_parent[vkeys[node]] = None if par is None else vkeys[par]
        for nb in reversed(adj[node]):
            if not visited[nb]:
                stack.append((nb, node))

    if len(dfs_order) != n:
        return None
    return dfs_order, dfs_parent


def _list_fragment_attribute_names(level_group) -> list:
    from zarr_vectors.constants import FRAGMENT_ATTRIBUTES
    try:
        g = level_group.zarr_group.get(FRAGMENT_ATTRIBUTES)
    except Exception:
        return []
    if g is None:
        return []
    try:
        # fragment_attributes/<name> are single vlen arrays.
        return sorted(set(g.array_keys()) | set(g.group_keys()))
    except Exception:
        try:
            return sorted([k for k in g.keys()])
        except Exception:
            return []


def _list_link_attribute_names(level_group, *, delta: int = 0) -> list:
    """Attribute names with a ``link_attributes/<name>/<delta>/`` family.

    Covers every link attribute at this delta — intra and cross-chunk
    alike — since the offsets arrays all hang under the one delta group.
    """
    from zarr_vectors.constants import LINK_ATTRIBUTES
    from zarr_vectors.core.paths import format_delta
    try:
        g = level_group.zarr_group.get(LINK_ATTRIBUTES)
    except Exception:
        return []
    if g is None:
        return []
    # format_delta, not str: delta=+1 is the directory "+1", so str(delta)
    # would look for "1" and silently match nothing.
    segment = format_delta(delta)
    names: list = []
    try:
        for name in sorted(list(g.group_keys())):
            sub = g.get(name)
            if sub is None:
                continue
            try:
                delta_children = set(sub.array_keys()) | set(sub.group_keys())
                if segment in delta_children:
                    names.append(name)
            except Exception:
                continue
    except Exception:
        return []
    return names


def _read_chunk_fragment_attributes_safe(level_group, attr_name, cc):
    from zarr_vectors.core.arrays import read_chunk_fragment_attributes
    try:
        return read_chunk_fragment_attributes(level_group, attr_name, cc)
    except Exception:
        return None


def _read_chunk_link_attribute_groups(level_group, attr_name, cc, n_groups):
    from zarr_vectors.core.arrays import read_chunk_link_attributes
    try:
        arr = read_chunk_link_attributes(
            level_group, attr_name, cc, delta=0,
        )
    except Exception:
        return None
    # arr is the full per-chunk attribute array; the caller cares about
    # row-by-row attribute values aligned with link rows. We return a
    # list with one block per fragment group, but since
    # read_chunk_link_attributes returns a flat array and link rows are
    # also flat per group, we slice using the link fragment index.
    try:
        from zarr_vectors.core.arrays import read_link_fragment_index
        lfi = read_link_fragment_index(level_group, cc)
    except Exception:
        return [arr]
    groups: list = []
    for f in range(lfi.num_fragments):
        if lfi.is_range(f):
            s, c = lfi.range(f)
            groups.append(arr[int(s):int(s) + int(c)])
        else:
            idx = lfi.indices(f)
            groups.append(arr[np.asarray(idx)])
    return groups


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _check_link_convention(
    session: EditSession,
    *,
    write_branch_entry: bool,
    level: int = 0,
) -> None:
    """Refuse — or auto-materialise — link edits on stores that can't
    represent them.

    ``implicit_sequential`` is the strictest: no link rows are stored
    at all.  ``implicit_sequential_with_branches`` accepts edits that
    write to the branch table.

    Default behaviour when an edit needs an explicit convention the
    store doesn't have: run
    :func:`materialise_object_links_explicit` for every object at
    ``level``, flip ``links_convention`` to ``"explicit"`` in-place,
    and emit one :class:`UserWarning` so the conversion is visible.
    Set ``session.auto_materialise_links = False`` to fall back to
    :class:`EditError`.
    """
    from zarr_vectors.core.metadata import RootMetadata
    meta = RootMetadata.from_dict(session.root.attrs.to_dict())
    conv = meta.links_convention or LINKS_EXPLICIT
    if conv == LINKS_EXPLICIT:
        return
    if conv == LINKS_IMPLICIT_BRANCHES and write_branch_entry:
        return
    if getattr(session, "auto_materialise_links", True):
        _auto_materialise_to_explicit(session, level=level, prior_conv=conv)
        return
    raise EditError(
        f"link edits on store with links_convention={conv!r} are not "
        f"supported.  Either run materialise_object_links_explicit(root, "
        f"level, object_id, flip_convention=True) yourself, set "
        f"session.auto_materialise_links=True (the default), or rewrite "
        f"the store with links_convention='explicit'."
    )


def _auto_materialise_to_explicit(
    session: EditSession,
    *,
    level: int,
    prior_conv: str,
) -> None:
    """Materialise every object's implicit chain into explicit rows
    and flip ``links_convention`` to ``"explicit"``.  Emits a single
    UserWarning summarising the conversion so callers can audit it.

    Runs lazily on the first link-edit that requires explicit
    semantics, so stores that never get link-edited never pay the
    materialisation cost.
    """
    import warnings

    from zarr_vectors.core.arrays import read_all_object_manifests
    from zarr_vectors.core.store import get_resolution_level

    level_group = get_resolution_level(session.root, level)
    try:
        manifests = read_all_object_manifests(level_group)
    except Exception:
        manifests = []
    n_added = 0
    n_objects = 0
    for oid, manifest in enumerate(manifests):
        if not manifest:
            continue
        n_objects += 1
        # ``flip_convention=False`` per call — we flip once at the end
        # so partial state is never visible to readers.
        n_added += materialise_object_links_explicit(
            session.root, level=level, object_id=oid, flip_convention=False,
        )
    # Flip the convention now that every object has its rows.
    attrs = session.root.attrs.to_dict()
    zv = dict(attrs.get("zarr_vectors", {}))
    zv["links_convention"] = LINKS_EXPLICIT
    session.root.attrs.update({"zarr_vectors": zv})

    warnings.warn(
        f"zarr-vectors: auto-materialised {n_added} branch-table rows "
        f"across {n_objects} object(s) at level {level} to support a "
        f"link edit that requires links_convention='explicit'.  The "
        f"store's links_convention was {prior_conv!r}; now flipped to "
        f"'explicit'.  Set EditSession(auto_materialise_links=False) "
        f"to fail-fast instead.",
        UserWarning,
        stacklevel=4,
    )


def _write_link_attr_row(
    session: EditSession,
    link_ref: LinkRef,
    attrs: dict[str, npt.ArrayLike],
) -> None:
    """RMW a per-link attribute row in the cell mirroring ``link_ref``.

    Delegates to the attributes module's per-link path.
    """
    from zarr_vectors.ops.attributes import _edit_link_attr
    from zarr_vectors.ops.refs import AttributeRef
    for name, val in attrs.items():
        _edit_link_attr(
            session,
            AttributeRef(scope="link", name=name, target=link_ref),
            val,
        )


