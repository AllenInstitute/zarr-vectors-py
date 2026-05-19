"""High-level edit API: :class:`EditSession` and free functions.

This module provides the two surfaces approved in the plan:

1. **Free functions** (``edit_vertex``, ``add_vertex``, ``remove_vertex``,
   ``edit_link``, ``edit_attribute``, ``edit_object``, ...) — apply
   immediately, no pyramid refresh, ``atomic=True`` by default.  Each
   call commits in one transaction when the store is icechunk-backed.

2. :class:`EditSession` — a context manager that buffers edits into a
   change set keyed by ``(level, chunk_coords)``, coalesces multiple
   edits to the same chunk into a single read-modify-write, optionally
   refreshes the pyramid on exit, and commits once on flush.

Both surfaces share the same underlying ``_apply_*`` helpers so
behaviour is consistent.

Edit semantics (atomic vs minimal, source-row retention, propagate-to-
objects, refresh policy) are documented in the approved plan at
``.claude/plans/i-need-to-consider-zany-wand.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import EditError
from zarr_vectors.ops.change_set import (
    ChunkChangeBuilder,
    CrossChunkLinkOp,
    EditReport,
    ManifestOp,
    OidPrefix,
)
from zarr_vectors.ops.refs import (
    AttributeRef,
    CrossChunkLinkRef,
    FragmentRef,
    LinkRef,
    ObjectRef,
    VertexRef,
)
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


RefreshPolicy = Literal[True, False, "batch"]
"""Allowed values for ``EditSession.refresh_pyramid``."""

PropagateTo = Literal["all"] | list[int]
"""Either the sentinel ``"all"`` or an explicit list of OIDs."""

# Sentinel used in ``EditSession._fragment_owners`` to mark which
# levels have been folded into the index.  The value is a chunk-coord
# tuple that can never appear in a real manifest (a chunk index of
# ``-1``); together with the ``-1`` fragment slot in the key, this
# never collides with a real fragment-owner entry.
_LEVEL_MARKER_CHUNK: ChunkCoords = (-1,)


class EditSession:
    """Transactional edit handle for one ZV store.

    Acquire with ``with EditSession(root, atomic=True, refresh_pyramid=
    "batch") as ed: ...``.  Inside the block every ``ed.edit_*`` /
    ``ed.add_*`` / ``ed.remove_*`` call accumulates an edit in the
    in-memory change set.  On ``__exit__`` (clean) the session:

    1. Flushes every dirty chunk via ``Group.batched_writes()``.
    2. Applies pending cross-chunk-link and object-manifest ops.
    3. Runs ``rebuild_pyramid_from_level`` if ``refresh_pyramid``
       requires it.
    4. Calls ``commit(root, message)`` once on icechunk-backed stores.

    Exceptions inside the block trigger a discard: the change set is
    dropped and ``discard_changes`` is called on the icechunk session
    so no partial state leaks to the next snapshot.
    """

    def __init__(
        self,
        root: Group,
        *,
        atomic: bool = True,
        refresh_pyramid: RefreshPolicy = False,
        propagate_to_objects: PropagateTo = "all",
        message: str = "zarr-vectors edit session",
        concurrent_writers: bool = False,
        oid_prefix: OidPrefix | tuple[str, int] | tuple[int, int] | None = None,
        auto_materialise_links: bool = True,
    ) -> None:
        if refresh_pyramid not in (True, False, "batch"):
            raise EditError(
                f"refresh_pyramid must be one of True / False / 'batch'; "
                f"got {refresh_pyramid!r}"
            )
        self.root = root
        self.atomic = bool(atomic)
        self.refresh_pyramid: RefreshPolicy = refresh_pyramid
        self.propagate_to_objects: PropagateTo = propagate_to_objects
        self.message = message
        self.concurrent_writers = bool(concurrent_writers)
        self.oid_prefix: OidPrefix | None = _coerce_oid_prefix(oid_prefix)
        # When True (the default), a link edit that needs an explicit
        # convention (e.g. add_link / remove_link / split-at-link on
        # an ``implicit_sequential*`` store) auto-runs
        # ``materialise_object_links_explicit`` for every object at the
        # affected level and flips ``links_convention`` to ``"explicit"``
        # in-place, then proceeds with the edit and emits one
        # UserWarning to make the conversion visible.  Set to False
        # for fail-fast (raises EditError) behaviour.
        self.auto_materialise_links = bool(auto_materialise_links)

        # Per-(level, chunk) builders, lazily filled.
        self._builders: dict[tuple[int, ChunkCoords], ChunkChangeBuilder] = {}
        # Per-level pending CCL operations.
        self._ccl_ops: list[CrossChunkLinkOp] = []
        # Per-level pending manifest ops, keyed by (level, oid).
        self._manifest_ops: dict[tuple[int, int], ManifestOp] = {}
        # Per-level manifest cache (lazy, populated on first lookup).
        self._all_manifests: dict[int, list[ObjectManifest]] = {}
        # Per-level next-available OID (atomic mode appends new OIDs here).
        self._next_oid: dict[int, int] = {}
        # Fragment → OIDs inverted index (lazy, built on first lookup
        # in ``_fragment_owners_for``).  Keyed by
        # ``(level, chunk_coords, fragment_index)``.  Kept consistent
        # with on-disk + pending-op state via surgical updates in
        # ``_stage_manifest``.
        self._fragment_owners: dict[
            tuple[int, ChunkCoords, int], list[int]
        ] | None = None
        # Per-level attribute-name cache (populated by
        # ``relocate._list_attr_names_from_store`` and invalidated when
        # a new per-vertex attribute is created during the session).
        self._attr_names_per_level: dict[int, list[str]] = {}

        self._report = EditReport(oid_prefix=self.oid_prefix)
        self._touched_levels: set[int] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> EditSession:
        from zarr_vectors.core.store import session_for
        if (
            self.concurrent_writers
            and session_for(self.root) is None
        ):
            from zarr_vectors.exceptions import ConcurrentEditError
            raise ConcurrentEditError(
                "concurrent_writers=True requires an icechunk-backed "
                "store; non-transactional backends are single-writer "
                "only."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self._discard()
            return
        try:
            self.flush()
        except Exception:
            self._discard()
            raise

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def report(self) -> EditReport:
        """Cumulative :class:`EditReport` for this session."""
        return self._report

    def change_set(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the pending diff.

        Useful for review / queueing pipelines that want to inspect the
        edits before flush.  Mutating the returned dict does *not*
        affect the session.
        """
        return {
            "dirty_chunks": [
                {"level": lv, "chunk": list(cc), "appended_fragments": list(
                    b.appended_fragments,
                )}
                for (lv, cc), b in self._builders.items()
                if b.is_dirty()
            ],
            "ccl_ops": [
                {"op": op.op, "delta": op.delta, "index": op.index}
                for op in self._ccl_ops
            ],
            "manifest_ops": [
                {
                    "level": op.level,
                    "object_id": op.object_id,
                    "new_oid": op.new_oid,
                    "deleted": op.new_manifest is None,
                }
                for op in self._manifest_ops.values()
            ],
            "atomic": self.atomic,
            "refresh_pyramid": self.refresh_pyramid,
        }

    # ------------------------------------------------------------------
    # Vertex edits
    # ------------------------------------------------------------------

    def edit_vertex(
        self,
        ref: VertexRef,
        *,
        new_pos: npt.ArrayLike | None = None,
        new_attrs: dict[str, npt.ArrayLike] | None = None,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> None:
        """Edit one vertex.

        Args:
            ref: Physical :class:`VertexRef` (use ``VertexRef.from_object``
                / ``VertexRef.from_position`` if you don't have one).
            new_pos: New position.  When ``None``, only attributes change.
            new_attrs: ``{attr_name: new_value}`` to update alongside
                the position.
            atomic: Override the session default for this edit.
            propagate_to_objects: Override the session default.
        """
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )

        if new_pos is None and not new_attrs:
            return  # noop

        if new_pos is None:
            # Pure attribute edit on the existing row; no fragment churn,
            # no manifest changes regardless of atomic flag.
            builder = self._builder(ref.level, ref.chunk)
            self._ensure_attrs_loaded(builder, list((new_attrs or {}).keys()))
            self._overwrite_attributes_in_place(
                builder, ref.fragment, ref.local,
                {k: np.asarray(v) for k, v in (new_attrs or {}).items()},
            )
            self._mark_edit(ref.level)
            return

        new_pos_arr = np.asarray(new_pos, dtype=np.float64).reshape(-1)
        # Detect chunk-cross.
        new_cc = _chunk_for(self.root, new_pos_arr)
        if tuple(new_cc) == tuple(ref.chunk):
            self._edit_vertex_in_chunk(
                ref, new_pos_arr, new_attrs,
                atomic=atomic_eff, propagate=prop_eff,
            )
        else:
            from zarr_vectors.ops.relocate import relocate_vertex_in_session
            relocate_vertex_in_session(
                self, ref, new_pos_arr, new_cc,
                new_attrs=new_attrs,
                atomic=atomic_eff, propagate=prop_eff,
            )
        self._mark_edit(ref.level)

    def remove_vertex(
        self,
        ref: VertexRef,
        *,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> None:
        """Remove one vertex row.

        Under ``atomic=True`` this is implemented as a soft-delete:
        the row stays but every referring object's manifest is updated
        to skip it (new OIDs allocated for the propagated objects).
        Under ``atomic=False`` the row is physically deleted; subsequent
        rows in the fragment shift down and link/manifest indices
        targeting later rows are rewritten.  Today we only support the
        atomic path; ``atomic=False`` removal is rejected with a clear
        error message to avoid silent index drift.
        """
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )
        if not atomic_eff:
            raise EditError(
                "remove_vertex(atomic=False) requires shifting downstream "
                "row indices in links and manifests; this destructive path "
                "is not implemented yet.  Use atomic=True (soft-delete) or "
                "remove the whole object via remove_object()."
            )
        # Atomic soft-delete: update manifests so propagated objects no
        # longer point at (chunk, fragment, local).  Keep the row on disk
        # so non-propagated OIDs still see it.
        affected_oids = self._oids_referencing(ref.level, ref.chunk, ref.fragment)
        targets = self._select_targets(affected_oids, prop_eff)
        for oid in targets:
            manifest = self._get_manifest(ref.level, oid)
            new_manifest = [
                (cc, fi) for (cc, fi) in manifest
                if not (tuple(cc) == tuple(ref.chunk) and fi == ref.fragment)
            ]
            self._stage_manifest(ref.level, oid, new_manifest, atomic=True)
        self._mark_edit(ref.level)

    def add_vertex(
        self,
        *,
        level: int,
        pos: npt.ArrayLike,
        attrs: dict[str, npt.ArrayLike] | None = None,
        object_id: int | None = None,
        manifest_position: int | None = None,
    ) -> VertexRef:
        """Add a new vertex at ``pos`` to ``level``.

        The vertex goes into the chunk that owns ``pos`` (per
        ``floor(pos / chunk_shape)``) as a new single-row fragment.

        When ``object_id`` is given, the new fragment is appended to
        that object's manifest by default.  Pass ``manifest_position=K``
        to splice the fragment at index ``K`` of the manifest instead —
        under ``links_convention="implicit_sequential*"`` this is the
        right way to insert a new vertex into the middle of a sequence
        (e.g. inserting H between C and D in ``[A,B,C,D,E,F]`` so the
        reader reconstructs ``A->B->C->H->D->E->F``).
        """
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        cc = _chunk_for(self.root, pos_arr)
        builder = self._builder(level, cc)
        new_frag = builder.append_fragment(
            pos_arr[np.newaxis, :],
            attrs={
                k: np.atleast_1d(np.asarray(v))
                for k, v in (attrs or {}).items()
            },
        )
        if object_id is not None:
            manifest = self._get_manifest(level, object_id)
            entry = (tuple(cc), new_frag)
            if manifest_position is None:
                new_manifest = list(manifest) + [entry]
            else:
                if manifest_position < 0 or manifest_position > len(manifest):
                    raise EditError(
                        f"add_vertex(manifest_position={manifest_position}): "
                        f"out of range [0, {len(manifest)}] for object "
                        f"{object_id}"
                    )
                new_manifest = (
                    list(manifest[:manifest_position])
                    + [entry]
                    + list(manifest[manifest_position:])
                )
            self._stage_manifest(level, object_id, new_manifest, atomic=False)
        elif manifest_position is not None:
            raise EditError(
                "add_vertex(manifest_position=...) requires object_id= "
                "to identify which manifest to splice into."
            )
        self._mark_edit(level)
        return VertexRef(level=level, chunk=tuple(cc), fragment=new_frag, local=0)

    # ------------------------------------------------------------------
    # Link edits
    # ------------------------------------------------------------------

    def edit_link(
        self,
        ref: LinkRef,
        *,
        new_endpoints: tuple[int, int] | None = None,
        new_attrs: dict[str, npt.ArrayLike] | None = None,
        atomic: bool | None = None,
        update_objects: bool = False,
    ) -> None:
        from zarr_vectors.ops.links import edit_link_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        edit_link_in_session(
            self, ref,
            new_endpoints=new_endpoints,
            new_attrs=new_attrs,
            atomic=atomic_eff,
            update_objects=update_objects,
        )

    def add_link(
        self,
        *,
        level: int,
        src: int,
        dst: int,
        chunk: ChunkCoords | None = None,
        fragment: int | None = None,
        delta: int = 0,
        attrs: dict[str, npt.ArrayLike] | None = None,
        update_objects: bool = False,
    ) -> LinkRef | CrossChunkLinkRef:
        from zarr_vectors.ops.links import add_link_in_session
        return add_link_in_session(
            self,
            level=level, src=src, dst=dst,
            chunk=chunk, fragment=fragment,
            delta=delta, attrs=attrs,
            update_objects=update_objects,
        )

    def remove_link(
        self,
        ref: LinkRef,
        *,
        atomic: bool | None = None,
        update_objects: bool = False,
    ) -> None:
        from zarr_vectors.ops.links import remove_link_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        remove_link_in_session(
            self, ref, atomic=atomic_eff, update_objects=update_objects,
        )

    def edit_cross_chunk_link(
        self,
        ref: CrossChunkLinkRef,
        *,
        new_endpoints: list[tuple[ChunkCoords, int]],
        atomic: bool | None = None,
    ) -> None:
        from zarr_vectors.ops.links import edit_cross_chunk_link_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        edit_cross_chunk_link_in_session(
            self, ref, new_endpoints=new_endpoints, atomic=atomic_eff,
        )

    def add_cross_chunk_link(
        self,
        *,
        level: int,
        endpoints: list[tuple[ChunkCoords, int]],
        delta: int = 0,
    ) -> CrossChunkLinkRef:
        from zarr_vectors.ops.links import add_cross_chunk_link_in_session
        return add_cross_chunk_link_in_session(
            self, level=level, endpoints=endpoints, delta=delta,
        )

    def remove_cross_chunk_link(
        self,
        ref: CrossChunkLinkRef,
    ) -> None:
        from zarr_vectors.ops.links import remove_cross_chunk_link_in_session
        remove_cross_chunk_link_in_session(self, ref)

    # ------------------------------------------------------------------
    # Attribute edits
    # ------------------------------------------------------------------

    def edit_attribute(self, ref: AttributeRef, value: npt.ArrayLike) -> None:
        from zarr_vectors.ops.attributes import edit_attribute_in_session
        edit_attribute_in_session(self, ref, value)

    def add_attribute(self, ref: AttributeRef, value: npt.ArrayLike) -> None:
        from zarr_vectors.ops.attributes import add_attribute_in_session
        add_attribute_in_session(self, ref, value)

    def remove_attribute(self, ref: AttributeRef) -> None:
        from zarr_vectors.ops.attributes import remove_attribute_in_session
        remove_attribute_in_session(self, ref)

    # ------------------------------------------------------------------
    # Object edits
    # ------------------------------------------------------------------

    def edit_object(
        self,
        oid: int,
        *,
        level: int = 0,
        new_manifest: list[tuple[ChunkCoords, int]] | None = None,
        atomic: bool | None = None,
    ) -> int | None:
        from zarr_vectors.ops.objects import edit_object_manifest_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        if new_manifest is None:
            raise EditError(
                "edit_object: new_manifest is required.  Per-object "
                "attribute edits go through edit_attribute(AttributeRef "
                "(scope='object', ...))."
            )
        return edit_object_manifest_in_session(
            self, oid=oid, level=level,
            new_manifest=new_manifest, atomic=atomic_eff,
        )

    def add_object(
        self,
        *,
        level: int,
        vertices: npt.ArrayLike,
        attrs: dict[str, npt.ArrayLike] | None = None,
    ) -> ObjectRef:
        from zarr_vectors.ops.objects import add_object_in_session
        return add_object_in_session(
            self, level=level, vertices=vertices, attrs=attrs,
        )

    def remove_object(
        self,
        oid: int,
        *,
        level: int = 0,
        atomic: bool | None = None,
    ) -> int | None:
        from zarr_vectors.ops.objects import remove_object_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        return remove_object_in_session(
            self, oid=oid, level=level, atomic=atomic_eff,
        )

    # ------------------------------------------------------------------
    # Fragment edits
    # ------------------------------------------------------------------

    def edit_fragment(
        self,
        ref: FragmentRef,
        *,
        new_vertices: npt.ArrayLike,
        new_attrs: dict[str, npt.ArrayLike] | None = None,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> None:
        from zarr_vectors.ops.fragments import edit_fragment_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )
        edit_fragment_in_session(
            self, ref,
            new_vertices=new_vertices, new_attrs=new_attrs,
            atomic=atomic_eff, propagate=prop_eff,
        )

    def add_fragment(
        self,
        *,
        level: int,
        chunk: ChunkCoords,
        vertices: npt.ArrayLike,
        attrs: dict[str, npt.ArrayLike] | None = None,
    ) -> FragmentRef:
        from zarr_vectors.ops.fragments import add_fragment_in_session
        return add_fragment_in_session(
            self, level=level, chunk=chunk, vertices=vertices, attrs=attrs,
        )

    def remove_fragment(
        self,
        ref: FragmentRef,
        *,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> None:
        from zarr_vectors.ops.fragments import remove_fragment_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )
        remove_fragment_in_session(self, ref, atomic=atomic_eff, propagate=prop_eff)

    def split_fragment(
        self,
        ref: FragmentRef,
        *,
        row_partition,
        atomic: bool | None = None,
        propagate_to_objects: PropagateTo | None = None,
    ) -> list[FragmentRef]:
        """Physically slice ``ref`` into N fragments by row_partition.

        See [zarr_vectors/ops/fragments.py:split_fragment_in_session]
        and [zarr_vectors/ops/graph.py:partition_fragment_rows] for the
        full contract.  ``row_partition`` is either a list of N
        row-index arrays or a 1-D label array assigning each row to a
        slice.  Returns the new :class:`FragmentRef` list in slice
        order.
        """
        from zarr_vectors.ops.fragments import split_fragment_in_session
        atomic_eff = self.atomic if atomic is None else bool(atomic)
        prop_eff = (
            self.propagate_to_objects
            if propagate_to_objects is None
            else propagate_to_objects
        )
        return split_fragment_in_session(
            self, ref,
            row_partition=row_partition,
            atomic=atomic_eff, propagate=prop_eff,
        )

    # ------------------------------------------------------------------
    # Internal vertex-edit helpers
    # ------------------------------------------------------------------

    def _edit_vertex_in_chunk(
        self,
        ref: VertexRef,
        new_pos: npt.NDArray[np.floating],
        new_attrs: dict[str, npt.ArrayLike] | None,
        *,
        atomic: bool,
        propagate: PropagateTo,
    ) -> None:
        builder = self._builder(ref.level, ref.chunk)
        if not atomic:
            # Minimal overwrite — row indices stay; manifests untouched.
            self._ensure_attrs_loaded(
                builder, list((new_attrs or {}).keys()),
            )
            builder.overwrite_vertex_row(
                ref.fragment, ref.local, new_pos,
                new_attrs={
                    k: np.asarray(v) for k, v in (new_attrs or {}).items()
                },
            )
            return

        # Atomic: append a new single-row fragment, update propagated
        # object manifests to point at it.
        attrs_dict: dict[str, npt.NDArray] = {}
        if new_attrs:
            for name, val in new_attrs.items():
                attrs_dict[name] = np.atleast_1d(np.asarray(val))
        new_frag = builder.append_fragment(
            new_pos[np.newaxis, :],
            attrs=attrs_dict or None,
        )

        affected_oids = self._oids_referencing(
            ref.level, ref.chunk, ref.fragment,
        )
        targets = self._select_targets(affected_oids, propagate)
        for oid in targets:
            manifest = self._get_manifest(ref.level, oid)
            new_manifest = [
                (tuple(ref.chunk), new_frag) if (
                    tuple(cc) == tuple(ref.chunk) and fi == ref.fragment
                ) else (cc, fi)
                for (cc, fi) in manifest
            ]
            self._stage_manifest(
                ref.level, oid, new_manifest, atomic=True,
            )

    def _overwrite_attributes_in_place(
        self,
        builder: ChunkChangeBuilder,
        fragment: int,
        local: int,
        attrs: dict[str, npt.NDArray],
    ) -> None:
        for name, val in attrs.items():
            attr_list = builder.attr_groups.get(name)
            if attr_list is None:
                continue
            attr_list[fragment][local] = val.astype(
                builder.attr_dtype.get(name, np.float32),
                copy=False,
            )
            builder.attrs_dirty[name] = True

    def _ensure_attrs_loaded(
        self,
        builder: ChunkChangeBuilder,
        names: Iterable[str],
    ) -> None:
        for name in names:
            builder.require_attribute(self.root, name)

    # ------------------------------------------------------------------
    # Manifest / OID bookkeeping
    # ------------------------------------------------------------------

    def _all_manifests_for(self, level: int) -> list[ObjectManifest]:
        if level in self._all_manifests:
            return self._all_manifests[level]
        from zarr_vectors.core.arrays import read_all_object_manifests
        from zarr_vectors.core.store import get_resolution_level
        level_group = get_resolution_level(self.root, level)
        try:
            manifests = read_all_object_manifests(level_group)
        except Exception:
            manifests = []
        self._all_manifests[level] = manifests
        self._next_oid[level] = len(manifests)
        return manifests

    def _get_manifest(self, level: int, oid: int) -> ObjectManifest:
        # Honour any pending in-session edit for this OID first.
        pending = self._manifest_ops.get((level, oid))
        if pending is not None and pending.new_manifest is not None:
            return list(pending.new_manifest)
        manifests = self._all_manifests_for(level)
        if oid < 0 or oid >= len(manifests):
            raise EditError(
                f"object_id {oid} out of range at level {level} "
                f"(have {len(manifests)} OIDs)"
            )
        return list(manifests[oid])

    def _oids_referencing(
        self,
        level: int,
        chunk: ChunkCoords,
        fragment: int,
    ) -> list[int]:
        """Return the OIDs whose manifest references ``(chunk, fragment)``.

        Backed by the lazily-built ``_fragment_owners`` index — O(1)
        per lookup after the first call in a session, vs the previous
        O(N_objects × manifest_len) per-call scan.  The index is kept
        consistent with pending-op state via surgical updates in
        :meth:`_stage_manifest`.
        """
        index = self._build_fragment_owners(level)
        chunk_t = tuple(int(c) for c in chunk)
        return list(index.get((level, chunk_t, int(fragment)), ()))

    def _fragment_owners_for(
        self,
        level: int,
        chunk: ChunkCoords,
        fragment: int,
    ) -> list[int]:
        """Public-flavoured alias of :meth:`_oids_referencing`."""
        return self._oids_referencing(level, chunk, fragment)

    def _build_fragment_owners(
        self,
        level: int,
    ) -> dict[tuple[int, ChunkCoords, int], list[int]]:
        """Return the fragment-owners index, building it on first call.

        The index is keyed by ``(level, chunk_coords, fragment_idx)``
        and maps to a list of OIDs whose manifest references that
        fragment.  Built from the already-cached
        ``_all_manifests_for(level)`` result, so the first lookup pays
        the on-disk read once; subsequent lookups are O(1).

        The index covers *all* known levels — when a new level is
        requested, that level's manifests are folded in alongside the
        existing entries.
        """
        if self._fragment_owners is None:
            self._fragment_owners = {}
        index = self._fragment_owners
        # Has this level already been folded in?  Cheapest sentinel:
        # we add a marker entry so subsequent calls are a constant-time
        # dict-membership check.
        marker = (level, _LEVEL_MARKER_CHUNK, -1)
        if marker in index:
            return index
        index[marker] = []  # sentinel — must not collide with real keys
        manifests = self._all_manifests_for(level)
        for oid, manifest in enumerate(manifests):
            for cc, fi in manifest:
                key = (level, tuple(int(c) for c in cc), int(fi))
                index.setdefault(key, []).append(oid)
        return index

    def _index_apply_manifest_delta(
        self,
        level: int,
        oid: int,
        old_manifest: ObjectManifest,
        new_manifest: ObjectManifest,
    ) -> None:
        """Surgically update ``_fragment_owners`` for one manifest write.

        Removes ``oid`` from entries that appear in ``old_manifest`` but
        not in ``new_manifest``; adds it to entries that appear in
        ``new_manifest`` but not in ``old_manifest``.  Entries present
        in both are untouched.  Atomic-mode callers pass the new-OID's
        old manifest as ``[]`` (treat the whole new manifest as added).

        No-op when the index hasn't been built yet for ``level`` — the
        first lookup will build a fresh index from the cached manifests
        plus any pending ops applied at that point.
        """
        if self._fragment_owners is None:
            return
        marker = (level, _LEVEL_MARKER_CHUNK, -1)
        if marker not in self._fragment_owners:
            return  # level not folded in yet; nothing to keep consistent
        old_set = {
            (tuple(int(c) for c in cc), int(fi)) for cc, fi in old_manifest
        }
        new_set = {
            (tuple(int(c) for c in cc), int(fi)) for cc, fi in new_manifest
        }
        for cc, fi in old_set - new_set:
            key = (level, cc, fi)
            lst = self._fragment_owners.get(key)
            if lst and oid in lst:
                lst.remove(int(oid))
                if not lst:
                    del self._fragment_owners[key]
        for cc, fi in new_set - old_set:
            key = (level, cc, fi)
            self._fragment_owners.setdefault(key, []).append(int(oid))

    def _select_targets(
        self,
        affected_oids: list[int],
        propagate: PropagateTo,
    ) -> list[int]:
        if propagate == "all":
            return list(affected_oids)
        wanted = set(int(o) for o in propagate)
        return [oid for oid in affected_oids if oid in wanted]

    def _stage_manifest(
        self,
        level: int,
        oid: int,
        new_manifest: ObjectManifest | None,
        *,
        atomic: bool,
    ) -> None:
        """Schedule a manifest write.

        Under ``atomic=True`` allocate a new OID via the prefix-aware
        allocator and leave the original OID's manifest untouched;
        record the remap.  Under ``atomic=False`` overwrite the
        manifest at ``oid``.

        Keeps the lazily-built fragment-owners index consistent via a
        surgical diff so it survives the whole session.
        """
        new_mani: ObjectManifest = list(new_manifest or [])
        if atomic:
            new_oid = self._allocate_atomic_oid(level)
            self._manifest_ops[(level, new_oid)] = ManifestOp(
                level=level, object_id=new_oid,
                new_manifest=new_mani, new_oid=new_oid,
            )
            self._report.oid_remap[int(oid)] = int(new_oid)
            # Index update: the new OID inherits all of ``new_mani``'s
            # entries; the source OID's entries are unchanged.
            self._index_apply_manifest_delta(
                level, int(new_oid),
                old_manifest=[],
                new_manifest=new_mani,
            )
        else:
            # Capture the pre-edit manifest for the diff before
            # overwriting the op store.  Honour any pending in-session
            # op at this OID (chains of non-atomic edits compose).
            pending = self._manifest_ops.get((level, oid))
            if pending is not None and pending.new_manifest is not None:
                old_mani = list(pending.new_manifest)
            else:
                manifests = self._all_manifests_for(level)
                old_mani = (
                    list(manifests[oid]) if 0 <= oid < len(manifests) else []
                )
            self._manifest_ops[(level, oid)] = ManifestOp(
                level=level, object_id=oid,
                new_manifest=new_mani, new_oid=None,
            )
            self._index_apply_manifest_delta(
                level, int(oid),
                old_manifest=old_mani,
                new_manifest=new_mani,
            )

    def _allocate_atomic_oid(self, level: int) -> int:
        """Return the next atomic OID for ``level``.

        Without an ``oid_prefix`` this is ``len(manifests)`` and
        increments by 1 each call.  With an ``oid_prefix`` it skips to
        the next OID in the prefix's residue class so two cooperating
        sessions with disjoint prefixes never collide.
        """
        self._all_manifests_for(level)  # populates _next_oid
        if self.oid_prefix is None:
            new_oid = self._next_oid[level]
            self._next_oid[level] = new_oid + 1
            return new_oid
        new_oid = self.oid_prefix.next_after(self._next_oid[level])
        # Advance the next-free cursor past the allocated OID.  Sparse
        # slots between the previous cursor and ``new_oid`` stay empty
        # (write_object_index fills them with empty manifests).
        self._next_oid[level] = new_oid + 1
        return new_oid

    # ------------------------------------------------------------------
    # Builders & touch tracking
    # ------------------------------------------------------------------

    def _builder(self, level: int, chunk: ChunkCoords) -> ChunkChangeBuilder:
        key = (level, tuple(chunk))
        existing = self._builders.get(key)
        if existing is not None:
            return existing
        builder = ChunkChangeBuilder.from_disk(self.root, level, key[1])
        self._builders[key] = builder
        return builder

    def _mark_edit(self, level: int) -> None:
        self._touched_levels.add(level)
        self._report.n_edits += 1
        if self.refresh_pyramid is True:
            self._refresh_now(level)

    # ------------------------------------------------------------------
    # Flush / commit / discard
    # ------------------------------------------------------------------

    def flush(self) -> EditReport:
        """Write every pending edit to disk and (if applicable) commit.

        Returns the cumulative :class:`EditReport`.  Idempotent: calling
        ``flush`` a second time is a no-op once everything is clean.
        """
        if self._closed:
            return self._report

        # 1. Apply dirty chunks via batched_writes for parallelism.
        from zarr_vectors.core.store import commit, get_resolution_level
        from zarr_vectors.core.arrays import (
            write_chunk_attributes,
            write_chunk_links,
            write_chunk_vertices,
        )

        with self.root.batched_writes():
            for (level, cc), builder in self._builders.items():
                if not builder.is_dirty():
                    continue
                level_group = get_resolution_level(self.root, level)
                if builder.vertices_dirty:
                    write_chunk_vertices(
                        level_group, cc, builder.vertex_groups,
                        dtype=builder.vertex_dtype,
                    )
                for delta, dirty in builder.links_dirty.items():
                    if not dirty:
                        continue
                    write_chunk_links(
                        level_group, cc, builder.link_groups[delta],
                        delta=delta,
                    )
                for name, dirty in builder.attrs_dirty.items():
                    if not dirty:
                        continue
                    write_chunk_attributes(
                        level_group, name, cc, builder.attr_groups[name],
                        dtype=builder.attr_dtype.get(name, np.float32),
                    )
                self._report.touched_chunks.append((level, tuple(cc)))

        # 2. Apply CCL ops grouped by (level, delta).
        self._flush_ccl_ops()

        # 3. Apply manifest ops grouped by level.
        self._flush_manifest_ops()

        # 4. Pyramid refresh.
        if self.refresh_pyramid == "batch" and self._touched_levels:
            min_level = min(self._touched_levels)
            self._refresh_now(min_level)
        elif self.refresh_pyramid is False and self._touched_levels:
            min_level = min(self._touched_levels)
            from zarr_vectors.core.store import list_resolution_levels
            all_levels = list_resolution_levels(self.root)
            self._report.dirty_pyramid_levels = [
                lv for lv in all_levels if lv > min_level
            ]

        # 5. Commit (icechunk-only).
        snapshot = commit(self.root, self.message)
        self._report.snapshot_id = snapshot

        self._closed = True
        return self._report

    def _flush_ccl_ops(self) -> None:
        if not self._ccl_ops:
            return
        from zarr_vectors.core.arrays import (
            read_cross_chunk_links,
            write_cross_chunk_links,
        )
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level

        meta = RootMetadata.from_dict(self.root.attrs.to_dict())
        sid_ndim = meta.sid_ndim

        # Group by (level, delta).  Only level 0 is the common edit
        # target today; we still group generically.
        from collections import defaultdict
        groups: dict[tuple[int, int], list[CrossChunkLinkOp]] = defaultdict(list)
        for op in self._ccl_ops:
            groups[(0, op.delta)].append(op)

        for (level, delta), ops in groups.items():
            level_group = get_resolution_level(self.root, level)
            current = read_cross_chunk_links(level_group, delta=delta)
            # Apply ops in submission order.
            rows: list[list[tuple[ChunkCoords, int]]] = [list(r) for r in current]
            for op in ops:
                if op.op == "append":
                    rows.append(list(op.payload or []))
                elif op.op == "delete":
                    if op.index is not None and 0 <= op.index < len(rows):
                        rows.pop(op.index)
                elif op.op == "overwrite":
                    if op.index is not None and 0 <= op.index < len(rows):
                        rows[op.index] = list(op.payload or [])
            write_cross_chunk_links(level_group, rows, sid_ndim, delta=delta)

    def _flush_manifest_ops(self) -> None:
        if not self._manifest_ops:
            return
        from zarr_vectors.core.arrays import write_object_index
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level

        meta = RootMetadata.from_dict(self.root.attrs.to_dict())
        sid_ndim = meta.sid_ndim

        # Group ops by level.
        from collections import defaultdict
        by_level: dict[int, list[ManifestOp]] = defaultdict(list)
        for op in self._manifest_ops.values():
            by_level[op.level].append(op)

        for level, ops in by_level.items():
            level_group = get_resolution_level(self.root, level)
            # Start from disk state then apply ops.
            manifests = list(self._all_manifests_for(level))
            max_oid = len(manifests) - 1
            for op in ops:
                oid = op.new_oid if op.new_oid is not None else op.object_id
                if oid > max_oid:
                    manifests.extend([] for _ in range(oid - max_oid))
                    max_oid = oid
                manifests[oid] = list(op.new_manifest or [])
            manifest_dict = {i: m for i, m in enumerate(manifests)}
            write_object_index(
                level_group, manifest_dict, sid_ndim,
                total_objects=len(manifests),
            )

    def _refresh_now(self, source_level: int) -> None:
        from zarr_vectors.ops.refresh import rebuild_pyramid_from_level
        rebuild_pyramid_from_level(self.root, source_level)

    def _discard(self) -> None:
        from zarr_vectors.core.store import discard_changes
        discard_changes(self.root)
        self._builders.clear()
        self._ccl_ops.clear()
        self._manifest_ops.clear()
        self._closed = True


# ======================================================================
# Free-function wrappers
# ======================================================================

def edit_vertex(
    root: Group,
    ref: VertexRef,
    *,
    new_pos: npt.ArrayLike | None = None,
    new_attrs: dict[str, npt.ArrayLike] | None = None,
    atomic: bool = True,
    propagate_to_objects: PropagateTo = "all",
    message: str = "edit_vertex",
) -> EditReport:
    """Edit one vertex; immediately flush and (if applicable) commit.

    No pyramid refresh — call :func:`zarr_vectors.ops.refresh.
    rebuild_pyramid_from_level` if you need the coarser levels to track.
    """
    with EditSession(
        root,
        atomic=atomic,
        refresh_pyramid=False,
        propagate_to_objects=propagate_to_objects,
        message=message,
    ) as ed:
        ed.edit_vertex(
            ref,
            new_pos=new_pos,
            new_attrs=new_attrs,
        )
    return ed.report


def add_vertex(
    root: Group,
    *,
    level: int,
    pos: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
    object_id: int | None = None,
    message: str = "add_vertex",
) -> tuple[VertexRef, EditReport]:
    """Append a new single-row fragment containing ``pos``."""
    with EditSession(
        root, atomic=True, refresh_pyramid=False, message=message,
    ) as ed:
        ref = ed.add_vertex(
            level=level, pos=pos, attrs=attrs, object_id=object_id,
        )
    return ref, ed.report


def remove_vertex(
    root: Group,
    ref: VertexRef,
    *,
    atomic: bool = True,
    propagate_to_objects: PropagateTo = "all",
    message: str = "remove_vertex",
) -> EditReport:
    """Remove one vertex (soft-delete under ``atomic=True``)."""
    with EditSession(
        root,
        atomic=atomic,
        refresh_pyramid=False,
        propagate_to_objects=propagate_to_objects,
        message=message,
    ) as ed:
        ed.remove_vertex(ref)
    return ed.report


def edit_link(
    root: Group,
    ref: LinkRef,
    *,
    new_endpoints: tuple[int, int] | None = None,
    new_attrs: dict[str, npt.ArrayLike] | None = None,
    atomic: bool = True,
    update_objects: bool = False,
    message: str = "edit_link",
) -> EditReport:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ed.edit_link(
            ref,
            new_endpoints=new_endpoints,
            new_attrs=new_attrs,
            update_objects=update_objects,
        )
    return ed.report


def add_link(
    root: Group,
    *,
    level: int,
    src: int,
    dst: int,
    chunk: ChunkCoords | None = None,
    fragment: int | None = None,
    delta: int = 0,
    attrs: dict[str, npt.ArrayLike] | None = None,
    update_objects: bool = False,
    atomic: bool = True,
    message: str = "add_link",
) -> tuple[LinkRef | CrossChunkLinkRef, EditReport]:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ref = ed.add_link(
            level=level, src=src, dst=dst, chunk=chunk,
            fragment=fragment, delta=delta, attrs=attrs,
            update_objects=update_objects,
        )
    return ref, ed.report


def remove_link(
    root: Group,
    ref: LinkRef,
    *,
    atomic: bool = True,
    update_objects: bool = False,
    message: str = "remove_link",
) -> EditReport:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ed.remove_link(ref, update_objects=update_objects)
    return ed.report


def add_cross_chunk_link(
    root: Group,
    *,
    level: int,
    endpoints: list[tuple[ChunkCoords, int]],
    delta: int = 0,
    message: str = "add_cross_chunk_link",
) -> tuple[CrossChunkLinkRef, EditReport]:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ref = ed.add_cross_chunk_link(level=level, endpoints=endpoints, delta=delta)
    return ref, ed.report


def remove_cross_chunk_link(
    root: Group,
    ref: CrossChunkLinkRef,
    *,
    message: str = "remove_cross_chunk_link",
) -> EditReport:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ed.remove_cross_chunk_link(ref)
    return ed.report


def edit_attribute(
    root: Group,
    ref: AttributeRef,
    value: npt.ArrayLike,
    *,
    message: str = "edit_attribute",
) -> EditReport:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ed.edit_attribute(ref, value)
    return ed.report


def add_attribute(
    root: Group,
    ref: AttributeRef,
    value: npt.ArrayLike,
    *,
    message: str = "add_attribute",
) -> EditReport:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ed.add_attribute(ref, value)
    return ed.report


def remove_attribute(
    root: Group,
    ref: AttributeRef,
    *,
    message: str = "remove_attribute",
) -> EditReport:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ed.remove_attribute(ref)
    return ed.report


def edit_object(
    root: Group,
    oid: int,
    *,
    level: int = 0,
    new_manifest: list[tuple[ChunkCoords, int]] | None = None,
    atomic: bool = True,
    message: str = "edit_object",
) -> EditReport:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ed.edit_object(oid, level=level, new_manifest=new_manifest)
    return ed.report


def add_object(
    root: Group,
    *,
    level: int,
    vertices: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
    message: str = "add_object",
) -> tuple[ObjectRef, EditReport]:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ref = ed.add_object(level=level, vertices=vertices, attrs=attrs)
    return ref, ed.report


def remove_object(
    root: Group,
    oid: int,
    *,
    level: int = 0,
    atomic: bool = True,
    message: str = "remove_object",
) -> EditReport:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ed.remove_object(oid, level=level)
    return ed.report


def edit_fragment(
    root: Group,
    ref: FragmentRef,
    *,
    new_vertices: npt.ArrayLike,
    new_attrs: dict[str, npt.ArrayLike] | None = None,
    atomic: bool = True,
    message: str = "edit_fragment",
) -> EditReport:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ed.edit_fragment(ref, new_vertices=new_vertices, new_attrs=new_attrs)
    return ed.report


def add_fragment(
    root: Group,
    *,
    level: int,
    chunk: ChunkCoords,
    vertices: npt.ArrayLike,
    attrs: dict[str, npt.ArrayLike] | None = None,
    message: str = "add_fragment",
) -> tuple[FragmentRef, EditReport]:
    with EditSession(root, atomic=True, refresh_pyramid=False, message=message) as ed:
        ref = ed.add_fragment(level=level, chunk=chunk, vertices=vertices, attrs=attrs)
    return ref, ed.report


def remove_fragment(
    root: Group,
    ref: FragmentRef,
    *,
    atomic: bool = True,
    message: str = "remove_fragment",
) -> EditReport:
    with EditSession(root, atomic=atomic, refresh_pyramid=False, message=message) as ed:
        ed.remove_fragment(ref)
    return ed.report


def split_fragment(
    root: Group,
    ref: FragmentRef,
    *,
    row_partition,
    atomic: bool = True,
    propagate_to_objects: PropagateTo = "all",
    message: str = "split_fragment",
) -> tuple[list[FragmentRef], EditReport]:
    """Physically slice ``ref`` into N fragments by ``row_partition``.

    See ``EditSession.split_fragment`` for the contract.  Returns
    ``(new_refs, report)`` where ``new_refs`` is the list of
    :class:`FragmentRef` objects in slice order.
    """
    with EditSession(
        root, atomic=atomic, refresh_pyramid=False,
        propagate_to_objects=propagate_to_objects, message=message,
    ) as ed:
        new_refs = ed.split_fragment(
            ref, row_partition=row_partition,
        )
    return new_refs, ed.report


# ======================================================================
# Internal helpers
# ======================================================================

def _chunk_for(root: Group, pos: npt.NDArray[np.floating]) -> ChunkCoords:
    """Return ``floor(pos / chunk_shape)`` as a tuple."""
    from zarr_vectors.core.metadata import RootMetadata
    from zarr_vectors.spatial.chunking import compute_chunk_coords

    meta = RootMetadata.from_dict(root.attrs.to_dict())
    chunk_shape = tuple(meta.chunk_shape)
    return compute_chunk_coords(np.asarray(pos, dtype=np.float64), chunk_shape)


def _coerce_oid_prefix(
    spec: OidPrefix | tuple[str, int] | tuple[int, int] | None,
) -> OidPrefix | None:
    """Accept the three constructor shapes for ``oid_prefix=``."""
    if spec is None or isinstance(spec, OidPrefix):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2:
        first, modulus = spec
        if isinstance(first, str):
            return OidPrefix.from_name(first, int(modulus))
        if isinstance(first, int):
            return OidPrefix(residue=int(first), modulus=int(modulus))
    raise EditError(
        f"oid_prefix must be OidPrefix, (name, k), (residue, k), or None; "
        f"got {spec!r}"
    )
