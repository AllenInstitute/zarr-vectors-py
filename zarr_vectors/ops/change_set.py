"""In-memory representation of pending edits.

A :class:`ChunkChangeBuilder` lazily decodes one chunk's ragged
state (vertices, fragment sidecar, per-vertex attributes, links) the
first time the edit engine touches it, lets callers mutate the decoded
state in Python, and re-encodes everything in one shot at flush time.
This is what gives the :class:`~zarr_vectors.ops.edit.EditSession` its
"coalesce many edits to the same chunk into one read-modify-write"
property.

The builder is the source chunk's view of ``links/<delta>/<offsets>/``:
it holds one entry per **cell** it has touched, keyed by
``(delta, offsets)`` — the pair naming the array — with the builder's
own chunk as the cell.  Intra- and cross-chunk links are the same
family and flush through the same per-chunk path; ``offsets is None``
is the all-zero (intra) array.

:class:`EditReport` is the user-facing summary of what changed in a
session: touched chunks, OID remap (atomic edits), dirty pyramid
levels, and (on icechunk) the snapshot id.  It is also the
serialisable diff returned by ``EditSession.change_set()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.paths import is_intra
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


# One touched ``links/<delta>/<offsets>/`` array, as seen from the
# builder's chunk (which is the cell).  ``offsets is None`` is the
# all-zero intra array — the single canonical spelling, so a cell has
# exactly one key.
LinkCell = tuple[int, tuple[ChunkCoords, ...] | None]


def link_cell(
    delta: int, offsets: Sequence[ChunkCoords] | None = None,
) -> LinkCell:
    """Normalise ``(delta, offsets)`` into a :data:`LinkCell` key.

    All-zero offsets collapse to ``None``: they name the intra array, so
    both spellings must key the same builder entry.
    """
    if offsets is None:
        return (int(delta), None)
    norm = tuple(tuple(int(c) for c in offset) for offset in offsets)
    return (int(delta), None if is_intra(norm) else norm)


@dataclass
class ChunkChangeBuilder:
    """Mutable in-memory state for one ``(level, chunk_coords)`` chunk.

    The builder holds three parallel kinds of state:

    - ``vertex_groups``: list of fragments, each a ``(N_k, D)`` float
      array.  The list's length is the chunk's fragment count.
    - ``link_groups``: ``{LinkCell: list_of_row_groups}`` — the decoded
      rows of every ``links/<delta>/<offsets>/`` cell this chunk sources.
      Row groups of the intra cell are kept 1:1 with ``vertex_groups``;
      other cells carry no such alignment and their rows may lead with a
      ``perm_idx`` column, so only the intra cell is safe to interpret as
      chunk-local ``(src, dst)`` pairs.
    - ``attr_groups``: ``{attr_name: list_of_fragment_arrays}`` aligned
      with ``vertex_groups``.  Empty when no per-vertex attributes are
      touched.

    Methods that mutate state — :meth:`append_fragment`,
    :meth:`overwrite_vertex_row`, :meth:`drop_fragment` — flip
    ``dirty=True`` so the flush phase knows the chunk needs a rewrite.

    A builder is created lazily by :class:`~zarr_vectors.ops.edit.
    EditSession` when it first sees an edit targeting this chunk; if
    the chunk doesn't exist on disk yet (an ``add_vertex`` to a fresh
    chunk), the builder starts with empty lists.
    """

    level: int
    chunk: ChunkCoords
    vertex_dtype: np.dtype
    vertex_ndim: int

    vertex_groups: list[npt.NDArray[np.floating]] = field(default_factory=list)
    link_groups: dict[LinkCell, list[npt.NDArray[np.integer]]] = field(
        default_factory=dict,
    )
    # Per-cell ``(link_width, directed, store)`` captured on first touch.
    # The flush needs them to re-create the array with the family's own
    # policy rather than the writer defaults, and to know the physical row
    # width (``links_has_perm``).
    link_policy: dict[LinkCell, tuple[int, bool, str]] = field(
        default_factory=dict,
    )
    attr_groups: dict[str, list[npt.NDArray]] = field(default_factory=dict)

    # Per-attribute dtype captured on first touch so the encoder uses the
    # store's on-disk dtype (and a future encoder hop doesn't widen it).
    attr_dtype: dict[str, np.dtype] = field(default_factory=dict)
    # Per-attribute ncols (1 for scalar attrs, K for vector attrs).
    attr_ncols: dict[str, int] = field(default_factory=dict)

    # When True, the flush phase rewrites ``vertices/<cc>`` +
    # ``vertex_fragments/<cc>`` for this chunk.
    vertices_dirty: bool = False
    # ``{LinkCell: True}`` for link cells that need rewrite.
    links_dirty: dict[LinkCell, bool] = field(default_factory=dict)
    # Per-attribute dirty flags.
    attrs_dirty: dict[str, bool] = field(default_factory=dict)

    # Fragments that were appended in this session (used by the
    # object-manifest update logic to translate "local in old fragment"
    # references into "local 0 in new fragment").
    appended_fragments: list[int] = field(default_factory=list)

    @classmethod
    def from_disk(
        cls,
        root: Group,
        level: int,
        chunk: ChunkCoords,
    ) -> ChunkChangeBuilder:
        """Decode the chunk's vertex + fragment-sidecar state from disk.

        Per-vertex attributes and link arrays are *not* eagerly decoded
        — they're loaded by :meth:`require_attribute` / :meth:`require_links`
        the first time an edit touches them.
        """
        from zarr_vectors.core.arrays import read_chunk_vertices
        from zarr_vectors.core.metadata import RootMetadata
        from zarr_vectors.core.store import get_resolution_level

        meta = RootMetadata.from_dict(root.attrs.to_dict())
        ndim = meta.sid_ndim
        level_group = get_resolution_level(root, level)

        try:
            vmeta = level_group.read_array_meta("vertices")
            vdtype = np.dtype(vmeta.get("dtype", "float32"))
        except Exception:
            vdtype = np.dtype(np.float32)

        try:
            groups = read_chunk_vertices(
                level_group, chunk, dtype=vdtype, ndim=ndim,
            )
        except Exception:
            # Chunk doesn't exist yet on disk — start empty.
            groups = []

        return cls(
            level=level,
            chunk=tuple(int(c) for c in chunk),
            vertex_dtype=vdtype,
            vertex_ndim=ndim,
            vertex_groups=[g.astype(vdtype, copy=True) for g in groups],
        )

    def require_attribute(
        self,
        root: Group,
        name: str,
    ) -> list[npt.NDArray]:
        """Lazily decode per-vertex attribute ``name`` for this chunk.

        Returns the existing fragment-aligned list.  Subsequent calls
        return the same in-memory list (mutations stick).
        """
        if name in self.attr_groups:
            return self.attr_groups[name]

        from zarr_vectors.core.arrays import read_chunk_attributes
        from zarr_vectors.core.store import get_resolution_level

        level_group = get_resolution_level(root, self.level)
        try:
            ameta = level_group.read_array_meta(
                f"vertex_attributes/{name}",
            )
            adtype = np.dtype(ameta.get("dtype", "float32"))
            shape = ameta.get("shape", [])
            ncols = int(shape[-1]) if len(shape) >= 2 else 1
        except Exception:
            adtype = np.dtype(np.float32)
            ncols = 1

        try:
            groups = read_chunk_attributes(
                level_group, name, self.chunk,
                dtype=adtype, ncols=ncols,
                vert_dtype=self.vertex_dtype,
                vert_ndim=self.vertex_ndim,
            )
        except Exception:
            groups = [
                np.zeros((g.shape[0], ncols) if ncols > 1 else (g.shape[0],),
                         dtype=adtype)
                for g in self.vertex_groups
            ]

        self.attr_groups[name] = [g.astype(adtype, copy=True) for g in groups]
        self.attr_dtype[name] = adtype
        self.attr_ncols[name] = ncols
        return self.attr_groups[name]

    def require_links(
        self,
        root: Group,
        delta: int = 0,
        link_width: int = 2,
        offsets: Sequence[ChunkCoords] | None = None,
    ) -> list[npt.NDArray[np.integer]]:
        """Lazily decode one ``links/<delta>/<offsets>/`` cell for this chunk.

        ``offsets`` names the array; ``None`` (default) is the all-zero
        intra one.  This chunk is always the cell — i.e. the **source**
        chunk of every record decoded here.

        For the intra cell the result is padded to one group per vertex
        fragment: link rows there are chunk-local, so the edit engine
        indexes them per vertex fragment, and the on-disk table may hold
        fewer groups (e.g. a graph writer that consolidated every edge
        into one link fragment).  ``write_chunk_links`` does not require
        1:1 alignment — the padding is internal bookkeeping.

        Other cells get no padding: their rows have no per-vertex-fragment
        meaning.  An absent one starts as a single empty group so callers
        have a group 0 to append into.
        """
        cell = link_cell(delta, offsets)
        if cell in self.link_groups:
            return self.link_groups[cell]

        from zarr_vectors.core.arrays import (
            _decode_link_cell,
            link_family_policy,
            links_has_perm,
        )
        from zarr_vectors.core.paths import intra_offsets
        from zarr_vectors.core.store import get_resolution_level

        level_group = get_resolution_level(root, self.level)

        # The family group carries the policy every cell under it shares.
        # Absent means the family is this session's to create.
        policy = link_family_policy(level_group, delta)
        if policy is not None:
            link_width, _sid_ndim, directed, store = policy
        else:
            directed, store = False, "canonical"
        self.link_policy[cell] = (int(link_width), bool(directed), str(store))

        resolved = (
            intra_offsets(len(self.chunk), link_width)
            if cell[1] is None else cell[1]
        )
        # Physical width: links_has_perm is the single definition writer
        # and reader consult, so the decode never guesses.
        width = link_width + (
            1 if links_has_perm(
                resolved, delta=delta, directed=directed, store=store,
            ) else 0
        )
        groups = _decode_link_cell(
            level_group, self.chunk, delta=delta, offsets=resolved,
            dtype=np.int64, width=width, default=[],
        )
        decoded: list[npt.NDArray[np.integer]] = [
            np.asarray(g).copy() for g in groups
        ]
        if cell[1] is None:
            target_len = len(self.vertex_groups)
            while len(decoded) < target_len:
                decoded.append(np.empty((0, width), dtype=np.int64))
        elif not decoded:
            decoded.append(np.empty((0, width), dtype=np.int64))
        self.link_groups[cell] = decoded
        return self.link_groups[cell]

    # ----- vertex mutations --------------------------------------------

    def overwrite_vertex_row(
        self,
        fragment: int,
        local: int,
        new_pos: npt.NDArray[np.floating],
        *,
        new_attrs: dict[str, npt.NDArray] | None = None,
    ) -> None:
        """Overwrite one row of one existing fragment in place."""
        if fragment < 0 or fragment >= len(self.vertex_groups):
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"fragment index {fragment} out of range for chunk "
                f"{self.chunk} (has {len(self.vertex_groups)} fragments)"
            )
        group = self.vertex_groups[fragment]
        if local < 0 or local >= group.shape[0]:
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"local row {local} out of range in fragment {fragment} "
                f"(size {group.shape[0]})"
            )
        group[local] = np.asarray(new_pos, dtype=self.vertex_dtype)
        self.vertices_dirty = True
        if new_attrs:
            for name, value in new_attrs.items():
                attr_list = self.attr_groups.get(name)
                if attr_list is None:
                    continue
                attr_list[fragment][local] = np.asarray(
                    value, dtype=self.attr_dtype.get(name, np.float32),
                )
                self.attrs_dirty[name] = True

    def append_fragment(
        self,
        rows: npt.NDArray[np.floating],
        *,
        attrs: dict[str, npt.NDArray] | None = None,
    ) -> int:
        """Append a new fragment containing ``rows`` to the chunk.

        Returns the new fragment's index.
        """
        new_idx = len(self.vertex_groups)
        arr = np.atleast_2d(np.asarray(rows, dtype=self.vertex_dtype))
        if arr.shape[1] != self.vertex_ndim:
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"new fragment row arity {arr.shape[1]} != ndim "
                f"{self.vertex_ndim}"
            )
        self.vertex_groups.append(arr)
        self.vertices_dirty = True
        self.appended_fragments.append(new_idx)
        # Extend every loaded attribute list so the per-fragment alignment
        # stays consistent.
        for name, attr_list in self.attr_groups.items():
            dtype = self.attr_dtype.get(name, np.float32)
            ncols = self.attr_ncols.get(name, 1)
            if attrs and name in attrs:
                vals = np.asarray(attrs[name], dtype=dtype)
            else:
                vals = np.zeros(
                    (arr.shape[0], ncols) if ncols > 1 else (arr.shape[0],),
                    dtype=dtype,
                )
            attr_list.append(vals)
            self.attrs_dirty[name] = True
        # Extend the intra cell with an empty per-fragment group so this
        # builder keeps one link group per vertex fragment (internal
        # bookkeeping; write_chunk_links no longer requires 1:1 alignment).
        # Only the intra cell tracks vertex fragments — other cells hold
        # records sourced here but indexed against other chunks, so a new
        # vertex fragment says nothing about their grouping.
        for cell, groups in self.link_groups.items():
            if cell[1] is not None:
                continue
            width = groups[0].shape[1] if groups and groups[0].ndim == 2 else 2
            groups.append(np.empty((0, width), dtype=np.int64))
            self.links_dirty[cell] = True
        return new_idx

    def drop_fragment_row(self, fragment: int, local: int) -> None:
        """Delete one row from one fragment.

        Used by chunk-cross relocation in the "delete source row" path
        of the source-row retention rule.  Adjacent rows shift down.
        """
        if fragment < 0 or fragment >= len(self.vertex_groups):
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"fragment index {fragment} out of range for chunk "
                f"{self.chunk}"
            )
        group = self.vertex_groups[fragment]
        if local < 0 or local >= group.shape[0]:
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"local row {local} out of range in fragment {fragment}"
            )
        self.vertex_groups[fragment] = np.delete(group, local, axis=0)
        self.vertices_dirty = True
        for name, attr_list in self.attr_groups.items():
            attr_list[fragment] = np.delete(attr_list[fragment], local, axis=0)
            self.attrs_dirty[name] = True

    # ----- link mutations ----------------------------------------------

    def _cell_groups(
        self, cell: LinkCell, fragment: int,
    ) -> list[npt.NDArray[np.integer]]:
        """Loaded row groups of ``cell``, with ``fragment`` bounds-checked."""
        from zarr_vectors.exceptions import EditError
        if cell not in self.link_groups:
            raise EditError(
                f"link cell delta={cell[0]} offsets={cell[1]} not loaded — "
                f"call require_links first"
            )
        groups = self.link_groups[cell]
        if fragment < 0 or fragment >= len(groups):
            raise EditError(
                f"fragment {fragment} out of range for link cell "
                f"delta={cell[0]} offsets={cell[1]} ({len(groups)} groups)"
            )
        return groups

    def append_link_row(
        self,
        cell: LinkCell,
        fragment: int,
        row: npt.NDArray[np.integer],
    ) -> int:
        """Append a row to one row group of one link cell.

        Returns the new row index inside that group.
        """
        groups = self._cell_groups(cell, fragment)
        arr = np.atleast_2d(np.asarray(row, dtype=np.int64))
        groups[fragment] = np.concatenate([groups[fragment], arr], axis=0)
        self.links_dirty[cell] = True
        return groups[fragment].shape[0] - 1

    def drop_link_row(self, cell: LinkCell, fragment: int, row: int) -> None:
        from zarr_vectors.exceptions import EditError
        groups = self._cell_groups(cell, fragment)
        if row < 0 or row >= groups[fragment].shape[0]:
            raise EditError(f"row {row} out of range in fragment {fragment}")
        groups[fragment] = np.delete(groups[fragment], row, axis=0)
        self.links_dirty[cell] = True

    def overwrite_link_row(
        self,
        cell: LinkCell,
        fragment: int,
        row: int,
        new_row: npt.NDArray[np.integer],
    ) -> None:
        from zarr_vectors.exceptions import EditError
        groups = self._cell_groups(cell, fragment)
        if row < 0 or row >= groups[fragment].shape[0]:
            raise EditError(f"row {row} out of range in fragment {fragment}")
        groups[fragment][row] = np.asarray(new_row, dtype=groups[fragment].dtype)
        self.links_dirty[cell] = True

    # ----- introspection -----------------------------------------------

    def is_dirty(self) -> bool:
        """True iff anything in this chunk has been modified."""
        return (
            self.vertices_dirty
            or any(self.links_dirty.values())
            or any(self.attrs_dirty.values())
        )


@dataclass
class ManifestOp:
    """One pending edit to ``object_index/manifests``.

    ``new_manifest`` is the post-edit manifest (None for a delete which
    writes an empty manifest).  ``new_oid`` is set under atomic mode
    when a fresh OID is allocated; the original OID's manifest is
    preserved.
    """

    level: int
    object_id: int
    new_manifest: ObjectManifest | None
    new_oid: int | None = None  # atomic mode: append at this OID instead


@dataclass(frozen=True)
class OidPrefix:
    """Disjoint-OID-range allocator for cooperating editors.

    Two ``EditSession``s configured with the same modulus ``k`` and
    different residues ``r`` will never collide on atomic-OID
    allocation: each session emits new OIDs ``n`` such that
    ``n % k == r``.

    Constructors:

    - ``OidPrefix.from_name(name, k)`` — hash ``name`` to a residue
      ``r ∈ [0, k)``.  Useful when editors agree on the modulus and
      pick disjoint identifiers (e.g. ``"alice"`` / ``"bob"``).
    - ``OidPrefix(residue=r, modulus=k)`` — explicit residue.
    """

    residue: int
    modulus: int

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"OidPrefix.modulus must be > 0, got {self.modulus}"
            )
        if not (0 <= self.residue < self.modulus):
            from zarr_vectors.exceptions import EditError
            raise EditError(
                f"OidPrefix.residue must be in [0, {self.modulus}), "
                f"got {self.residue}"
            )

    @classmethod
    def from_name(cls, name: str, modulus: int) -> OidPrefix:
        """Stable hash of ``name`` modulo ``modulus``."""
        import hashlib
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        residue = int.from_bytes(digest[:8], "little") % modulus
        return cls(residue=residue, modulus=modulus)

    def next_after(self, lower_bound: int) -> int:
        """Return the smallest OID ``n >= lower_bound`` with
        ``n % modulus == residue``."""
        r = lower_bound % self.modulus
        if r <= self.residue:
            return lower_bound + (self.residue - r)
        return lower_bound + (self.modulus - r) + self.residue


@dataclass
class VacuumReport:
    """Result of :func:`zarr_vectors.ops.vacuum.vacuum`.

    Vacuum is destructive: applying it invalidates external references
    to old OIDs unless the caller composes them through ``oid_remap``.
    """

    oid_remap: dict[int, int] = field(default_factory=dict)
    """``{old_oid: new_oid}`` after OID compaction.  Identity for OIDs
    that were already dense at the head of the table.
    """

    dropped_fragments_per_chunk: dict[tuple[int, ChunkCoords], list[int]] = field(
        default_factory=dict,
    )
    """``{(level, chunk_coords): [dropped_fragment_indices]}`` — empty
    this iteration (filled by the deferred tombstone-GC pass).
    """

    bytes_freed: int = 0
    """Aggregate bytes reclaimed by the vacuum pass.  Best-effort
    estimate based on the pre-vs-post ``object_index/manifests`` size
    and any chunk re-encodes."""

    def to_dict(self) -> dict:
        return {
            "oid_remap": {str(k): v for k, v in self.oid_remap.items()},
            "dropped_fragments_per_chunk": {
                f"{lv}/{'.'.join(str(c) for c in cc)}": list(v)
                for (lv, cc), v in self.dropped_fragments_per_chunk.items()
            },
            "bytes_freed": int(self.bytes_freed),
        }


@dataclass
class EditReport:
    """User-facing summary of what an :class:`EditSession` did.

    Returned by ``EditSession.__exit__`` (via ``ed.report``) and by every
    free-function edit so callers can audit, replay, or queue diffs.
    """

    touched_chunks: list[tuple[int, ChunkCoords]] = field(default_factory=list)
    """``[(level, chunk_coords), ...]`` for every chunk whose bytes
    were rewritten during the session.  Useful for partial-pyramid
    refresh and crash-recovery replay on non-transactional backends.
    """

    oid_remap: dict[int, int] = field(default_factory=dict)
    """``{old_oid: new_oid}`` for atomic edits that allocated a new
    OID.  Empty under ``atomic=False``.
    """

    dirty_pyramid_levels: list[int] = field(default_factory=list)
    """Resolution levels above the edited level that are now stale
    because they were *not* refreshed in this session (i.e. a
    downstream call to ``rebuild_pyramid_from_level`` is required).
    Empty when ``refresh_pyramid`` was ``"batch"`` or ``True``.
    """

    snapshot_id: str | None = None
    """icechunk snapshot id of the commit that flushed this session;
    ``None`` for non-transactional backends.
    """

    n_edits: int = 0
    """Total number of individual edits applied in the session."""

    oid_prefix: OidPrefix | None = None
    """The OID-prefix allocator used during this session (if any).
    ``None`` means atomic OIDs were appended at ``len(manifests)`` with
    no residue constraint.  ``merge_edit_reports`` cross-checks the
    prefix of each input to ensure their atomic OIDs cannot collide.
    """

    def to_dict(self) -> dict:
        """Return a JSON-serialisable view of the report."""
        return {
            "touched_chunks": [
                {"level": lv, "chunk": list(cc)}
                for lv, cc in self.touched_chunks
            ],
            "oid_remap": {str(k): v for k, v in self.oid_remap.items()},
            "dirty_pyramid_levels": list(self.dirty_pyramid_levels),
            "snapshot_id": self.snapshot_id,
            "n_edits": self.n_edits,
            "oid_prefix": (
                None if self.oid_prefix is None
                else {"residue": self.oid_prefix.residue,
                      "modulus": self.oid_prefix.modulus}
            ),
        }
