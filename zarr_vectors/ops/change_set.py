"""In-memory representation of pending edits.

A :class:`ChunkChangeBuilder` lazily decodes one chunk's ragged
state (vertices, fragment sidecar, per-vertex attributes, intra-chunk
links) the first time the edit engine touches it, lets callers mutate
the decoded state in Python, and re-encodes everything in one shot at
flush time.  This is what gives the :class:`~zarr_vectors.ops.edit.
EditSession` its "coalesce many edits to the same chunk into one
read-modify-write" property.

:class:`EditReport` is the user-facing summary of what changed in a
session: touched chunks, OID remap (atomic edits), dirty pyramid
levels, and (on icechunk) the snapshot id.  It is also the
serialisable diff returned by ``EditSession.change_set()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.core.group import Group


@dataclass
class ChunkChangeBuilder:
    """Mutable in-memory state for one ``(level, chunk_coords)`` chunk.

    The builder holds three parallel kinds of state:

    - ``vertex_groups``: list of fragments, each a ``(N_k, D)`` float
      array.  The list's length is the chunk's fragment count.
    - ``link_groups``: per-fragment ``(M_k, L)`` integer arrays for the
      intra-level ``links/0/`` array.  ``None`` when the chunk has no
      links yet.
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
    link_groups: dict[int, list[npt.NDArray[np.integer]]] = field(
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
    # ``{delta: True}`` for link arrays that need rewrite.
    links_dirty: dict[int, bool] = field(default_factory=dict)
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
    ) -> list[npt.NDArray[np.integer]]:
        """Lazily decode ``links/<delta>/<chunk>`` for this chunk.

        Always returns a list whose length matches ``self.vertex_groups``:
        when the on-disk link layer has fewer fragments than the chunk
        has vertex fragments (e.g. a graph writer that consolidated all
        edges into a single link fragment), the result is padded with
        empty per-fragment groups so subsequent mutations preserve the
        1:1 alignment invariant ``write_chunk_links`` enforces.
        """
        if delta in self.link_groups:
            return self.link_groups[delta]

        from zarr_vectors.core.arrays import read_chunk_links
        from zarr_vectors.core.store import get_resolution_level

        level_group = get_resolution_level(root, self.level)
        try:
            groups = read_chunk_links(
                level_group, self.chunk, link_width=link_width, delta=delta,
            )
        except Exception:
            groups = []
        # Normalise to one group per vertex fragment.  When the on-disk
        # link table has fewer fragments than the chunk has vertex
        # fragments (e.g. a graph writer that consolidated all edges
        # into the first link slot), pad the tail with empty groups so
        # the write-back invariant holds.
        target_len = len(self.vertex_groups)
        padded: list[npt.NDArray[np.integer]] = [g.copy() for g in groups]
        while len(padded) < target_len:
            padded.append(np.empty((0, link_width), dtype=np.int64))
        self.link_groups[delta] = padded
        return self.link_groups[delta]

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
        # Also extend every link delta with an empty per-fragment group
        # so the 1:1 link/fragment alignment invariant holds.
        for delta, groups in self.link_groups.items():
            link_width = groups[0].shape[1] if groups and groups[0].ndim == 2 else 2
            groups.append(np.empty((0, link_width), dtype=np.int64))
            self.links_dirty[delta] = True
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

    def append_link_row(
        self,
        delta: int,
        fragment: int,
        row: npt.NDArray[np.integer],
    ) -> int:
        """Append a row to the per-fragment intra-chunk link group.

        Returns the new row index inside the fragment group.
        """
        from zarr_vectors.exceptions import EditError
        if delta not in self.link_groups:
            raise EditError(
                f"link delta={delta} not loaded — call require_links first"
            )
        groups = self.link_groups[delta]
        if fragment < 0 or fragment >= len(groups):
            raise EditError(
                f"fragment {fragment} out of range for link delta={delta}"
            )
        arr = np.atleast_2d(np.asarray(row, dtype=np.int64))
        groups[fragment] = np.concatenate([groups[fragment], arr], axis=0)
        self.links_dirty[delta] = True
        return groups[fragment].shape[0] - 1

    def drop_link_row(self, delta: int, fragment: int, row: int) -> None:
        from zarr_vectors.exceptions import EditError
        if delta not in self.link_groups:
            raise EditError(
                f"link delta={delta} not loaded — call require_links first"
            )
        groups = self.link_groups[delta]
        if fragment < 0 or fragment >= len(groups):
            raise EditError(f"fragment {fragment} out of range")
        if row < 0 or row >= groups[fragment].shape[0]:
            raise EditError(f"row {row} out of range in fragment {fragment}")
        groups[fragment] = np.delete(groups[fragment], row, axis=0)
        self.links_dirty[delta] = True

    def overwrite_link_row(
        self,
        delta: int,
        fragment: int,
        row: int,
        new_row: npt.NDArray[np.integer],
    ) -> None:
        from zarr_vectors.exceptions import EditError
        if delta not in self.link_groups:
            raise EditError(
                f"link delta={delta} not loaded — call require_links first"
            )
        groups = self.link_groups[delta]
        if fragment < 0 or fragment >= len(groups):
            raise EditError(f"fragment {fragment} out of range")
        if row < 0 or row >= groups[fragment].shape[0]:
            raise EditError(f"row {row} out of range in fragment {fragment}")
        groups[fragment][row] = np.asarray(new_row, dtype=groups[fragment].dtype)
        self.links_dirty[delta] = True

    # ----- introspection -----------------------------------------------

    def is_dirty(self) -> bool:
        """True iff anything in this chunk has been modified."""
        return (
            self.vertices_dirty
            or any(self.links_dirty.values())
            or any(self.attrs_dirty.values())
        )


@dataclass
class CrossChunkLinkOp:
    """One pending edit to the global ``cross_chunk_links/<delta>/data``
    array.

    ``op`` is ``"append"`` (add a new row), ``"delete"`` (drop the row
    at ``index``), or ``"overwrite"`` (replace the row at ``index``).
    """

    op: str
    delta: int
    payload: list[tuple[ChunkCoords, int]] | None = None
    index: int | None = None


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
