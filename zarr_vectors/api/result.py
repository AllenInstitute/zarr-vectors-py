"""One result type for every geometry.

The five readers in :mod:`zarr_vectors.types` return five differently
shaped dicts.  They use four names for the vertex array
(``positions`` / ``vertices`` / ``endpoints`` / ``polylines``), seven
distinct count keys, and only one of them carries attributes on every
path.  A caller who wants to handle two geometry types therefore writes
two code paths for what is, structurally, the same answer: some vertices,
optionally cut into parts, optionally connected.

:class:`ReadResult` is that answer said once.  The per-geometry shapes
survive as *derived* properties, so a polyline user still writes
``res.polylines`` and gets the list they had before — but a tool that
does not care which geometry it is given can read ``res.positions`` and
be correct for all of them.

**Parts, not per-type containers.**  A "part" is one emitted run of
consecutive vertices: a polyline, a line segment, a mesh (whole), a point
cloud (whole).  Storing them as slices into one contiguous ``positions``
array rather than as a list of arrays means the common case — "give me
all the coordinates" — needs no concatenation, and the per-part view
costs a slice rather than a copy.  It is also what makes the mapping from
each legacy reader *total*: every one of them can be expressed as
positions plus a cut, and every one can be read back out unchanged.

**Errors are values.**  The lazy layer swallows per-chunk failures in
bare ``except Exception: continue`` blocks, so a corrupt chunk and an
absent one are indistinguishable and both look like "no data here".  A
read that partially failed reports which cells failed in ``errors``
rather than quietly returning less than was asked for.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["Attributes", "ReadError", "ReadResult"]


class Attributes(Mapping[str, "npt.NDArray[Any]"]):
    """Per-vertex attribute arrays, keyed by name.

    A real :class:`~collections.abc.Mapping` rather than a bare dict so
    that a later phase can make it lazy — reading an attribute only when
    it is asked for — without changing a single call site.  Today every
    value is already materialised.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, npt.NDArray[Any]] | None = None) -> None:
        self._data: dict[str, npt.NDArray[Any]] = dict(data or {})

    def __getitem__(self, key: str) -> npt.NDArray[Any]:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(
                f"no attribute {key!r}; this result carries "
                f"{sorted(self._data) or 'none'}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._data))

    def __repr__(self) -> str:
        return f"Attributes({self.names()!r})"


@dataclass(frozen=True, slots=True)
class ReadError:
    """One cell that could not be read, and why.

    Carried rather than raised because a partial answer is usually more
    useful than none — but only if the caller can tell it apart from a
    complete one.
    """

    cell: str
    """Chunk key, in the usual dotted form."""

    array: str
    """Which array the failure was in, relative to the level."""

    error: str
    """The exception's message."""


def _remap_indices(
    table: npt.NDArray[Any] | None, remap: npt.NDArray[np.int64],
) -> npt.NDArray[Any] | None:
    """Renumber an ``(M, W)`` index table, dropping rows that reference a
    vertex ``remap`` removed (marked ``-1``)."""
    if table is None:
        return None
    table = np.asarray(table)
    if table.size == 0:
        return table
    moved = np.asarray(remap[table])
    kept: npt.NDArray[Any] = moved[(moved >= 0).all(axis=1)]
    return kept


def replace_truncated(result: ReadResult, value: bool) -> ReadResult:
    """``result`` with :attr:`ReadResult.truncated` set.

    A free function rather than ``dataclasses.replace`` because
    ``slots=True`` frozen dataclasses and ``replace`` interact badly with
    the default factory on ``attributes``.
    """
    return ReadResult(
        kind=result.kind,
        positions=result.positions,
        parts=result.parts,
        part_objects=result.part_objects,
        object_ids=result.object_ids,
        edges=result.edges,
        faces=result.faces,
        attributes=result.attributes,
        attributes_read=result.attributes_read,
        truncated=value,
        errors=result.errors,
    )


def _join_segments(part: Any) -> npt.NDArray[Any]:
    """One part's vertices as a single ``(N, D)`` array.

    ``read_polylines`` yields each polyline as a list of per-chunk
    segments; other readers yield a plain array.  Both are accepted so
    the adapter does not have to care which reader it came from.
    """
    if isinstance(part, (list, tuple)):
        segments = [np.asarray(s) for s in part if len(s)]
        if not segments:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(segments, axis=0)
    return np.asarray(part)


def _slices_from_lengths(lengths: Sequence[int]) -> tuple[slice, ...]:
    """Consecutive slices covering ``sum(lengths)`` rows."""
    out: list[slice] = []
    start = 0
    for n in lengths:
        out.append(slice(start, start + int(n)))
        start += int(n)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class ReadResult:
    """What a read returned, independent of which geometry it was.

    Construct via the ``from_*`` adapters rather than directly; they are
    the total mapping from each legacy reader's output, and keeping them
    in one place is what makes that mapping checkable.
    """

    kind: str
    """Geometry type, one of :mod:`zarr_vectors.constants`' ``GEOM_*``."""

    positions: npt.NDArray[Any]
    """``(N, D)`` vertex coordinates.  Always this name, for every kind."""

    parts: tuple[slice, ...] = ()
    """One slice into ``positions`` per emitted part.

    A point cloud, mesh or graph is one part covering everything; a
    polyline store has one per polyline; a line store has one per
    segment.  Empty only when there are no vertices.
    """

    part_objects: npt.NDArray[Any] | None = None
    """``(P,)`` source object id of each part, when the store tracks them.

    Distinct from :attr:`object_ids`, and the distinction is load-bearing:
    ``read_polylines`` returns one id per *polyline*, while
    ``read_points``' object path returns one per *vertex*.  Collapsing
    them would silently misalign one of the two.
    """

    object_ids: npt.NDArray[Any] | None = None
    """``(N,)`` per-vertex object id, when the reader supplied one."""

    edges: npt.NDArray[Any] | None = None
    """``(M, 2)`` vertex-index pairs, for graph-like geometry."""

    faces: npt.NDArray[Any] | None = None
    """``(F, L)`` vertex-index tuples, for mesh geometry."""

    attributes: Attributes = field(default_factory=Attributes)
    """Per-vertex attributes.  Empty is not the same as unavailable — see
    :attr:`attributes_read`."""

    attributes_read: bool = True
    """Whether attributes were even attempted.

    ``read_points``' object-id path returns ``vertex_attributes={}``
    unconditionally: the attributes are not absent, they were never
    looked at.  A caller that cannot distinguish those two will conclude
    a store has no attributes when it has plenty.
    """

    truncated: bool = False
    """A ``limit=`` cut the result short."""

    errors: tuple[ReadError, ...] = ()
    """Cells that failed.  Empty means the answer is complete."""

    # ---------------- derived views ----------------

    @property
    def vertex_count(self) -> int:
        return int(len(self.positions))

    @property
    def part_count(self) -> int:
        return len(self.parts)

    @property
    def ndim(self) -> int:
        return int(self.positions.shape[1]) if self.positions.ndim == 2 else 0

    @property
    def polylines(self) -> list[npt.NDArray[Any]]:
        """Each part as its own ``(N_k, D)`` array.

        The shape ``read_polylines`` returns.  Views, not copies.
        """
        return [self.positions[s] for s in self.parts]

    @property
    def endpoints(self) -> npt.NDArray[Any]:
        """``(M, 2, D)`` line endpoints — the shape ``read_lines`` returns.

        Raises when the parts are not uniformly two vertices long, since
        a "line" that is not a pair is a polyline and the caller has
        reached for the wrong view.
        """
        if any(s.stop - s.start != 2 for s in self.parts):
            raise ValueError(
                "endpoints is only defined when every part is a vertex pair; "
                "this result has parts of length "
                f"{sorted({s.stop - s.start for s in self.parts})}. Use .polylines."
            )
        return self.positions.reshape(len(self.parts), 2, self.ndim)

    @property
    def complete(self) -> bool:
        return not self.errors and not self.truncated

    def __repr__(self) -> str:
        bits = [f"kind={self.kind!r}", f"vertices={self.vertex_count}"]
        if self.part_count != 1:
            bits.append(f"parts={self.part_count}")
        if self.edges is not None:
            bits.append(f"edges={len(self.edges)}")
        if self.faces is not None:
            bits.append(f"faces={len(self.faces)}")
        if self.attributes:
            bits.append(f"attributes={list(self.attributes.names())}")
        if self.errors:
            bits.append(f"errors={len(self.errors)}")
        return f"ReadResult({', '.join(bits)})"

    def restrict(self, keep: npt.NDArray[np.bool_], *, truncated: bool = False) -> ReadResult:
        """This result with only the vertices ``keep`` selects.

        Every index-bearing field is repaired, not merely sliced.  Edges
        and faces are renumbered into the new positions array and any
        that reference a dropped vertex are removed — a filtered mesh
        whose faces still point at the old row numbers is silently
        corrupt, and silently corrupt geometry is the worst thing this
        layer could hand back.

        Parts are re-cut from what survives, so a partly-kept polyline
        stays one part and simply gets shorter.
        """
        keep = np.asarray(keep, dtype=bool)
        if keep.all():
            return self if not truncated else replace_truncated(self, True)

        # old row -> new row, with -1 for dropped rows.
        remap = np.full(len(keep), -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)

        kept_per_part = [int(keep[s].sum()) for s in self.parts]
        parts = _slices_from_lengths([n for n in kept_per_part if n])
        part_objects = self.part_objects
        if part_objects is not None:
            surviving = [i for i, n in enumerate(kept_per_part) if n]
            part_objects = np.asarray(part_objects)[surviving]

        return ReadResult(
            kind=self.kind,
            positions=self.positions[keep],
            parts=parts,
            part_objects=part_objects,
            object_ids=(
                None if self.object_ids is None else np.asarray(self.object_ids)[keep]
            ),
            edges=_remap_indices(self.edges, remap),
            faces=_remap_indices(self.faces, remap),
            attributes=Attributes(
                {k: np.asarray(v)[keep] for k, v in self.attributes.items()}
            ),
            attributes_read=self.attributes_read,
            truncated=self.truncated or truncated,
            errors=self.errors,
        )

    # ---------------- adapters from the legacy readers ----------------
    #
    # Each of these is total over the corresponding reader's return
    # paths, including its empty-store path.  They are the correctness
    # proof for this phase: the facade delegates to the reader and then
    # comes through here, so if an adapter is wrong the facade is wrong.

    @classmethod
    def from_points(cls, raw: Mapping[str, Any], *, kind: str) -> ReadResult:
        """``{positions, vertex_attributes, vertex_count}``, plus
        ``object_ids`` on the object-id path.

        That path returns ``vertex_attributes={}`` unconditionally
        (``points.py:653``) — not because there are none, but because it
        never reads them.  ``attributes_read`` records the difference.
        """
        positions = np.asarray(raw["positions"])
        object_ids = raw.get("object_ids")
        by_object = object_ids is not None
        return cls(
            kind=kind,
            positions=positions,
            parts=(slice(0, len(positions)),) if len(positions) else (),
            object_ids=None if object_ids is None else np.asarray(object_ids),
            attributes=Attributes(raw.get("vertex_attributes") or {}),
            attributes_read=not by_object,
        )

    @classmethod
    def from_lines(cls, raw: Mapping[str, Any], *, kind: str) -> ReadResult:
        """``{endpoints, line_count}`` — the only reader with no
        attribute key at all on any path.

        ``endpoints`` is ``(M, 2, D)``.  Flattening it to ``(2M, D)`` with
        one two-vertex part per line loses nothing (:attr:`endpoints`
        reverses it exactly) and gains the shared shape.  The synthesised
        ``edges`` makes the connectivity explicit, which the tuple form
        left implicit in the axis layout.
        """
        endpoints = np.asarray(raw["endpoints"])
        n_lines = int(endpoints.shape[0])
        ndim = int(endpoints.shape[2]) if endpoints.ndim == 3 else 0
        positions = endpoints.reshape(n_lines * 2, ndim)
        edges = (
            np.arange(n_lines * 2, dtype=np.int64).reshape(n_lines, 2)
            if n_lines else np.zeros((0, 2), dtype=np.int64)
        )
        return cls(
            kind=kind,
            positions=positions,
            parts=_slices_from_lengths([2] * n_lines),
            edges=edges,
            attributes_read=False,
        )

    @classmethod
    def from_polylines(cls, raw: Mapping[str, Any], *, kind: str) -> ReadResult:
        """``{polylines, object_ids, polyline_count, vertex_count}``.

        ``object_ids`` here is one id per *polyline*, so it becomes
        :attr:`part_objects`.  Reading it as per-vertex would misalign it
        against ``positions`` by the length of every polyline.

        Each element is a **list of per-chunk segments**, not a single
        array — even for a polyline that fits in one chunk, which comes
        back as a list of one.  (The quickstart's
        ``result["polylines"][0].shape`` is wrong; there is no ``.shape``
        on a list.)  Concatenating the segments reconstructs the polyline
        exactly: measured across single- and multi-chunk stores, the
        segment lengths sum to the input length with no vertex duplicated
        at a boundary.
        """
        raw_polylines = list(raw.get("polylines") or [])
        arrays = [_join_segments(p) for p in raw_polylines]
        if arrays:
            positions = np.concatenate(arrays, axis=0)
        else:
            positions = np.zeros((0, 3), dtype=np.float32)
        part_objects = raw.get("object_ids")
        return cls(
            kind=kind,
            positions=positions,
            parts=_slices_from_lengths([len(a) for a in arrays]),
            part_objects=(
                None if part_objects is None or len(part_objects) == 0
                else np.asarray(part_objects)
            ),
            attributes_read=False,
        )

    @classmethod
    def from_mesh(cls, raw: Mapping[str, Any], *, kind: str) -> ReadResult:
        """``{vertices, faces, vertex_count, face_count}`` — note
        ``vertices``, not ``positions``."""
        positions = np.asarray(raw["vertices"])
        return cls(
            kind=kind,
            positions=positions,
            parts=(slice(0, len(positions)),) if len(positions) else (),
            faces=np.asarray(raw["faces"]),
            attributes_read=False,
        )

    @classmethod
    def from_graph(cls, raw: Mapping[str, Any], *, kind: str) -> ReadResult:
        """``{positions, edges, node_count, edge_count}``."""
        positions = np.asarray(raw["positions"])
        return cls(
            kind=kind,
            positions=positions,
            parts=(slice(0, len(positions)),) if len(positions) else (),
            edges=np.asarray(raw["edges"]),
            attributes_read=False,
        )
