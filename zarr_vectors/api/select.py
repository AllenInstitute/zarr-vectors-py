"""Asking for data by what it is, not by where it is stored.

The five readers already share a selection vocabulary — ``level``,
``bbox``, ``object_ids``, ``group_ids``, ``attribute_filter`` — but they
expose it inconsistently (only ``read_points`` takes ``attribute_names``;
only ``read_lines`` lacks ``chunks``) and two of the terms are storage,
not data: ``chunks=[(3, 1, 2), ...]`` is a list of grid coordinates that
the caller has to know the grid to construct.

:class:`Selection` is that vocabulary said once, with the storage terms
removed.  :class:`Query` is a lazy, chainable handle on one — nothing is
read until a terminal is called, and intersecting two selections is a
value operation, so a caller can build a query up in one place and run it
in another.

**Laziness is not decoration.**  It is what lets this phase delegate to
the eager readers and a later phase run resolvers against the batching
engine, with no visible difference.  If ``select()`` did I/O, that switch
would be a behaviour change and every caller would have to be revisited.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.exceptions import ZVError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.api.level import Level
    from zarr_vectors.api.result import ReadResult

__all__ = ["Query", "Selection"]


def _as_bbox(
    value: tuple[Sequence[float], Sequence[float]],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    lo, hi = value
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Selection:
    """What to read.  A value: no store, no I/O, comparable and mergeable.

    Every field narrows.  Combining two selections intersects them, so
    ``level.select(bbox=b).select(objects=ids)`` means "in that box *and*
    among those objects" — the reading a caller expects from chaining.
    """

    level: int = 0
    bbox: tuple[Sequence[float], Sequence[float]] | None = None
    near: tuple[Sequence[float], float] | None = None
    objects: Sequence[int] | None = None
    groups: Sequence[int] | None = None
    where: Mapping[str, Any] | None = None
    cells: Any = None
    """Opaque cell references, from ``level.grid.cells_in(bbox)``.

    The honest replacement for ``chunks=[(3, 1, 2), ...]``: a caller who
    genuinely needs to target storage regions gets the references from
    the grid rather than constructing coordinates they had to derive.
    """
    attributes: Sequence[str] | str = "all"
    limit: int | None = None

    def intersect(self, other: Selection) -> Selection:
        """Narrow by ``other``.

        Two bboxes intersect geometrically; two id sets intersect as
        sets.  A field set on only one side is simply adopted — there is
        nothing to reconcile.
        """
        bbox = self.bbox
        if other.bbox is not None:
            if bbox is None:
                bbox = other.bbox
            else:
                (a_lo, a_hi), (b_lo, b_hi) = _as_bbox(bbox), _as_bbox(other.bbox)
                bbox = (np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi))
        return Selection(
            level=other.level if other.level != 0 else self.level,
            bbox=bbox,
            near=other.near if other.near is not None else self.near,
            objects=_intersect_ids(self.objects, other.objects),
            groups=_intersect_ids(self.groups, other.groups),
            where={**(self.where or {}), **(other.where or {})} or None,
            cells=other.cells if other.cells is not None else self.cells,
            attributes=(
                other.attributes if other.attributes != "all" else self.attributes
            ),
            limit=(
                min(x for x in (self.limit, other.limit) if x is not None)
                if self.limit is not None or other.limit is not None else None
            ),
        )

    def to_reader_kwargs(self, *, supports: Sequence[str]) -> dict[str, Any]:
        """Translate into the legacy reader's keyword set.

        ``supports`` names the kwargs that reader actually accepts — they
        differ per reader, which is one of the inconsistencies this class
        exists to hide.  A term the reader cannot express is dropped here
        and applied afterwards in Python, so the *answer* is the same
        either way and only the amount of data read differs.
        """
        out: dict[str, Any] = {"level": self.level}
        effective_bbox: tuple[Any, Any] | None = self.bbox
        if self.near is not None:
            centre, radius = self.near
            c = np.asarray(centre, dtype=np.float64)
            sphere = (c - float(radius), c + float(radius))
            effective_bbox = (
                sphere if effective_bbox is None
                else (
                    np.maximum(_as_bbox(effective_bbox)[0], sphere[0]),
                    np.minimum(_as_bbox(effective_bbox)[1], sphere[1]),
                )
            )
        if effective_bbox is not None and "bbox" in supports:
            out["bbox"] = _as_bbox(effective_bbox)
        if self.objects is not None and "object_ids" in supports:
            out["object_ids"] = list(self.objects)
        if self.groups is not None and "group_ids" in supports:
            out["group_ids"] = list(self.groups)
        if self.where and "attribute_filter" in supports:
            out["attribute_filter"] = dict(self.where)
        if (
            self.attributes != "all"
            and not isinstance(self.attributes, str)
            and "attribute_names" in supports
        ):
            out["attribute_names"] = list(self.attributes)
        return out

    @property
    def needs_post_filter(self) -> bool:
        """Whether a term has to be applied after the read.

        ``near`` always does — it degrades to its bounding box on the way
        down, and the corners of that box are outside the sphere.
        """
        return (
            self.near is not None
            or self.limit is not None
            or self.cells is not None
        )


def _intersect_ids(
    a: Sequence[int] | None, b: Sequence[int] | None,
) -> Sequence[int] | None:
    if a is None:
        return b
    if b is None:
        return a
    keep = set(a) & set(b)
    return [i for i in a if i in keep]


class Query:
    """A pending read.  Nothing happens until a terminal is called.

    Terminals are :meth:`read`, :meth:`aread`, :meth:`count` and
    :meth:`object_ids`.  Everything else returns a new ``Query``.
    """

    __slots__ = ("_level", "_selection")

    def __init__(self, level: Level, selection: Selection) -> None:
        self._level = level
        self._selection = selection

    # ---------------- narrowing ----------------

    def select(self, **kw: Any) -> Query:
        """Narrow further.  Returns a new query; this one is unchanged."""
        return Query(self._level, self._selection.intersect(Selection(**kw)))

    def limit(self, n: int) -> Query:
        return Query(self._level, replace(self._selection, limit=int(n)))

    @property
    def selection(self) -> Selection:
        """What this query is asking for.  Inspectable, comparable."""
        return self._selection

    # ---------------- terminals ----------------

    def read(self) -> ReadResult:
        """Execute and return the whole answer."""
        return self._level._execute(self._selection)

    async def aread(self) -> ReadResult:
        """Execute without ever blocking on zarr's ``sync()`` bridge.

        For hosts where a synchronous zarr call cannot work — Pyodide,
        where it needs WebAssembly stack switching and deadlocks the JS
        event loop under concurrency.
        """
        return await self._level._aexecute(self._selection)

    def count(self) -> int:
        """How many vertices this query would return."""
        return self.read().vertex_count

    def object_ids(self) -> npt.NDArray[Any]:
        """The distinct object ids this query touches."""
        result = self.read()
        for source in (result.part_objects, result.object_ids):
            if source is not None and len(source):
                return np.unique(np.asarray(source))
        return np.zeros((0,), dtype=np.int64)

    # ---------------- diagnostics ----------------

    def explain(self) -> str:
        """A human-readable account of what this query will do."""
        sel = self._selection
        bits = [f"level {sel.level}"]
        if sel.bbox is not None:
            lo, hi = _as_bbox(sel.bbox)
            bits.append(f"bbox {lo.tolist()}..{hi.tolist()}")
        if sel.near is not None:
            bits.append(f"within {sel.near[1]} of {list(sel.near[0])}")
        if sel.objects is not None:
            bits.append(f"{len(sel.objects)} object id(s)")
        if sel.groups is not None:
            bits.append(f"{len(sel.groups)} group(s)")
        if sel.where:
            bits.append(f"where {dict(sel.where)}")
        if sel.limit is not None:
            bits.append(f"limit {sel.limit}")
        how = self._level._explain_execution(sel)
        kind = self._level.dataset.kind_of(sel.level)
        return f"{kind} read: " + ", ".join(bits) + f"\n  via {how}"

    def plan(self) -> Any:
        """The I/O this query would perform, without performing any of it.

        Both a debugging tool and the decoupling test: an internal change
        that alters the shape of a read shows up here, in one comparable
        value, rather than as a performance mystery downstream.
        """
        return self._level.plan(self._selection)

    def iter_cells(self) -> Any:
        """Stream the result one chunk at a time.

        Not available yet: streaming needs the facade to drive the read
        itself.  Delegating to a reader that materialises everything and
        then yielding from it would have the signature of streaming and
        none of the benefit, which is worse than not offering it.
        """
        raise NotImplementedError(
            "Query.iter_cells() arrives with the resolver phase. The legacy "
            "readers materialise the whole result, so a generator over one "
            "would use the same peak memory while implying it does not."
        )

    def __repr__(self) -> str:
        return f"Query({self.explain().splitlines()[0]})"


class SelectionError(ZVError):
    """A selection cannot be expressed against this store."""
