"""One resolution level, addressed by what is in it.

A :class:`Level` is where reads actually happen.  It knows which legacy
reader serves this store's geometry, which of that reader's keyword
arguments exist, and how to repair the difference between what the
caller asked for and what the reader could express.

Nothing here mentions a chunk, a bin, a shard or a backend.  Where a
selection term has no equivalent in the reader — ``near`` has none, and
``attribute_names`` exists on exactly one of the five — the term is
applied afterwards in Python.  That costs reading more than was needed;
it never costs a wrong answer, and it is temporary: once the facade
drives its own reads, the same public behaviour is served by a plan that
asks for less.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_vectors.api.result import ReadResult
from zarr_vectors.api.select import Query, Selection
from zarr_vectors.constants import (
    GEOM_GRAPH,
    GEOM_LINE,
    GEOM_MESH,
    GEOM_POINT_CLOUD,
    GEOM_POLYLINE,
    GEOM_SKELETON,
    GEOM_STREAMLINE,
)
from zarr_vectors.exceptions import ZVError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.api.dataset import Dataset

__all__ = ["Level"]

# Which reader serves which geometry, which of its keyword arguments
# exist, and how its result dict maps into the uniform one.  The kwarg
# lists are not decoration: the five readers genuinely differ (only
# read_points takes attribute_names, only read_lines lacks chunks), and
# passing one a keyword it does not have is a TypeError at the call site.
_READERS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    GEOM_POINT_CLOUD: (
        "zarr_vectors.types.points", "read_points",
        ("bbox", "object_ids", "group_ids", "attribute_names", "attribute_filter"),
        "from_points",
    ),
    GEOM_LINE: (
        "zarr_vectors.types.lines", "read_lines",
        ("bbox", "object_ids", "attribute_filter"),
        "from_lines",
    ),
    GEOM_POLYLINE: (
        "zarr_vectors.types.polylines", "read_polylines",
        ("bbox", "object_ids", "group_ids", "attribute_filter"),
        "from_polylines",
    ),
    GEOM_STREAMLINE: (
        "zarr_vectors.types.polylines", "read_polylines",
        ("bbox", "object_ids", "group_ids", "attribute_filter"),
        "from_polylines",
    ),
    GEOM_MESH: (
        "zarr_vectors.types.meshes", "read_mesh",
        ("bbox", "object_ids", "attribute_filter"),
        "from_mesh",
    ),
    GEOM_GRAPH: (
        "zarr_vectors.types.graphs", "read_graph",
        ("bbox", "object_ids", "attribute_filter"),
        "from_graph",
    ),
    GEOM_SKELETON: (
        "zarr_vectors.types.graphs", "read_graph",
        ("bbox", "object_ids", "attribute_filter"),
        "from_graph",
    ),
}

# Attribute families, by the group each lives under in a level.
_ATTR_GROUPS: dict[str, str] = {
    "vertex": "vertex_attributes",
    "fragment": "fragment_attributes",
    "object": "object_attributes",
    "group": "groupings_attributes",
    "link": "link_attributes",
}


class Level:
    """One resolution level of a :class:`~zarr_vectors.api.dataset.Dataset`."""

    __slots__ = ("_dataset", "_index", "_meta")

    def __init__(self, dataset: Dataset, index: int) -> None:
        self._dataset = dataset
        self._index = int(index)
        self._meta: Any = None

    # ---------------- identity ----------------

    @property
    def index(self) -> int:
        return self._index

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def kind(self) -> str:
        return self._dataset.kind_of(self._index)

    @property
    def vertex_count(self) -> int:
        """Vertices at this level, from metadata — no data is read."""
        return int(getattr(self._metadata(), "vertex_count", 0) or 0)

    @property
    def scale(self) -> tuple[float, ...]:
        """Physical size of one grid cell at this level.

        The honest data-shaped reading of what the storage layer calls a
        chunk shape: it is a length in the same units as the coordinates,
        and it is what a caller needs to write a compatible store or to
        choose a level for a viewport.
        """
        meta = self._metadata()
        shape = getattr(meta, "chunk_shape", None) or self._dataset._root_meta.chunk_shape
        return tuple(float(v) for v in (shape or ()))

    @property
    def resolution(self) -> tuple[float, ...]:
        """Physical size of one spatial-query cell — the finest region a
        bbox read can isolate without over-reading.

        Assembled rather than read from one place, because it is not
        stored in one place: a level carries its own ``bin_shape`` only
        once it has been coarsened, while level 0 leaves it implicit in
        the root's ``base_bin_shape``.  A caller should not have to know
        which of the two a given store happens to have.
        """
        meta = self._metadata()
        own = getattr(meta, "bin_shape", None)
        if own:
            return tuple(float(v) for v in own)
        base = getattr(self._dataset._root_meta, "base_bin_shape", None)
        if not base:
            return self.scale
        ratio = getattr(meta, "bin_ratio", None) or (1,) * len(base)
        return tuple(float(b) * float(r) for b, r in zip(base, ratio))

    def _metadata(self) -> Any:
        if self._meta is None:
            from zarr_vectors.core.store import read_level_metadata

            self._meta = read_level_metadata(self._dataset._group, self._index)
        return self._meta

    @property
    def objects(self) -> Any:
        """The objects at this level, addressed by id."""
        from zarr_vectors.api.objects import ObjectCatalog

        return ObjectCatalog(self)

    @property
    def groups(self) -> Any:
        """The object groups at this level, keyed by name."""
        from zarr_vectors.api.groups import GroupCatalog

        return GroupCatalog(self)

    @property
    def metadata(self) -> Any:
        """Namespaced application metadata stored beside this level.

        A sanctioned home for what consumers currently put directly in
        the level's ``attrs``, where it can collide with the format's own
        keys and where two writers clobber each other.
        """
        from zarr_vectors.core.store import get_resolution_level
        from zarr_vectors.core.user_metadata import Metadata

        return Metadata(
            get_resolution_level(self._dataset._group, self._index)
        )

    @property
    def grid(self) -> Any:
        """This level's cell grid, in physical units.

        Exposed because two questions about it are genuinely a caller's
        business — "will my data fit this allocation" and "which cells is
        this region" — and answering them by hand means re-deriving the
        allocator, which one consumer does and documents as fragile.
        """
        from zarr_vectors.api.grid import Grid

        bounds = getattr(self._dataset._root_meta, "bounds", None)
        if not bounds or not self.scale:
            return Grid(shape=(), cell_shape=tuple(self.scale))
        return Grid.plan(bounds, cell_size=self.scale)

    # ---------------- introspection ----------------

    def attribute_names(self, kind: str = "vertex") -> tuple[str, ...]:
        """Names of the attributes of one family present at this level.

        Downstream reaches into ``group.zarr_group[...]`` and unions
        ``array_keys()`` with ``group_keys()`` in at least six places
        because there was no way to ask this.  There is now.

        Args:
            kind: ``"vertex"``, ``"fragment"``, ``"object"``, ``"group"``
                or ``"link"``.
        """
        if kind not in _ATTR_GROUPS:
            raise ValueError(
                f"unknown attribute family {kind!r}; expected one of "
                f"{sorted(_ATTR_GROUPS)}"
            )
        from zarr_vectors.core.store import get_resolution_level

        level_group = get_resolution_level(self._dataset._group, self._index)
        try:
            family = level_group[_ATTR_GROUPS[kind]]
        except Exception:
            return ()
        try:
            return tuple(sorted(family.children()))
        except Exception:
            return ()

    # ---------------- selection ----------------

    def select(self, **kw: Any) -> Query:
        """Build a query against this level.  Nothing is read yet."""
        kw.setdefault("level", self._index)
        return Query(self, Selection(**kw))

    def read(self, **kw: Any) -> ReadResult:
        """Shorthand for ``select(**kw).read()``."""
        return self.select(**kw).read()

    async def aread(self, **kw: Any) -> ReadResult:
        """Shorthand for ``select(**kw).aread()``."""
        return await self.select(**kw).aread()

    # ---------------- execution ----------------

    def _reader(self) -> tuple[Any, tuple[str, ...], str]:
        kind = self.kind
        try:
            module_name, func_name, supports, adapter = _READERS[kind]
        except KeyError:
            raise ZVError(
                f"no reader for geometry type {kind!r}; this store declares "
                f"{self._dataset.kinds}"
            ) from None
        import importlib

        return getattr(importlib.import_module(module_name), func_name), supports, adapter

    def _reader_kwargs(self, selection: Selection, supports: Sequence[str]) -> dict[str, Any]:
        """Selection terms this reader can express, with ``"all"`` resolved.

        ``read_points`` returns **no** attributes when ``attribute_names``
        is omitted — the parameter defaults to ``None`` and ``None`` means
        none, not all.  So ``attributes="all"`` has to be turned into the
        actual list, which means asking the level what it holds.  Getting
        this wrong is silent: the read succeeds and simply has no
        attributes in it.
        """
        kwargs = selection.to_reader_kwargs(supports=supports)
        if (
            selection.attributes == "all"
            and "attribute_names" in supports
            and "attribute_names" not in kwargs
        ):
            names = self.attribute_names("vertex")
            if names:
                kwargs["attribute_names"] = list(names)
        return kwargs

    def plan(self, selection: Selection) -> Any:
        """The I/O this selection implies, before any of it is performed.

        Pure metadata arithmetic — building a plan reads nothing.
        """
        from zarr_vectors._engine.resolve import context_from_level, resolve

        return resolve(selection, context_from_level(self))

    def _execute(self, selection: Selection) -> ReadResult:
        """Fetch in batches, then decode.

        The plan is prefetched through the engine and the ordinary
        synchronous reader is then replayed against the snapshot — the
        same prime-and-replay the async path uses, and for the same
        reason: the readers are mostly pure, and only their I/O ever
        needed rearranging.

        ``offline_reads``, not ``batched_reads``, because the readers open
        a ``batched_reads`` block of their own on the level group and
        nesting those raises.  An offline snapshot propagates to derived
        groups instead, and the reader's own block becomes a no-op
        passthrough.

        Any failure falls back to calling the reader directly, and that
        is safe by construction: the fallback is the *identical*
        computation without the optimisation, so either it succeeds — in
        which case the batched attempt was at fault and no caller should
        care — or it raises the genuine error, unobscured.  The cost is
        having done the work twice on the rare read that needs it.
        """
        reader, supports, adapter = self._reader()
        kwargs = self._reader_kwargs(selection, supports)
        group = self._dataset._group
        try:
            from zarr_vectors._engine.execute import execute
            from zarr_vectors._engine.fetch import GroupFetcher

            raw = execute(
                group,
                lambda g: reader(g, **kwargs),
                fetcher=GroupFetcher(group),
                plan=self.plan(selection),
                strict=True,
                label=reader.__name__,
            )
        except Exception:
            raw = reader(group, **kwargs)
        return self._finish(raw, adapter, selection)

    async def _aexecute(self, selection: Selection) -> ReadResult:
        """The same read, with its I/O awaited.

        Reuses :func:`zarr_vectors.core.aio.read_async`, which supplies
        any reader's I/O up front and replays it offline — so the async
        path is the sync reader, not a second implementation of it.
        """
        from zarr_vectors.core.aio import read_async

        reader, supports, adapter = self._reader()
        kwargs = self._reader_kwargs(selection, supports)
        raw = await read_async(reader, self._dataset._group, **kwargs)
        return self._finish(raw, adapter, selection)

    def _finish(self, raw: Any, adapter: str, selection: Selection) -> ReadResult:
        result = getattr(ReadResult, adapter)(raw, kind=self.kind)
        return self._post_filter(result, selection)

    def _post_filter(self, result: ReadResult, selection: Selection) -> ReadResult:
        """Apply the selection terms the reader could not express.

        ``near`` degrades to its bounding box on the way down — the box's
        corners lie outside the sphere — so the sphere is enforced here.
        ``limit`` has no reader equivalent at all.
        """
        if not selection.needs_post_filter or result.vertex_count == 0:
            return result

        keep = np.ones(result.vertex_count, dtype=bool)
        if selection.cells is not None:
            # No reader takes a cell list any more, so the region is
            # enforced here. The plan already narrowed what was fetched;
            # this narrows what is returned.
            grid = self.grid
            wanted = {ref.key for ref in selection.cells}
            keep &= np.array([
                grid.cell_of(p).key in wanted for p in result.positions
            ], dtype=bool)
        if selection.near is not None:
            centre, radius = selection.near
            offset = result.positions - np.asarray(centre, dtype=result.positions.dtype)
            keep &= (offset ** 2).sum(axis=1) <= float(radius) ** 2

        truncated = False
        if selection.limit is not None:
            allowed = int(selection.limit)
            chosen = np.flatnonzero(keep)
            if len(chosen) > allowed:
                keep = np.zeros_like(keep)
                keep[chosen[:allowed]] = True
                truncated = True

        return result.restrict(keep, truncated=truncated)

    def _explain_execution(self, selection: Selection) -> str:
        _, supports, _ = self._reader()
        _, func_name, _, _ = _READERS[self.kind]
        passed = sorted(self._reader_kwargs(selection, supports))
        note = " then filtered in memory" if selection.needs_post_filter else ""
        return f"{func_name}({', '.join(passed)}){note}"

    def __repr__(self) -> str:
        return f"Level({self._index}, kind={self.kind!r}, vertices={self.vertex_count})"


def known_kinds() -> tuple[str, ...]:
    """Geometry types this facade can read."""
    return tuple(sorted(_READERS))


def attribute_families() -> tuple[str, ...]:
    return tuple(sorted(_ATTR_GROUPS))


def _unused(_: Sequence[Any]) -> None:  # pragma: no cover - import hygiene
    pass
