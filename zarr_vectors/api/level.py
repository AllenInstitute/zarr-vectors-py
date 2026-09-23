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
import numpy.typing as npt

from zarr_vectors.api.result import Attributes, ReadResult
from zarr_vectors.api.select import Query, Selection
from zarr_vectors.constants import (
    FRAGMENT_ATTRIBUTES,
    GEOM_GRAPH,
    GEOM_LINE,
    GEOM_MESH,
    GEOM_POINT_CLOUD,
    GEOM_POLYLINE,
    GEOM_SKELETON,
    GEOM_STREAMLINE,
    GROUP_ATTRIBUTES,
    LINK_ATTRIBUTES,
    OBJECT_ATTRIBUTES,
    VERTEX_ATTRIBUTES,
)
from zarr_vectors.exceptions import (
    ArrayError,
    MetadataError,
    StoreError,
    ZVError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.api.dataset import Dataset

__all__ = ["Level"]

# Which reader serves which geometry, which of its keyword arguments
# exist, and how its result dict maps into the uniform one.  The kwarg
# lists are not decoration: the five readers genuinely differ (only
# read_points takes attribute_names, only read_lines lacks chunks), and
# passing one a keyword it does not have is a TypeError at the call site.
#
# ``chunks`` was omitted from every entry even though four of the five
# readers accept it, so a ``cells=`` selection could only ever be applied
# after assembly -- which silently changed how a partly-kept object was
# counted.  See ``Selection.cells_split``.
_READERS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    GEOM_POINT_CLOUD: (
        "zarr_vectors.types.points", "read_points",
        ("bbox", "object_ids", "group_ids", "chunks", "attribute_names",
         "attribute_filter"),
        "from_points",
    ),
    GEOM_LINE: (
        "zarr_vectors.types.lines", "read_lines",
        ("bbox", "object_ids", "attribute_filter"),
        "from_lines",
    ),
    GEOM_POLYLINE: (
        "zarr_vectors.types.polylines", "read_polylines",
        ("bbox", "object_ids", "group_ids", "chunks", "attribute_filter"),
        "from_polylines",
    ),
    GEOM_STREAMLINE: (
        "zarr_vectors.types.polylines", "read_polylines",
        ("bbox", "object_ids", "group_ids", "chunks", "attribute_filter"),
        "from_polylines",
    ),
    # ``object_ids`` stays listed for mesh/graph/skeleton even though
    # those readers raise NotImplementedError for it.  Dropping it does
    # NOT help: ``to_reader_kwargs`` silently discards a term the reader
    # cannot express, and ``_post_filter`` has nothing to filter on
    # (neither reader returns per-vertex object ids), so the read would
    # come back as the whole level with the caller believing they had
    # scoped it.  Raising is the right answer; ``_check_supported``
    # below just raises it with a message about what the CALLER typed.
    GEOM_MESH: (
        "zarr_vectors.types.meshes", "read_mesh",
        ("bbox", "object_ids", "chunks", "attribute_filter"),
        "from_mesh",
    ),
    GEOM_GRAPH: (
        "zarr_vectors.types.graphs", "read_graph",
        ("bbox", "object_ids", "chunks", "attribute_filter"),
        "from_graph",
    ),
    GEOM_SKELETON: (
        "zarr_vectors.types.graphs", "read_graph",
        ("bbox", "object_ids", "chunks", "attribute_filter"),
        "from_graph",
    ),
}

# Attribute families, by the group each lives under in a level.
#
# Spelled with the constants, not with string literals.  ``"group"``
# pointed at ``groupings_attributes``, which is not a path this format has
# -- the family is ``group_attributes``, which is what
# ``write_groupings_attributes`` creates and what the README promises
# ``attribute_names("group")`` will find.  It silently answered ``()`` for
# every store, and a literal is exactly the kind of thing that drifts
# without anything noticing.
_ATTR_GROUPS: dict[str, str] = {
    "vertex": VERTEX_ATTRIBUTES,
    "fragment": FRAGMENT_ATTRIBUTES,
    "object": OBJECT_ATTRIBUTES,
    "group": GROUP_ATTRIBUTES,
    "link": LINK_ATTRIBUTES,
}


def _narrows(selection: Selection) -> bool:
    """Does ``selection`` cut the result down to less than the whole level?

    Every field of a :class:`~zarr_vectors.api.select.Selection` narrows
    except ``level`` (which chooses one) and the presentation flags, so
    this is the full set that makes a level-ordered column unalignable.
    No I/O: the Selection is a value.
    """
    return any((
        selection.bbox is not None,
        selection.near is not None,
        selection.objects is not None,
        selection.groups is not None,
        selection.where is not None,
        selection.cells is not None,
    ))


def _rows_in(
    rows: npt.NDArray[np.int64], wanted: set[tuple[int, ...]],
) -> npt.NDArray[np.bool_]:
    """Which of ``rows`` appear in ``wanted``, without a Python loop.

    A cell post-filter asks this once per vertex, and asking it with
    ``tuple(row) in wanted`` builds a Python tuple per vertex: 4.4s and
    358 MB to keep one sixty-fourth of five million points.  Viewing each
    row as a single structured scalar turns it into one ``np.isin``.
    """
    if not wanted:
        return np.zeros(len(rows), dtype=bool)
    width = rows.shape[1]
    dt = np.dtype([(f"f{i}", np.int64) for i in range(width)])
    want = np.asarray(sorted(wanted), dtype=np.int64)
    if want.ndim != 2 or want.shape[1] != width:
        # A ref of a different arity cannot match any row, and reshaping
        # it to force a comparison would invent an answer.
        return np.zeros(len(rows), dtype=bool)
    return np.isin(
        np.ascontiguousarray(rows).view(dt).ravel(),
        np.ascontiguousarray(want).view(dt).ravel(),
    )


class Level:
    """One resolution level of a :class:`~zarr_vectors.api.dataset.Dataset`."""

    __slots__ = ("_dataset", "_index", "_meta", "_attr_names")

    def __init__(self, dataset: Dataset, index: int) -> None:
        self._dataset = dataset
        self._index = int(index)
        self._meta: Any = None
        # Per-family attribute names, listed once.  Lives and dies with
        # this handle exactly as ``_meta`` does: the Dataset drops its
        # Levels on every write it performs.
        self._attr_names: dict[str, tuple[str, ...]] = {}

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
        from zarr_vectors.core.user_metadata import Metadata

        return Metadata(self.store)

    @property
    def grid(self) -> Any:
        """This level's cell grid, in physical units.

        Exposed because two questions about it are genuinely a caller's
        business — "will my data fit this allocation" and "which cells is
        this region" — and answering them by hand means re-deriving the
        allocator, which one consumer does and documents as fragile.

        The grid is **spatial**.  On a level chunked by an attribute the
        arrays carry a leading bin axis as well, so ``grid.shape`` is the
        spatial tail of their shape and ``grid.cells`` counts spatial
        cells, not keys.  A cell from the grid selects that cell in every
        bin; :meth:`cells` lists the keys the level actually holds, bin
        included, and one of those selects one bin's cell.
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
        # Listed once per handle.  A listing reads every child's
        # ``zarr.json``, and a read asked for it twice -- to plan, then to
        # resolve ``attributes="all"`` -- so on a one-cell query the two
        # listings were a third of the whole read.
        cached = self._attr_names.get(kind)
        if cached is not None:
            return cached
        names = self._list_attribute_names(kind)
        self._attr_names[kind] = names
        return names

    def _list_attribute_names(self, kind: str) -> tuple[str, ...]:
        level_group = self.store
        try:
            family = level_group[_ATTR_GROUPS[kind]]
        except Exception:
            return ()
        try:
            return tuple(sorted(family.children()))
        except Exception:
            return ()

    # ---------------- selection ----------------

    @property
    def store(self) -> Any:
        """This LEVEL's :class:`~zarr_vectors.core.group.Group` handle.

        The same escape hatch :attr:`Dataset.store` offers, at the
        granularity every :mod:`zarr_vectors.building` function takes:
        those want a *level* group, so reaching them from an api ``Level``
        otherwise meant ``get_resolution_level(level._dataset._group,
        level.index)`` — two internal reaches to express "read through the
        api, patch through building".

        Deliberately the level group and not the root: the legacy readers
        take a root handle plus ``level=``, so a ``store`` that returned
        the root would look interchangeable with this one and hand every
        ``building`` writer a handle it reads as a level — allocating the
        array families at the store root.
        """
        from zarr_vectors.core.store import get_resolution_level

        return get_resolution_level(self._dataset._group, self._index)

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

    def _check_supported(self, selection: Selection) -> None:
        """Refuse a selection term this geometry's reader cannot honour.

        ``read_mesh`` and ``read_graph`` raise for ``object_ids=``, which
        is right -- silently handing back the whole level is the worst
        outcome -- but their message names a parameter of a function the
        caller never called.  Say it in the caller's own terms instead.
        """
        if selection.objects is None and selection.groups is None:
            return
        if self.kind not in (GEOM_MESH, GEOM_GRAPH, GEOM_SKELETON):
            return
        term = "objects=" if selection.objects is not None else "groups="
        raise ZVError(
            f"select({term}...) is not supported for {self.kind!r}: the "
            f"reader behind it cannot scope a read by object, and applying "
            f"the filter afterwards is impossible because the result "
            f"carries no per-vertex object ids. Read the level and filter "
            f"the arrays yourself, or use level.objects[...] on a geometry "
            f"that supports it (point clouds, polylines, lines)."
        )

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
        # A Selection built directly (rather than through Level.select)
        # may carry level=None, meaning "not specified" -- resolve it to
        # the level this query is actually bound to.
        if kwargs.get("level") is None:
            kwargs["level"] = self._index
        if (
            selection.attributes == "all"
            and "attribute_names" in supports
            and "attribute_names" not in kwargs
        ):
            names = self.attribute_names("vertex")
            if names:
                kwargs["attribute_names"] = list(names)
        if "chunks" in supports and "chunks" not in kwargs:
            prefix = self._cells_covering_limit(selection)
            if prefix is not None:
                kwargs["chunks"] = prefix
        return kwargs

    def _cells_covering_limit(
        self, selection: Selection,
    ) -> list[tuple[int, ...]] | None:
        """Enough leading cells to satisfy ``limit``, or ``None``.

        A limit was applied only after the read, so ``select(limit=10)``
        against five million points read all five million and then kept
        ten -- 239 MB of peak memory for ten rows.  The reader emits
        cells in sorted key order and the post-filter keeps the first
        ``limit`` survivors, so the same rows come from a prefix of that
        order.  Which prefix is decided by the fragment indices, walked
        in order and stopped at the first cell that makes the total
        enough -- index reads, not vertex reads.

        ``None`` whenever anything else narrows the selection, since then
        the surviving rows are not a prefix of the cells, and whenever
        the indices cannot be read, which simply leaves the previous
        read-everything behaviour in place.
        """
        limit = selection.limit
        if (
            limit is None
            or selection.bbox is not None
            or selection.near is not None
            or selection.objects is not None
            or selection.groups is not None
            or selection.cells is not None
            or selection.where is not None
        ):
            return None
        if int(limit) <= 0:
            return None
        from zarr_vectors.core.arrays import (
            _fragment_vertex_extent,
            list_chunk_keys,
            read_vertex_fragment_index,
        )

        target = (
            self if selection.level in (None, self._index)
            else self._dataset.level(selection.level)
        )
        level_group = target.store
        try:
            keys = list_chunk_keys(level_group)
        except (ArrayError, StoreError, KeyError):
            return None
        if not keys:
            return None
        taken: list[tuple[int, ...]] = []
        total = 0
        for cc in keys:
            taken.append(tuple(int(c) for c in cc))
            try:
                total += _fragment_vertex_extent(
                    read_vertex_fragment_index(level_group, cc),
                )
            except (ArrayError, StoreError, KeyError):
                return None
            if total >= int(limit):
                break
        # Every cell was needed, so naming them narrows nothing.
        return None if len(taken) == len(keys) else taken

    def cells(
        self, bbox: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> Any:
        """The cells this level actually holds, optionally within a box.

        The counterpart to :meth:`Grid.cells_in`, which enumerates the
        *allocation*: a grid is a pure value and cannot know what is in
        the store, so a sparse level -- a specimen bounding box with data
        in part of it -- answers ``cells_in`` with a reference per
        allocated cell. Building a million of those takes about nine
        seconds before any read happens, and all but a thousand of them
        name nothing.

        This asks the level instead, so the answer is proportional to the
        data. It is the better argument to ``select(cells=...)`` whenever
        the caller wants "what is here", and ``Grid.cells_in`` remains
        the right one for "what could be here".
        """
        import numpy as _np

        from zarr_vectors.api.grid import CellRef, CellSet
        from zarr_vectors.core.arrays import list_chunk_keys, resolve_chunk_keys

        level_group = self.store
        if bbox is None:
            coords = list_chunk_keys(level_group)
        else:
            coords = resolve_chunk_keys(
                level_group,
                tuple(self.scale),
                bbox=(
                    _np.asarray(bbox[0], dtype=_np.float64),
                    _np.asarray(bbox[1], dtype=_np.float64),
                ),
            )
        return CellSet(CellRef(tuple(int(c) for c in cc)) for cc in coords)

    def _count_without_reading(self, selection: Selection) -> int | None:
        """The vertex count, when it can be had from metadata alone.

        ``None`` when the selection narrows anything, because then the
        answer depends on the data and there is nothing stored that
        knows it.  An unnarrowed count is the level's own
        ``vertex_count``, which every writer stamps -- reading five
        million points to count them cost 236 MB of peak memory to
        return a number already on disk.
        """
        if (
            selection.bbox is not None
            or selection.near is not None
            or selection.objects is not None
            or selection.groups is not None
            or selection.cells is not None
            or selection.limit is not None
            or selection.where is not None
        ):
            return None
        target = (
            self if selection.level in (None, self._index)
            else self._dataset.level(selection.level)
        )
        try:
            count = getattr(target._metadata(), "vertex_count", None)
        except (ArrayError, StoreError, MetadataError, KeyError):
            return None
        return None if count is None else int(count)

    def plan(self, selection: Selection) -> Any:
        """The I/O this selection implies, before any of it is performed.

        Metadata arithmetic: the level's attrs and its presence manifest
        are read so a narrowed plan can name the cells that actually
        hold data, and no chunk is fetched.  Inside a
        :meth:`~zarr_vectors.core.group.Group.cached_nodes` block those
        reads are already paid for.
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
        self._check_supported(selection)
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
        except (ArrayError, StoreError, ZVError, KeyError, ValueError) as e:
            # Narrowed from a bare ``except Exception``.  The fallback is
            # sound -- it is the identical computation without the
            # batching -- but catching everything meant an engine bug and
            # a genuine store failure were indistinguishable, each costing
            # the read twice and neither ever being seen.  A
            # NotImplementedError or a KeyboardInterrupt now propagates.
            import warnings

            warnings.warn(
                f"batched read failed ({type(e).__name__}: {e}); falling "
                f"back to the direct reader. The answer is the same; the "
                f"read is slower, and this is worth reporting.",
                RuntimeWarning, stacklevel=2,
            )
            raw = reader(group, **kwargs)
        return self._finish(raw, adapter, selection)

    async def _aexecute(self, selection: Selection) -> ReadResult:
        """The same read, with its I/O awaited.

        Reuses :func:`zarr_vectors.core.aio.read_async`, which supplies
        any reader's I/O up front and replays it offline — so the async
        path is the sync reader, not a second implementation of it.
        """
        from zarr_vectors.core.aio import read_async

        self._check_supported(selection)
        reader, supports, adapter = self._reader()
        kwargs = self._reader_kwargs(selection, supports)
        raw = await read_async(reader, self._dataset._group, **kwargs)
        return self._finish(raw, adapter, selection)

    def _finish(self, raw: Any, adapter: str, selection: Selection) -> ReadResult:
        result = getattr(ReadResult, adapter)(raw, kind=self.kind)
        result = self._gather_attributes(result, selection)
        return self._post_filter(result, selection)

    def _wanted_attributes(self, selection: Selection) -> tuple[str, ...]:
        """The vertex attributes this selection asks for, resolved."""
        if isinstance(selection.attributes, str):
            if selection.attributes != "all":
                return (selection.attributes,)
            return self.attribute_names("vertex")
        return tuple(selection.attributes)

    def _gather_attributes(
        self, result: ReadResult, selection: Selection,
    ) -> ReadResult:
        """Attach per-vertex attributes for a reader that cannot return them.

        Only ``read_points`` takes ``attribute_names``; the other four
        adapters hardcode ``attributes_read=False``.  So a polyline level
        advertised ``level.attribute_names("vertex") == ("fa",)`` and then
        answered ``level.read().attributes == {}`` — the request was
        dropped by ``to_reader_kwargs`` (which silently discards a term
        the reader cannot express) and nothing downstream recovered it.

        The gather is level-ordered: ``chunk_local_to_global_offsets``
        defines the same vertex order the readers concatenate in, so the
        columns line up with ``result.positions`` before any post-filter
        runs.  ``restrict`` then cuts attributes alongside positions.

        Anything unreadable leaves ``attributes_read`` False rather than
        reporting an empty mapping as a complete answer.
        """
        if result.attributes_read or result.vertex_count == 0:
            return result
        wanted = self._wanted_attributes(selection)
        if not wanted:
            return result
        if _narrows(selection):
            # A level-ordered column cannot be aligned to a narrowed
            # reading, so the gather below would be discarded by the
            # count check.  Answered from the Selection, before the
            # scan, because finding it out the expensive way meant
            # decompressing every vertices chunk in the level and
            # throwing the result away — on every narrowed read that
            # asked for attributes.
            return result

        from dataclasses import replace as _replace

        from zarr_vectors.core.arrays import (
            _maybe_batched_reads,
            read_chunk_attribute_rows,
        )
        from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets

        try:
            level_group = (
                self.store if selection.level in (None, self._index)
                else self._dataset.level(selection.level).store
            )
            _offsets, chunk_keys, total = chunk_local_to_global_offsets(level_group)
            if total != result.vertex_count:
                # The reader already narrowed the result (bbox, objects,
                # cells): a level-ordered column cannot be aligned to it
                # without knowing which vertices survived, and guessing
                # would silently mis-pair values with positions.
                return result
            # One gather for every (attribute, cell) pair rather than a
            # sequential read each. On an object store the unbatched form
            # is one round-trip per cell per attribute, and even locally
            # the fixed per-cell cost dominates: reading one attribute of
            # a 64-cell level took 0.96s against 0.04s for the positions.
            from zarr_vectors.constants import VERTEX_ATTRIBUTES, VERTEX_FRAGMENTS

            key_strs = [
                cc if isinstance(cc, str) else ".".join(str(int(c)) for c in cc)
                for cc in chunk_keys
            ]
            plan = [
                (f"{VERTEX_ATTRIBUTES}/{name}", key_strs) for name in wanted
            ]
            plan.append((VERTEX_FRAGMENTS, key_strs))
            gathered: dict[str, Any] = {}
            with _maybe_batched_reads(level_group, plan):
                for name in wanted:
                    # Each chunk's rows whole, not partitioned per
                    # fragment and re-joined: see read_chunk_attribute_rows.
                    flat = [
                        rows for rows in (
                            read_chunk_attribute_rows(level_group, name, cc)
                            for cc in chunk_keys
                        )
                        if len(rows)
                    ]
                    if not flat:
                        continue
                    col = np.concatenate(flat, axis=0)
                    if col.shape[0] == result.vertex_count:
                        gathered[name] = col
            if not gathered:
                return result
            return _replace(
                result,
                attributes=Attributes({**dict(result.attributes), **gathered}),
                attributes_read=True,
            )
        except Exception:
            return result

    def _post_filter(self, result: ReadResult, selection: Selection) -> ReadResult:
        """Apply the selection terms the reader could not express.

        ``near`` degrades to its bounding box on the way down — the box's
        corners lie outside the sphere — so the sphere is enforced here.
        ``limit`` has no reader equivalent at all.
        """
        if not selection.needs_post_filter or result.vertex_count == 0:
            return result

        _, supports, _ = self._reader()
        keep = np.ones(result.vertex_count, dtype=bool)
        if selection.cells is not None and not selection.cells_pushed_down(
            supports=supports
        ):
            # Only when the reader could not take the cells itself.
            #
            # Vectorised: the per-point ``grid.cell_of(p).key in wanted``
            # this replaces ran a Python call per vertex (~3.9 us), which
            # is minutes on a hundred-million-vertex level -- for a
            # membership test that is one floor-divide and one lookup.
            #
            # Against the grid of the level the SELECTION names, not
            # self.grid: on a pyramid with per-level chunk_scale_factors
            # the two have different cell sizes.
            target = (
                self if selection.level in (None, self._index)
                else self._dataset.level(selection.level)
            )
            grid = target.grid
            cell = np.asarray(grid.cell_shape, dtype=np.float64)
            if cell.size != result.positions.shape[1]:
                raise ZVError(
                    f"level {target.index} declares no cell size, so a "
                    f"cells= selection cannot be resolved against it"
                )
            # On the spatial tail: a ref from ``Level.cells`` on a level
            # chunked by an attribute leads with a bin, and a position has
            # no bin to compare it with.  Such a ref therefore keeps its
            # cell in every bin here -- only a reader that takes the cells
            # itself can tell bins apart.
            rank = cell.size
            wanted = {
                tuple(int(c) for c in getattr(ref, "coords", ref))[-rank:]
                for ref in selection.cells
            }
            # Absolute, matching ref.coords and the keys on disk.  There is
            # no origin term because ``assign_chunks`` has none either; the
            # subtraction that used to be here made this the one place that
            # disagreed with every reference it was comparing against, and
            # it also crashed outright on a level with no declared bounds,
            # where ``grid.origin`` is ``()`` rather than the ``None`` the
            # old guard tested for.
            idx = np.floor(
                np.asarray(result.positions, dtype=np.float64) / cell
            ).astype(np.int64)
            keep &= _rows_in(idx, wanted)
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
