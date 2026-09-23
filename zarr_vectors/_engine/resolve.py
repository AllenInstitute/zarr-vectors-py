"""Turning "what I want" into "what to fetch".

This is the inversion the engine exists for.  Today a reader that wants
to be fast hand-builds its own prefetch plan inline — ``read_points``
does it at ``points.py:712``, and the other four do not do it at all,
which is why they are slower.  Worse, a *caller* who wants a targeted
read has to pass ``chunks=[(3, 1, 2), ...]``: grid coordinates they can
only construct by knowing the grid.

A resolver moves that knowledge inside.  It takes a
:class:`~zarr_vectors.api.select.Selection` and a
:class:`LevelContext` — both pure metadata — and returns a
:class:`~zarr_vectors._engine.plan.ReadPlan`.  It touches no store, so
it is a plain function that can be tested against a golden, and the same
plan serves the sync, batched and async paths without variation.

**``expand`` means "fan out"; ``cells`` means "exactly these".**
Resolving an array's node yields its ``nonempty_chunks``, from which the
fetcher can imply every cell it holds — right for "read the whole
level", ruinous for a bbox query over a large store.  So a resolver that
knows which cells it wants names them in ``cells`` and puts the array in
``nodes`` (which resolves it, for the grid origin, without implying
anything); one that wants everything puts it in ``expand``.

Leaving the node out of a targeted plan does not work, and it was tried:
the reader resolves it anyway on the next round, the miss comes back as
a discovery, and discovery fans out.  The fan-out has to be refused
explicitly, which is why it is a field rather than an inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors._engine.plan import CellRequest, ReadPlan
from zarr_vectors.constants import (
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)

__all__ = ["LevelContext", "resolve"]

# Arrays every geometry read touches, in the order a reader touches them.
_CORE_ARRAYS = (VERTICES, VERTEX_FRAGMENTS)

# The row -> object id table under ``object_index/`` --
# ``core.arrays.OBJECT_IDS_ARRAY``, spelled here so this module keeps
# depending on constants alone.
_OBJECT_IDS_TABLE = "object_ids"


@dataclass(frozen=True, slots=True)
class LevelContext:
    """Everything a resolver needs, and nothing that requires I/O.

    Built once from metadata the facade has already read.  Keeping it a
    value means a resolver is a pure function of it, so a plan can be
    compared against a checked-in golden and a change in the I/O a read
    performs shows up as a diff in one file rather than as a performance
    mystery.
    """

    level: int
    ndim: int
    chunk_shape: tuple[float, ...] | None = None
    bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    attribute_names: tuple[str, ...] = ()
    has_object_index: bool = False
    known_cells: tuple[str, ...] = ()
    """Cells known to be populated, when the caller already knows.

    Empty means "not known", which is different from "none": a resolver
    that cannot enumerate falls back to naming the node and letting the
    fetcher fan out.
    """

    @property
    def prefix(self) -> str:
        return str(self.level)

    def path(self, *parts: str) -> str:
        """Root-relative path of an array in this level."""
        return "/".join((self.prefix, *parts))


@dataclass(frozen=True, slots=True)
class _Wanted:
    """Which arrays a read will touch, before deciding how to fetch them."""

    arrays: tuple[str, ...] = ()
    whole_arrays: tuple[str, ...] = ()
    listings: tuple[str, ...] = field(default=())


def _arrays_for(
    ctx: LevelContext, attributes: Sequence[str] | str, *, by_object: bool = False,
) -> _Wanted:
    """The arrays a read of this level touches.

    Attributes are included only when asked for.  ``read_points`` returns
    none unless named, so requesting every attribute array by default
    would fetch data the reader will then not even decode.

    A read scoped by object does not take the manifests whole.  It wants
    the rows of the ids it names, and those it discovers on the first
    pass (see ``Group.read_vlen_elements``) and fetches by coordinate
    selection -- the difference between reading 200 of 21 million
    manifests and reading all of them.  What it does take whole is the
    id table, which maps those ids to rows and is one integer per
    object.
    """
    arrays = list(_CORE_ARRAYS)
    if attributes == "all":
        arrays += [f"{VERTEX_ATTRIBUTES}/{n}" for n in ctx.attribute_names]
    elif not isinstance(attributes, str):
        wanted = set(attributes)
        arrays += [
            f"{VERTEX_ATTRIBUTES}/{n}" for n in ctx.attribute_names if n in wanted
        ]
    whole = []
    if ctx.has_object_index:
        if by_object:
            whole.append(ctx.path(OBJECT_INDEX, _OBJECT_IDS_TABLE))
        else:
            whole.append(ctx.path(OBJECT_INDEX, "manifests"))
    return _Wanted(arrays=tuple(arrays), whole_arrays=tuple(whole))


def _match_known(
    named: Sequence[str], known: Sequence[str], rank: int,
) -> tuple[str, ...]:
    """The cells in ``known`` that the keys in ``named`` select, sorted.

    A key of the level's own rank selects itself.  A spatial key -- what
    ``Grid`` hands out, and what a box expands to -- selects every known
    cell with that spatial tail: on a level chunked by an attribute, one
    per bin, compared on the trailing components as :func:`_key_in_box`
    is.  On a spatial level the tail is the whole key, so this is an
    exact match there.
    """
    known_set = set(known)
    by_tail: dict[str, list[str]] | None = None
    out: set[str] = set()
    for key in named:
        if key in known_set:
            out.add(key)
            continue
        if key.count(".") + 1 != rank:
            continue
        if by_tail is None:
            by_tail = {}
            for k in known:
                by_tail.setdefault(".".join(k.split(".")[-rank:]), []).append(k)
        out.update(by_tail.get(key, ()))
    return tuple(sorted(out))


def _key_in_box(
    key: str, lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64],
) -> bool:
    """Whether a dotted chunk key falls inside an inclusive cell box.

    Compared on the **trailing** ``len(lo)`` components, because a store
    written with ``chunk_by_attribute`` prefixes every key with a bin
    axis and the box is spatial.  For an un-binned store the tail is the
    whole key, so this is an identity there.
    """
    parts = key.split(".")
    if len(parts) < lo.size:
        return False
    try:
        tail = [int(p) for p in parts[-lo.size:]]
    except ValueError:
        return False
    return all(
        int(lo[d]) <= v <= int(hi[d]) for d, v in enumerate(tail)
    )


def _cells_in_bbox(
    ctx: LevelContext, bbox: tuple[Any, Any],
) -> tuple[str, ...] | None:
    """Chunk keys a bounding box touches, or ``None`` if not derivable.

    ``None`` when the level does not declare a chunk shape, in which case
    the caller falls back to fanning out.

    When the context carries :attr:`LevelContext.known_cells` the answer
    is the *intersection* of the box with what the level actually holds,
    and the cheaper of the two enumerations is used to compute it —
    walking the presence manifest when the box spans more cells than the
    level has, and the box otherwise.  That matters more than it sounds:
    the box is derived from the declared grid, so a sparse store whose
    grid allocates a million cells for a thousand occupied ones would
    otherwise plan a million cell fetches, and the fetcher would perform
    them.  The same probe-or-scan choice is made by
    :func:`zarr_vectors.core.arrays._chunks_in_box` for the sync readers.
    """
    if not ctx.chunk_shape:
        return None
    from zarr_vectors.spatial.chunking import chunks_intersecting_bbox

    lo = np.asarray(bbox[0], dtype=np.float64)
    hi = np.asarray(bbox[1], dtype=np.float64)
    if lo.shape != hi.shape or lo.size != len(ctx.chunk_shape):
        return None

    if ctx.known_cells:
        cs = np.asarray(ctx.chunk_shape, dtype=np.float64)
        lo_c = np.floor(lo / cs).astype(np.int64)
        hi_c = np.floor(hi / cs).astype(np.int64)
        n_candidates = int(np.prod(hi_c - lo_c + 1, dtype=np.int64))
        if n_candidates > len(ctx.known_cells):
            return tuple(sorted(
                k for k in ctx.known_cells if _key_in_box(k, lo_c, hi_c)
            ))

    coords = chunks_intersecting_bbox(lo, hi, tuple(ctx.chunk_shape))
    keys = [".".join(str(int(c)) for c in cc) for cc in coords]
    if ctx.known_cells:
        return _match_known(keys, ctx.known_cells, len(ctx.chunk_shape))
    return tuple(sorted(set(keys)))


def resolve(selection: Any, ctx: LevelContext) -> ReadPlan:
    """The plan for one selection against one level.

    Args:
        selection: A :class:`~zarr_vectors.api.select.Selection`.  Typed
            loosely so this module does not import the public API and
            create a cycle.
        ctx: Metadata about the level.  No I/O is performed on it.

    Returns:
        A plan.  It may be incomplete — a decoder that wants more will
        say so and the executor will fetch it — but every cell it does
        name is one the read is very likely to need.
    """
    by_object = (
        getattr(selection, "objects", None) is not None
        or getattr(selection, "groups", None) is not None
    )
    wanted = _arrays_for(
        ctx, getattr(selection, "attributes", "all"),
        by_object=by_object and ctx.has_object_index,
    )
    plan = ReadPlan.of(
        nodes=[ctx.prefix],
        arrays=wanted.whole_arrays,
    )

    bbox = getattr(selection, "bbox", None)
    near = getattr(selection, "near", None)
    if near is not None:
        centre = np.asarray(near[0], dtype=np.float64)
        radius = float(near[1])
        sphere = (centre - radius, centre + radius)
        bbox = sphere if bbox is None else (
            np.maximum(np.asarray(bbox[0], dtype=np.float64), sphere[0]),
            np.minimum(np.asarray(bbox[1], dtype=np.float64), sphere[1]),
        )

    explicit = getattr(selection, "cells", None)
    if explicit is not None:
        # The caller named the region in grid terms, so the region is
        # theirs and is not re-derived. Cells the level is known not to
        # hold are still dropped: fetching one returns nothing, so the
        # result is identical and the request is not made. That is what
        # keeps ``select(cells=grid.cells_in(...))`` proportional to the
        # data rather than to the grid, which for a sparse store are
        # different by orders of magnitude.
        named = sorted({ref.key for ref in explicit})
        cells: tuple[str, ...] | None
        if ctx.known_cells and ctx.chunk_shape:
            # A spatial ref names that cell in every bin of a binned
            # level; matched as a box is, on the spatial tail.
            cells = _match_known(named, ctx.known_cells, len(ctx.chunk_shape))
        elif ctx.known_cells:
            known = set(ctx.known_cells)
            cells = tuple(k for k in named if k in known)
        else:
            cells = tuple(named)
    else:
        cells = _cells_in_bbox(ctx, bbox) if bbox is not None else None

    if cells is None:
        if by_object and ctx.has_object_index:
            # Scoped by the manifests, not by the grid: which cells an
            # object read touches is written in the object index, and the
            # reader names them -- all of them, in one plan -- once it
            # has read the manifests.  Fanning out here instead read the
            # entire level to answer for one object: 3.8 s per
            # ``level.objects[i]`` on a million-point store, the same as
            # reading everything.  The nodes are still resolved, so the
            # reader has its metadata and the fetcher its grid.
            return plan.merge(ReadPlan.of(nodes=[ctx.path(a) for a in wanted.arrays]))
        # Nothing narrows the read, or the grid is not declared. Fan out
        # and take the whole level in one round-trip, which is what a
        # full read wants anyway.
        return plan.merge(ReadPlan.of(expand=[ctx.path(a) for a in wanted.arrays]))

    # A region is known, so name its cells exactly. The nodes are still
    # resolved -- the reader needs their metadata -- but they are not in
    # ``expand``, so resolving them implies nothing and the narrowing
    # survives.
    return plan.merge(ReadPlan.of(
        nodes=[ctx.path(a) for a in wanted.arrays],
        cells=[
            CellRequest(ctx.path(array), key)
            for array in wanted.arrays
            for key in cells
        ],
    ))


def context_from_level(level: Any) -> LevelContext:
    """Build a :class:`LevelContext` from a facade ``Level``.

    Reads metadata only — the level's attrs, and the presence manifest
    that names its populated cells — never chunk data.  Both come off
    nodes the resulting plan resolves anyway, and inside a
    :meth:`~zarr_vectors.core.group.Group.cached_nodes` block they are
    already in hand, so this does not add a round-trip to a read.

    The presence manifest is what stops a narrowed read from scaling
    with the *declared grid* instead of the data: see
    :func:`_cells_in_bbox`.  When it cannot be read, ``known_cells``
    stays empty, which means "not known" and preserves the previous
    fan-out behaviour rather than pretending the level is empty.
    """
    dataset = level.dataset
    root = dataset._root_meta
    bounds = getattr(root, "bounds", None)
    return LevelContext(
        level=level.index,
        ndim=dataset.ndim,
        chunk_shape=tuple(level.scale) or None,
        bounds=(
            (tuple(float(v) for v in bounds[0]), tuple(float(v) for v in bounds[1]))
            if bounds else None
        ),
        attribute_names=level.attribute_names("vertex"),
        has_object_index=True,
        known_cells=_known_cells(level),
    )


def _known_cells(level: Any) -> tuple[str, ...]:
    """The level's populated ``vertices`` cells, or ``()`` when unknown.

    ``()`` is the honest answer for a level with no ``vertices`` array,
    an unreadable manifest, or a natively sharded array whose manifest
    the listing cannot resolve — in every case the resolver falls back
    to fanning out, which is correct but unnarrowed.
    """
    from zarr_vectors.exceptions import ArrayError, StoreError

    try:
        return tuple(level.store.list_chunks(VERTICES))
    except (ArrayError, StoreError, KeyError, AttributeError):
        return ()


def _unused(_: tuple[str, ...] = (OBJECT_ATTRIBUTES,)) -> None:  # pragma: no cover
    """Keeps the constant import honest until object attributes are planned."""
