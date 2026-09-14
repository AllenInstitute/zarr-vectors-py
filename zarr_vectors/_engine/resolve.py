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


def _arrays_for(ctx: LevelContext, attributes: Sequence[str] | str) -> _Wanted:
    """The arrays a read of this level touches.

    Attributes are included only when asked for.  ``read_points`` returns
    none unless named, so requesting every attribute array by default
    would fetch data the reader will then not even decode.
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
        whole.append(ctx.path(OBJECT_INDEX, "manifests"))
    return _Wanted(arrays=tuple(arrays), whole_arrays=tuple(whole))


def _cells_in_bbox(
    ctx: LevelContext, bbox: tuple[Any, Any],
) -> tuple[str, ...] | None:
    """Chunk keys a bounding box touches, or ``None`` if not derivable.

    Pure arithmetic on the grid the store declares: no listing, no
    metadata read.  ``None`` when the level does not declare a chunk
    shape, in which case the caller falls back to fanning out.
    """
    if not ctx.chunk_shape:
        return None
    from zarr_vectors.spatial.chunking import chunks_intersecting_bbox

    lo = np.asarray(bbox[0], dtype=np.float64)
    hi = np.asarray(bbox[1], dtype=np.float64)
    if lo.shape != hi.shape or lo.size != len(ctx.chunk_shape):
        return None
    coords = chunks_intersecting_bbox(lo, hi, tuple(ctx.chunk_shape))
    keys = [".".join(str(int(c)) for c in cc) for cc in coords]
    if ctx.known_cells:
        known = set(ctx.known_cells)
        keys = [k for k in keys if k in known]
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
    wanted = _arrays_for(ctx, getattr(selection, "attributes", "all"))
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
        # The caller named the region in grid terms. Honour it exactly:
        # they got the references from the grid, so second-guessing them
        # would only mean re-deriving what they already resolved.
        cells: tuple[str, ...] | None = tuple(sorted(
            ref.key for ref in explicit
        ))
    else:
        cells = _cells_in_bbox(ctx, bbox) if bbox is not None else None

    if cells is None:
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

    Reads only metadata the facade has already cached, so this is cheap
    and does not turn plan construction into I/O.
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
    )


def _unused(_: tuple[str, ...] = (OBJECT_ATTRIBUTES,)) -> None:  # pragma: no cover
    """Keeps the constant import honest until object attributes are planned."""
