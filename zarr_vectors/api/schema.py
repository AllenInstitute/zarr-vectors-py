"""Declaring what the data *is*, and — separately — how it is stored.

Today a caller who wants a point cloud must also decide a chunk shape, a
bin shape, a shard shape, a compressor and a backend, in the same call,
with no defaults that work.  Four of those five are storage decisions
that depend on the store's size and location, not on the data; the fifth
is not a decision at all, since the backend is derivable from the URL.

So they are split.  :class:`Schema` says what the data is and carries no
storage term.  :class:`Layout` says how to store it, every field has a
working ``"auto"``, and it is the single place the four physical
parameters are computed.  A caller who never touches ``Layout`` gets a
sensible store; a caller with an externally fixed grid sets one field.

The precedence rule for :class:`StorageOptions` mirrors
:func:`zarr_vectors.core.backends.resolve_backend_name` — explicit
argument, then ``$ZARR_VECTORS_BACKEND``, then scheme detection.  Keeping
the environment variable as the second step is what lets ``backend=``
disappear from every public signature without any capability being lost.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from zarr_vectors.constants import GEOM_POINT_CLOUD
from zarr_vectors.exceptions import MetadataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.metadata import LevelMetadata, RootMetadata

__all__ = [
    "AttributeSpec",
    "Axis",
    "Layout",
    "ResolvedLayout",
    "Schema",
    "SchemaConflict",
    "SizeHints",
    "StorageOptions",
]

# One storage object per chunk cell is fine locally, where an open() is
# cheap; against an object store each one is a request, so cells get
# packed into shards until a shard is worth a round-trip.  16 MiB is the
# usual sweet spot for S3/GCS throughput.
_DEFAULT_TARGET_OBJECT_BYTES = 16 << 20

# Bins are the spatial-query unit inside a chunk. Four per axis means a
# bbox read touches ~1/64th of a 3-D chunk instead of all of it, at the
# cost of a slightly larger fragment index.
_DEFAULT_SUBCELLS = 4

# With no bounds and no hint there is nothing to divide, so one cell per
# axis is the only honest answer: a single chunk holding everything.
_DEFAULT_CELLS = 1


class SchemaConflict(MetadataError):
    """An existing store disagrees with the schema it was opened with.

    Raised rather than ignored.  ``_create_or_open_store`` currently
    *silently* drops ``bounds`` / ``chunk_shape`` / ``axes`` when the path
    already exists, which is how ``write_polylines(path, chunk_shape=X)``
    can produce a store whose declared grid contradicts its own chunk
    keys — and then a later bbox read looks for coordinates that were
    never written and returns nothing, with no error anywhere.
    """


@dataclass(frozen=True, slots=True)
class Axis:
    """One spatial axis: what it is called and what it is measured in."""

    name: str
    unit: str | None = None
    type: str = "space"

    def to_ngff(self) -> dict[str, str]:
        out = {"name": self.name, "type": self.type}
        if self.unit:
            out["unit"] = self.unit
        return out


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    """One attribute's declared type.

    ``channels`` is the trailing width: 1 for a scalar per vertex, 3 for
    an RGB colour, and so on.
    """

    dtype: str = "float32"
    channels: int = 1
    categorical: bool = False
    unit: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SizeHints:
    """Roughly how much data is coming.

    Only used to pick a layout.  Wrong hints cost performance, never
    correctness, so a caller who does not know should leave them unset
    rather than guess.
    """

    n_vertices: int | None = None
    n_objects: int | None = None
    vertices_per_object: float | None = None


@dataclass(frozen=True, slots=True)
class ResolvedLayout:
    """The physical parameters, computed.

    This is the only object in the package that carries all four of
    ``chunk_shape`` / ``bin_shape`` / ``shard_shape`` / ``compressor``,
    and it is internal: the public surface never names them.
    """

    chunk_shape: tuple[float, ...]
    bin_shape: tuple[float, ...]
    shard_shape: int | tuple[int, ...] | None
    compressor: Any
    bounds: tuple[list[float], list[float]]


@dataclass(frozen=True, slots=True)
class Layout:
    """How the data is physically stored.  Every field has a working
    ``"auto"``.

    ``cells`` is the data-shaped spelling — "cut the volume into about
    this many pieces per axis" — and is what most callers should use.
    ``cell_size`` is the escape hatch for a grid fixed from outside, such
    as a pipeline whose chunks must line up with an image volume's.
    """

    cells: int | Sequence[int] | Literal["auto"] = "auto"
    cell_size: Sequence[float] | None = None
    subcells: int | Literal["auto"] = "auto"
    pack: bool | Literal["auto"] = "auto"
    compression: str | None | Literal["auto"] = "auto"
    target_object_bytes: int = _DEFAULT_TARGET_OBJECT_BYTES

    def resolve(self, schema: Schema, *, store_kind: str = "local") -> ResolvedLayout:
        """Turn this into the four physical parameters.

        ``store_kind`` decides packing: one object per cell is cheap on a
        local filesystem and expensive on an object store, so ``pack``
        defaults on for the latter and off for the former.
        """
        if schema.bounds is None:
            raise MetadataError(
                "Layout.resolve needs Schema.bounds: the cell size is a "
                "fraction of the extent, so there is nothing to divide."
            )
        lo, hi = ([float(v) for v in schema.bounds[0]], [float(v) for v in schema.bounds[1]])
        ndim = len(lo)
        extent = [h - low for low, h in zip(lo, hi)]

        chunk_shape = self._chunk_shape(extent, ndim)
        bin_shape = self._bin_shape(chunk_shape)
        pack = store_kind != "local" if self.pack == "auto" else bool(self.pack)
        shard_shape = self._shard_shape(extent, chunk_shape, schema) if pack else None
        compressor = None if self.compression == "auto" else self.compression
        if self.compression == "auto":
            compressor = os.environ.get("ZARR_VECTORS_COMPRESSION") or None
        return ResolvedLayout(
            chunk_shape=chunk_shape,
            bin_shape=bin_shape,
            shard_shape=shard_shape,
            compressor=compressor,
            bounds=(lo, hi),
        )

    def _chunk_shape(self, extent: list[float], ndim: int) -> tuple[float, ...]:
        if self.cell_size is not None:
            size = tuple(float(v) for v in self.cell_size)
            if len(size) != ndim:
                raise MetadataError(
                    f"Layout.cell_size has {len(size)} axes but the bounds have {ndim}."
                )
            return size
        cells = _DEFAULT_CELLS if self.cells == "auto" else self.cells
        counts = (
            [int(cells)] * ndim if isinstance(cells, int)
            else [int(c) for c in cells]
        )
        if len(counts) != ndim:
            raise MetadataError(
                f"Layout.cells has {len(counts)} axes but the bounds have {ndim}."
            )
        return tuple(
            (e / c if c > 0 else e) or 1.0 for e, c in zip(extent, counts)
        )

    def _bin_shape(self, chunk_shape: tuple[float, ...]) -> tuple[float, ...]:
        # bin_shape must divide chunk_shape exactly, so this divides
        # rather than picking an absolute size a caller might get wrong.
        n = _DEFAULT_SUBCELLS if self.subcells == "auto" else int(self.subcells)
        n = max(1, n)
        return tuple(c / n for c in chunk_shape)

    def _shard_shape(
        self, extent: list[float], chunk_shape: tuple[float, ...], schema: Schema,
    ) -> int | tuple[int, ...] | None:
        grid = [max(1, math.ceil(e / c)) for e, c in zip(extent, chunk_shape)]
        hint = schema.expected.n_vertices if schema.expected else None
        if hint is None:
            # No idea how dense it is; pack the whole grid into one shard
            # per axis-run of 4, which is a safe middle.
            return tuple(min(g, 4) for g in grid)
        total_cells = max(1, math.prod(grid))
        bytes_per_cell = (hint / total_cells) * schema.ndim * 4
        if bytes_per_cell <= 0:
            return tuple(min(g, 4) for g in grid)
        per_shard = max(1, int(self.target_object_bytes // max(1.0, bytes_per_cell)))
        side = max(1, int(round(per_shard ** (1.0 / max(1, schema.ndim)))))
        return tuple(min(g, side) for g in grid)


@dataclass(frozen=True, slots=True)
class StorageOptions:
    """Where the bytes live, and how to reach them.

    Separate from :class:`Schema` because it is not a property of the
    data: the same dataset can be moved between a local disk and a bucket
    without its schema changing at all.
    """

    backend: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    branch: str | None = None

    def resolve_backend(self, url: str) -> str | None:
        """Explicit, then ``$ZARR_VECTORS_BACKEND``, then scheme detection.

        Returns ``None`` when nothing is forced, which is the signal to
        let :func:`zarr_vectors.core.store.open_store` auto-detect.
        """
        if self.backend:
            return self.backend
        return os.environ.get("ZARR_VECTORS_BACKEND") or None


@dataclass(frozen=True, slots=True)
class Schema:
    """What the data is.  No storage term appears here.

    ``profile`` stands in for the three convention strings
    (``links_convention``, ``object_index_convention``,
    ``cross_chunk_strategy``) that callers are currently expected to set
    and that the type writers lazily fill in anyway.  Naming the *use*
    rather than the encoding means a store stays correct when the
    encoding changes.
    """

    ndim: int = 3
    bounds: tuple[Sequence[float], Sequence[float]] | None = None
    axes: tuple[Axis, ...] | None = None
    kind: str = GEOM_POINT_CLOUD
    profile: str = "auto"
    position_dtype: str = "float32"
    vertex_attributes: Mapping[str, AttributeSpec] = field(default_factory=dict)
    object_attributes: Mapping[str, AttributeSpec] = field(default_factory=dict)
    link_attributes: Mapping[str, AttributeSpec] = field(default_factory=dict)
    expected: SizeHints | None = None
    layout: Layout = field(default_factory=Layout)

    def __post_init__(self) -> None:
        if self.bounds is not None:
            lo, hi = self.bounds
            if len(lo) != len(hi):
                raise MetadataError(
                    f"bounds corners disagree: {len(lo)} vs {len(hi)} axes."
                )
            if len(lo) != self.ndim:
                object.__setattr__(self, "ndim", len(lo))
        if self.axes is not None and len(self.axes) != self.ndim:
            raise MetadataError(
                f"{len(self.axes)} axes declared but ndim is {self.ndim}."
            )

    def with_bounds(
        self, lo: Sequence[float], hi: Sequence[float],
    ) -> Schema:
        return replace(self, bounds=(tuple(lo), tuple(hi)))

    @classmethod
    def from_store(
        cls, root_meta: RootMetadata, level_meta: LevelMetadata | None = None,
    ) -> Schema:
        """Reconstruct what a store on disk declares itself to be.

        The inverse of using a Schema to create one, so
        ``open_or_create`` can compare intent against reality rather than
        silently ignoring the caller's.  Physical fields come back inside
        :class:`Layout` as explicit overrides — a store already has a
        grid, and re-deriving one would be a different grid.
        """
        from zarr_vectors.core.metadata import RootMetadata  # noqa: F401

        bounds = getattr(root_meta, "bounds", None)
        chunk_shape = getattr(root_meta, "chunk_shape", None)
        base_bin = getattr(root_meta, "base_bin_shape", None)
        subcells: int | Literal["auto"] = "auto"
        if chunk_shape and base_bin:
            ratios = {
                int(round(c / b)) for c, b in zip(chunk_shape, base_bin) if b
            }
            if len(ratios) == 1:
                subcells = ratios.pop()
        kinds = list(getattr(root_meta, "geometry_types", None) or [])
        return cls(
            # sid_ndim, not spatial_index_dims: the latter is the axes list.
            ndim=int(getattr(root_meta, "sid_ndim", 3) or 3),
            bounds=(tuple(bounds[0]), tuple(bounds[1])) if bounds else None,
            kind=kinds[0] if kinds else GEOM_POINT_CLOUD,
            layout=Layout(
                cell_size=tuple(chunk_shape) if chunk_shape else None,
                subcells=subcells,
            ),
        )

    def diff(self, other: Schema) -> list[str]:
        """Fields on which ``self`` and ``other`` disagree.

        Only the fields a store actually pins are compared: declaring an
        attribute the store does not yet have is a legitimate way to
        extend it, not a conflict.
        """
        out: list[str] = []
        if self.ndim != other.ndim:
            out.append(f"ndim: {self.ndim} != {other.ndim}")
        if self.bounds and other.bounds:
            for i, (a, b) in enumerate(zip(self.bounds, other.bounds)):
                if [float(v) for v in a] != [float(v) for v in b]:
                    corner = "min" if i == 0 else "max"
                    out.append(f"bounds {corner}: {list(a)} != {list(b)}")
        mine, theirs = self.layout.cell_size, other.layout.cell_size
        if mine is not None and theirs is not None:
            if [float(v) for v in mine] != [float(v) for v in theirs]:
                out.append(f"cell size: {list(mine)} != {list(theirs)}")
        return out
