"""The entry point: a dataset you can ask questions of.

``zarr_vectors``' top-level namespace currently exports ten names, all of
them store-shaped — ``Group``, ``create_store``, ``open_store``,
``rebind``.  Not one of them is about the data.  The consequence shows up
downstream: across four consuming repositories only three import sites go
through the top-level package, while ``core.store`` is imported in
seventy-seven files and ``core.arrays`` in sixty-seven.  Every internal
change breaks them, because the internals *are* the interface they were
given.

:class:`Dataset` is the interface they should have been given.  It talks
about bounds, axes, geometry kinds and resolution levels; it never
mentions a chunk, a bin, a shard, a compressor or a backend.  Underneath,
in this phase, it delegates to exactly the readers and writers that exist
today — so adopting it is safe before any internals move, which is the
whole point of introducing it first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from zarr_vectors.api.level import Level
from zarr_vectors.api.result import ReadResult
from zarr_vectors.api.schema import Layout, Schema, SchemaConflict, StorageOptions
from zarr_vectors.api.select import Query
from zarr_vectors.constants import (
    GEOM_GRAPH,
    GEOM_LINE,
    GEOM_MESH,
    GEOM_POINT_CLOUD,
    GEOM_POLYLINE,
)
from zarr_vectors.exceptions import MetadataError, StoreError, ZVError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group

__all__ = [
    "Dataset",
    "FormatError",
    "aopen",
    "create",
    "open",
    "open_or_create",
    "require_format",
]


class FormatError(ZVError):
    """The store's on-disk format is not one this code can serve."""


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(text).split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts or (0,))


class Dataset:
    """A zarr-vectors store, addressed by what is in it.

    Build one with :func:`open`, :func:`create` or :func:`open_or_create`
    rather than by calling the constructor.
    """

    __slots__ = ("_group", "_meta", "_levels", "_storage", "_url")

    def __init__(self, group: Group, *, storage: StorageOptions | None = None) -> None:
        self._group = group
        self._storage = storage or StorageOptions()
        self._meta: Any = None
        self._levels: dict[int, Level] = {}
        self._url: str | None = None

    # ---------------- identity ----------------

    @property
    def url(self) -> str:
        """Where this dataset lives.

        A string for every backend.  ``Group.path`` raises for anything
        that is not a local store, which is why nothing here uses it.
        """
        if self._url is None:
            self._url = str(self._group.url)
        return self._url

    @property
    def _root_meta(self) -> Any:
        if self._meta is None:
            from zarr_vectors.core.store import read_root_metadata

            self._meta = read_root_metadata(self._group)
        return self._meta

    @property
    def format_version(self) -> tuple[int, ...]:
        """On-disk format version, parsed and comparable.

        Downstream currently hard-codes ``REQUIRED_ZV_FORMAT = (0, 9)``
        and reaches for ``constants.FORMAT_VERSION`` to check it, because
        the store never offered the number in a comparable form.
        """
        return _parse_version(getattr(self._root_meta, "zv_version", "0.0.0") or "0.0.0")

    @property
    def capabilities(self) -> frozenset[str]:
        """Optional format features this store declares."""
        return frozenset(getattr(self._root_meta, "format_capabilities", None) or ())

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    # ---------------- shape ----------------

    @property
    def ndim(self) -> int:
        # sid_ndim, not spatial_index_dims: the latter is the axes list.
        return int(getattr(self._root_meta, "sid_ndim", 3) or 3)

    @property
    def bounds(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """``(min_corner, max_corner)`` in coordinate units.

        Downstream reads this today as
        ``root.attrs.to_dict()["zarr_vectors"]["bounds"]`` in at least
        three repositories.  That expression encodes the attribute
        layout, so it breaks whenever the layout moves.
        """
        raw = getattr(self._root_meta, "bounds", None)
        if not raw:
            raise MetadataError(f"{self.url} declares no bounds")
        return np.asarray(raw[0], dtype=np.float64), np.asarray(raw[1], dtype=np.float64)

    @property
    def axes(self) -> tuple[dict[str, str], ...]:
        """Axis descriptors: name, type, and unit where declared."""
        declared = getattr(self._root_meta, "spatial_index_dims", None) or ()
        return tuple(dict(a) for a in declared)

    @property
    def kinds(self) -> tuple[str, ...]:
        """Geometry types present, e.g. ``("streamline",)``."""
        return tuple(getattr(self._root_meta, "geometry_types", None) or ())

    def kind_of(self, level: int = 0) -> str:
        """The geometry type a read at ``level`` should be decoded as."""
        kinds = self.kinds
        if not kinds:
            raise MetadataError(
                f"{self.url} declares no geometry_types, so there is no way to "
                f"know how to decode it. Write geometry to it first."
            )
        return kinds[0]

    # ---------------- navigation ----------------

    @property
    def levels(self) -> tuple[int, ...]:
        from zarr_vectors.core.store import list_resolution_levels

        return tuple(list_resolution_levels(self._group))

    def level(self, index: int = 0) -> Level:
        if index not in self._levels:
            self._levels[index] = Level(self, index)
        return self._levels[index]

    def __getitem__(self, index: int) -> Level:
        return self.level(index)

    def __iter__(self) -> Any:
        return iter(self.level(i) for i in self.levels)

    def resolution(self, *, scale: float) -> Level:
        """The level whose cell size is closest to ``scale``.

        What a viewer actually wants: not "level 2" but "roughly this
        many units per cell", so the choice survives a change in how many
        levels the pyramid has.
        """
        best: tuple[float, int] | None = None
        for index in self.levels:
            cell = self.level(index).scale
            if not cell:
                continue
            distance = abs(float(np.mean(cell)) - float(scale))
            if best is None or distance < best[0]:
                best = (distance, index)
        if best is None:
            raise MetadataError(f"{self.url} has no level with a known cell size")
        return self.level(best[1])

    @property
    def metadata(self) -> Any:
        """Namespaced application metadata stored in the root.

        Replaces reaching for ``root.attrs`` directly, which three
        downstream modules do today and which has no protection against
        two writers of different keys losing each other's.
        """
        from zarr_vectors.core.user_metadata import Metadata

        return Metadata(self._group)

    @property
    def groups(self) -> Any:
        """Object groups at level 0, keyed by name."""
        return self.level(0).groups

    @property
    def headers(self) -> Any:
        """Opaque per-format header blocks."""
        from zarr_vectors.headers.registry import HeaderRegistry

        return HeaderRegistry(self._group)

    # ---------------- reading ----------------

    def select(self, **kw: Any) -> Query:
        return self.level(int(kw.get("level", 0))).select(**kw)

    def read(self, **kw: Any) -> ReadResult:
        return self.select(**kw).read()

    async def aread(self, **kw: Any) -> ReadResult:
        return await self.select(**kw).aread()

    # ---------------- writing ----------------

    def add_points(
        self,
        positions: npt.ArrayLike,
        *,
        attributes: Mapping[str, npt.ArrayLike] | None = None,
        object_ids: npt.ArrayLike | None = None,
        object_attributes: Mapping[str, npt.ArrayLike] | None = None,
        groups: Mapping[int, Sequence[int]] | None = None,
        layout: Layout | None = None,
        on_out_of_bounds: Literal["raise", "ignore", "expand"] = "raise",
    ) -> dict[str, Any]:
        """Write a point cloud."""
        from zarr_vectors.types.points import write_points

        return self._write(
            write_points, positions,
            vertex_attributes=dict(attributes or {}) or None,
            object_ids=object_ids,
            object_attributes=dict(object_attributes or {}) or None,
            groups=dict(groups or {}) or None,
            layout=layout, out_of_bounds=on_out_of_bounds,
        )

    def add_polylines(
        self,
        polylines: Sequence[npt.ArrayLike],
        *,
        attributes: Mapping[str, npt.ArrayLike] | None = None,
        object_attributes: Mapping[str, npt.ArrayLike] | None = None,
        groups: Mapping[int, Sequence[int]] | None = None,
        layout: Layout | None = None,
        streamlines: bool = False,
        on_out_of_bounds: Literal["raise", "ignore", "expand"] = "raise",
    ) -> dict[str, Any]:
        """Write polylines, or streamlines when ``streamlines=True``."""
        from zarr_vectors.constants import GEOM_STREAMLINE
        from zarr_vectors.types.polylines import write_polylines

        return self._write(
            write_polylines, polylines,
            vertex_attributes=dict(attributes or {}) or None,
            object_attributes=dict(object_attributes or {}) or None,
            groups=dict(groups or {}) or None,
            geometry_type=GEOM_STREAMLINE if streamlines else GEOM_POLYLINE,
            layout=layout, out_of_bounds=on_out_of_bounds,
        )

    def add_lines(
        self,
        endpoints: npt.ArrayLike,
        *,
        attributes: Mapping[str, npt.ArrayLike] | None = None,
        object_attributes: Mapping[str, npt.ArrayLike] | None = None,
        layout: Layout | None = None,
        on_out_of_bounds: Literal["raise", "ignore", "expand"] = "raise",
    ) -> dict[str, Any]:
        """Write line segments, as an ``(M, 2, D)`` array."""
        from zarr_vectors.types.lines import write_lines

        return self._write(
            write_lines, endpoints,
            vertex_attributes=dict(attributes or {}) or None,
            object_attributes=dict(object_attributes or {}) or None,
            layout=layout, out_of_bounds=on_out_of_bounds,
        )

    def add_mesh(
        self,
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        *,
        attributes: Mapping[str, npt.ArrayLike] | None = None,
        object_ids: npt.ArrayLike | None = None,
        layout: Layout | None = None,
        on_out_of_bounds: Literal["raise", "ignore", "expand"] = "raise",
    ) -> dict[str, Any]:
        """Write a triangulated surface."""
        from zarr_vectors.types.meshes import write_mesh

        return self._write(
            write_mesh, vertices, faces,
            vertex_attributes=dict(attributes or {}) or None,
            object_ids=object_ids,
            layout=layout, out_of_bounds=on_out_of_bounds,
        )

    def add_graph(
        self,
        positions: npt.ArrayLike,
        edges: npt.ArrayLike,
        *,
        attributes: Mapping[str, npt.ArrayLike] | None = None,
        edge_attributes: Mapping[str, npt.ArrayLike] | None = None,
        object_ids: npt.ArrayLike | None = None,
        tree: bool = False,
        layout: Layout | None = None,
        on_out_of_bounds: Literal["raise", "ignore", "expand"] = "raise",
    ) -> dict[str, Any]:
        """Write a graph, or a skeleton when ``tree=True``."""
        from zarr_vectors.types.graphs import write_graph

        return self._write(
            write_graph, positions, edges,
            vertex_attributes=dict(attributes or {}) or None,
            link_attributes=dict(edge_attributes or {}) or None,
            object_ids=object_ids, is_tree=tree,
            layout=layout, out_of_bounds=on_out_of_bounds,
        )

    def _write(
        self, writer: Any, *data: Any, layout: Layout | None = None, **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolve the layout, then call the legacy writer.

        This is the one place the four physical parameters are produced,
        and they are produced from the store's own declared bounds — so a
        second write into an existing store lands on the same grid as the
        first, rather than on whatever grid the caller happened to pass.
        """
        schema = Schema.from_store(self._root_meta)
        effective = layout or schema.layout
        resolved = effective.resolve(schema, store_kind=self._store_kind())
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._meta = None  # geometry_types and bounds may change
        self._levels.clear()
        report: dict[str, Any] = writer(
            self._group, *data,
            chunk_shape=resolved.chunk_shape,
            bin_shape=resolved.bin_shape,
            shard_shape=resolved.shard_shape,
            compressor=resolved.compressor,
            **kwargs,
        )
        return report

    def _store_kind(self) -> str:
        from zarr_vectors.core.backends import detect_scheme

        try:
            return "local" if detect_scheme(self.url) in ("", "file") else "object"
        except Exception:
            return "local"

    def editing(self, *, level: int = 0, in_place: bool = False, **kw: Any) -> Any:
        """Open a batch of edits addressed by intent.

        ``with ds.editing() as edit: edit.move_vertex(object=7, index=3,
        to=...)`` -- no chunk coordinates, no fragment indices.

        Edits are copy-on-write by default: the edited object is
        reallocated under a new id and the original is left readable.
        ``in_place=True`` asks for overwrite instead, but object-bearing
        geometry may reallocate regardless -- always check
        ``edit.renamed()``, or the object you edited will appear
        unchanged.
        """
        from zarr_vectors.api.edit import EditPlan

        return EditPlan(self, level=level, **kw)

    # ---------------- maintenance ----------------

    def build_pyramid(
        self, *, factors: Sequence[tuple[float, float]], method: str = "per_object",
        **kw: Any,
    ) -> dict[str, Any]:
        """Build coarser levels.

        ``method`` selects the coarsener: ``"per_object"`` is core's, and
        anything :func:`coarsen_methods` also lists comes from an
        installed strategy package.  Pass that strategy's own knobs as
        ``options={...}``.
        """
        from zarr_vectors.multiresolution.coarsen import build_pyramid

        out = build_pyramid(self.url, factors=list(factors), method=method, **kw)
        self._levels.clear()
        self._meta = None
        return out

    def validate(self, *, level: int = 3) -> Any:
        from zarr_vectors.validate import validate as _validate

        return _validate(self.url, level=level)

    def commit(self, message: str = "zarr-vectors write") -> str | None:
        """Commit, for backends that have transactions.  ``None`` otherwise."""
        from zarr_vectors.core.store import commit as _commit

        return _commit(self._group, message)

    # ---------------- escape hatch ----------------

    @property
    def store(self) -> Group:
        """The underlying storage handle.

        Deliberately named as an escape hatch and deliberately present:
        pretending the layer below does not exist would only mean
        consumers reach past the facade in ways nobody can find later.
        Reaching for this is a signal that something is missing here.
        """
        return self._group

    def __repr__(self) -> str:
        try:
            kinds = ", ".join(self.kinds) or "empty"
            return f"Dataset({self.url!r}, {kinds}, levels={list(self.levels)})"
        except Exception:
            return f"Dataset({self.url!r})"


# =====================================================================
# Entry points
# =====================================================================


def open(
    source: Any,
    *,
    mode: Literal["r", "r+"] = "r",
    storage: StorageOptions | None = None,
) -> Dataset:
    """Open an existing dataset.

    Accepts a path, a URL, an already-open store handle, or a
    pre-constructed ``zarr`` Store — the last of which is what a browser
    host supplies, since a fetch-backed Store is the only way in there.

    The backend is resolved, never asked for: explicit
    :attr:`StorageOptions.backend`, then ``$ZARR_VECTORS_BACKEND``, then
    the URL scheme.
    """
    from zarr_vectors.core.store import open_store

    storage = storage or StorageOptions()
    group = open_store(
        source, mode,
        backend=storage.resolve_backend(str(source)),
        storage_options=dict(storage.options) or None,
    )
    return Dataset(group, storage=storage)


def create(
    target: Any,
    *,
    schema: Schema,
    storage: StorageOptions | None = None,
) -> Dataset:
    """Create a new dataset from a :class:`Schema`.

    Raises if something already exists at ``target`` — use
    :func:`open_or_create` when either outcome is acceptable.
    """
    from zarr_vectors.core.store import create_store

    storage = storage or StorageOptions()
    resolved = schema.layout.resolve(schema, store_kind=_kind_of_url(str(target)))
    group = create_store(
        target,
        bounds=(list(resolved.bounds[0]), list(resolved.bounds[1])),
        chunk_shape=resolved.chunk_shape,
        base_bin_shape=resolved.bin_shape,
        compressor=resolved.compressor,
        ndim=schema.ndim,
        vertex_dtype=schema.position_dtype,
        axes=cast("Any", [a.to_ngff() for a in schema.axes]) if schema.axes else None,
        geometry_types=[schema.kind] if schema.kind else None,
        backend=storage.resolve_backend(str(target)),
        storage_options=dict(storage.options) or None,
    )
    return Dataset(group, storage=storage)


def open_or_create(
    target: Any,
    *,
    schema: Schema,
    storage: StorageOptions | None = None,
    on_conflict: Literal["raise", "keep"] = "raise",
) -> Dataset:
    """Open ``target`` if it exists, otherwise create it — idempotently.

    When it exists, ``schema`` is *checked* against what is on disk and a
    disagreement raises :class:`SchemaConflict`.  That deliberately
    overturns the existing behaviour, in which ``_create_or_open_store``
    silently drops ``bounds`` and ``chunk_shape`` for an existing path:
    that silence is how a store ends up with a declared grid that
    contradicts its own chunk keys, after which a bbox read looks for
    coordinates that were never written and returns nothing, with no
    error raised anywhere.

    Pass ``on_conflict="keep"`` to accept what is on disk instead.
    """
    from zarr_vectors.core.store import open_store

    storage = storage or StorageOptions()
    try:
        group = open_store(
            target, "r+",
            backend=storage.resolve_backend(str(target)),
            storage_options=dict(storage.options) or None,
        )
    except (StoreError, FileNotFoundError, KeyError):
        return create(target, schema=schema, storage=storage)

    dataset = Dataset(group, storage=storage)
    if on_conflict == "keep":
        return dataset
    existing = Schema.from_store(dataset._root_meta)
    differences = schema.diff(existing)
    if differences:
        raise SchemaConflict(
            f"{dataset.url} already exists and disagrees with the schema it was "
            f"opened with:\n  " + "\n  ".join(differences)
            + "\nPass on_conflict='keep' to use the store's own values."
        )
    return dataset


async def aopen(source: Any, *, storage: StorageOptions | None = None) -> Dataset:
    """Open without ever entering zarr's blocking ``sync()`` bridge.

    For Pyodide, where a synchronous zarr call needs WebAssembly stack
    switching and deadlocks the JS event loop under concurrency.  Pair it
    with :meth:`Query.aread`.
    """
    from zarr_vectors.core.aio import open_store_async

    group = await open_store_async(source)
    return Dataset(group, storage=storage or StorageOptions())


def require_format(dataset: Dataset, spec: str) -> None:
    """Raise unless ``dataset``'s on-disk format satisfies ``spec``.

    ``spec`` is a comma-separated list of ``>=``/``>``/``<=``/``<``/``==``
    clauses, e.g. ``">=0.9,<0.11"``.

    This exists because there is no backward-compatible reader: a store
    written before 0.9.0 does not raise a clear error, it raises a
    ``MetadataError`` from somewhere deep in the metadata parser.  One
    consumer carries a hundred-and-eighteen-line module to turn that into
    a sentence.
    """
    found = dataset.format_version
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        for op in (">=", "<=", "==", ">", "<"):
            if clause.startswith(op):
                want = _parse_version(clause[len(op):])
                width = max(len(found), len(want))
                lhs = found + (0,) * (width - len(found))
                rhs = want + (0,) * (width - len(want))
                ok = {
                    ">=": lhs >= rhs, "<=": lhs <= rhs, "==": lhs == rhs,
                    ">": lhs > rhs, "<": lhs < rhs,
                }[op]
                if not ok:
                    raise FormatError(
                        f"{dataset.url} is on-disk format "
                        f"{'.'.join(map(str, found))}, which does not satisfy "
                        f"{spec!r}. There is no backward-compatible reader: an "
                        f"older store must be rewritten from source, and a newer "
                        f"one needs a newer zarr-vectors."
                    )
                break
        else:
            raise ValueError(f"cannot parse version clause {clause!r} in {spec!r}")


def _kind_of_url(url: str) -> str:
    from zarr_vectors.core.backends import detect_scheme

    try:
        return "local" if detect_scheme(url) in ("", "file") else "object"
    except Exception:
        return "local"


# Kinds this module knows how to create stores for, exported so callers
# can validate a Schema.kind without importing constants.
KINDS = (GEOM_POINT_CLOUD, GEOM_LINE, GEOM_POLYLINE, GEOM_MESH, GEOM_GRAPH)
