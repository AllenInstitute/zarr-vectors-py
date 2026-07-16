"""ZVLevel — lazy handle to a single resolution level."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zarr_vectors.constants import VERTEX_ATTRIBUTES
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import FsGroup
from zarr_vectors.lazy.arrays import ZVAttributeCollection, ZVVertexCollection, ZVObjectIndex
from zarr_vectors.typing import ChunkCoords

import numpy.typing as npt
import numpy as np


class ZVLevel:
    """Lazy handle to one resolution level.

    Chunk listings are cached on first access.  Vertex and attribute
    data are not read until ``.compute()`` is called on the
    corresponding collection.

    Args:
        group: The level's :class:`FsGroup`.
        level_index: Integer level number (0, 1, ...).
        root_meta: Root-level metadata (for chunk/bin shapes).
        level_meta: Level-specific metadata, or None if unavailable.
    """

    def __init__(
        self,
        group: FsGroup,
        level_index: int,
        root_meta: RootMetadata,
        level_meta: LevelMetadata | None,
    ) -> None:
        self._group = group
        self._level_index = level_index
        self._root_meta = root_meta
        self._level_meta = level_meta
        self._chunk_keys_cache: list[ChunkCoords] | None = None
        self._vertices_cache: ZVVertexCollection | None = None
        self._attributes_cache: dict[str, ZVAttributeCollection] = {}
        self._object_index_cache: ZVObjectIndex | None = None

    # ---------------------------------------------------------------
    # Metadata properties
    # ---------------------------------------------------------------

    @property
    def level_index(self) -> int:
        return self._level_index

    @property
    def vertex_count(self) -> int:
        """Vertex count from metadata (no data I/O)."""
        if self._level_meta is not None:
            return self._level_meta.vertex_count
        # Fallback: read from attrs
        attrs = self._group.attrs.to_dict()
        zvl = attrs.get("zarr_vectors_level", attrs)
        return zvl.get("vertex_count", 0)

    @property
    def bin_shape(self) -> tuple[float, ...] | None:
        if self._level_meta is not None:
            return self._level_meta.bin_shape
        return None

    @property
    def chunk_shape(self) -> tuple[float, ...]:
        """Effective physical chunk shape for this level.

        Returns the per-level override if set (v0.7+), else falls back
        to ``RootMetadata.chunk_shape``.
        """
        from zarr_vectors.core.metadata import get_level_chunk_shape
        return get_level_chunk_shape(self._root_meta, self._level_meta)

    @property
    def bin_ratio(self) -> tuple[int, ...] | None:
        if self._level_meta is not None:
            return self._level_meta.bin_ratio
        return None

    @property
    def object_sparsity(self) -> float:
        if self._level_meta is not None:
            return self._level_meta.object_sparsity
        return 1.0

    @property
    def chunk_keys(self) -> list[ChunkCoords]:
        """List of chunk coordinate tuples with data at this level."""
        if self._chunk_keys_cache is None:
            self._chunk_keys_cache = list_chunk_keys(self._group)
        return self._chunk_keys_cache

    @property
    def chunk_count(self) -> int:
        return len(self.chunk_keys)

    # ---------------------------------------------------------------
    # Attribute-chunking accessors
    # ---------------------------------------------------------------

    @property
    def chunk_dims(self) -> list[str] | None:
        """Names of chunk-key axes, leading axis first.

        ``None`` for legacy spatial-only stores (chunk keys are
        ``[dim0, dim1, ...]``).  Non-None when the store was written
        with ``chunk_by_attribute`` or rechunked along a non-spatial
        dimension.
        """
        if self._level_meta is not None:
            return self._level_meta.chunk_dims
        return None

    @property
    def chunk_attribute_name(self) -> str | None:
        """Name of the per-vertex attribute used as the leading chunk axis."""
        if self._level_meta is not None:
            return self._level_meta.chunk_attribute_name
        return None

    @property
    def attribute_values(self) -> list[Any] | None:
        """Ordered list mapping leading-axis bin index to attribute value.

        ``attribute_values[i]`` is the original attribute value for any
        chunk whose key starts with ``i``.  ``None`` for non-attribute
        stores.
        """
        if self._level_meta is not None:
            return self._level_meta.chunk_attribute_values
        return None

    # ---------------------------------------------------------------
    # ID-preserving pyramid accessors
    # ---------------------------------------------------------------

    @property
    def preserves_object_ids(self) -> bool:
        """True when this level inherits the parent level's OID space.

        On such a level, OIDs map 1:1 to their level-0 identity and
        dropped objects appear as empty manifest slots.  Set by the
        per-object pyramid writer.
        """
        if self._level_meta is not None:
            return bool(self._level_meta.preserves_object_ids)
        return False

    @property
    def shared_fragments(self) -> bool:
        """True when per-chunk fragments may be referenced by
        multiple objects' manifests (shared metavertices)."""
        if self._level_meta is not None:
            return bool(self._level_meta.shared_fragments)
        return False

    @property
    def inherited_num_objects(self) -> int | None:
        """OID-space size inherited from the parent level (or None)."""
        if self._level_meta is not None:
            return self._level_meta.inherited_num_objects
        return None

    def has_object(self, oid: int) -> bool:
        """Return True if ``oid`` is present at this level.

        Cheap: probes the object's manifest and treats an empty
        manifest (or out-of-range OID) as absent.  A store with no object
        index at all — a plain point cloud — has no objects, so this
        returns False rather than raising.
        """
        from zarr_vectors.constants import OBJECT_INDEX
        from zarr_vectors.core.arrays import read_object_manifest
        from zarr_vectors.exceptions import ArrayError, StoreError
        # No object index => no objects.  read_object_manifest requires one
        # (it reads sid_ndim off its meta), so guard rather than let it
        # KeyError on a point-cloud store.
        if not self._group.array_exists(OBJECT_INDEX):
            return False
        try:
            manifest = read_object_manifest(self._group, int(oid))
        except (ArrayError, StoreError, KeyError, IndexError):
            return False
        return bool(manifest)

    @property
    def present_oids(self) -> "np.ndarray":
        """Sorted array of OIDs present at this level."""
        from zarr_vectors.core.arrays import read_all_object_manifests
        try:
            manifests = read_all_object_manifests(self._group)
        except Exception:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(
            [i for i, m in enumerate(manifests) if m], dtype=np.int64,
        )

    def read_attribute_chunk(self, value: Any) -> list[npt.NDArray]:
        """Read all fragments for chunks whose attribute equals ``value``.

        Only valid on stores written with ``chunk_by_attribute`` (or
        rechunked by attribute).  Returns the concatenated vertex
        groups for every chunk whose leading coord maps to ``value``.

        Args:
            value: One of the entries from :attr:`attribute_values`.

        Returns:
            List of vertex-group arrays.
        """
        from zarr_vectors.core.arrays import read_chunk_vertices

        vals = self.attribute_values
        if vals is None:
            raise ValueError(
                "read_attribute_chunk: this level is not attribute-chunked"
            )
        try:
            bin_idx = vals.index(value)
        except ValueError:
            return []
        groups: list[npt.NDArray] = []
        ndim = self._root_meta.sid_ndim
        for cc in self.chunk_keys:
            if not cc or cc[0] != bin_idx:
                continue
            try:
                groups.extend(
                    read_chunk_vertices(self._group, cc, ndim=ndim)
                )
            except Exception:
                continue
        return groups

    # ---------------------------------------------------------------
    # Lazy collections
    # ---------------------------------------------------------------

    @property
    def vertices(self) -> ZVVertexCollection:
        """Lazy vertex collection for this level."""
        if self._vertices_cache is None:
            self._vertices_cache = ZVVertexCollection(
                level_group=self._group,
                chunk_keys=self.chunk_keys,
                ndim=self._root_meta.sid_ndim,
                dtype="float32",
                vertex_count=self.vertex_count,
            )
        return self._vertices_cache

    @property
    def attributes(self) -> _AttributeAccessor:
        """Dict-like access to lazy attribute collections.

        Usage::

            level.attributes["intensity"].compute()
        """
        return _AttributeAccessor(self)

    def _get_attribute(self, name: str) -> ZVAttributeCollection:
        if name not in self._attributes_cache:
            self._attributes_cache[name] = ZVAttributeCollection(
                level_group=self._group,
                attr_name=name,
                chunk_keys=self.chunk_keys,
            )
        return self._attributes_cache[name]

    @property
    def object_index(self) -> ZVObjectIndex:
        """Lazy object index accessor."""
        if self._object_index_cache is None:
            self._object_index_cache = ZVObjectIndex(self._group)
        return self._object_index_cache

    # ---------------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------------

    def filter(
        self,
        *,
        bbox: tuple[npt.NDArray, npt.NDArray] | None = None,
        object_ids: list[int] | None = None,
        group_ids: list[int] | None = None,
    ) -> "ZVView":
        """Apply filter constraints, returning a lazy filtered view.

        Filters can be chained: ``level.filter(group_ids=[0]).filter(bbox=roi)``.

        Args:
            bbox: Bounding box ``(min_corner, max_corner)``.
            object_ids: Keep only these object IDs.
            group_ids: Keep only objects in these groups.

        Returns:
            A :class:`ZVView` with the specified constraints.
        """
        from zarr_vectors.lazy.views import ZVView, FilterSpec
        view = ZVView(
            self._group, self._root_meta, self._level_meta,
            self.chunk_keys, FilterSpec(),
        )
        return view.filter(
            bbox=bbox, object_ids=object_ids, group_ids=group_ids,
        )

    # ---------------------------------------------------------------
    # Mutation (write-back) handle
    # ---------------------------------------------------------------

    def writer(self) -> "ZVWriter":
        """Return a :class:`ZVWriter` for mutating this level.

        Use as an async or sync context manager::

            async with zv[0].writer() as w:
                await w.add_attribute("normal", normals)

        Single-writer-only — concurrent writers on the same level can
        race on object_index sidecar batch numbering.
        """
        from zarr_vectors.lazy.writer import ZVWriter
        return ZVWriter(self)

    # ---------------------------------------------------------------
    # Geometry-specific collections
    # ---------------------------------------------------------------

    @property
    def polylines(self) -> "ZVPolylineCollection":
        """Lazy polyline collection for streamline/polyline geometry.

        Each polyline is accessible by object ID::

            level.polylines[42].compute()  # full streamline
        """
        from zarr_vectors.lazy.views import ZVPolylineCollection
        return ZVPolylineCollection(self._group, ndim=self._root_meta.sid_ndim)

    def rechunk(
        self,
        by: str,
        *,
        bins: list[float] | None = None,
        output: str | None = None,
        categorical: bool = False,
    ) -> dict:
        """Rechunk this level's store along a non-spatial dimension.

        Convenience wrapper around :func:`zarr_vectors.rechunk.rechunk`.

        Args:
            by: Dimension to rechunk by (``"group"``, ``"object_id"``,
                ``"attribute:<name>"``).
            bins: Bin edges for continuous values.
            output: Output path. If None, rechunks in-place.
            categorical: If True, treat the binning dimension as
                categorical (one bin per unique value, no quartile
                fallback).  Required for high-cardinality attributes
                like gene labels.

        Returns:
            Summary dict from the rechunk engine.
        """
        from zarr_vectors.rechunk import rechunk as _rechunk, RechunkSpec
        store_path = str(self._group.path.parent)
        spec = RechunkSpec(by=by, bins=bins, categorical=categorical)
        return _rechunk(store_path, spec, output=output)

    # ---------------------------------------------------------------
    # Repr
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        bs = self.bin_shape or self._root_meta.effective_bin_shape
        return (
            f"ZVLevel({self._level_index}, "
            f"vertices={self.vertex_count}, "
            f"chunks={self.chunk_count}, "
            f"bin_shape={bs})"
        )


class _AttributeAccessor(Mapping):
    """Mapping proxy for lazy per-vertex attribute access.

    A full :class:`~collections.abc.Mapping`, so ``list()``, ``dict()``,
    ``for name in attrs``, ``len()`` and membership all work.  Hand-rolling
    only ``__getitem__`` used to hang: with no ``__iter__``/``__len__``,
    Python fell back to the legacy sequence protocol (``self[0]``,
    ``self[1]``, …), and ``__getitem__`` never raised, so the sequence
    never ended.
    """

    def __init__(self, level: ZVLevel) -> None:
        self._level = level

    def _names(self) -> list[str]:
        """The per-vertex attribute names actually present, sorted."""
        group = self._level._group
        if not group.array_exists(VERTEX_ATTRIBUTES):
            return []
        try:
            return list(group[VERTEX_ATTRIBUTES].children())
        except Exception:
            return []

    def __getitem__(self, name: str) -> ZVAttributeCollection:
        if name not in self._names():
            raise KeyError(name)
        return self._level._get_attribute(name)

    def __iter__(self):
        return iter(self._names())

    def __len__(self) -> int:
        return len(self._names())

    def __contains__(self, name: object) -> bool:
        return name in self._names()

    def __repr__(self) -> str:
        return (
            f"AttributeAccessor(level={self._level.level_index}, "
            f"names={self._names()})"
        )
