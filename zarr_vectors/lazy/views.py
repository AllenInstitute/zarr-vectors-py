"""Lazy filtered views and geometry-specific collections.

``ZVView`` is a filtered projection of a ``ZVLevel`` that narrows
which chunks, bins, objects, or groups will be read.  Filters chain:
each ``.filter()`` returns a new view with the intersection of all
constraints.  Data is materialised only on ``.compute()``.

``ZVPolylineCollection`` provides per-object lazy access to
polylines/streamlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.core.arrays import (
    read_all_object_manifests,
    read_chunk_vertices,
    read_fragment,
    read_object_vertices,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import FsGroup
from zarr_vectors.typing import BinCoords, ChunkCoords

try:
    import dask
    from dask import delayed as dask_delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    dask = None  # type: ignore

    def dask_delayed(func):  # type: ignore
        class _FakeDelayed:
            def __init__(self, *a, **kw):
                self._f, self._a, self._kw = func, a, kw
            def compute(self):
                return self._f(*self._a, **self._kw)
            def __repr__(self):
                return f"Delayed({self._f.__name__})"
        class _W:
            def __call__(self, *a, **kw):
                return _FakeDelayed(*a, **kw)
        return _W()


# ===================================================================
# Filter specification
# ===================================================================

@dataclass
class FilterSpec:
    """Accumulated filter constraints for a lazy view."""

    target_chunks: set[ChunkCoords] | None = None
    target_bins: set[BinCoords] | None = None
    target_object_ids: set[int] | None = None
    bbox: tuple[npt.NDArray, npt.NDArray] | None = None

    def intersect(self, other: FilterSpec) -> FilterSpec:
        """Return a new FilterSpec that is the intersection of self and other."""
        def _intersect_sets(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return a & b

        bbox = other.bbox if other.bbox is not None else self.bbox

        return FilterSpec(
            target_chunks=_intersect_sets(self.target_chunks, other.target_chunks),
            target_bins=_intersect_sets(self.target_bins, other.target_bins),
            target_object_ids=_intersect_sets(self.target_object_ids, other.target_object_ids),
            bbox=bbox,
        )


# ===================================================================
# ZVView — filtered lazy view
# ===================================================================

class ZVView:
    """A filtered lazy view of a resolution level.

    Created by calling ``.filter()`` on a ``ZVLevel`` or another
    ``ZVView``.  Each filter narrows the read plan; data is only
    loaded on ``.compute()``.

    Args:
        group: Resolution level FsGroup.
        root_meta: Root metadata.
        level_meta: Level metadata (or None).
        all_chunk_keys: Full list of chunk keys at this level.
        spec: The accumulated filter constraints.
    """

    def __init__(
        self,
        group: FsGroup,
        root_meta: RootMetadata,
        level_meta: LevelMetadata | None,
        all_chunk_keys: list[ChunkCoords],
        spec: FilterSpec,
    ) -> None:
        self._group = group
        self._root_meta = root_meta
        self._level_meta = level_meta
        self._all_chunk_keys = all_chunk_keys
        self._spec = spec

    def filter(
        self,
        *,
        bbox: tuple[npt.NDArray, npt.NDArray] | None = None,
        object_ids: list[int] | None = None,
        group_ids: list[int] | None = None,
    ) -> ZVView:
        """Apply additional filter constraints, returning a new view.

        Args:
            bbox: Bounding box ``(min_corner, max_corner)``.
            object_ids: Keep only these object IDs.
            group_ids: Keep only objects in these groups (resolved
                to object IDs via groupings).

        Returns:
            A new ``ZVView`` with the intersection of all constraints.
        """
        new_spec = FilterSpec()

        if bbox is not None:
            new_spec.bbox = (np.asarray(bbox[0]), np.asarray(bbox[1]))
            # Chunks the box touches, filtered from the ones this view
            # already knows the level holds rather than enumerated from
            # the declared grid. The product has no clamp, so on a sparse
            # store -- a specimen bounding box with data in part of it --
            # a wide box materialised one tuple per allocated cell to
            # build a set used only for membership tests against chunks
            # that exist. The keys are in hand, so this needs no I/O.
            cs = np.asarray(self._root_meta.chunk_shape, dtype=np.float64)
            lo_c = np.floor(new_spec.bbox[0] / cs).astype(np.int64)
            hi_c = np.floor(new_spec.bbox[1] / cs).astype(np.int64)
            nd = len(cs)
            new_spec.target_chunks = {
                cc for cc in self._all_chunk_keys
                if all(
                    int(lo_c[d]) <= int(v) <= int(hi_c[d])
                    for d, v in enumerate(cc[-nd:])
                )
            }

            # If bins are available, compute bin-level targets
            bins_per_chunk = self._root_meta.bins_per_chunk
            if any(b > 1 for b in bins_per_chunk):
                from zarr_vectors.spatial.chunking import (
                    bins_intersecting_bbox,
                )
                effective_bin = self._root_meta.effective_bin_shape
                target_bins = set(bins_intersecting_bbox(
                    new_spec.bbox[0], new_spec.bbox[1], effective_bin,
                ))
                new_spec.target_bins = target_bins

        if object_ids is not None:
            new_spec.target_object_ids = set(object_ids)

        if group_ids is not None:
            from zarr_vectors.core.arrays import read_group_object_ids
            resolved: set[int] = set()
            for gid in group_ids:
                try:
                    members = read_group_object_ids(self._group, gid)
                    resolved.update(members)
                except Exception:
                    pass
            new_spec.target_object_ids = resolved

        merged = self._spec.intersect(new_spec)
        return ZVView(
            self._group, self._root_meta, self._level_meta,
            self._all_chunk_keys, merged,
        )

    @property
    def vertices(self) -> _FilteredVertices:
        """Lazy filtered vertex accessor."""
        return _FilteredVertices(self)

    def compute(self) -> dict[str, Any]:
        """Materialise the filtered data.

        Returns:
            Dict with ``positions``, ``vertex_count``, and optionally
            ``object_ids``.
        """
        ndim = self._root_meta.sid_ndim
        dtype = np.float32

        # Path 1: object-ID based read
        if self._spec.target_object_ids is not None:
            return self._compute_by_objects(ndim, dtype)

        # Path 2: bin/chunk spatial read
        return self._compute_spatial(ndim, dtype)

    def _compute_by_objects(self, ndim: int, dtype: np.dtype) -> dict[str, Any]:
        """Read by object ID using manifests."""
        all_positions: list[npt.NDArray] = []
        all_oids: list[npt.NDArray] = []

        for oid in sorted(self._spec.target_object_ids):
            try:
                verts_list = read_object_vertices(
                    self._group, oid, dtype=dtype, ndim=ndim,
                )
            except Exception:
                continue
            for fragment in verts_list:
                if len(fragment) > 0:
                    all_positions.append(fragment)
                    all_oids.append(np.full(len(fragment), oid, dtype=np.int64))

        if not all_positions:
            return {"positions": np.zeros((0, ndim), dtype=dtype),
                    "vertex_count": 0, "object_ids": np.array([], dtype=np.int64)}

        positions = np.concatenate(all_positions, axis=0)
        object_ids = np.concatenate(all_oids)

        # Apply bbox post-filter
        if self._spec.bbox is not None:
            mask = np.all(
                (positions >= self._spec.bbox[0]) & (positions <= self._spec.bbox[1]),
                axis=1,
            )
            positions = positions[mask]
            object_ids = object_ids[mask]

        return {
            "positions": positions,
            "vertex_count": len(positions),
            "object_ids": object_ids,
        }

    def _compute_spatial(self, ndim: int, dtype: np.dtype) -> dict[str, Any]:
        """Read by spatial targeting (bins or chunks)."""
        bins_per_chunk = self._root_meta.bins_per_chunk
        has_bins = any(b > 1 for b in bins_per_chunk)

        # Determine which chunks to read
        chunk_keys_set = set(self._all_chunk_keys)
        if self._spec.target_chunks is not None:
            active_chunks = [ck for ck in self._all_chunk_keys
                            if ck in self._spec.target_chunks]
        else:
            active_chunks = self._all_chunk_keys

        all_positions: list[npt.NDArray] = []

        if has_bins and self._spec.target_bins is not None:
            # Bin-level read
            from zarr_vectors.spatial.chunking import bin_to_chunk, bin_to_fragment_index
            chunk_fragment_targets: dict[ChunkCoords, list[int]] = {}
            for bc in self._spec.target_bins:
                cc = bin_to_chunk(bc, bins_per_chunk)
                fragment_index = bin_to_fragment_index(bc, cc, bins_per_chunk)
                if cc not in chunk_fragment_targets:
                    chunk_fragment_targets[cc] = []
                chunk_fragment_targets[cc].append(fragment_index)

            for cc, fragment_indices in chunk_fragment_targets.items():
                if cc not in chunk_keys_set:
                    continue
                for fragment_index in fragment_indices:
                    try:
                        fragment = read_fragment(
                            self._group, cc, fragment_index,
                            dtype=dtype, ndim=ndim,
                        )
                        if len(fragment) > 0:
                            all_positions.append(fragment)
                    except Exception:
                        continue
        else:
            # Chunk-level read
            for ck in active_chunks:
                try:
                    groups = read_chunk_vertices(self._group, ck, dtype=dtype, ndim=ndim)
                    for fragment in groups:
                        if len(fragment) > 0:
                            all_positions.append(fragment)
                except Exception:
                    continue

        if not all_positions:
            return {"positions": np.zeros((0, ndim), dtype=dtype), "vertex_count": 0}

        positions = np.concatenate(all_positions, axis=0)

        # Final bbox mask
        if self._spec.bbox is not None:
            mask = np.all(
                (positions >= self._spec.bbox[0]) & (positions <= self._spec.bbox[1]),
                axis=1,
            )
            positions = positions[mask]

        return {"positions": positions, "vertex_count": len(positions)}

    def __repr__(self) -> str:
        parts = []
        if self._spec.target_object_ids is not None:
            parts.append(f"objects={len(self._spec.target_object_ids)}")
        if self._spec.target_chunks is not None:
            parts.append(f"chunks={len(self._spec.target_chunks)}")
        if self._spec.target_bins is not None:
            parts.append(f"bins={len(self._spec.target_bins)}")
        if self._spec.bbox is not None:
            parts.append("bbox=set")
        desc = ", ".join(parts) if parts else "unfiltered"
        return f"ZVView({desc})"


class _FilteredVertices:
    """Vertex accessor on a filtered view."""

    def __init__(self, view: ZVView) -> None:
        self._view = view

    def compute(self) -> npt.NDArray[np.floating]:
        result = self._view.compute()
        return result["positions"]

    def __repr__(self) -> str:
        return f"FilteredVertices({self._view!r})"


# ===================================================================
# ZVPolylineCollection — per-object lazy polyline access
# ===================================================================

class ZVPolylineCollection:
    """Lazy collection of polylines accessible by object ID.

    Each polyline is reconstructed by following its object_index
    manifest and concatenating fragments from the relevant chunks.

    Args:
        group: Resolution level FsGroup.
        ndim: Number of spatial dimensions.
    """

    def __init__(self, group: FsGroup, ndim: int = 3) -> None:
        self._group = group
        self._ndim = ndim
        self._manifests: list | None = None

    def _ensure_manifests(self) -> list:
        if self._manifests is None:
            try:
                self._manifests = read_all_object_manifests(self._group)
            except Exception:
                self._manifests = []
        return self._manifests

    @property
    def count(self) -> int:
        """Number of polylines."""
        return len(self._ensure_manifests())

    @property
    def object_ids(self) -> list[int]:
        """List of polyline object IDs."""
        return list(range(self.count))

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, object_id: int) -> Any:
        """Get a delayed polyline by object ID.

        Returns a delayed object that, when computed, returns the full
        reconstructed polyline as an ``(N, D)`` numpy array.
        """
        manifests = self._ensure_manifests()
        if object_id < 0 or object_id >= len(manifests):
            raise IndexError(
                f"Polyline {object_id} out of range [0, {len(manifests)})"
            )
        return _delayed_read_polyline(
            self._group, object_id, self._ndim,
        )

    def items(self):
        """Iterate over ``(object_id, delayed_polyline)`` pairs."""
        for oid in range(self.count):
            yield oid, self[oid]

    def compute(self) -> list[npt.NDArray[np.floating]]:
        """Materialise all polylines.

        Returns:
            List of ``(N_k, D)`` arrays, one per polyline.
        """
        delayed_list = [self[oid] for oid in range(self.count)]
        if HAS_DASK and len(delayed_list) > 1:
            return list(dask.compute(*delayed_list))
        return [d.compute() for d in delayed_list]

    def filter(
        self,
        *,
        object_ids: list[int] | None = None,
        length_range: tuple[float, float] | None = None,
    ) -> FilteredPolylineCollection:
        """Filter polylines by ID or length range.

        Returns a new filtered collection (lazy).

        Args:
            object_ids: Keep only these IDs.
            length_range: ``(min_length, max_length)`` — keep polylines
                whose Euclidean path length falls in this range.
                Requires computing lengths on first access.
        """
        return FilteredPolylineCollection(
            self, object_ids=object_ids, length_range=length_range,
        )

    def __repr__(self) -> str:
        return f"ZVPolylineCollection(count={self.count})"


class FilteredPolylineCollection:
    """A filtered subset of a polyline collection."""

    def __init__(
        self,
        parent: ZVPolylineCollection,
        *,
        object_ids: list[int] | None = None,
        length_range: tuple[float, float] | None = None,
    ) -> None:
        self._parent = parent
        self._explicit_ids = set(object_ids) if object_ids is not None else None
        self._length_range = length_range
        self._resolved_ids: list[int] | None = None

    def _resolve(self) -> list[int]:
        """Resolve which object IDs pass all filters."""
        if self._resolved_ids is not None:
            return self._resolved_ids

        candidates = (
            sorted(self._explicit_ids)
            if self._explicit_ids is not None
            else list(range(self._parent.count))
        )

        if self._length_range is not None:
            lo, hi = self._length_range
            kept: list[int] = []
            for oid in candidates:
                poly = self._parent[oid].compute()
                if len(poly) < 2:
                    length = 0.0
                else:
                    diffs = np.diff(poly, axis=0)
                    length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
                if lo <= length <= hi:
                    kept.append(oid)
            candidates = kept

        self._resolved_ids = candidates
        return candidates

    @property
    def count(self) -> int:
        return len(self._resolve())

    def __len__(self) -> int:
        return self.count

    def compute(self) -> list[npt.NDArray[np.floating]]:
        """Materialise the filtered polylines."""
        ids = self._resolve()
        delayed_list = [self._parent[oid] for oid in ids]
        if HAS_DASK and len(delayed_list) > 1:
            return list(dask.compute(*delayed_list))
        return [d.compute() for d in delayed_list]

    def items(self):
        for oid in self._resolve():
            yield oid, self._parent[oid]

    def __repr__(self) -> str:
        parts = []
        if self._explicit_ids is not None:
            parts.append(f"ids={len(self._explicit_ids)}")
        if self._length_range is not None:
            parts.append(f"length={self._length_range}")
        return f"FilteredPolylineCollection({', '.join(parts)}, count={self.count})"


# ===================================================================
# Delayed helpers
# ===================================================================

@dask_delayed
def _read_polyline(
    group: FsGroup,
    object_id: int,
    ndim: int,
) -> npt.NDArray[np.floating]:
    """Read and reconstruct a single polyline from its manifest."""
    try:
        verts_list = read_object_vertices(
            group, object_id, dtype=np.float32, ndim=ndim,
        )
        non_empty = [v for v in verts_list if len(v) > 0]
        if not non_empty:
            return np.zeros((0, ndim), dtype=np.float32)
        return np.concatenate(non_empty, axis=0)
    except Exception:
        return np.zeros((0, ndim), dtype=np.float32)


def _delayed_read_polyline(group: FsGroup, object_id: int, ndim: int) -> Any:
    return _read_polyline(group, object_id, ndim)
