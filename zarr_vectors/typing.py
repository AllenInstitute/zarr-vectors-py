"""Shared type aliases used throughout the zarr-vectors package.

These provide consistent typing across modules without importing
heavy dependencies at type-check time.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------

Vertices: TypeAlias = npt.NDArray[np.floating]
"""(N, D) array of vertex positions in D-dimensional space."""

Faces: TypeAlias = npt.NDArray[np.integer]
"""(F, L) array of face indices — L vertices per face (3=tri, 4=quad)."""

Edges: TypeAlias = npt.NDArray[np.integer]
"""(M, 2) array of edge endpoint indices."""

ParentArray: TypeAlias = npt.NDArray[np.integer]
"""(N,) array where parent[i] is the index of node i's parent (-1 for root)."""

# ---------------------------------------------------------------------------
# Spatial indexing types
# ---------------------------------------------------------------------------

ChunkCoords: TypeAlias = tuple[int, ...]
"""Integer coordinates identifying a spatial chunk, e.g. (0, 1, 2) for 3D."""

ChunkShape: TypeAlias = tuple[float, ...]
"""Physical size of a spatial chunk per dimension, e.g. (100.0, 100.0, 50.0)."""

BinCoords: TypeAlias = tuple[int, ...]
"""Integer coordinates identifying a supervoxel bin, e.g. (0, 3, 7) for 3D.
Bins are a finer spatial grid than chunks — multiple bins tile each chunk."""

BinShape: TypeAlias = tuple[float, ...]
"""Physical edge lengths of a supervoxel bin, e.g. (25.0, 25.0, 25.0).
Must evenly divide the chunk shape in every dimension."""

BinRatio: TypeAlias = tuple[int, ...]
"""Per-dimension integer fold-change of bin shape relative to level 0,
e.g. (2, 2, 2) means the bin is 2× wider in each axis at this level."""

BoundingBox: TypeAlias = tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
"""(min_corner, max_corner) arrays, each shape (D,)."""

# ---------------------------------------------------------------------------
# Object/group reference types
# ---------------------------------------------------------------------------

FragmentIndex: TypeAlias = int
"""Index of a fragment within a spatial chunk (0-based)."""

FragmentRef: TypeAlias = tuple[ChunkCoords, FragmentIndex]
"""Reference to a fragment: (chunk_coordinates, fragment_index)."""

ObjectManifest: TypeAlias = list[FragmentRef]
"""Ordered list of fragment references composing one object."""

ObjectID: TypeAlias = int
"""Integer identifier for an object (streamline, neuron, mesh, cell, ...)."""

GroupID: TypeAlias = int
"""Integer identifier for a group of objects (tract, cell type, region, ...)."""

CrossChunkLink: TypeAlias = tuple[
    tuple[ChunkCoords, int],  # (chunk_A_coords, local_vertex_index_A)
    tuple[ChunkCoords, int],  # (chunk_B_coords, local_vertex_index_B)
]
"""One link record as a pair of ``(chunk_coords, local_vertex_index)``.

Named for the pre-0.9 layout, where cross-chunk links were their own
family.  Since 0.9.0 connectivity is one family and a record whose
endpoints share a chunk is simply one whose relative offset is zero, so
this shape describes *any* 2-endpoint link -- ``write_links`` accepts it
for both.  Kept under the old name because it is the spelling every
caller already uses.
"""

# ---------------------------------------------------------------------------
# Convention / metadata string types
# ---------------------------------------------------------------------------

LinksConvention: TypeAlias = str
"""One of: 'explicit', 'implicit_sequential', 'implicit_sequential_with_branches'."""

ObjectIndexConvention: TypeAlias = str
"""One of: 'standard', 'identity'."""

CrossChunkStrategy: TypeAlias = str
"""One of: 'boundary_deduplication', 'explicit_links', 'both'."""

GeometryType: TypeAlias = str
"""One of: 'point_cloud', 'line', 'polyline', 'streamline', 'skeleton', 'graph', 'mesh'."""

# ---------------------------------------------------------------------------
# Aggregation for multi-resolution
# ---------------------------------------------------------------------------

AggregationMethod: TypeAlias = str
"""One of: 'mean', 'sum', 'mode', 'count', 'min', 'max'."""

AttributeAggregation: TypeAlias = dict[str, AggregationMethod]
"""Mapping from attribute name to aggregation method for coarsening."""
