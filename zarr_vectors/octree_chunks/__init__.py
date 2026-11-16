"""Octree chunks: an optional, convertible chunking of a level.

:func:`to_octree` regroups a level's uniform cells into octree leaves of
at most ``target`` vertices (one chunk each); :func:`to_uniform` restores
the uniform chunks exactly.  See ``docs/spec/chunking/octree_chunks.md``.
"""

from zarr_vectors.octree_chunks.convert import (
    cell_table,
    leaf_table,
    refresh_capabilities,
    to_octree,
    to_uniform,
)
from zarr_vectors.octree_chunks.guard import (
    converting,
    is_converting,
    is_octree_level,
    octree_levels,
    octree_meta,
    refuse_octree_levels,
    refuse_octree_read,
    refuse_octree_write,
    refuse_unfinished_conversions,
)
from zarr_vectors.octree_chunks.read import leaves_in_box, read_leaves, read_octree, read_seams
from zarr_vectors.octree_chunks.table import (
    LeafTable,
    OctreeError,
    format_link_segment,
    parse_link_segment,
    plan_leaves,
)

__all__ = [
    "LeafTable", "OctreeError", "cell_table", "converting", "format_link_segment",
    "is_converting", "is_octree_level", "leaf_table", "leaves_in_box", "octree_levels",
    "octree_meta", "parse_link_segment", "plan_leaves", "read_leaves", "read_octree",
    "read_seams", "refresh_capabilities", "refuse_octree_levels", "refuse_octree_read", "refuse_octree_write", "refuse_unfinished_conversions",
    "to_octree", "to_uniform",
]
