"""Octree chunks: an optional, convertible chunking of a level.

A level normally has uniform chunks.  :func:`to_octree` regroups its cells
into octree leaves sized by vertex count -- each leaf one chunk, so no
chunk holds more than ``target`` vertices unless a single cell does --
and :func:`to_uniform` restores the uniform chunks exactly.  See
:mod:`zarr_vectors.octree_chunks.table` for addresses and link naming, and
:mod:`zarr_vectors.octree_chunks.guard` for telling an octree level, and
the conversion's own writes to it, apart -- and for the refusals every
other writer and uniform-only reader raises on an octree level.
"""

from zarr_vectors.octree_chunks.convert import (
    cell_table,
    leaf_table,
    octree_meta,
    to_octree,
    to_uniform,
)
from zarr_vectors.octree_chunks.guard import (
    converting,
    is_converting,
    is_octree_level,
    octree_levels,
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
    "read_seams", "refuse_octree_levels", "refuse_octree_read", "refuse_octree_write", "refuse_unfinished_conversions",
    "to_octree", "to_uniform",
]
