"""Octree chunks: an optional, convertible chunking of a level.

A level normally has uniform chunks.  :func:`to_octree` regroups its cells
into octree leaves sized by vertex count -- each leaf one chunk, so no
chunk holds more than ``target`` vertices unless a single cell does --
and :func:`to_uniform` restores the uniform chunks exactly.  See
:mod:`zarr_vectors.octree_chunks.table` for addresses and link naming.
"""

from zarr_vectors.octree_chunks.convert import (
    cell_table,
    leaf_table,
    octree_meta,
    to_octree,
    to_uniform,
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
    "LeafTable", "OctreeError", "cell_table", "format_link_segment", "leaf_table",
    "leaves_in_box", "octree_meta", "parse_link_segment", "plan_leaves", "read_leaves",
    "read_octree", "read_seams", "to_octree", "to_uniform",
]
