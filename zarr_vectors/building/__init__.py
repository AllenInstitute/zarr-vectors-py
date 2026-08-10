"""The supported low-level surface, for tools that *build* stores.

Two audiences, two contracts.

:mod:`zarr_vectors.api` is for reading and querying data.  It says
nothing about chunks or fragments, and it is where an application
belongs.

This module is for the other audience: ingest converters, pyramid
builders, exporters — code whose job *is* the physical layout, and for
which "hide the storage" would be hiding the thing they are working on.
Forcing them onto the data-oriented API would be wrong, so instead they
get an honest promise about a smaller surface.

**The distinction this makes real.**  ``zarr_vectors.core``,
``zarr_vectors.lazy``, ``zarr_vectors.ops``, ``zarr_vectors.encoding``
and ``zarr_vectors.spatial`` are *internal*.  They change without notice;
they always did, and saying so is the only honest description of a
package whose recent history includes three breaking layout refactors.
What is re-exported here does not.

The contents are not a guess.  They are the symbols the two consuming
repositories actually import from ``core.arrays`` and ``core.store``,
counted — every name imported at three or more sites, plus the handful
below that threshold that have no alternative.  Migration is therefore a
change of import line and nothing else::

    -from zarr_vectors.core.arrays import list_chunk_keys, read_fragment
    +from zarr_vectors.building import list_chunk_keys, read_fragment

If something you need is missing, that is a bug in this list rather than
an invitation to import from ``core`` — say so and it will be added,
because a name used from ``core`` is a name nobody knows is load-bearing.
"""

from __future__ import annotations

# --- array names and layout sentinels --------------------------------
from zarr_vectors.constants import (
    CAP_FRAGMENT_INDEX,
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    FORMAT_VERSION,
    GROUPS,
    LINKS,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V1,
    # --- links ---
    cell_endpoint_chunks,
    # --- allocation ---
    create_attribute_array,
    create_fragment_attribute_array,
    create_groupings_array,
    create_groupings_attributes_array,
    create_link_attributes_array,
    create_links_array,
    create_links_family,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    # --- manifests ---
    expand_manifest_blocks,
    finalize_links,
    iter_link_cells,
    level_grid_layout,
    link_family_policy,
    links_has_perm,
    list_chunk_keys,
    list_link_attribute_offsets,
    list_link_deltas,
    list_link_offsets,
    object_count,
    # --- write sessions ---
    open_write_session,
    read_all_groupings,
    read_all_object_manifests,
    read_chunk_attributes,
    read_chunk_fragment_attributes,
    read_chunk_link_attributes,
    read_chunk_links,
    # --- reading ---
    read_chunk_vertices,
    read_fragment,
    read_group_object_ids,
    read_groupings_attributes,
    read_link_attributes,
    read_links,
    read_links_for_tuple,
    read_object_attribute_present_mask,
    read_object_attributes,
    read_object_manifests,
    read_object_vertices,
    read_vertex_fragment_index,
    # --- writing ---
    write_chunk_attributes,
    write_chunk_fragment_attributes,
    write_chunk_fragments,
    write_chunk_link_attributes,
    write_chunk_links,
    write_chunk_vertices,
    write_groupings,
    write_groupings_attributes,
    write_link_attributes,
    write_link_cells,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.group import Group
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    chunk_scale_factor,
    compute_bin_ratio,
    get_level_chunk_shape,
    validate_level_chunk_shape_against_root,
)
from zarr_vectors.core.multiscale import (
    get_level_scale,
    get_level_translation,
    read_multiscale_metadata,
    upsert_level_transform,
    write_multiscale_metadata,
)
from zarr_vectors.core.paths import (
    intra_offsets,
    is_intra,
    link_attributes_group_path,
    link_attributes_path,
    links_group_path,
    links_path,
    parse_offsets,
)
from zarr_vectors.core.store import (
    commit,
    create_resolution_level,
    create_store,
    get_resolution_level,
    list_available_ratios,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
    remove_resolution_level,
    session_for,
)
from zarr_vectors.core.streaming import ObjectIndexAppender
from zarr_vectors.encoding.fragments import (
    decode_object_manifest_blocks,
    encode_object_manifest_blocks,
)
from zarr_vectors.multiresolution.registry import (
    register_coarsen_strategy,
    register_selection_strategy,
)
from zarr_vectors.sharding.io import (
    get_shard_info,
    reshard,
    shard_store,
    unshard_store,
)
from zarr_vectors.spatial.boundary import (
    apply_perm_inverse,
    build_vertex_chunk_mapping,
    chunk_local_to_global_offsets,
    partition_cross_level_edges,
    split_polyline_at_boundaries,
)
from zarr_vectors.spatial.chunking import (
    assign_chunks,
    chunks_intersecting_bbox,
    compute_grid_shape,
    neighbouring_chunk_keys,
)


def is_sharded(level_group: Group, array_name: str) -> bool:
    """Whether one array is stored with the native sharding codec.

    Takes a *name*, not a resolved zarr node.  The internal helper takes
    the node, which means a caller has to obtain one — and obtaining one
    is exactly the reach past the API that this module exists to remove.
    """
    from zarr_vectors.sharding.io import _is_native_sharded

    try:
        node = level_group._lookup_node(array_name)
    except Exception:
        return False
    return bool(node is not None and _is_native_sharded(node))


def write_object_manifests(
    level_group: Group,
    manifest_blobs: list[bytes],
) -> None:
    """Write ``object_index/manifests`` as one ragged vlen-bytes array.

    The promoted spelling of an internal that two consuming call sites
    use, so that writing an object index does not require a private name.
    Takes already-encoded blobs -- one per object, in object-id order --
    which is what those call sites already have; encode them with
    :func:`encode_object_manifest_blocks`.
    """
    from zarr_vectors.core.arrays import _write_object_index_manifests

    _write_object_index_manifests(level_group, manifest_blobs)


def rebuild_presence(level_group: Group, array_name: str) -> list[str]:
    """Rebuild one array's ``nonempty_chunks`` manifest from the store.

    The coordinator half of decentralised writing: workers writing
    disjoint cells pass ``record_presence=False`` because the manifest is
    one attribute shared by every cell, so stamping it races.  This
    rebuilds it once, afterwards.

    A downstream module re-implements this by walking the raw zarr group;
    the version here is the one that already gathers asynchronously.
    """
    return level_group.derive_nonempty_chunks(array_name)


__all__ = [
    "CAP_FRAGMENT_INDEX",
    "CAP_MULTISCALE_LINKS",
    "CAP_PRESERVED_OBJECT_IDS",
    "CAP_SHARED_FRAGMENTS",
    "FORMAT_VERSION",
    "GROUPS",
    "Group",
    "LINKS",
    "LevelMetadata",
    "OBJECT_ATTRIBUTES",
    "OBJECT_INDEX",
    "OBJECT_INDEX_LAYOUT_V1",
    "ObjectIndexAppender",
    "RootMetadata",
    "VERTEX_ATTRIBUTES",
    "VERTEX_FRAGMENTS",
    "VERTICES",
    "apply_perm_inverse",
    "assign_chunks",
    "build_vertex_chunk_mapping",
    "cell_endpoint_chunks",
    "chunk_local_to_global_offsets",
    "chunk_scale_factor",
    "chunks_intersecting_bbox",
    "commit",
    "compute_bin_ratio",
    "compute_grid_shape",
    "create_attribute_array",
    "create_fragment_attribute_array",
    "create_groupings_array",
    "create_groupings_attributes_array",
    "create_link_attributes_array",
    "create_links_array",
    "create_links_family",
    "create_object_attributes_array",
    "create_object_index_array",
    "create_resolution_level",
    "create_store",
    "create_vertices_array",
    "decode_object_manifest_blocks",
    "encode_object_manifest_blocks",
    "expand_manifest_blocks",
    "finalize_links",
    "get_level_chunk_shape",
    "get_level_scale",
    "get_level_translation",
    "get_resolution_level",
    "get_shard_info",
    "intra_offsets",
    "is_intra",
    "is_sharded",
    "iter_link_cells",
    "level_grid_layout",
    "link_attributes_group_path",
    "link_attributes_path",
    "link_family_policy",
    "links_group_path",
    "links_has_perm",
    "links_path",
    "list_available_ratios",
    "list_chunk_keys",
    "list_link_attribute_offsets",
    "list_link_deltas",
    "list_link_offsets",
    "list_resolution_levels",
    "neighbouring_chunk_keys",
    "object_count",
    "open_store",
    "open_write_session",
    "parse_offsets",
    "partition_cross_level_edges",
    "read_all_groupings",
    "read_all_object_manifests",
    "read_chunk_attributes",
    "read_chunk_fragment_attributes",
    "read_chunk_link_attributes",
    "read_chunk_links",
    "read_chunk_vertices",
    "read_fragment",
    "read_group_object_ids",
    "read_groupings_attributes",
    "read_level_metadata",
    "read_link_attributes",
    "read_links",
    "read_links_for_tuple",
    "read_multiscale_metadata",
    "read_object_attribute_present_mask",
    "read_object_attributes",
    "read_object_manifests",
    "read_object_vertices",
    "read_root_metadata",
    "read_vertex_fragment_index",
    "rebuild_presence",
    "register_coarsen_strategy",
    "register_selection_strategy",
    "remove_resolution_level",
    "reshard",
    "session_for",
    "shard_store",
    "split_polyline_at_boundaries",
    "unshard_store",
    "upsert_level_transform",
    "validate_level_chunk_shape_against_root",
    "write_chunk_attributes",
    "write_chunk_fragment_attributes",
    "write_chunk_fragments",
    "write_chunk_link_attributes",
    "write_chunk_links",
    "write_chunk_vertices",
    "write_groupings",
    "write_groupings_attributes",
    "write_link_attributes",
    "write_link_cells",
    "write_links",
    "write_multiscale_metadata",
    "write_object_attributes",
    "write_object_index",
    "write_object_manifests",
]
