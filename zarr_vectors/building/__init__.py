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
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    FRAGMENT_ATTRIBUTES,
    GROUP_ATTRIBUTES,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
    XLEVEL_EXPLICIT,
    XLEVEL_NONE,
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    FORMAT_VERSION,
    GROUPS,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    LINKS,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V1,
    attribute_layout,
    # --- links ---
    cell_endpoint_chunks,
    chunk_vertex_count,
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
    read_attribute_fragment,
    read_chunk_attributes,
    read_chunk_fragment_attributes,
    read_chunk_link_attributes,
    read_chunk_links,
    # --- reading ---
    read_chunk_vertex_buffer,
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
    write_link_attribute_cells,
    write_link_cells,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.group import Group
from zarr_vectors.exceptions import StoreError
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    chunk_scale_factor,
    chunk_scale_from_root,
    compute_bin_ratio,
    compute_bin_shape,
    get_level_bin_shape,
    level_chunk_scale,
    level_factor,
    get_level_chunk_shape,
    validate_bin_shape_divides_chunk,
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
    update_level_metadata,
    update_root_metadata,
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
# Re-layout, alongside the sharding verbs it sits next to.  Retired from
# ``zarr_vectors.__all__`` (it describes where bytes live, not what the
# data is), so this is where a tool gets it.  Note it REWRITES the chunk
# grid of every array it touches -- a coordinator-side operation, never
# something to run while workers are writing.
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute
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

# --- store-creating writers -------------------------------------------
#
# Promoted from ``zarr_vectors.types``, unchanged.  They were in neither
# tier: not internal, not on either surface, and the thing this package's
# entire ingest layer writes through -- 88 import statements in the one
# consumer measured.  ``Dataset.add_*`` does not supersede them
# (``write_points`` takes 18 keyword parameters against add_points' six,
# and the skeleton group's four-call streaming writer has no counterpart
# at all), and their subject IS the physical layout -- chunk_shape,
# bin_shape, shard_shape, compressor, dtype -- which is this module's
# stated charter.  So they belong here rather than behind a third surface.
#
# The READERS are not promoted: ``ReadResult.from_*`` covers all five.
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.lines import write_lines
from zarr_vectors.types.meshes import write_mesh
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.types.skeletons import (
    decompose_tree_to_paths,
    finalize_skeleton_store,
    get_coordinate_offset,
    init_skeleton_store,
    read_skeleton_by_segment_id,
    set_coordinate_offset,
    write_skeleton_chunk,
    write_skeleton_cross_chunk_links,
)


#: The subset of :class:`Group`'s methods this module promises.
#:
#: ``Group`` is re-exported here, but it carries ~40 public methods and a
#: consumer cannot tell which are contract and which are the storage
#: layer's own working surface.  These are the ones a builder needs and
#: that will not change without a deprecation path; everything else on the
#: class -- ``zarr_group``, ``_lookup_node``, the offline-session plumbing
#: -- is internal even though Python cannot say so.
#:
#: Note ``commit`` is deliberately absent: it is a module-level function
#: in ``core.store`` (re-exported here), not a ``Group`` method.
GROUP_SUPPORTED_METHODS: frozenset[str] = frozenset({
    # hierarchy
    "attrs", "children", "create_group", "require_group",
    "array_exists", "standalone_array_exists", "delete_subtree",
    # per-chunk cell I/O
    "read_bytes", "write_bytes", "chunk_exists", "list_chunks",
    "create_sharded_chunk_array",
    # per-array metadata
    "read_array_meta", "write_array_meta", "read_array_attrs",
    "read_array_fill_value",
    # whole standalone arrays
    "read_array", "write_array", "read_vlen_array", "write_vlen_array",
    "read_vlen_element", "read_vlen_elements",
    # batching + presence
    "batched_reads", "batched_writes", "offline_reads", "chunk_array_codecs",
    "derive_nonempty_chunks", "native_sharded_arrays",
    # identity
    "url", "prefix", "path",
})


def refresh_arrays_present(level_group: Group) -> list[str]:
    """Re-derive the level's ``arrays_present`` from what is on disk.

    ``arrays_present`` is hand-listed at every write site, so it drifts:
    an ordinary ``write_points`` declared ``["vertices",
    "vertex_attributes"]`` while ``vertex_fragments`` sat on disk
    undeclared, and no allocation function registers anything at all.  A
    reader that gates on the list therefore cannot see arrays that exist.

    This walks the level instead and writes back the FAMILY names found --
    the granularity every reader actually tests.  The single owner of the
    field; a coordinator calls it once after a parallel phase, alongside
    :func:`rebuild_presence`.

    Returns the sorted family names now recorded.
    """
    families: set[str] = set()

    # Per-chunk families: the family is the first path segment for every
    # one of them, including the nested link families
    # (``links/<delta>/<offsets>`` -> ``links``).
    for path in per_chunk_array_paths(level_group):
        families.add(path.split("/")[0])

    # The arrays with no spatial chunk grid, which the walk excludes but
    # which writers do declare.
    for name in (OBJECT_INDEX, OBJECT_ATTRIBUTES, GROUPS, GROUP_ATTRIBUTES):
        try:
            if level_group.array_exists(name):
                families.add(name)
        except Exception:  # noqa: BLE001 - a probe, never fatal
            continue

    ordered = sorted(families)
    update_level_metadata(level_group, arrays_present=ordered)
    return ordered


def array_is_sharded(level_group: Group, array_name: str) -> bool:
    """Whether ONE ARRAY is stored with the native sharding codec.

    Takes a *name*, not a resolved zarr node.  The internal helper takes
    the node, which means a caller has to obtain one — and obtaining one
    is exactly the reach past the API that this module exists to remove.

    Named for the array, not for the store, because
    :func:`zarr_vectors.sharding.io.is_sharded` answers a different
    question — "does any array in this store shard?" — over a *store
    path*.  While both were spelled ``is_sharded`` a caller who swapped
    one import line for the other got no error: the path went in as
    ``level_group``, the lookup failed, and the old bare ``except``
    returned ``False``.  For a whole store that is a plausible answer, so
    nothing surfaced.  Use :func:`get_shard_info` for the store-wide
    question; it is already exported here.

    Raises:
        StoreError: If ``level_group`` is not a Group. A missing or
            non-array *path* is still ``False`` — that is a real answer,
            unlike being handed the wrong kind of object.
    """
    from zarr_vectors.sharding.io import _is_native_sharded

    if not isinstance(level_group, Group):
        raise StoreError(
            f"array_is_sharded() takes a level Group and an array name, got "
            f"{type(level_group).__name__}. For 'does this store shard at "
            f"all', pass the store path to get_shard_info()."
        )
    try:
        node = level_group._lookup_node(array_name)
    except (StoreError, KeyError):
        return False
    return bool(node is not None and _is_native_sharded(node))


def is_sharded(level_group: Group, array_name: str) -> bool:
    """Deprecated alias for :func:`array_is_sharded`."""
    import warnings

    warnings.warn(
        "building.is_sharded() is renamed array_is_sharded(). The old name "
        "collided with sharding.io.is_sharded(store_path), which answers a "
        "different question over a different argument, and swapping the two "
        "returned False instead of raising.",
        DeprecationWarning,
        stacklevel=2,
    )
    return array_is_sharded(level_group, array_name)


def link_endpoint_scales(
    level_group: Group, delta: int, sid_ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The ``(scale_src, scale_trg)`` pair the link placement arithmetic takes.

    Each is the per-axis integer multiple of the root ``chunk_shape``
    that a level's own ``chunk_shape`` represents: ``scale_src`` for the
    level ``level_group`` names, ``scale_trg`` for ``level + delta``.
    They exist because two levels of a pyramid built with
    ``chunk_scale_factor > 1`` index grids of different cell sizes, so
    their chunk coordinates cannot be differenced directly.

    Promoted because :func:`cell_endpoint_chunks` is exported and *needs*
    these two, and until now the only way to obtain them was a private
    helper — an argument on a supported function with no supported
    provenance.  Reconstructing them from :func:`read_root_metadata` and
    :func:`chunk_scale_from_root` is possible but gets the padding rule
    wrong: a store chunked by an attribute prepends a bin axis to every
    chunk key, so the scales must be padded with leading 1s to
    ``sid_ndim`` or the rank check rejects the store.  This is the same
    value the readers and writers anchor through, so a caller
    reconstructing a cell agrees with the code that placed it.

    Args:
        level_group: The level the *source* endpoint lives at.
        delta: Level delta of the link family — ``0`` for links within
            one level, non-zero for cross-level ones.
        sid_ndim: Rank of the chunk keys in this family, which is
            ``link_family_policy(level_group, delta)[1]`` and may exceed
            the spatial rank on an attribute-binned store.

    Returns:
        ``(scale_src, scale_trg)``, each ``sid_ndim`` ints.  Both are
        all-ones when the metadata cannot be read or the target level
        does not exist yet.  That is exactly right for a default pyramid
        and makes the anchor a no-op; it is *wrong* for one built with
        ``chunk_scale_factor > 1``, so call this only once the target
        level and its ``LevelMetadata`` are on disk.
    """
    from zarr_vectors.core.arrays import _link_scales

    return _link_scales(level_group, int(delta), int(sid_ndim))


def write_object_manifests(
    level_group: Group,
    manifest_blobs: list[bytes],
    *,
    mode: str = "replace",
    at: int | None = None,
) -> None:
    """Write ``object_index/manifests`` as one ragged vlen-bytes array.

    The promoted spelling of an internal that two consuming call sites
    use, so that writing an object index does not require a private name.
    Takes already-encoded blobs -- one per object, in object-id order --
    which is what those call sites already have; encode them with
    :func:`encode_object_manifest_blocks`.

    Args:
        level_group: Resolution level group.
        manifest_blobs: Encoded manifests, one per object.  Under the
            default ``mode="replace"`` these are the WHOLE index; under
            ``mode="append"`` they are only the new objects.
        mode: ``"replace"`` rewrites every row.  ``"append"`` extends the
            existing array, touching only the zarr chunks the new rows fall
            in — the write a per-chunk emitter wants, since replacing costs
            the whole index once per chunk.
        at: Row index the appended blobs must start at (``mode="append"``
            only).  Pass the object-id being claimed so a torn previous
            flush cannot shift ids; ``None`` appends at the current end.
    """
    from zarr_vectors.core.arrays import _write_object_index_manifests

    _write_object_index_manifests(level_group, manifest_blobs, mode=mode, at=at)


def per_chunk_array_paths(level_group: Group) -> list[str]:
    """Every per-spatial-chunk array path in ``level_group``, recursively.

    Walks the hierarchy rather than guessing names: the link families nest
    two levels deeper than everything else, and their ``<offsets>``
    segments are discoverable only by listing.  Excludes the arrays with
    no spatial chunk grid — ``object_index``, ``object_attributes/…``,
    ``groups``, ``group_attributes/…`` — which have no presence manifest.

    Walks via :meth:`Group.children` / :meth:`Group.standalone_array_exists`
    rather than the underlying zarr group's ``array_keys`` / ``group_keys``:
    those go straight to the backing store, missing the node cache and the
    offline-read snapshot every other read here passes through.

    Offered because a consumer that needs it otherwise re-implements both
    this walk and the private :func:`_is_per_chunk_array` it depends on,
    and a fork of a private predicate is a fork nobody diffs.
    """
    from zarr_vectors.core.arrays import _is_per_chunk_array

    names: list[str] = []

    def _walk(prefix: str, group: Group) -> None:
        for name in group.children():
            path = f"{prefix}{name}"
            if level_group.standalone_array_exists(path):
                if _is_per_chunk_array(path):
                    names.append(path)
            else:
                _walk(f"{path}/", group[name])

    _walk("", level_group)
    return sorted(names)


def rebuild_presence(
    level_group: Group,
    array_name: str | None = None,
    *,
    on_sharded: str = "skip",
) -> list[str]:
    """Rebuild ``nonempty_chunks`` from the store — one array, or the level.

    The coordinator half of decentralised writing: workers writing
    disjoint cells pass ``record_presence=False`` because the manifest is
    one attribute shared by every cell, so stamping it races.  This
    rebuilds it once, afterwards.

    Args:
        array_name: One array's path, or ``None`` (the default) to walk
            every per-chunk array in the level — the shape a coordinator
            actually needs after a parallel phase.
        on_sharded: Forwarded to
            :meth:`Group.derive_nonempty_chunks`.  Defaults to ``"skip"``
            here, not ``"raise"``: this verb legitimately runs over a
            level's mixed arrays, and a sharded one already has a correct
            manifest because sharding runs after the rebuild by contract.
            Pass ``"raise"`` to assert that ordering instead.

    Returns:
        The sorted keys recorded, for the single-array form; for the
        level-wide form, the array paths whose manifests were rebuilt
        (sharded arrays are skipped and therefore absent).
    """
    if array_name is not None:
        return level_group.derive_nonempty_chunks(
            array_name, on_sharded=on_sharded,
        )

    rebuilt: list[str] = []
    for name in per_chunk_array_paths(level_group):
        if array_is_sharded(level_group, name):
            # Consult before the call so the caller learns which arrays
            # were rebuilt; "skip" alone would return a manifest and make
            # a skipped array indistinguishable from a rebuilt one.
            continue
        level_group.derive_nonempty_chunks(name, on_sharded=on_sharded)
        rebuilt.append(name)
    return rebuilt


__all__ = [
    "CAP_FRAGMENT_INDEX",
    "CAP_MULTISCALE_LINKS",
    "CAP_PRESERVED_OBJECT_IDS",
    "CAP_SHARED_FRAGMENTS",
    "COARSEN_PER_OBJECT",
    "DEFAULT_CROSS_LEVEL_DEPTH",
    "DEFAULT_CROSS_LEVEL_STORAGE",
    "FORMAT_VERSION",
    "FRAGMENT_ATTRIBUTES",
    "GROUPS",
    "GROUP_ATTRIBUTES",
    "GROUP_SUPPORTED_METHODS",
    "Group",
    "LINKS",
    "LINKS_IMPLICIT_BRANCHES",
    "LINKS_IMPLICIT_SEQUENTIAL",
    "LINK_ATTRIBUTES",
    "LINK_FRAGMENTS",
    "LevelMetadata",
    "OBJECT_ATTRIBUTES",
    "OBJECT_INDEX",
    "OBJECT_INDEX_LAYOUT_V1",
    "ObjectIndexAppender",
    "RechunkSpec",
    "RootMetadata",
    "VERTEX_ATTRIBUTES",
    "VERTEX_FRAGMENTS",
    "VERTICES",
    "XLEVEL_EXPLICIT",
    "XLEVEL_NONE",
    "apply_perm_inverse",
    "array_is_sharded",
    "assign_chunks",
    "attribute_layout",
    "build_vertex_chunk_mapping",
    "cell_endpoint_chunks",
    "chunk_local_to_global_offsets",
    "chunk_vertex_count",
    "chunk_scale_factor",
    "chunk_scale_from_root",
    "chunks_intersecting_bbox",
    "commit",
    "compute_bin_ratio",
    "compute_bin_shape",
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
    "decompose_tree_to_paths",
    "encode_object_manifest_blocks",
    "expand_manifest_blocks",
    "finalize_links",
    "finalize_skeleton_store",
    "get_coordinate_offset",
    "get_level_bin_shape",
    "get_level_chunk_shape",
    "get_level_scale",
    "get_level_translation",
    "get_resolution_level",
    "get_shard_info",
    "init_skeleton_store",
    "intra_offsets",
    "is_intra",
    "is_sharded",
    "iter_link_cells",
    "level_chunk_scale",
    "level_factor",
    "level_grid_layout",
    "link_attributes_group_path",
    "link_attributes_path",
    "link_endpoint_scales",
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
    "per_chunk_array_paths",
    "read_all_groupings",
    "read_all_object_manifests",
    "read_attribute_fragment",
    "read_chunk_attributes",
    "read_chunk_fragment_attributes",
    "read_chunk_link_attributes",
    "read_chunk_links",
    "read_chunk_vertex_buffer",
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
    "read_skeleton_by_segment_id",
    "read_vertex_fragment_index",
    "rebuild_presence",
    "rechunk",
    "rechunk_by_attribute",
    "refresh_arrays_present",
    "register_coarsen_strategy",
    "register_selection_strategy",
    "remove_resolution_level",
    "reshard",
    "session_for",
    "set_coordinate_offset",
    "shard_store",
    "split_polyline_at_boundaries",
    "unshard_store",
    "update_level_metadata",
    "update_root_metadata",
    "upsert_level_transform",
    "validate_bin_shape_divides_chunk",
    "validate_level_chunk_shape_against_root",
    "write_chunk_attributes",
    "write_chunk_fragment_attributes",
    "write_chunk_fragments",
    "write_chunk_link_attributes",
    "write_chunk_links",
    "write_chunk_vertices",
    "write_graph",
    "write_groupings",
    "write_groupings_attributes",
    "write_lines",
    "write_link_attribute_cells",
    "write_link_attributes",
    "write_link_cells",
    "write_links",
    "write_mesh",
    "write_multiscale_metadata",
    "write_object_attributes",
    "write_object_index",
    "write_object_manifests",
    "write_points",
    "write_polylines",
    "write_skeleton_chunk",
    "write_skeleton_cross_chunk_links",
]
