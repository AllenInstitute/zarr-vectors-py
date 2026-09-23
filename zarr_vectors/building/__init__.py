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

from pathlib import Path
from typing import Any

# --- array names and layout sentinels --------------------------------
from zarr_vectors.constants import (
    CAP_FRAGMENT_INDEX,
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    FORMAT_VERSION,
    FRAGMENT_ATTRIBUTES,
    GROUP_ATTRIBUTES,
    GROUPS,
    LINK_ATTRIBUTES,
    LINK_FRAGMENTS,
    LINKS,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
    XLEVEL_EXPLICIT,
    XLEVEL_NONE,
)
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V1,
    OBJECT_INDEX_MANIFEST_BUCKET,
    ManifestCSR,
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
    link_endpoints_to_rows,
    link_family_policy,
    links_has_perm,
    list_chunk_keys,
    list_link_attribute_offsets,
    list_link_deltas,
    list_link_offsets,
    object_count,
    # --- write sessions ---
    open_write_session,
    patch_object_manifests,
    read_all_groupings,
    read_all_object_manifests,
    read_all_object_manifests_csr,
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
    read_link_arrays,
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
    write_link_attribute_cells,
    write_link_attributes,
    write_link_cells,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.cells import (
    CellBatch,
    CellColumn,
    CellReadError,
    read_cells,
    read_neighbourhood,
)
from zarr_vectors.core.group import Group, observe_presence_writes
from zarr_vectors.core.metadata import (
    LevelMetadata,
    RootMetadata,
    chunk_scale_factor,
    chunk_scale_from_root,
    compute_bin_ratio,
    compute_bin_shape,
    get_level_bin_shape,
    get_level_chunk_shape,
    level_chunk_scale,
    level_factor,
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
    decode_object_manifests_csr,
    encode_object_manifest_blocks,
    encode_object_manifests_csr,
)
from zarr_vectors.exceptions import ArrayError, StoreError
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
    "read_bytes", "write_bytes", "write_cells", "chunk_exists", "list_chunks",
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
    "collect_presence", "apply_presence", "presence_deferred",
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


def stamp_ome_node(
    store: str | Path | Group,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """Add (or refresh) the RFC 8 ``ome`` node on an existing store.

    What makes a store nameable by an OME *collection*: a resolver
    following ``{"type": "zarr", "path": "./x.zarrvectors"}`` fetches the
    root ``zarr.json`` and looks for a legal node under ``ome``.  Stores
    written before 0.9.2 carry none, and stores written since carry one
    already -- calling this on either leaves a correct block, so it is
    safe to run over a whole directory of stores.

    Metadata-only, and the one upgrade in this format's history that can
    be applied in place.  Every prior version bump moved bytes, so the
    migration story was "rewrite from source"; this one only adds a root
    attribute, so a store is brought up to date without its data being
    read, let alone rewritten.

    Args:
        store: Store path, URL, or an open root group.  A path or URL is
            opened ``mode="r+"``.
        name: Store name for the node.  ``None`` keeps the name already
            recorded, and otherwise derives one from the store URL.

    Returns:
        The ``ome`` block as written.
    """
    from zarr_vectors.core.group import Group as _Group
    from zarr_vectors.core.ome import refresh_root_node

    root = store if isinstance(store, _Group) else open_store(store, mode="r+")
    return refresh_root_node(root, name=name)


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
    manifest_blobs: Any = None,
    *,
    chunk_coords: Any = None,
    fragment_idx: Any = None,
    manifest_offsets: Any = None,
    ids: Any = None,
    mode: str = "replace",
    at: int | None = None,
) -> tuple[int, int]:
    """Write ``object_index/manifests`` as one ragged vlen-bytes array.

    Give the manifests one of two ways:

    - ``manifest_blobs``: already-encoded blobs, one per object, in
      object order (encode them with :func:`encode_object_manifest_blocks`);
    - as arrays: object ``o`` owns blocks
      ``manifest_offsets[o]:manifest_offsets[o + 1]``, block ``b`` naming
      fragment ``fragment_idx[b]`` of chunk ``chunk_coords[b]``. Without
      ``manifest_offsets`` every object owns one block. Encoded with no
      Python object per object, to the same bytes; device arrays are
      copied to the host once each.

    Only the manifests array is written: committing the index's metadata
    (``num_objects`` and the rest) stays the caller's step.

    Args:
        level_group: Resolution level group.
        manifest_blobs: Encoded manifests (list of bytes or object array).
        chunk_coords: ``(M, sid_ndim)`` block chunk coordinates.
        fragment_idx: ``(M,)`` block fragment indices.
        manifest_offsets: ``(n + 1,)`` CSR offsets of blocks per object.
        ids: Object ids of the rows written, for an index that stores
            ids rather than using rows as ids; see below.
        mode: ``"replace"`` rewrites every row.  ``"append"`` extends the
            existing array, touching only the zarr chunks the new rows fall
            in — the write a per-chunk emitter wants, since replacing costs
            the whole index once per chunk.
        at: Row index the appended blobs must start at (``mode="append"``
            only).  Pass the object-id being claimed so a torn previous
            flush cannot shift ids; ``None`` appends at the current end.

    On an index that stores its object ids (layout V2), an append extends
    the id table with ``ids``, or with the rows when the table is simply
    the rows; otherwise it raises, where it used to add rows with no id
    that a read by id could not find.

    Returns:
        ``(first_row, n)``: where the written manifests start, and how many.
    """
    from zarr_vectors import _xp
    from zarr_vectors.core.arrays import _write_object_index_manifests
    from zarr_vectors.encoding.fragments import encode_object_manifests_csr

    as_arrays = chunk_coords is not None or fragment_idx is not None
    if as_arrays == (manifest_blobs is not None) or (
        as_arrays and (chunk_coords is None or fragment_idx is None)
    ):
        raise ArrayError(
            "give manifest_blobs, or chunk_coords and fragment_idx "
            "(optionally with manifest_offsets), not both"
        )
    if as_arrays:
        try:
            sid = int((level_group.read_array_meta(OBJECT_INDEX) or {}).get("sid_ndim"))
        except Exception:
            sid = None
        manifest_blobs = encode_object_manifests_csr(
            _xp.to_host(chunk_coords), _xp.to_host(fragment_idx),
            None if manifest_offsets is None else _xp.to_host(manifest_offsets),
            sid_ndim=sid,
        )
    first = _write_object_index_manifests(
        level_group, manifest_blobs, mode=mode, at=at,
        ids=None if ids is None else _xp.to_host(ids),
    )
    return int(first), len(manifest_blobs)


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


def build_fragment_owner_index(store_path, *, level: int | None = None) -> int:
    """Record which object owns each fragment, as a readable column.

    Writes ``fragment_attributes/object_id``, the slot the format
    already reserves for it.  With the column in place "which objects
    reference this fragment" is one cell read; without it, the only way
    to answer is to decode every manifest in the level, which is what
    an edit session does on its first lookup and what four other places
    in this package each reinvent.

    Offered as a maintenance verb rather than run by the writers,
    because building it costs a full manifest scan and every consumer
    falls back cleanly when it is absent.  Re-running it is safe; it
    rewrites whatever it finds.

    Args:
        store_path: Store path, URL or open group.
        level: Only this resolution level.  ``None`` does every level.

    Returns:
        How many fragments were recorded.
    """
    from zarr_vectors.core.arrays import write_fragment_owner_column
    from zarr_vectors.core.store import (
        get_resolution_level,
        list_resolution_levels,
        open_store,
    )

    root = open_store(store_path, mode="r+")
    levels = (
        list_resolution_levels(root) if level is None else [int(level)]
    )
    total = 0
    for lvl in levels:
        total += write_fragment_owner_column(get_resolution_level(root, lvl))
    return total


def defer_presence(level_group: Group) -> list[str]:
    """Declare this level's presence deferred to one coordinator rebuild.

    The build mode for many workers writing one level.  ``nonempty_chunks``
    is ONE attribute per array, shared by every cell, so each stamp is a
    read-modify-write of the whole list: two workers writing disjoint
    cells still race, and every stamp costs the full list again, which
    makes a build of N cells O(N²) in metadata bytes.  Declaring the
    level deferred removes both at the source:

    * every per-chunk array in the level drops its manifest, and arrays
      allocated afterwards -- a links segment a worker creates on first
      write -- are born without one;
    * a write into an array with no manifest stamps nothing, whatever
      ``record_presence`` says, so a worker needs no flag threaded
      through every call, including the writers that do not expose one;
    * readers ask the store instead: :meth:`Group.chunk_exists` reads the
      cell, :meth:`Group.list_chunks` lists the array.  A cell is visible
      the moment its payload lands, not only after the rebuild.

    Then one :func:`rebuild_presence` over the level, after the workers
    finish, writes every manifest once and clears the declaration.  The
    finished store records the same presence an undeferred build does.

    Call it from the coordinator BEFORE the workers open the level: the
    declaration is read from the level handle a worker holds, so a handle
    opened earlier still allocates arrays with a manifest.  Those stay
    correct -- the rebuild rewrites them too -- but their stamps are back
    to racing.

    While deferred, readers pay for asking the store: a listing per
    array per :meth:`Group.list_chunks` (cached inside
    :meth:`Group.cached_nodes`), a request per :meth:`Group.chunk_exists`,
    and a sharded array is read shard by shard.  Three kinds of reader
    see nothing until the rebuild: an offline-read session, which must
    not touch the store; one on a store that cannot list; and a
    zarr-vectors older than 0.9.3, which reads a missing manifest as an
    empty array.  A reader of the manifest itself may do better:
    neuroglancer's datasource probes every cell when there is none,
    which finds them all, slowly.

    Returns:
        Every per-chunk array path in the level.
    """
    from zarr_vectors.core.group import (
        _NONEMPTY_CHUNKS_ATTR,
        _PRESENCE_DECL_ATTR,
        _PRESENCE_DEFERRED,
    )

    # Declared first, so an array allocated while the manifests are
    # being dropped is born without one rather than slipping between.
    level_group.attrs.update({_PRESENCE_DECL_ATTR: _PRESENCE_DEFERRED})
    names = per_chunk_array_paths(level_group)
    for name in names:
        arr = level_group._sharded_chunk_array(name)
        if arr is not None and _NONEMPTY_CHUNKS_ATTR in arr.attrs:
            del arr.attrs[_NONEMPTY_CHUNKS_ATTR]
            level_group._invalidate_node(name)
    return names


def rebuild_presence(
    level_group: Group,
    array_name: str | None = None,
    *,
    on_sharded: str = "derive",
) -> list[str]:
    """Rebuild ``nonempty_chunks`` from the store — one array, or the level.

    The coordinator half of decentralised writing: workers writing
    disjoint cells pass ``record_presence=False`` because the manifest is
    one attribute shared by every cell, so stamping it races.  This
    rebuilds it once, afterwards.

    Covers sharded arrays.  It did not, and defaulted to ``"skip"``,
    while a sharded array's manifest could not be rebuilt from a per-cell
    listing — safe then, because sharding ran after the rebuild by
    contract, so anything sharded was already correct.  A store can now
    be born sharded, and under that premise "skip" quietly declined to
    repair the arrays most in need of it.

    The level-wide form also ends a :func:`defer_presence` declaration,
    once every manifest is written: arrays allocated afterwards get a
    manifest again, and stamps resume.  The single-array form leaves the
    declaration alone.

    Args:
        array_name: One array's path, or ``None`` (the default) to walk
            every per-chunk array in the level — the shape a coordinator
            actually needs after a parallel phase.
        on_sharded: Forwarded to
            :meth:`Group.derive_nonempty_chunks`.  ``"derive"`` (the
            default) rebuilds sharded arrays like any other; ``"skip"``
            leaves them alone; ``"raise"`` asserts none are sharded.

    Returns:
        The sorted keys recorded, for the single-array form; for the
        level-wide form, every per-chunk array path walked.  Under
        ``on_sharded="skip"`` a sharded array is left alone and therefore
        absent, so the return still says what was actually rebuilt.
    """
    if array_name is not None:
        return level_group.derive_nonempty_chunks(
            array_name, on_sharded=on_sharded,
        )

    rebuilt: list[str] = []
    for name in per_chunk_array_paths(level_group):
        if on_sharded == "skip" and array_is_sharded(level_group, name):
            # Consult before the call so the caller learns which arrays
            # were rebuilt; "skip" alone would return a manifest and make
            # a skipped array indistinguishable from a rebuilt one.
            continue
        level_group.derive_nonempty_chunks(name, on_sharded=on_sharded)
        rebuilt.append(name)

    from zarr_vectors.core.group import _PRESENCE_DECL_ATTR

    # Last, so a rebuild that fails part-way leaves the level still
    # declared -- and its remaining arrays still derived, not trusted.
    if level_group.presence_deferred():
        del level_group.zarr_group.attrs[_PRESENCE_DECL_ATTR]
    return rebuilt


__all__ = [
    "CAP_FRAGMENT_INDEX",
    "CellBatch",
    "CellColumn",
    "CellReadError",
    "ManifestCSR",
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
    # Exported so a consumer can audit an existing store's manifest chunking
    # without importing from ``core``: the number is fixed at array-creation
    # time and cannot be changed afterwards, so "is this store chunked
    # correctly?" is a question only an outside reader can answer, and it needs
    # the reference value to answer it against.
    "OBJECT_INDEX_MANIFEST_BUCKET",
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
    "defer_presence",
    "decompose_tree_to_paths",
    "decode_object_manifests_csr",
    "encode_object_manifest_blocks",
    "encode_object_manifests_csr",
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
    "link_endpoints_to_rows",
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
    "observe_presence_writes",
    "open_store",
    "open_write_session",
    "parse_offsets",
    "partition_cross_level_edges",
    "patch_object_manifests",
    "per_chunk_array_paths",
    "read_all_groupings",
    "read_all_object_manifests",
    "read_all_object_manifests_csr",
    "read_attribute_fragment",
    "read_cells",
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
    "read_link_arrays",
    "read_neighbourhood",
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
    "build_fragment_owner_index",
    "rebuild_presence",
    "rechunk",
    "rechunk_by_attribute",
    "refresh_arrays_present",
    "stamp_ome_node",
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
