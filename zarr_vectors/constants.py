"""Format constants for the Zarr Vectors (ZV) format.

These are the canonical names, prefixes, and default values used
throughout the specification.  Changing a value here changes it
everywhere in the package.
"""

# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

FORMAT_VERSION: str = "0.8.1"
"""Current ZV specification version.

0.8.1: flat single-array layout for dense / ragged blobs.  Removes the
``group-with-data-child`` pattern used by every non-spatial array
(object_attributes, group_attributes, groups, parametric blobs,
composite links) and writes each logical array as a single standard
Zarr v3 array at its logical path.  Sparse coverage is expressed via
the array's ``fill_value`` (NaN for floats, sentinel for integers)
instead of a sibling ``present_mask`` child array.  Ragged arrays move
to vlen-bytes codec, matching the layout ``object_index/manifests``
already used.  Custom per-array attrs move from the parent group's
``attributes`` into the array's own ``attributes`` block.  Hard break:
0.8.0 stores are not readable; rewrite from source.

0.8.1 also adds three **optional, backward-compatible** fields to the
``cross_chunk_links/<delta>/`` family ``.zattrs`` (absent ⇒ the prior
behaviour, so existing 0.8.1 stores read unchanged): ``directed``
(bool; when true endpoint order is preserved and ``A→B`` / ``B→A`` are
distinct cells — used by streamline and skeleton parent→child edges),
``store`` (``"canonical"`` = one cell per record, or ``"duplicate"`` =
one cell per distinct incident chunk for prefix-scan incidence reads),
and ``num_physical_records`` (on-disk row count, ``> num_links`` when
duplicated).  ``num_links`` remains the *logical* record count.  A
decentralized ``write_cross_chunk_link_cells`` + ``finalize_cross_chunk_links``
pair lets independent workers append into disjoint cells race-free.
See ``docs/spec/object_model/cross_chunk_links.md``.

0.8.0: per-tuple ``cross_chunk_links`` layout.  The global flat
``cross_chunk_links/<delta>/data`` blob is replaced by per-cell
arrays keyed on the canonical-sorted L-tuple of endpoint chunks
(``cross_chunk_links/<delta>/<chunk_0.x.y.z>.<chunk_1.x.y.z>...``)
where L = ``link_width``.  Each cell stores only the records
spanning that exact chunk-tuple — readers scale with cell size,
not store size.  Endpoint order is preserved via a Lehmer-coded
``perm_idx`` int64 per record so mesh-face winding and directed-
edge direction survive the canonical sort.  ``write_cross_chunk_links``
returns a :class:`CrossChunkLinkPartition` instead of an ``int``;
legacy callers can still ``int(partition)`` for the ``first_new``
field.  Hard break: 0.7.x stores are not readable; rewrite from
source.

0.7.0: per-level ``chunk_shape``.  ``RootMetadata.chunk_shape`` remains
the level-0 default; ``LevelMetadata`` gains an optional
``chunk_shape`` field that, when set, overrides root for that level.
Required invariant: per-level ``chunk_shape`` must be a positive
integer multiple of the root ``chunk_shape`` along every axis (nested
chunk grids).  This lets coarser pyramid levels use larger chunks the
way OME-Zarr image pyramids grow physical chunk extent via voxel-size
scaling.  Cross-level link arrays keep the single-chunk-key
convention; the differing level's chunk coord is computed by integer
division using the per-level chunk_shape ratio.  Hard break: 0.6.x
stores are not readable; rewrite from source.

0.6.0: fragment-index schema.  Replaces ``vertex_group_offsets`` with
``vertex_fragments`` and splits the v0.5 inline-header link blob into
a flat ``links/0/<chunk>`` payload plus a sibling
``link_fragments/<chunk>`` group.  The fragment-index byte layout
expresses each per-group boundary as either a contiguous index range
``[start, count)`` or an explicit list of row indices, supporting
vertex re-use across fragments.  ``object_index/data`` now uses a
per-chunk manifest-block format with single / range / explicit modes;
cross-level link arrays (``cross_chunk_links``, ``delta != 0``) are
unchanged.  Hard break: 0.5.x stores are not readable; rewrite from
source.

0.5.0: NGFF-alignment cleanup + format simplification.  The 0.5
series went through several on-disk simplifications without a
version bump (consumers should pin to a specific point release):

- ``vertex_counts/`` per-chunk sidecars removed; per-chunk vertex
  counts are derived from ``vertex_group_offsets`` and the
  ``vertices/<key>`` blob size.
- ``vertex_group_offsets/<key>`` is a plain ``(K,)`` int64 array of
  vertex byte offsets (the legacy ``(K, 2)`` paired layout with a
  link-offset column is gone).
- ``attributes/<name>/<key>_offsets`` sibling blobs removed.
  Attribute groups align 1:1 with vertex groups; per-group byte
  offsets are computed at read time.
- ``metanode_children/`` removed.  Pyramid drill-down uses the
  ``links/<+1>/`` + ``cross_chunk_links/<+1>/`` arrays emitted inline
  during coarsening (mirrored as ``-1`` on the coarse side under
  ``cross_level_storage="explicit"``).
- ``cross_chunk_faces/`` removed.  Cross-chunk face identity uses
  ``cross_chunk_links/<delta>/`` with ``link_width=3``.  The
  ``cross_chunk_links`` array carries a ``link_width`` metadata
  field (default 2 for edges).
- ``object_index/pending/`` staging tree removed.  Incremental
  writes go directly into ``object_index/``; transactional backends
  (icechunk) make this cheap.

Earlier 0.5 changes (now baseline): renamed ``format_version`` to
``zv_version``, moved axes to ``multiscales[0].axes``, dropped
per-array dtype duplication.

0.4.1: bare-integer resolution-level group names (``0/``, ``1/``).
"""

# Capability tokens stored in RootMetadata.format_capabilities.  Readers
# inspect these to know which optional features the store uses.
CAP_PRESERVED_OBJECT_IDS: str = "preserved_object_ids"
"""At least one resolution level was written with ID-preserving
sparsification (``preserves_object_ids=True`` on the level metadata).
Dropped objects appear as empty manifest slots and zero
``present_mask`` bytes; ``parent_level`` carries semantic weight."""

CAP_SHARED_FRAGMENTS: str = "shared_fragments"
"""At least one resolution level stores per-chunk fragments that may
be referenced by multiple objects' manifests (the v0.6 successor to
``shared_vertex_groups``; the sharing primitive is now a fragment
rather than a contiguous-byte vertex group)."""

CAP_FRAGMENT_INDEX: str = "fragment_index"
"""The store uses the v0.6 fragment-index encoding for ``vertex_fragments``
and ``link_fragments`` (single uint8 blob per chunk; see
:mod:`zarr_vectors.encoding.fragments`)."""

CAP_MULTISCALE_LINKS: str = "multiscale_links"
"""Store uses the 0.4 multiscale links layout (``links/<delta>/``,
``cross_chunk_links/<delta>/``, ``link_attributes/<name>/<delta>/`` and
``cross_chunk_link_attributes/<name>/<delta>/``) and may contain
cross-pyramid-level edges (``delta != 0``)."""

DEFAULT_AXES_NAMES: tuple[str, ...] = ("x", "y", "z", "w")
"""Default axis names used when ``create_store`` is called without an
explicit ``axes`` kwarg.  Indexed by ``sid_ndim`` (1 → ``("x",)``, 2 →
``("x", "y")``, ...).  Stops at 4 dims; higher-dim stores must pass
axes explicitly."""

DEFAULT_BOUNDS_SIDE: float = 128.0
"""Default per-dimension extent for a freshly-warmed store.  When
``create_store(path)`` is called with no ``bounds`` kwarg, the store is
created with ``bounds = ([0,...,0], [128,...,128])`` for the resolved
``sid_ndim``.  Out-of-bounds writes raise unless an ``out_of_bounds=``
policy is supplied by the caller."""

DEFAULT_OOB_POLICY: str = "raise"
"""Default ``out_of_bounds`` policy applied by the top-level write
functions when the caller does not specify one.  Values: ``"raise"``
(reject the write), ``"ignore"`` (silently drop out-of-bounds vertices),
``"expand"`` (grow store ``bounds`` to include the new data)."""

VALID_OOB_POLICIES: frozenset[str] = frozenset({"raise", "ignore", "expand"})

# ---------------------------------------------------------------------------
# Store layout names
# ---------------------------------------------------------------------------

RESOLUTION_PREFIX: str = ""
"""Empty under the 0.4.1 layout — resolution level groups are bare
integer names (``0``, ``1``, ...) to mirror OME-Zarr's convention.
Retained as a symbol so callers that build paths via
``f"{RESOLUTION_PREFIX}{n}"`` keep working without import churn."""

PARAMETRIC_GROUP: str = "parametric"
"""Name of the root-level parametric objects group."""

# ---------------------------------------------------------------------------
# Per-level array names
# ---------------------------------------------------------------------------

VERTICES: str = "vertices"
VERTEX_FRAGMENTS: str = "vertex_fragments"
"""Per-chunk fragment-index group describing how rows of ``vertices/<chunk>``
group into fragments.  See :mod:`zarr_vectors.encoding.fragments`."""

LINK_FRAGMENTS: str = "link_fragments"
"""Per-chunk fragment-index group describing how rows of
``links/0/<chunk>`` group into fragments.  Exists at ``delta == 0``
only; cross-level link arrays keep their inline self-describing
header."""

LINKS: str = "links"
VERTEX_ATTRIBUTES: str = "vertex_attributes"
FRAGMENT_ATTRIBUTES: str = "fragment_attributes"
"""Per-fragment attribute arrays.  Dense per-chunk blob with one row per
fragment in that chunk (``(F,)`` or ``(F, C)``).  Optional — used when
callers want to materialize per-fragment data, including parent-IDs
(e.g. an ``object_id`` fragment attribute carrying the OID that owns
each fragment)."""
OBJECT_INDEX: str = "object_index"
OBJECT_ATTRIBUTES: str = "object_attributes"
GROUPS: str = "groups"
GROUP_ATTRIBUTES: str = "group_attributes"
CROSS_CHUNK_LINKS: str = "cross_chunk_links"
LINK_ATTRIBUTES: str = "link_attributes"
CROSS_CHUNK_LINK_ATTRIBUTES: str = "cross_chunk_link_attributes"

# Parametric sub-arrays
PARAMETRIC_OBJECTS: str = "objects"
PARAMETRIC_OBJECT_ATTRIBUTES: str = "object_attributes"
PARAMETRIC_GROUPS: str = "groups"
PARAMETRIC_GROUP_ATTRIBUTES: str = "group_attributes"

ALL_ARRAY_NAMES: frozenset[str] = frozenset({
    VERTICES,
    VERTEX_FRAGMENTS,
    LINK_FRAGMENTS,
    LINKS,
    VERTEX_ATTRIBUTES,
    FRAGMENT_ATTRIBUTES,
    OBJECT_INDEX,
    OBJECT_ATTRIBUTES,
    GROUPS,
    GROUP_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    LINK_ATTRIBUTES,
    CROSS_CHUNK_LINK_ATTRIBUTES,
})

# Array names whose on-disk layout includes a ``<level_delta>`` segment
# between the array prefix and the chunk-key / data subpath (multiscale
# links, 0.4+).  Use ``zarr_vectors.core.paths`` to compose paths.
MULTISCALE_LINK_ARRAY_NAMES: frozenset[str] = frozenset({
    LINKS,
    CROSS_CHUNK_LINKS,
    LINK_ATTRIBUTES,
    CROSS_CHUNK_LINK_ATTRIBUTES,
})

# ---------------------------------------------------------------------------
# Convention values
# ---------------------------------------------------------------------------

# links_convention
LINKS_EXPLICIT: str = "explicit"
LINKS_IMPLICIT_SEQUENTIAL: str = "implicit_sequential"
LINKS_IMPLICIT_BRANCHES: str = "implicit_sequential_with_branches"

VALID_LINKS_CONVENTIONS: frozenset[str] = frozenset({
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_SEQUENTIAL,
    LINKS_IMPLICIT_BRANCHES,
})

# object_index_convention
OBJIDX_STANDARD: str = "standard"
OBJIDX_IDENTITY: str = "identity"

VALID_OBJIDX_CONVENTIONS: frozenset[str] = frozenset({
    OBJIDX_STANDARD,
    OBJIDX_IDENTITY,
})

# cross_chunk_strategy
CROSS_CHUNK_DEDUP: str = "boundary_deduplication"
CROSS_CHUNK_EXPLICIT: str = "explicit_links"
CROSS_CHUNK_BOTH: str = "both"

VALID_CROSS_CHUNK_STRATEGIES: frozenset[str] = frozenset({
    CROSS_CHUNK_DEDUP,
    CROSS_CHUNK_EXPLICIT,
    CROSS_CHUNK_BOTH,
})

# cross_level_storage (0.4 multiscale links).  Controls whether
# ``cross_level_depth`` materializes both directions or only positive
# deltas.  See ``schema/zarr_vectors.linkml.yaml`` ``CrossLevelStorage``.
XLEVEL_NONE: str = "none"
XLEVEL_IMPLICIT: str = "implicit"
XLEVEL_EXPLICIT: str = "explicit"

VALID_XLEVEL_STORAGE: frozenset[str] = frozenset({
    XLEVEL_NONE,
    XLEVEL_IMPLICIT,
    XLEVEL_EXPLICIT,
})

# Defaults applied by ``build_pyramid`` when the caller does not override.
DEFAULT_CROSS_LEVEL_DEPTH: int = 1
"""Default ``cross_level_depth``: emit ``±1`` cross-level link arrays at
every adjacent level pair.  ``0`` disables, ``-1`` means all available
pyramid levels."""

DEFAULT_CROSS_LEVEL_STORAGE: str = XLEVEL_EXPLICIT
"""Default ``cross_level_storage``: materialize both ``+N`` (at the
finer level) and ``-N`` (at the coarser level)."""

# ---------------------------------------------------------------------------
# Geometry types
# ---------------------------------------------------------------------------

GEOM_POINT_CLOUD: str = "point_cloud"
GEOM_LINE: str = "line"
GEOM_POLYLINE: str = "polyline"
GEOM_STREAMLINE: str = "streamline"
GEOM_SKELETON: str = "skeleton"
GEOM_GRAPH: str = "graph"
GEOM_MESH: str = "mesh"

VALID_GEOMETRY_TYPES: frozenset[str] = frozenset({
    GEOM_POINT_CLOUD,
    GEOM_LINE,
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
    GEOM_SKELETON,
    GEOM_GRAPH,
    GEOM_MESH,
})

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

ENCODING_RAW: str = "raw"
ENCODING_DRACO: str = "draco"

VALID_ENCODINGS: frozenset[str] = frozenset({
    ENCODING_RAW,
    ENCODING_DRACO,
})

# ---------------------------------------------------------------------------
# Default compression
# ---------------------------------------------------------------------------

DEFAULT_COMPRESSOR: str = "blosc"
DEFAULT_COMPRESSOR_OPTS: dict[str, object] = {
    "cname": "zstd",
    "clevel": 5,
    "shuffle": 1,  # SHUFFLE_BYTE
}

# ---------------------------------------------------------------------------
# Multi-resolution defaults
# ---------------------------------------------------------------------------

DEFAULT_REDUCTION_FACTOR: int = 8
"""Default: only emit a new resolution level when vertex count drops by 8×."""

DEFAULT_BIN_RATIO: tuple[int, ...] = (1, 1, 1)
"""Bin ratio at level 0 (no downsampling)."""

DEFAULT_COARSENING_METHOD: str = "per_object"

# Valid values for LevelMetadata.coarsening_method.  Open-set: future
# strategies (e.g. mesh edge-collapse decimation) may add tokens here.
COARSEN_PER_OBJECT: str = "per_object"
"""Per-object pyramid: each surviving object's vertices are aggregated
into bin centroids (metavertices).  Metavertices may be shared between
objects; OIDs are preserved across levels."""

COARSEN_MANUAL: str = "manual"
COARSEN_NONE: str = "none"

VALID_COARSENING_METHODS: frozenset[str] = frozenset({
    COARSEN_PER_OBJECT,
    COARSEN_MANUAL,
    COARSEN_NONE,
})

# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------

AGG_MEAN: str = "mean"
AGG_SUM: str = "sum"
AGG_MODE: str = "mode"
AGG_COUNT: str = "count"
AGG_MIN: str = "min"
AGG_MAX: str = "max"

VALID_AGGREGATIONS: frozenset[str] = frozenset({
    AGG_MEAN, AGG_SUM, AGG_MODE, AGG_COUNT, AGG_MIN, AGG_MAX,
})
