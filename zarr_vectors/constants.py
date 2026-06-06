"""Format constants for the Zarr Vectors (ZV) format.

These are the canonical names, prefixes, and default values used
throughout the specification.  Changing a value here changes it
everywhere in the package.
"""

# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

FORMAT_VERSION: str = "0.8.0"
"""Current ZV specification version.

0.8.0: partitioned cross-chunk-link layout.  The single monolithic
``cross_chunk_links/<delta>/data`` int64 blob (and its parallel
``cross_chunk_link_attributes/<name>/<delta>/data``) is replaced by
**K-deep leaves keyed by the sorted unique chunks each record touches**:
``cross_chunk_links/<delta>/<chunk_sorted_0>/.../<chunk_sorted_{K-1}>/data``
where ``K`` is the number of distinct chunks the record involves
(``1 ≤ K ≤ link_width``).  Each record's encoding drops from
``link_width * (sid_ndim + 1) * 8`` bytes to ``9 * link_width`` bytes by
recovering each endpoint's chunk identity from the leaf path's K
segments via a per-endpoint ``uint8`` chunk-index in the record (no
chunk coords in the payload).  Canonicalization rule: ``delta=0,
link_width=2`` records MUST emit ``ci = [0, 1]`` (one orientation per
undirected edge).  New ``layout = "partitioned_v1"`` discriminator
stamped on every ``cross_chunk_links/<delta>/`` group ``.zattrs``; new
``CAP_PARTITIONED_CROSS_CHUNK_LINKS`` capability token, coupled with
``CAP_MULTISCALE_LINKS`` (any store with a ``cross_chunk_links/<delta>/``
group MUST carry both).  ``num_links`` is no longer at the group level
— per-leaf counts are derivable from leaf byte length.  Migration:
**in-place helper** ``zarr_vectors.migration.partition_legacy_cross_chunk_links``
regroups records by their sorted unique chunks and rewrites the leaves.
Hard break: 0.7.x stores require migration before a 0.8 reader can
open them.

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

CAP_PARTITIONED_CROSS_CHUNK_LINKS: str = "partitioned_cross_chunk_links"
"""Store uses the v0.8 partitioned cross-chunk-link layout:
``cross_chunk_links/<delta>/<chunk_sorted_0>/.../<chunk_sorted_{K-1}>/data``
leaves keyed by the sorted unique set of chunks each record touches;
each record is ``L * uint8`` chunk-indices followed by ``L * int64``
vertex indices (``9 * link_width`` bytes per record).  Coupled with
``multiscale_links``: any store with a ``cross_chunk_links/<delta>/``
group in the partitioned layout MUST carry both tokens.  Stores tagged
only with ``multiscale_links`` (without ``partitioned_cross_chunk_links``)
are v0.7-era monolithic-blob stores and require migration before a v0.8
reader can open them; see
:func:`zarr_vectors.migration.partition_legacy_cross_chunk_links`."""

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
