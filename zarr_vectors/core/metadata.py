"""ZV metadata: dataclasses, serialisation, and validation.

This module is pure Python — it does not import zarr.  Metadata objects
are serialised to/from plain dicts (JSON-compatible) and written to
stores by ``core.store``.

Three tiers of metadata:
- **RootMetadata**: format version, spatial index, CRS, conventions
- **LevelMetadata**: per-resolution-level configuration
- **ArrayMetadata**: per-array dtype, encoding, compression

Plus helpers for OME-Zarr–compatible axes and coordinate transforms,
convention validation, and parametric type registries.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np


class NgffAxis(TypedDict, total=False):
    """An NGFF (OME-Zarr RFC 4/5) axis descriptor.

    Three string fields:

    - ``name`` (required): axis identifier — e.g. ``"x"``, ``"z"``,
      ``"channel"``.  Must be unique within the axis list.
    - ``type``: one of ``"space"``, ``"time"``, ``"channel"``, or a
      custom type.  NGFF prescribes the order
      ``time → channel → custom → space``.
    - ``unit``: UDUNITS-2 name (``"micrometer"``, ``"second"``, ...) or
      omitted.  Do **not** stamp a placeholder string here; NGFF
      validators will reject it.
    """

    name: str
    type: str
    unit: str

from zarr_vectors.constants import (
    DEFAULT_COARSENING_METHOD,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    DEFAULT_REDUCTION_FACTOR,
    FORMAT_VERSION,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_STANDARD,
    CROSS_CHUNK_EXPLICIT,
    VALID_CROSS_CHUNK_STRATEGIES,
    VALID_GEOMETRY_TYPES,
    VALID_LINKS_CONVENTIONS,
    VALID_OBJIDX_CONVENTIONS,
    VALID_XLEVEL_STORAGE,
    VALID_ENCODINGS,
    ENCODING_RAW,
)
from zarr_vectors.exceptions import ConventionError, MetadataError


# ===================================================================
# Axes / CRS helpers (OME-Zarr RFC 4/5)
# ===================================================================

def build_axes_metadata(
    dim_names: list[str],
    dim_types: list[str],
    dim_units: list[str | None],
) -> list[dict[str, str]]:
    """Build OME-Zarr–style axes list.

    Args:
        dim_names: Axis names, e.g. ``["x", "y", "z"]``.
        dim_types: Axis types, e.g. ``["space", "space", "space"]``.
        dim_units: Axis units, e.g. ``["micrometer", "micrometer", "micrometer"]``.
            Each entry may be ``None`` or an empty string to omit the
            ``unit`` field — NGFF requires ``unit`` to be a valid
            UDUNITS-2 name when present, so unknown units MUST be
            omitted rather than stamped with a placeholder.

    Returns:
        List of axis dicts: ``[{"name":"x","type":"space","unit":"micrometer"}, ...]``

    Raises:
        MetadataError: If list lengths are inconsistent or no space axes.
    """
    if not (len(dim_names) == len(dim_types) == len(dim_units)):
        raise MetadataError(
            f"Axis list lengths must match: names={len(dim_names)}, "
            f"types={len(dim_types)}, units={len(dim_units)}"
        )
    if not any(t == "space" for t in dim_types):
        raise MetadataError("At least one axis must have type 'space'")
    out: list[dict[str, str]] = []
    for n, t, u in zip(dim_names, dim_types, dim_units):
        ax: dict[str, str] = {"name": n, "type": t}
        if u:
            ax["unit"] = u
        out.append(ax)
    return out


_NGFF_AXIS_TYPE_ORDER: dict[str, int] = {
    "time": 0,
    "channel": 1,
    # Any non-time/channel/space type is treated as "custom" and slots
    # between channel (1) and space (3).
    "space": 3,
}


def validate_axes(axes: list[dict[str, str]]) -> None:
    """Validate axes metadata structure against NGFF conventions.

    NGFF (RFC 4/5) prescribes axis ordering ``time → channel → custom
    → space (z, y, x)``.  Stores written with axes in any other order
    will not load correctly in NGFF-aware viewers.

    Raises:
        MetadataError: If axes are malformed or not in NGFF order.
    """
    if not axes or len(axes) < 2:
        raise MetadataError("At least 2 axes required")
    space_count = sum(1 for a in axes if a.get("type") == "space")
    if space_count < 2:
        raise MetadataError(f"Need ≥2 space axes, found {space_count}")
    seen_names: set[str] = set()
    last_rank = -1
    for i, a in enumerate(axes):
        if "name" not in a:
            raise MetadataError(f"Axis {i} missing 'name'")
        name = a["name"]
        if name in seen_names:
            raise MetadataError(f"Duplicate axis name {name!r}")
        seen_names.add(name)
        # NGFF rank: time(0) < channel(1) < custom(2) < space(3).
        a_type = a.get("type", "space")
        rank = _NGFF_AXIS_TYPE_ORDER.get(a_type, 2)
        if rank < last_rank:
            raise MetadataError(
                f"Axis {i} {name!r} has type {a_type!r} but follows a "
                f"higher-rank axis; NGFF requires order time → channel "
                f"→ custom → space."
            )
        last_rank = rank


def build_coordinate_transforms(
    scale: list[float],
    translation: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build coordinateTransformations list for a resolution level.

    Args:
        scale: Scale factors per axis (voxel-to-physical).
        translation: Optional offset per axis.

    Returns:
        List of transform dicts.
    """
    transforms: list[dict[str, Any]] = [{"type": "scale", "scale": list(scale)}]
    if translation is not None:
        transforms.append({"type": "translation", "translation": list(translation)})
    return transforms


# ===================================================================
# Convention validation
# ===================================================================

def validate_conventions(
    links_convention: str,
    object_index_convention: str,
    cross_chunk_strategy: str,
    num_spatial_chunks: int | None = None,
    has_links_array: bool | None = None,
    has_object_index: bool | None = None,
    geometry_type: str | None = None,
) -> None:
    """Validate that convention flags are consistent with store contents.

    Args:
        links_convention: Value of links_convention metadata field.
        object_index_convention: Value of object_index_convention field.
        cross_chunk_strategy: Value of cross_chunk_strategy field.
        num_spatial_chunks: Number of spatial chunks (if known).
        has_links_array: Whether the links array is present (if known).
        has_object_index: Whether the object_index array is present (if known).
        geometry_type: Primary geometry type (if known).

    Raises:
        ConventionError: If any convention is invalid or inconsistent.
    """
    if links_convention not in VALID_LINKS_CONVENTIONS:
        raise ConventionError(
            f"Invalid links_convention '{links_convention}', "
            f"must be one of {VALID_LINKS_CONVENTIONS}"
        )
    if object_index_convention not in VALID_OBJIDX_CONVENTIONS:
        raise ConventionError(
            f"Invalid object_index_convention '{object_index_convention}', "
            f"must be one of {VALID_OBJIDX_CONVENTIONS}"
        )
    if cross_chunk_strategy not in VALID_CROSS_CHUNK_STRATEGIES:
        raise ConventionError(
            f"Invalid cross_chunk_strategy '{cross_chunk_strategy}', "
            f"must be one of {VALID_CROSS_CHUNK_STRATEGIES}"
        )
    # Identity convention only valid for single-chunk stores
    if object_index_convention == "identity":
        if num_spatial_chunks is not None and num_spatial_chunks > 1:
            raise ConventionError(
                "object_index_convention='identity' is only valid for "
                f"single-chunk stores, but found {num_spatial_chunks} chunks"
            )


def requires_links_array(convention: str) -> bool:
    """Return whether the links array is required for this convention."""
    return convention == "explicit"


def requires_object_index(convention: str, num_chunks: int) -> bool:
    """Return whether the object_index array is required."""
    if convention == "identity":
        return False
    return num_chunks > 1 or convention == "standard"


# ===================================================================
# RootMetadata
# ===================================================================

@dataclass
class RootMetadata:
    """Root-level metadata for a ZV store.

    Attributes:
        zv_version: ZV specification version (renamed from
            ``format_version`` in 0.5.0 to disambiguate from Zarr v3's
            ``zarr_format`` field).
        spatial_index_dims: Axes definitions (NGFF / OME-Zarr style).
            On disk, axes live in ``multiscales[0].axes`` (NGFF) and
            are NOT duplicated under the ``zarr_vectors`` namespace as
            of 0.5.0.  This Python attribute remains the canonical
            in-memory accessor.
        chunk_shape: Spatial chunk size per dimension.
        bounds: Global bounding box as ``(min_corner, max_corner)``.
        geometry_types: List of geometry types in the store.
        crs: Coordinate reference system dict (or None).
        links_convention: How intra-chunk links are stored.
        object_index_convention: How object→vertex-group mapping works.
        cross_chunk_strategy: How cross-chunk connectivity is handled.
        reduction_factor: Multi-resolution threshold factor.
    """

    spatial_index_dims: list[dict[str, str]]
    chunk_shape: tuple[float, ...]
    bounds: tuple[list[float], list[float]]
    geometry_types: list[str]
    zv_version: str = FORMAT_VERSION
    crs: dict[str, Any] | None = None
    links_convention: str = LINKS_IMPLICIT_SEQUENTIAL
    object_index_convention: str = OBJIDX_STANDARD
    cross_chunk_strategy: str = CROSS_CHUNK_EXPLICIT
    reduction_factor: int = DEFAULT_REDUCTION_FACTOR
    base_bin_shape: tuple[float, ...] | None = None
    """Supervoxel bin edge lengths at level 0. When None, defaults to
    chunk_shape (one bin per chunk — backward compatible)."""
    cross_level_depth: int = DEFAULT_CROSS_LEVEL_DEPTH
    """0.4 multiscale links: max absolute level delta materialized by
    ``build_pyramid``.  ``0`` = none, ``N`` = up to ``±N`` (or ``+N``
    only when ``cross_level_storage='implicit'``), ``-1`` = all."""
    cross_level_storage: str = DEFAULT_CROSS_LEVEL_STORAGE
    """0.4 multiscale links: ``"none"`` / ``"implicit"`` / ``"explicit"``."""
    format_capabilities: list[str] = field(default_factory=list)
    """Optional capability tokens this store uses.  See
    :mod:`zarr_vectors.constants` for the canonical token names
    (``CAP_*``).  Empty list by default."""

    def validate(self) -> None:
        """Validate this metadata object.

        Raises:
            MetadataError: If any field is invalid.
            ConventionError: If conventions are inconsistent.
        """
        if not self.zv_version:
            raise MetadataError("zv_version is required")

        # Hard cuts:
        #   - pre-0.4   : multiscale-links layout (no ``<delta>`` segment)
        #   - pre-0.4.1 : resolution-level group names were prefixed
        #                 (``resolution_0/``); current writers use bare
        #                 integers (``0/``) to mirror OME-Zarr.
        #   - pre-0.5.0 : root-attr key was ``format_version`` and axes
        #                 were duplicated under
        #                 ``zarr_vectors.spatial_index_dims``; per-array
        #                 ``.zattrs`` carried a redundant ``dtype`` field.
        #   - pre-0.6.0 : ``vertex_fragments`` instead of
        #                 ``vertex_fragments``; links carried an inline
        #                 self-describing header; ``object_index`` used
        #                 the flat quad encoding.
        #   - pre-0.7.0 : ``chunk_shape`` was root-only.
        #   - pre-0.8.0 : ``cross_chunk_links`` was a single global flat
        #                 blob.  0.8 split it into per-tuple cells keyed
        #                 on canonical-sorted endpoint chunks and
        #                 introduced the ``perm_idx`` record header.
        #                 Old layout cannot be read by this build.
        #   - pre-0.8.1 : every non-spatial array was stored as a Zarr
        #                 group containing single-chunk ``uint8`` child
        #                 arrays (``data``, ``offsets``, ``present_mask``).
        #                 0.8.1 collapses each to a single standard Zarr
        #                 v3 array at the logical path, expresses sparse
        #                 coverage via ``fill_value`` instead of a
        #                 ``present_mask`` sibling, and moves ragged
        #                 arrays to vlen-bytes.  Old layout cannot be
        #                 read by this build.
        # No shims ship; older stores must be rewritten from source.
        parts = self.zv_version.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) >= 2 else 0
            patch = int(parts[2]) if len(parts) >= 3 else 0
        except (ValueError, AttributeError, IndexError) as exc:
            raise MetadataError(
                f"zv_version {self.zv_version!r} is not a valid X.Y[.Z] string"
            ) from exc
        if (major, minor, patch) < (0, 9, 0):
            raise MetadataError(
                f"store zv_version is {self.zv_version}; this build "
                f"requires {FORMAT_VERSION} — no backwards-compat shim. "
                f"Pre-0.9.0 stores wrote each per-spatial-chunk array "
                f"(vertices, fragments, links, attribute arrays) as a Zarr "
                f"group of single-chunk ``uint8`` sub-arrays, one per chunk "
                f"key; 0.9.0 collapses each to a single vlen-bytes Zarr "
                f"array whose cells are spatial chunks — the old layout "
                f"cannot be read.  Rewrite from source."
            )

        if self.cross_level_storage not in VALID_XLEVEL_STORAGE:
            raise MetadataError(
                f"cross_level_storage={self.cross_level_storage!r} not in "
                f"{sorted(VALID_XLEVEL_STORAGE)}"
            )
        if self.cross_level_depth < -1:
            raise MetadataError(
                f"cross_level_depth must be ≥ -1 (got {self.cross_level_depth})"
            )

        validate_axes(self.spatial_index_dims)

        sid_ndim = len(self.spatial_index_dims)
        if len(self.chunk_shape) != sid_ndim:
            raise MetadataError(
                f"chunk_shape length {len(self.chunk_shape)} != "
                f"spatial_index_dims length {sid_ndim}"
            )
        if any(c <= 0 for c in self.chunk_shape):
            raise MetadataError("All chunk_shape values must be > 0")

        if len(self.bounds) != 2:
            raise MetadataError("bounds must be (min_corner, max_corner)")
        if len(self.bounds[0]) != sid_ndim or len(self.bounds[1]) != sid_ndim:
            raise MetadataError(
                f"bounds dimensions must match sid_ndim={sid_ndim}"
            )

        for gt in self.geometry_types:
            if gt not in VALID_GEOMETRY_TYPES:
                raise MetadataError(
                    f"Unknown geometry_type '{gt}', "
                    f"must be one of {VALID_GEOMETRY_TYPES}"
                )

        validate_conventions(
            self.links_convention,
            self.object_index_convention,
            self.cross_chunk_strategy,
        )

        if self.reduction_factor < 2:
            raise MetadataError(
                f"reduction_factor must be ≥2, got {self.reduction_factor}"
            )

        # Validate base_bin_shape if set
        if self.base_bin_shape is not None:
            if len(self.base_bin_shape) != sid_ndim:
                raise MetadataError(
                    f"base_bin_shape length {len(self.base_bin_shape)} != "
                    f"spatial_index_dims length {sid_ndim}"
                )
            if any(b <= 0 for b in self.base_bin_shape):
                raise MetadataError("All base_bin_shape values must be > 0")
            # chunk_shape must be an integer multiple of base_bin_shape
            for i, (cs, bs) in enumerate(
                zip(self.chunk_shape, self.base_bin_shape)
            ):
                ratio = cs / bs
                if abs(ratio - round(ratio)) > 1e-9:
                    raise MetadataError(
                        f"chunk_shape[{i}]={cs} is not an integer multiple "
                        f"of base_bin_shape[{i}]={bs} (ratio={ratio:.6f})"
                    )

    @property
    def effective_bin_shape(self) -> tuple[float, ...]:
        """Base bin shape, defaulting to chunk_shape if not set."""
        return self.base_bin_shape if self.base_bin_shape is not None else self.chunk_shape

    @property
    def bins_per_chunk(self) -> tuple[int, ...]:
        """Number of supervoxel bins per chunk in each dimension."""
        bbs = self.effective_bin_shape
        return tuple(
            int(round(cs / bs))
            for cs, bs in zip(self.chunk_shape, bbs)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns the **non-axis** fields only — under 0.5.0 axes live in
        the NGFF ``multiscales[0].axes`` block, written separately by
        :func:`zarr_vectors.core.multiscale.write_multiscale_metadata`.
        Combine the two when writing root attrs.
        """
        d = {
            "zarr_vectors": {
                "zv_version": self.zv_version,
                "chunk_shape": list(self.chunk_shape),
                "bounds": [list(self.bounds[0]), list(self.bounds[1])],
                "geometry_types": self.geometry_types,
                "crs": self.crs,
                "links_convention": self.links_convention,
                "object_index_convention": self.object_index_convention,
                "cross_chunk_strategy": self.cross_chunk_strategy,
                "cross_level_depth": self.cross_level_depth,
                "cross_level_storage": self.cross_level_storage,
                "reduction_factor": self.reduction_factor,
            }
        }
        if self.base_bin_shape is not None:
            d["zarr_vectors"]["base_bin_shape"] = list(self.base_bin_shape)
        if self.format_capabilities:
            d["zarr_vectors"]["format_capabilities"] = list(self.format_capabilities)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, strict: bool = True) -> RootMetadata:
        """Deserialise from a dict (as stored in ``.zattrs``).

        Args:
            d: The full root-attrs dict — must contain a
                ``"zarr_vectors"`` key for ZV-specific fields and
                (when ``strict``) a top-level ``"multiscales"`` key
                holding the NGFF axes.
            strict: When True (default), every structural field
                (axes via ``multiscales[0].axes``, ``chunk_shape``,
                ``bounds``, ``geometry_types``) must be present.  When
                False, missing structural fields fall through as
                ``None``/``[]`` so a freshly-warmed store can
                round-trip; the resulting instance will fail
                :meth:`validate` until those fields are filled in.

        Raises:
            MetadataError: If the dict is malformed or (in strict mode)
                missing required keys.
        """
        if "zarr_vectors" not in d:
            raise MetadataError("Root metadata must contain 'zarr_vectors' key")
        zv = d["zarr_vectors"]

        # Axes now live in NGFF ``multiscales[0].axes`` (0.5.0+).
        ms = d.get("multiscales") or []
        axes: list[dict[str, str]] = []
        if ms and isinstance(ms, list) and ms[0].get("axes"):
            axes = list(ms[0]["axes"])

        if strict:
            required = ["zv_version", "chunk_shape", "bounds", "geometry_types"]
            for key in required:
                if key not in zv:
                    raise MetadataError(f"Missing required root metadata key: '{key}'")
            if not axes:
                raise MetadataError(
                    "Missing required root metadata: NGFF "
                    "``multiscales[0].axes`` block (axes are no longer "
                    "stored under ``zarr_vectors.spatial_index_dims`` "
                    "as of format 0.5.0)"
                )
        elif "zv_version" not in zv:
            raise MetadataError("Missing required root metadata key: 'zv_version'")

        bbs = zv.get("base_bin_shape")
        caps = zv.get("format_capabilities") or []
        chunk_shape_raw = zv.get("chunk_shape")
        bounds_raw = zv.get("bounds")
        return cls(
            zv_version=zv["zv_version"],
            spatial_index_dims=axes,
            chunk_shape=tuple(chunk_shape_raw) if chunk_shape_raw else (),
            bounds=(
                (list(bounds_raw[0]), list(bounds_raw[1]))
                if bounds_raw
                else ([], [])
            ),
            geometry_types=zv.get("geometry_types") or [],
            crs=zv.get("crs"),
            links_convention=zv.get("links_convention", LINKS_IMPLICIT_SEQUENTIAL),
            object_index_convention=zv.get("object_index_convention", OBJIDX_STANDARD),
            cross_chunk_strategy=zv.get("cross_chunk_strategy", CROSS_CHUNK_EXPLICIT),
            cross_level_depth=zv.get("cross_level_depth", DEFAULT_CROSS_LEVEL_DEPTH),
            cross_level_storage=zv.get("cross_level_storage", DEFAULT_CROSS_LEVEL_STORAGE),
            reduction_factor=zv.get("reduction_factor", DEFAULT_REDUCTION_FACTOR),
            base_bin_shape=tuple(bbs) if bbs else None,
            format_capabilities=list(caps),
        )

    def is_complete(self) -> bool:
        """True when all structural fields are populated and the meta
        passes :meth:`validate`.

        A freshly-warmed store (created via ``create_store(path)`` with
        no inference yet) has ``is_complete() is False`` until the first
        write fills in dims/chunk_shape/bounds.
        """
        if not self.spatial_index_dims or not self.chunk_shape:
            return False
        if not self.bounds or not self.bounds[0] or not self.bounds[1]:
            return False
        try:
            self.validate()
        except (MetadataError, ConventionError):
            return False
        return True

    @property
    def sid_ndim(self) -> int:
        """Number of spatial index dimensions."""
        return len(self.spatial_index_dims)


# ===================================================================
# LevelMetadata
# ===================================================================

@dataclass
class LevelMetadata:
    """Per-resolution-level metadata.

    Attributes:
        level: Integer level index (0 = full resolution).
        vertex_count: Total vertices at this level.
        arrays_present: FAMILY names present at this level -- ``"vertices"``,
            ``"vertex_attributes"``, ``"links"``, ``"object_index"`` -- never
            ``"<family>/<name>"``.  Readers gate on the family, so a per-name
            entry advertises nothing.  Hand-maintaining this list is how it
            drifts (``create_store`` long declared ``[vertices]`` while also
            creating ``vertex_fragments``); prefer
            :func:`zarr_vectors.building.refresh_arrays_present`, which
            derives it from what is actually on disk.
        bin_shape: Supervoxel edge lengths at this level (None for level 0,
            which inherits base_bin_shape from root).
        bin_ratio: Fold-change per axis relative to level 0.  Float,
            not int: a coarsen factor may be fractional, and this is
            what becomes the NGFF ``scale``.  ``compute_bin_ratio``
            keeps its integer contract for callers that want a whole
            fold.
            ``(1,1,1)`` for level 0, ``(2,2,2)`` for 2× coarser bins, etc.
        object_sparsity: Fraction of objects retained at this level (0,1].
        coarsening_method: How this level was generated.
        parent_level: Index of the source level (None for level 0).
        chunk_dims: Names of chunk-key axes, leading axis first.  When
            ``None`` (the default), chunk keys are spatial-only and have
            ``sid_ndim`` axes (e.g. ``["dim0", "dim1", "dim2"]``).  When
            set, the first entries name non-spatial leading axes (e.g.
            ``["gene", "dim0", "dim1", "dim2"]``).
        chunk_attribute_name: Name of the per-vertex attribute that is
            used as the leading chunk axis (single-axis attribute
            chunking; v1 supports only one).  ``None`` for spatial-only.
        chunk_attribute_values: Ordered list mapping attribute-bin index
            to the original attribute value.  ``chunk_attribute_values[i]``
            is the value of the attribute for any vertex in chunks with
            leading coord ``i``.  ``None`` for spatial-only.
    """

    level: int
    vertex_count: int
    arrays_present: list[str]
    bin_shape: tuple[float, ...] | None = None
    bin_ratio: tuple[float, ...] | None = None
    chunk_shape: tuple[float, ...] | None = None
    """Per-level physical chunk size override.  ``None`` means the level
    inherits :attr:`RootMetadata.chunk_shape`.  When set, each axis must
    be a positive integer multiple of the root chunk_shape (nested chunk
    grids).  Lets coarser pyramid levels carry larger chunks the way
    OME-Zarr image pyramids grow physical chunk extent via voxel-size
    scaling."""
    object_sparsity: float = 1.0
    coarsening_method: str = "none"
    parent_level: int | None = None
    chunk_dims: list[str] | None = None
    chunk_attribute_name: str | None = None
    chunk_attribute_values: list[Any] | None = None
    preserves_object_ids: bool = False
    """True for levels written by the per-object pyramid regime.

    Causes ``num_objects`` and ``object_attributes`` row count to
    inherit from the parent OID space; dropped objects leave empty
    manifest slots and ``present_mask`` byte = 0.  ``parent_level`` is
    load-bearing under this flag."""
    inherited_num_objects: int | None = None
    """OID-space size inherited from the parent level
    (= ``parent_level.num_objects``).  Lets readers allocate lookup
    arrays without traversing parent metadata."""
    shared_fragments: bool = False
    """True when per-chunk fragments represent metavertices that
    may be referenced by multiple objects' manifests (the shared-
    metavertex case)."""
    fragments_tile: bool = False
    """True when EVERY chunk in this level has a vertex fragment index
    that tiles its buffer — ``ChunkFragmentIndex.tiles(n_rows)`` holds for
    all of them.

    Under that condition the concatenation of a chunk's fragments *is*
    its vertex buffer, so a bulk read — no bbox, no object or group
    filter, no attribute filter — can return the buffer without reading
    ``vertex_fragments`` at all.  Measured: 19.0 ms -> 14.3 ms on a
    10^6-point / 125-chunk level, byte-identical result.  The saving is
    the per-chunk object reads and decompressions the index costs, not
    the index decode itself, which is why no reader-side cleverness can
    reach it.

    Unlike every other field here, this is a claim about what has *not*
    happened since — that nothing has retired a row.  It is therefore set
    only by a bulk writer that lays each chunk out bin by bin and writes
    nothing else, stamped after those writes; and it is cleared by
    :meth:`Group.write_bytes` on any later write to ``vertices`` or
    ``vertex_fragments``, the chokepoint every such write passes through.
    Readers additionally re-verify it on the first chunk they read, and
    ``validate_consistency`` checks it against the store.

    A stale True is a wrong answer, not a crash: rows no fragment
    references come back, and per-vertex attribute columns desync from
    the positions they describe.  Hence the three independent guards.
    Absent (the default) means "unknown", which is what every store
    written before this field says, and costs only the read it would
    otherwise have saved."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "level": self.level,
            "bin_shape": list(self.bin_shape) if self.bin_shape else None,
            "bin_ratio": list(self.bin_ratio) if self.bin_ratio else None,
            "object_sparsity": self.object_sparsity,
            "vertex_count": self.vertex_count,
            "coarsening_method": self.coarsening_method,
            "parent_level": self.parent_level,
            "arrays_present": self.arrays_present,
        }
        if self.chunk_shape is not None:
            d["chunk_shape"] = list(self.chunk_shape)
        if self.chunk_dims is not None:
            d["chunk_dims"] = list(self.chunk_dims)
        if self.chunk_attribute_name is not None:
            d["chunk_attribute_name"] = self.chunk_attribute_name
        if self.chunk_attribute_values is not None:
            # Cast numpy scalars to native Python so JSON is happy.
            d["chunk_attribute_values"] = [
                _to_python_scalar(v) for v in self.chunk_attribute_values
            ]
        if self.preserves_object_ids:
            d["preserves_object_ids"] = True
        if self.inherited_num_objects is not None:
            d["inherited_num_objects"] = int(self.inherited_num_objects)
        if self.shared_fragments:
            d["shared_fragments"] = True
        if self.fragments_tile:
            d["fragments_tile"] = True
        return {"zarr_vectors_level": d}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LevelMetadata:
        """Deserialise from a dict.

        Supports both new ``bin_shape``/``bin_ratio`` fields and the
        legacy ``bin_size`` field for backward compatibility.

        Raises:
            MetadataError: If the dict is malformed.
        """
        if "zarr_vectors_level" not in d:
            raise MetadataError(
                "Level metadata must contain 'zarr_vectors_level' key"
            )
        lv = d["zarr_vectors_level"]
        required = ["level", "vertex_count", "arrays_present"]
        for key in required:
            if key not in lv:
                raise MetadataError(
                    f"Missing required level metadata key: '{key}'"
                )
        # Read bin_shape — fall back to legacy bin_size
        bs = lv.get("bin_shape") or lv.get("bin_size")
        br = lv.get("bin_ratio")
        cs = lv.get("chunk_shape")
        cd = lv.get("chunk_dims")
        cav = lv.get("chunk_attribute_values")
        return cls(
            level=lv["level"],
            vertex_count=lv["vertex_count"],
            arrays_present=lv["arrays_present"],
            bin_shape=tuple(bs) if bs else None,
            bin_ratio=tuple(float(x) for x in br) if br else None,
            chunk_shape=tuple(float(x) for x in cs) if cs else None,
            object_sparsity=lv.get("object_sparsity", 1.0),
            coarsening_method=lv.get("coarsening_method", "none"),
            parent_level=lv.get("parent_level"),
            chunk_dims=list(cd) if cd else None,
            chunk_attribute_name=lv.get("chunk_attribute_name"),
            chunk_attribute_values=list(cav) if cav is not None else None,
            preserves_object_ids=bool(lv.get("preserves_object_ids", False)),
            inherited_num_objects=lv.get("inherited_num_objects"),
            shared_fragments=bool(lv.get("shared_fragments", False)),
            fragments_tile=bool(lv.get("fragments_tile", False)),
        )

    def validate(self) -> None:
        """Validate this metadata object.

        Raises:
            MetadataError: If fields are invalid.
        """
        if self.level < 0:
            raise MetadataError(f"Level must be ≥0, got {self.level}")
        if self.vertex_count < 0:
            raise MetadataError(
                f"vertex_count must be ≥0, got {self.vertex_count}"
            )
        if not (0.0 < self.object_sparsity <= 1.0):
            raise MetadataError(
                f"object_sparsity must be in (0, 1], got {self.object_sparsity}"
            )
        if self.level == 0:
            # Level 0 inherits bin_shape from root — must not set its own
            if self.bin_shape is not None:
                raise MetadataError(
                    "Level 0 should not have bin_shape set (inherits from root)"
                )
            if self.parent_level is not None:
                raise MetadataError("Level 0 should have parent_level=None")
        else:
            if self.bin_shape is None:
                raise MetadataError(
                    f"Level {self.level} must have a bin_shape"
                )
            if self.parent_level is None:
                raise MetadataError(
                    f"Level {self.level} must have a parent_level"
                )
            if self.bin_ratio is not None:
                if any(r < 1 for r in self.bin_ratio):
                    raise MetadataError(
                        f"bin_ratio values must be ≥1, got {self.bin_ratio}"
                    )

        # Per-level chunk_shape: self-consistency only.  Cross-level
        # validation (positive integer multiple of root chunk_shape,
        # divisibility by per-level bin_shape) lives in
        # :func:`validate_level_chunk_shape_against_root` since it needs
        # both metadata objects.
        if self.chunk_shape is not None:
            if any(c <= 0 for c in self.chunk_shape):
                raise MetadataError(
                    f"Level {self.level} chunk_shape values must be > 0, "
                    f"got {self.chunk_shape}"
                )

        # Attribute-chunking fields must be coherent.
        attr_fields = (
            self.chunk_attribute_name,
            self.chunk_attribute_values,
        )
        if any(f is not None for f in attr_fields) and not all(
            f is not None for f in attr_fields
        ):
            raise MetadataError(
                "chunk_attribute_name and chunk_attribute_values must be "
                "either both set or both None"
            )
        if (
            self.chunk_attribute_values is not None
            and len(self.chunk_attribute_values) == 0
        ):
            raise MetadataError("chunk_attribute_values must not be empty")

        # ID-preserving levels must point at a parent and declare the
        # inherited OID-space size.  Level 0 cannot preserve IDs (it
        # *defines* the OID space).
        if self.preserves_object_ids:
            if self.level == 0:
                raise MetadataError(
                    "Level 0 cannot have preserves_object_ids=True "
                    "(it defines the OID space)"
                )
            if self.parent_level is None:
                raise MetadataError(
                    "preserves_object_ids=True requires parent_level"
                )
            if self.inherited_num_objects is None:
                raise MetadataError(
                    "preserves_object_ids=True requires inherited_num_objects"
                )
            if self.inherited_num_objects < 0:
                raise MetadataError(
                    f"inherited_num_objects must be >= 0, got "
                    f"{self.inherited_num_objects}"
                )


    @classmethod
    def from_parent(
        cls,
        root_meta: "RootMetadata",
        parent_meta: "LevelMetadata | None",
        *,
        level: int,
        vertex_count: int,
        arrays_present: list[str],
        bin_scale_from_parent: float | Sequence[float] = 1.0,
        chunk_scale_from_parent: int | Sequence[int] = 1,
        object_sparsity: float = 1.0,
        coarsening_method: str | None = None,
        preserves_object_ids: bool = False,
        inherited_num_objects: int | None = None,
        shared_fragments: bool = False,
        parent_level: int | None = None,
    ) -> "LevelMetadata":
        """Build a coarser level's metadata from the level below it.

        This block is hand-written at twenty-odd sites across core and its
        consumers, and the copies have already diverged: one computed
        ``bin_shape`` against the ROOT instead of the parent, so every level
        of that pyramid got the same bin.  The arithmetic is small and the
        frames are easy to mix up (see the module comment above), which is
        exactly the shape of thing that belongs in one place.

        Both scale arguments are **parent-relative**, matching
        ``coarsen_level``'s keywords; ``bin_ratio`` is derived
        level-0-relative because that is what becomes the NGFF ``scale``,
        and ``chunk_shape`` is omitted when it equals the root's, matching
        the on-disk convention.

        A factor of ``1.0`` is a true no-op, so a coarsener that does not
        bin at all -- a skeleton pruner keeps the root bin at every level --
        is expressed by leaving the defaults alone rather than being
        assumed away.

        Args:
            parent_meta: The level below.  ``None`` when the parent is
                level 0, which carries no bin_shape of its own.
            bin_scale_from_parent: Bin factor against the PARENT's bin.
            chunk_scale_from_parent: Chunk factor against the PARENT's
                chunk grid.  Positive integers only (nested grids).

        Raises:
            MetadataError: If a factor is non-positive, or the resulting
                level fails its own cross-level validation.
        """
        parent_bin = get_level_bin_shape(root_meta, parent_meta)
        parent_chunk = get_level_chunk_shape(root_meta, parent_meta)
        ndim = len(parent_bin)

        def _per_axis(value: Any, what: str) -> tuple[float, ...]:
            seq = (
                tuple(float(v) for v in value)
                if isinstance(value, (tuple, list))
                else (float(value),) * ndim
            )
            if len(seq) != ndim:
                raise MetadataError(
                    f"{what} has {len(seq)} axes, level has {ndim}"
                )
            if any(v <= 0 for v in seq):
                raise MetadataError(f"{what} must be positive, got {seq}")
            return seq

        bin_f = _per_axis(bin_scale_from_parent, "bin_scale_from_parent")
        chunk_f = _per_axis(chunk_scale_from_parent, "chunk_scale_from_parent")
        for axis, f in enumerate(chunk_f):
            if abs(f - round(f)) > 1e-9:
                raise MetadataError(
                    f"chunk_scale_from_parent axis {axis} = {f} is not an "
                    f"integer; chunk grids nest"
                )

        bin_shape = tuple(b * f for b, f in zip(parent_bin, bin_f))
        chunk_shape = tuple(c * f for c, f in zip(parent_chunk, chunk_f))

        root_bin = tuple(float(b) for b in root_meta.effective_bin_shape)
        bin_ratio = tuple(
            (float(t) / float(r)) if r else 1.0
            for t, r in zip(bin_shape, root_bin)
        )

        # The on-disk convention: a level that matches root omits the field
        # rather than restating it.
        root_chunk = tuple(float(c) for c in root_meta.chunk_shape)
        chunk_override: tuple[float, ...] | None = (
            None
            if all(abs(a - b) < 1e-9 for a, b in zip(chunk_shape, root_chunk))
            else chunk_shape
        )

        meta = cls(
            level=level,
            vertex_count=int(vertex_count),
            arrays_present=list(arrays_present),
            bin_shape=bin_shape,
            bin_ratio=bin_ratio,
            chunk_shape=chunk_override,
            object_sparsity=object_sparsity,
            coarsening_method=coarsening_method,
            parent_level=(
                parent_level if parent_level is not None
                else (parent_meta.level if parent_meta is not None else 0)
            ),
            preserves_object_ids=preserves_object_ids,
            inherited_num_objects=inherited_num_objects,
            shared_fragments=shared_fragments,
        )
        meta.validate()
        validate_level_chunk_shape_against_root(root_meta, meta)
        return meta


def _to_python_scalar(v: Any) -> Any:
    """Convert numpy scalar / array-of-one to a JSON-friendly Python type."""
    if isinstance(v, np.generic):
        return v.item()
    return v


# ===================================================================
# Bin shape / ratio helpers
# ===================================================================

def compute_bin_shape(
    base_bin_shape: tuple[float, ...],
    bin_ratio: tuple[int, ...],
) -> tuple[float, ...]:
    """Compute the bin shape at a coarser level.

    Args:
        base_bin_shape: Supervoxel edge lengths at level 0.
        bin_ratio: Integer fold-change per dimension.

    Returns:
        ``tuple(base * ratio for each dimension)``.

    Raises:
        MetadataError: If lengths don't match.
    """
    if len(base_bin_shape) != len(bin_ratio):
        raise MetadataError(
            f"base_bin_shape has {len(base_bin_shape)} dims, "
            f"bin_ratio has {len(bin_ratio)} dims"
        )
    return tuple(b * r for b, r in zip(base_bin_shape, bin_ratio))


def compute_bin_ratio(
    base_bin_shape: tuple[float, ...],
    level_bin_shape: tuple[float, ...],
) -> tuple[int, ...]:
    """Compute the bin ratio from base and level bin shapes.

    Args:
        base_bin_shape: Supervoxel edge lengths at level 0.
        level_bin_shape: Supervoxel edge lengths at this level.

    Returns:
        Integer ratio per dimension.

    Raises:
        MetadataError: If the ratio is not integer in any dimension.
    """
    if len(base_bin_shape) != len(level_bin_shape):
        raise MetadataError(
            f"base_bin_shape has {len(base_bin_shape)} dims, "
            f"level_bin_shape has {len(level_bin_shape)} dims"
        )
    ratios: list[int] = []
    for i, (base, level) in enumerate(zip(base_bin_shape, level_bin_shape)):
        r = level / base
        if abs(r - round(r)) > 1e-9:
            raise MetadataError(
                f"Dimension {i}: level_bin_shape={level} / "
                f"base_bin_shape={base} = {r:.6f} is not an integer"
            )
        ratios.append(int(round(r)))
    return tuple(ratios)


def validate_bin_shape_divides_chunk(
    chunk_shape: tuple[float, ...],
    bin_shape: tuple[float, ...],
) -> None:
    """Validate that chunk_shape is an integer multiple of bin_shape.

    Raises:
        MetadataError: If any dimension is not evenly divisible.
    """
    for i, (cs, bs) in enumerate(zip(chunk_shape, bin_shape)):
        ratio = cs / bs
        if abs(ratio - round(ratio)) > 1e-9:
            raise MetadataError(
                f"chunk_shape[{i}]={cs} is not an integer multiple "
                f"of bin_shape[{i}]={bs} (ratio={ratio:.6f})"
            )


# ===================================================================
# ArrayMetadata
# ===================================================================

@dataclass
class ArrayMetadata:
    """Per-array metadata (stored in each array's ``.zattrs``).

    Attributes:
        name: Array name (e.g. ``"vertices"``, ``"attributes/radius"``).
        dtype: Numpy dtype string (e.g. ``"float32"``, ``"int64"``).
        encoding: ``"raw"`` or ``"draco"``.
        compression_codec: Codec name or None.
        chunk_shape: Zarr chunk shape or None (for ragged).
        channel_names: For attribute arrays, list of channel names.
        channel_dtype: For attribute arrays, the channel dtype.
    """

    name: str
    dtype: str
    encoding: str = ENCODING_RAW
    compression_codec: str | None = None
    chunk_shape: tuple[int, ...] | None = None
    channel_names: list[str] | None = None
    channel_dtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "name": self.name,
            "dtype": self.dtype,
            "encoding": self.encoding,
        }
        if self.compression_codec is not None:
            d["compression_codec"] = self.compression_codec
        if self.chunk_shape is not None:
            d["chunk_shape"] = list(self.chunk_shape)
        if self.channel_names is not None:
            d["channel_names"] = self.channel_names
        if self.channel_dtype is not None:
            d["channel_dtype"] = self.channel_dtype
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArrayMetadata:
        """Deserialise from a dict."""
        if "name" not in d or "dtype" not in d:
            raise MetadataError("ArrayMetadata requires 'name' and 'dtype'")
        cs = d.get("chunk_shape")
        return cls(
            name=d["name"],
            dtype=d["dtype"],
            encoding=d.get("encoding", ENCODING_RAW),
            compression_codec=d.get("compression_codec"),
            chunk_shape=tuple(cs) if cs else None,
            channel_names=d.get("channel_names"),
            channel_dtype=d.get("channel_dtype"),
        )

    def validate(self) -> None:
        """Validate this metadata object."""
        if self.encoding not in VALID_ENCODINGS:
            raise MetadataError(
                f"Invalid encoding '{self.encoding}', "
                f"must be one of {VALID_ENCODINGS}"
            )


# ===================================================================
# ParametricTypeDef
# ===================================================================

@dataclass
class ParametricTypeDef:
    """Definition of a parametric object type for the type registry.

    Attributes:
        type_id: Integer type tag stored in the objects array.
        name: Human-readable name (e.g. ``"plane"``, ``"line"``).
        coefficients: Ordered list of coefficient names.
    """

    type_id: int
    name: str
    coefficients: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "name": self.name,
            "coefficients": self.coefficients,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParametricTypeDef:
        return cls(
            type_id=d["type_id"],
            name=d["name"],
            coefficients=d["coefficients"],
        )


# ===================================================================
# Standard parametric type definitions
# ===================================================================

PARAMETRIC_PLANE = ParametricTypeDef(
    type_id=0, name="plane", coefficients=["A", "B", "C", "D"]
)
PARAMETRIC_LINE = ParametricTypeDef(
    type_id=1, name="line", coefficients=["x0", "y0", "z0", "dx", "dy", "dz"]
)
PARAMETRIC_SPHERE = ParametricTypeDef(
    type_id=2, name="sphere", coefficients=["cx", "cy", "cz", "r"]
)

DEFAULT_PARAMETRIC_TYPES = [PARAMETRIC_PLANE, PARAMETRIC_LINE, PARAMETRIC_SPHERE]


# ===================================================================
# Serialisation helpers (for store read/write)
# ===================================================================

def serialise_parametric_types(
    types: list[ParametricTypeDef],
) -> dict[str, Any]:
    """Serialise parametric type registry to a dict.

    Returns:
        ``{"parametric_types": {"0": {...}, "1": {...}, ...}}``
    """
    return {
        "parametric_types": {
            str(t.type_id): {"name": t.name, "coefficients": t.coefficients}
            for t in types
        }
    }


# ---------------------------------------------------------------------------
# Per-level chunk_shape helpers (0.7.0+)
# ---------------------------------------------------------------------------


def get_level_chunk_shape(
    root_meta: RootMetadata,
    level_meta: LevelMetadata | None,
) -> tuple[float, ...]:
    """Return the effective ``chunk_shape`` for a level.

    Resolves the v0.7 per-level override: when ``level_meta`` carries
    a non-None ``chunk_shape``, that value wins.  Otherwise the level
    inherits :attr:`RootMetadata.chunk_shape`.  Pass ``level_meta=None``
    to get the root chunk_shape (useful when only the level index is
    known but the LevelMetadata hasn't been read yet).
    """
    if level_meta is not None and level_meta.chunk_shape is not None:
        return level_meta.chunk_shape
    return root_meta.chunk_shape


def chunk_scale_from_root(
    root_meta: RootMetadata,
    level_meta: LevelMetadata | None,
) -> tuple[int, ...]:
    """Return the per-axis integer multiple of ROOT ``chunk_shape``.

    For a level whose ``chunk_shape`` is ``r_i × root_chunk_shape[i]``
    along axis ``i``, returns ``(r_0, r_1, ..., r_{ndim-1})``.  A level
    that inherits root returns all-ones.  Raises :class:`MetadataError`
    when the per-level chunk_shape isn't an integer multiple of root
    along some axis (nesting violation).

    **Root-relative**, and named so.  ``coarsen_level``'s
    ``chunk_scale_factor=`` keyword is *parent*-relative — it multiplies
    the SOURCE level's chunk_shape — so on a pyramid built with a repeated
    factor the two quantities diverge (``[2, 2]`` gives parent-factors
    2 and 2, but root-scales 2 and 4).  While both were called
    ``chunk_scale_factor`` nothing at the call site distinguished them,
    and reading one as the other compounds silently.  The old name remains
    as a deprecated alias.
    """
    eff = get_level_chunk_shape(root_meta, level_meta)
    root_cs = root_meta.chunk_shape
    if len(eff) != len(root_cs):
        raise MetadataError(
            f"Level chunk_shape rank {len(eff)} != root chunk_shape rank "
            f"{len(root_cs)}"
        )
    ratios: list[int] = []
    for axis, (e, r) in enumerate(zip(eff, root_cs)):
        if r <= 0:
            raise MetadataError(
                f"root chunk_shape axis {axis} = {r}; must be > 0"
            )
        ratio_f = float(e) / float(r)
        ratio_i = int(round(ratio_f))
        if ratio_i < 1 or abs(ratio_f - ratio_i) > 1e-9:
            raise MetadataError(
                f"Level chunk_shape axis {axis} = {e} is not a positive "
                f"integer multiple of root chunk_shape axis {axis} = {r} "
                f"(ratio {ratio_f}); nested chunk grids are required."
            )
        ratios.append(ratio_i)
    return tuple(ratios)


def chunk_scale_factor(
    root_meta: RootMetadata,
    level_meta: LevelMetadata | None,
) -> tuple[int, ...]:
    """Deprecated alias for :func:`chunk_scale_from_root`."""
    import warnings

    warnings.warn(
        "metadata.chunk_scale_factor() is renamed chunk_scale_from_root(). "
        "It is ROOT-relative, whereas coarsen_level(chunk_scale_factor=) is "
        "PARENT-relative; the shared name hid that they are different "
        "quantities on any pyramid with more than one coarser level.",
        DeprecationWarning,
        stacklevel=2,
    )
    return chunk_scale_from_root(root_meta, level_meta)


# ---------------------------------------------------------------------------
# Three reference frames, and how to move between them
# ---------------------------------------------------------------------------
#
# A pyramid level carries four quantities that all look like "a factor" and
# are measured against three different things:
#
#   bin_shape            ABSOLUTE  -- edge lengths, in world units.
#   chunk_shape          ABSOLUTE  -- edge lengths, in world units.
#   bin_ratio            vs LEVEL 0 -- what becomes the NGFF ``scale``.
#   coarsen_factor=      vs PARENT  -- what ``coarsen_level`` multiplies the
#                                     SOURCE level's bin by.
#   chunk_scale_factor=  vs PARENT  -- likewise for the chunk grid.
#
# Nothing in a name or a type distinguishes them, and reading one as another
# compounds silently.  Two real bugs came from exactly that: a pyramid
# refresh recovered ``coarsen_factor`` from the level-0-relative
# ``bin_ratio`` and squared it (levels came back at 2x, 8x, 64x instead of
# 2x, 4x, 8x), and a coarsener multiplied the ROOT bin instead of the
# parent's so every level got the same bin.
#
# The helpers below are the supported way to move between the frames.
# ``chunk_scale_from_root`` is the root-relative chunk counterpart and is
# named for its frame for the same reason.


def get_level_bin_shape(
    root_meta: RootMetadata,
    level_meta: LevelMetadata | None,
) -> tuple[float, ...]:
    """This level's ABSOLUTE bin shape.

    The bin counterpart of :func:`get_level_chunk_shape`, resolved in the
    order the store actually records things:

    1. ``level_meta.bin_shape`` when set -- the authoritative absolute value;
    2. else ``root.effective_bin_shape * level_meta.bin_ratio``, because a
       level round-tripped through a store that had no NGFF translation
       keeps its ratio but loses its shape;
    3. else the root's effective bin (level 0, or a level that does not bin).

    ``bin_shape`` is preferred over ``bin_ratio`` deliberately: the ratio is
    the level-0-relative view and reconstructing an absolute from it is one
    multiplication further from the source of truth.
    """
    if level_meta is not None and level_meta.bin_shape:
        return tuple(float(b) for b in level_meta.bin_shape)
    base = tuple(float(b) for b in root_meta.effective_bin_shape)
    if level_meta is not None and level_meta.bin_ratio:
        if len(level_meta.bin_ratio) != len(base):
            raise MetadataError(
                f"bin_ratio has {len(level_meta.bin_ratio)} dims, root bin "
                f"has {len(base)}"
            )
        return tuple(b * float(r) for b, r in zip(base, level_meta.bin_ratio))
    return base


def level_factor(
    root_meta: RootMetadata,
    level_meta: LevelMetadata,
    parent_meta: LevelMetadata | None,
) -> tuple[float, ...]:
    """The PARENT-relative bin factor -- what ``coarsen_factor`` means.

    ``bin_ratio`` is level-0-relative and ``bin_shape`` is absolute, so
    neither is the number ``coarsen_level`` wants.  This is that number.

    ``parent_meta`` is ``None`` for a level whose parent is level 0.

    Raises:
        MetadataError: If the per-axis factors are not uniform.  The
            caller reduces this to a scalar (``coarsen_level`` takes one),
            so letting axis 0 speak for the rest would silently discard
            the others.
    """
    child = get_level_bin_shape(root_meta, level_meta)
    parent = get_level_bin_shape(root_meta, parent_meta)
    factors = tuple(
        (float(c) / float(p)) if p else 1.0 for c, p in zip(child, parent)
    )
    if factors and any(abs(f - factors[0]) > 1e-9 for f in factors):
        raise MetadataError(
            f"per-axis bin factors {factors} are not uniform; a scalar "
            f"coarsen_factor cannot express this level. Pass the per-axis "
            f"bin_shape explicitly instead."
        )
    return factors


def level_chunk_scale(
    root_meta: RootMetadata,
    level_meta: LevelMetadata,
    parent_meta: LevelMetadata | None,
) -> tuple[int, ...]:
    """The PARENT-relative chunk factor -- what ``chunk_scale_factor`` means.

    The chunk counterpart of :func:`level_factor`.  Contrast
    :func:`chunk_scale_from_root`, which answers the level-0-relative
    question and is what ``LevelMetadata`` records.
    """
    child = get_level_chunk_shape(root_meta, level_meta)
    parent = get_level_chunk_shape(root_meta, parent_meta)
    out: list[int] = []
    for axis, (c, p) in enumerate(zip(child, parent)):
        if p <= 0:
            raise MetadataError(
                f"parent chunk_shape axis {axis} = {p}; must be > 0"
            )
        ratio_f = float(c) / float(p)
        ratio_i = int(round(ratio_f))
        if ratio_i < 1 or abs(ratio_f - ratio_i) > 1e-9:
            raise MetadataError(
                f"chunk_shape axis {axis} = {c} is not a positive integer "
                f"multiple of the parent's {p} (ratio {ratio_f})"
            )
        out.append(ratio_i)
    return tuple(out)


def validate_level_chunk_shape_against_root(
    root_meta: RootMetadata,
    level_meta: LevelMetadata,
) -> None:
    """Cross-level invariants for a per-level ``chunk_shape`` override.

    Validates:

    1. Per-axis positive integer multiple of root ``chunk_shape``
       (nested chunk grids).
    2. Per-axis divisibility by the effective per-level ``bin_shape``
       so bins still tile chunks cleanly at the level's resolution.

    Self-consistency (positivity, rank match against root) is checked
    independently by :meth:`LevelMetadata.validate`.

    Both this helper and that method are no-ops when
    ``level_meta.chunk_shape is None``.

    The no-op is deliberate and must stay.  Extending the divisibility
    check to levels that INHERIT root's ``chunk_shape`` looks like it
    would close a gap -- a coarse level takes its ``bin_shape`` from the
    coarsen factor while keeping root's grid -- but it rejects the normal
    case: a coarse level's bin is routinely LARGER than the chunk it sits
    in (bin 200 in a chunk of 100), and "bins tile chunks" is only an
    invariant while bins are the finer of the two.
    """
    if level_meta.chunk_shape is None:
        return
    effective_chunk = level_meta.chunk_shape
    # Triggers the nesting check.
    _ratios = chunk_scale_from_root(root_meta, level_meta)
    del _ratios

    # Per-level bin_shape must still divide the effective chunk_shape.
    bin_shape: tuple[float, ...] | None
    if level_meta.bin_shape is not None:
        bin_shape = level_meta.bin_shape
    elif level_meta.level == 0:
        bin_shape = root_meta.effective_bin_shape
    else:
        bin_shape = None
    if bin_shape is None:
        return
    for axis, (cs, bs) in enumerate(zip(effective_chunk, bin_shape)):
        if bs <= 0:
            raise MetadataError(
                f"Level {level_meta.level} bin_shape axis {axis} = {bs}; "
                "must be > 0"
            )
        if float(bs) > float(cs):
            # A coarse level's bin routinely exceeds the chunk it sits in
            # (bin 240 in a chunk of 120 -- one bin spanning two chunks is
            # exactly what coarsening produces).  "Bins tile chunks" is an
            # invariant only while bins are the FINER of the two; above
            # that the meaningful question is whether the bin is a whole
            # multiple of the chunk, which the grid does not require.
            #
            # This is why the helper had never been called from a write
            # path: run as written it rejects levels core's own coarsener
            # produces.  Checking only the bin <= chunk case is what makes
            # it safe to enforce.
            continue
        ratio_f = float(cs) / float(bs)
        ratio_i = int(round(ratio_f))
        if ratio_i < 1 or abs(ratio_f - ratio_i) > 1e-9:
            raise MetadataError(
                f"Level {level_meta.level} bin_shape axis {axis} = {bs} "
                f"does not divide chunk_shape axis {axis} = {cs} "
                f"(ratio {ratio_f})"
            )


def deserialise_parametric_types(
    d: dict[str, Any],
) -> list[ParametricTypeDef]:
    """Deserialise parametric type registry from a dict.

    Args:
        d: Dict with ``"parametric_types"`` key.

    Returns:
        List of ParametricTypeDef objects.
    """
    if "parametric_types" not in d:
        return []
    registry = d["parametric_types"]
    types: list[ParametricTypeDef] = []
    for tid_str, info in registry.items():
        types.append(ParametricTypeDef(
            type_id=int(tid_str),
            name=info["name"],
            coefficients=info["coefficients"],
        ))
    return sorted(types, key=lambda t: t.type_id)
