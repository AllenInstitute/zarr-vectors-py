"""Level 2 metadata validation — verify metadata is well-formed."""

from __future__ import annotations

from pathlib import Path

from zarr_vectors.constants import (
    CROSS_CHUNK_DEDUP, CROSS_CHUNK_BOTH, CROSS_CHUNK_EXPLICIT,
    LINKS_EXPLICIT, LINKS_IMPLICIT_BRANCHES, LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_IDENTITY, OBJIDX_STANDARD,
)
from zarr_vectors.core.store import (
    open_store, read_root_metadata, get_resolution_level, list_resolution_levels,
)
from zarr_vectors.validate.structure import ValidationResult

VALID_LINKS = {LINKS_EXPLICIT, LINKS_IMPLICIT_SEQUENTIAL, LINKS_IMPLICIT_BRANCHES}
VALID_OBJIDX = {OBJIDX_STANDARD, OBJIDX_IDENTITY}
VALID_CROSS = {CROSS_CHUNK_EXPLICIT, CROSS_CHUNK_DEDUP, CROSS_CHUNK_BOTH}


def validate_metadata(store_path: str | Path) -> ValidationResult:
    """Level 2: verify all metadata is well-formed."""
    result = ValidationResult(level=2)

    try:
        root = open_store(str(store_path))
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result

    try:
        meta = read_root_metadata(root)
    except Exception as e:
        result.add_error(f"Cannot read root metadata: {e}")
        return result
    result.add_pass("Root metadata parsed")

    sid_ndim = meta.sid_ndim
    if sid_ndim < 1:
        result.add_error(f"SID dimensionality is {sid_ndim}, must be >= 1")
    else:
        result.add_pass(f"SID dimensionality: {sid_ndim}")

    if len(meta.chunk_shape) != sid_ndim:
        result.add_error(f"chunk_shape has {len(meta.chunk_shape)} dims, expected {sid_ndim}")
    else:
        result.add_pass("chunk_shape dimensionality matches SID")

    for i, cs in enumerate(meta.chunk_shape):
        if cs <= 0:
            result.add_error(f"chunk_shape[{i}] = {cs}, must be > 0")

    if meta.bounds:
        bmin, bmax = meta.bounds
        if len(bmin) != sid_ndim or len(bmax) != sid_ndim:
            result.add_error(f"Bounds dim mismatch: min={len(bmin)}, max={len(bmax)}, expected {sid_ndim}")
        else:
            result.add_pass("Bounds dimensionality matches SID")

    lc = meta.links_convention
    if lc and lc not in VALID_LINKS:
        result.add_error(f"Unknown links_convention: '{lc}'")
    else:
        result.add_pass(f"links_convention: '{lc}'")

    oc = meta.object_index_convention
    if oc and oc not in VALID_OBJIDX:
        result.add_error(f"Unknown object_index_convention: '{oc}'")
    else:
        result.add_pass(f"object_index_convention: '{oc}'")

    cc = meta.cross_chunk_strategy
    if cc and cc not in VALID_CROSS:
        result.add_error(f"Unknown cross_chunk_strategy: '{cc}'")
    else:
        result.add_pass(f"cross_chunk_strategy: '{cc}'")

    if not meta.geometry_types:
        result.add_warning("No geometry_types specified")
    else:
        result.add_pass(f"geometry_types: {meta.geometry_types}")

    # Bin shape / chunk divisibility validation
    if meta.base_bin_shape is not None:
        if len(meta.base_bin_shape) != sid_ndim:
            result.add_error(
                f"base_bin_shape has {len(meta.base_bin_shape)} dims, "
                f"expected {sid_ndim}"
            )
        else:
            result.add_pass(
                f"base_bin_shape: {meta.base_bin_shape}"
            )
            for i, (cs, bs) in enumerate(zip(meta.chunk_shape, meta.base_bin_shape)):
                if bs <= 0:
                    result.add_error(
                        f"base_bin_shape[{i}]={bs}, must be > 0"
                    )
                else:
                    ratio = cs / bs
                    if abs(ratio - round(ratio)) > 1e-9:
                        result.add_error(
                            f"chunk_shape[{i}]={cs} not integer multiple "
                            f"of base_bin_shape[{i}]={bs}"
                        )
            result.add_pass(
                f"bins_per_chunk: {meta.bins_per_chunk}"
            )

    levels = list_resolution_levels(root)
    for li in levels:
        try:
            lg = get_resolution_level(root, li)
            la = lg.attrs
            result.add_pass(f"resolution_{li} metadata parsed")
            vc = la.get("vertex_count")
            if vc is not None:
                if not isinstance(vc, int) or vc < 0:
                    result.add_error(f"resolution_{li}: vertex_count={vc} invalid")
                else:
                    result.add_pass(f"resolution_{li}: vertex_count={vc}")

            # Cross-level chunk_shape invariants: a per-level chunk_shape
            # override must be a positive integer multiple of the root's,
            # and the per-level bin_shape must still tile it.  This
            # validator existed but was never called by any pass, so
            # per-level overrides went unvalidated.
            from zarr_vectors.core.metadata import (
                get_level_chunk_shape,
                validate_level_chunk_shape_against_root,
            )
            from zarr_vectors.core.store import read_level_metadata
            from zarr_vectors.exceptions import MetadataError

            level_meta = None
            try:
                level_meta = read_level_metadata(root, li)
                validate_level_chunk_shape_against_root(meta, level_meta)
            except MetadataError as e:
                result.add_error(f"resolution_{li}: {e}")
            except Exception:
                level_meta = None

            # Validate per-level bin_shape divides the EFFECTIVE per-level
            # chunk_shape.  Comparing against the root chunk_shape (as this
            # did before) tests the wrong quantity on a chunk_scale_factor>1
            # store, where the level's chunk_shape is a multiple of root's.
            eff_chunk_shape = (
                get_level_chunk_shape(meta, level_meta)
                if level_meta is not None else meta.chunk_shape
            )
            bin_shape = la.get("bin_shape")
            if bin_shape is None:
                bin_shape = la.get("bin_size")  # legacy fallback
            if bin_shape is not None:
                if len(bin_shape) != sid_ndim:
                    result.add_error(
                        f"resolution_{li}: bin_shape has {len(bin_shape)} dims"
                    )
                else:
                    for i, (cs, bs) in enumerate(zip(eff_chunk_shape, bin_shape)):
                        if bs <= 0:
                            result.add_error(
                                f"resolution_{li}: bin_shape[{i}]={bs} not > 0"
                            )
                        else:
                            ratio = cs / bs
                            if abs(ratio - round(ratio)) > 1e-9:
                                result.add_error(
                                    f"resolution_{li}: bin_shape[{i}]={bs} "
                                    f"does not divide chunk_shape[{i}]={cs}"
                                )

            # Validate bin_ratio values
            bin_ratio = la.get("bin_ratio")
            if bin_ratio is not None:
                if len(bin_ratio) != sid_ndim:
                    result.add_error(
                        f"resolution_{li}: bin_ratio has {len(bin_ratio)} dims"
                    )
                else:
                    for i, r in enumerate(bin_ratio):
                        if r < 1:
                            result.add_error(
                                f"resolution_{li}: bin_ratio[{i}]={r} < 1"
                            )

            # Validate object_sparsity in (0, 1]
            sparsity = la.get("object_sparsity", 1.0)
            if not (0.0 < sparsity <= 1.0):
                result.add_error(
                    f"resolution_{li}: object_sparsity={sparsity} not in (0, 1]"
                )
        except Exception as e:
            result.add_error(f"resolution_{li}: cannot read metadata: {e}")

    return result
