"""Level 2 metadata validation — verify metadata is well-formed."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zarr_vectors.constants import (
    CROSS_CHUNK_BOTH,
    CROSS_CHUNK_DEDUP,
    CROSS_CHUNK_EXPLICIT,
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJIDX_IDENTITY,
    OBJIDX_STANDARD,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.validate.structure import ValidationResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group

VALID_LINKS = {LINKS_EXPLICIT, LINKS_IMPLICIT_SEQUENTIAL, LINKS_IMPLICIT_BRANCHES}
VALID_OBJIDX = {OBJIDX_STANDARD, OBJIDX_IDENTITY}
VALID_CROSS = {CROSS_CHUNK_EXPLICIT, CROSS_CHUNK_DEDUP, CROSS_CHUNK_BOTH}


#: Level fields the validators check, and the legacy top-level spellings
#: some old stores used before they moved under ``zarr_vectors_level``.
_LEVEL_FIELDS = (
    "vertex_count", "bin_shape", "bin_size", "bin_ratio", "object_sparsity",
)


def level_fields(root: Group, li: int, lg: Group) -> dict:
    """The level's metadata fields as stored, for checking.

    The writer keeps per-level fields under the ``zarr_vectors_level``
    block, and ``bin_shape`` / ``bin_ratio`` only in the NGFF
    ``multiscales`` block (see :func:`read_level_metadata`).  Reading
    them off the level's top-level attributes -- as the validators once
    did -- found nothing, so every per-level check silently passed.

    Values are returned raw (not coerced through :class:`LevelMetadata`)
    so a malformed value reaches the check that reports it.  Precedence:
    NGFF transform > ``zarr_vectors_level`` > legacy top-level keys.
    """
    attrs = lg.attrs.to_dict()
    out = {k: attrs[k] for k in _LEVEL_FIELDS if k in attrs}
    block = attrs.get("zarr_vectors_level")
    if isinstance(block, dict):
        out.update({k: block[k] for k in _LEVEL_FIELDS if k in block})
    if out.get("bin_shape") is None and out.get("bin_size") is not None:
        out["bin_shape"] = out["bin_size"]
    out.pop("bin_size", None)
    try:
        from zarr_vectors.core.multiscale import read_level_transform
        scale, translation = read_level_transform(root, li)
    except Exception:
        scale, translation = None, None
    if scale is not None:
        # Mirrors read_level_metadata: level 0 is the identity by
        # convention and inherits the root's base_bin_shape.
        if li == 0:
            out.pop("bin_shape", None)
            out.pop("bin_ratio", None)
        else:
            out["bin_ratio"] = list(scale)
            if translation is not None:
                out["bin_shape"] = [2.0 * t for t in translation]
    return out


def validate_metadata(store_path: str | Path | Group) -> ValidationResult:
    """Level 2: verify all metadata is well-formed."""
    result = ValidationResult(level=2)

    try:
        root = open_store(store_path)
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result

    try:
        meta = read_root_metadata(root)
    except Exception as e:
        result.add_error(f"Cannot read root metadata: {e}")
        return result
    result.add_pass("Root metadata parsed")

    # Parsing checks only that the keys are there; RootMetadata.validate
    # holds the rules about their values (format version, conventions,
    # cross-level settings, manifest layout, ...).  Readers do not run it,
    # so this is where a store that breaks them is reported.
    from zarr_vectors.exceptions import ConventionError, MetadataError
    try:
        meta.validate()
    except (MetadataError, ConventionError) as e:
        result.add_error(f"Root metadata invalid: {e}")
    except Exception as e:
        result.add_error(
            f"Root metadata cannot be validated: {type(e).__name__}: {e}"
        )
    else:
        result.add_pass("Root metadata validated")

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
            result.add_error(
                f"Bounds dim mismatch: min={len(bmin)}, "
                f"max={len(bmax)}, expected {sid_ndim}"
            )
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
            la = level_fields(root, li, lg)
            result.add_pass(f"resolution_{li} metadata parsed")
            if lg.presence_deferred():
                # Legal -- readers derive presence from the store -- but a
                # build-time state: a finished store has run its rebuild,
                # and every reader that trusts the manifest sees nothing
                # of this level until it does.
                result.add_warning(
                    f"resolution_{li}: presence is deferred (a build that "
                    f"did not finish, or whose coordinator never ran "
                    f"rebuild_presence); its arrays carry no nonempty_chunks"
                )
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
            # A coarse level that inherits the root grid takes its bin
            # from the coarsen factor and is not required to tile the
            # chunk (bin 20 in a chunk of 50 is what the coarsener writes);
            # see validate_level_chunk_shape_against_root, the writer's
            # rule.  Divisibility is checked where the level declares its
            # own chunk_shape, or on a legacy level we could not parse.
            inherits_grid = (
                li > 0 and level_meta is not None
                and level_meta.chunk_shape is None
            )
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
                        elif inherits_grid or bs > cs:
                            # A coarse level's bin may span several chunks
                            # (bin 200 in a chunk of 100); tiling is only
                            # required while bins are the finer of the two,
                            # as validate_level_chunk_shape_against_root says.
                            continue
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
            if (
                isinstance(sparsity, bool)
                or not isinstance(sparsity, (int, float))
                or not (0.0 < sparsity <= 1.0)
            ):
                result.add_error(
                    f"resolution_{li}: object_sparsity={sparsity} not in (0, 1]"
                )
        except Exception as e:
            result.add_error(f"resolution_{li}: cannot read metadata: {e}")

    return result
