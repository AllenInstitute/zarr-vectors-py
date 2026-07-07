"""Level 3 consistency validation — verify data arrays are internally consistent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    list_chunk_keys, read_all_object_manifests, read_chunk_vertices, read_cross_chunk_links,
)
from zarr_vectors.core.store import (
    get_resolution_level, list_resolution_levels, open_store, read_root_metadata,
)
from zarr_vectors.validate.structure import ValidationResult


def validate_consistency(store_path: str | Path) -> ValidationResult:
    """Level 3: verify internal data consistency."""
    result = ValidationResult(level=3)

    try:
        root = open_store(str(store_path))
        meta = read_root_metadata(root)
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result

    ndim = meta.sid_ndim
    levels = list_resolution_levels(root)

    for li in levels:
        prefix = f"resolution_{li}"
        try:
            lg = get_resolution_level(root, li)
        except Exception as e:
            result.add_error(f"{prefix}: cannot open: {e}")
            continue

        chunk_keys = list_chunk_keys(lg)
        if not chunk_keys:
            result.add_warning(f"{prefix}: no chunk data")
            continue

        total_verts = 0
        chunk_fragment_counts: dict[tuple, int] = {}

        # Resolve per-level chunk_shape (v0.7 may override root).
        from zarr_vectors.core.metadata import get_level_chunk_shape
        from zarr_vectors.core.store import read_level_metadata
        try:
            level_meta_obj = read_level_metadata(root, li)
        except Exception:
            level_meta_obj = None
        level_chunk_shape = get_level_chunk_shape(meta, level_meta_obj)

        # Determine effective bin shape for this level
        try:
            la = lg.attrs
            level_bin_shape = la.get("bin_shape") or la.get("bin_size")
            if level_bin_shape is not None:
                level_bin_shape = tuple(float(x) for x in level_bin_shape)
            elif li == 0:
                level_bin_shape = meta.effective_bin_shape
            else:
                level_bin_shape = level_chunk_shape  # unknown — skip bin checks
        except Exception:
            level_bin_shape = level_chunk_shape

        # Compute bins_per_chunk for this level
        level_bins_per_chunk = tuple(
            int(round(cs / bs))
            for cs, bs in zip(level_chunk_shape, level_bin_shape)
        )
        max_fragments = 1
        for b in level_bins_per_chunk:
            max_fragments *= b

        # Per-bin fragment layout only applies to undifferentiated point-cloud stores.
        # Polylines / lines / graphs / meshes use fragments to represent segments,
        # endpoints, or per-object partitions — not bins.
        geom_types = meta.geometry_types or []
        is_point_cloud_only = (
            "point_cloud" in geom_types
            and not any(gt in geom_types for gt in [
                "polyline", "streamline", "line", "graph",
                "skeleton", "mesh",
            ])
        )
        # Also skip if the store has object_index (fragments are per-object, not per-bin)
        try:
            has_object_index = "object_index" in lg
        except Exception:
            has_object_index = False

        check_bin_layout = is_point_cloud_only and not has_object_index
        chunks_checked_for_bin_bounds = 0

        for ck in chunk_keys:
            try:
                groups = read_chunk_vertices(lg, ck, dtype=np.float32, ndim=ndim)
            except Exception as e:
                result.add_error(f"{prefix}: chunk {ck} decode failed: {e}")
                continue

            chunk_fragment_counts[ck] = len(groups)

            # Check fragment count doesn't exceed bins_per_chunk
            # (only for undifferentiated point clouds with explicit bins)
            has_bins = any(b > 1 for b in level_bins_per_chunk)
            if check_bin_layout and has_bins and len(groups) > max_fragments:
                result.add_error(
                    f"{prefix}: chunk {ck} has {len(groups)} fragments, "
                    f"exceeds bins_per_chunk product {max_fragments}"
                )

            for vi, fragment in enumerate(groups):
                if fragment.ndim != 2 or fragment.shape[1] != ndim:
                    result.add_error(f"{prefix}: chunk {ck} fragment[{vi}] shape {fragment.shape}")
                if len(fragment) > 0 and np.any(~np.isfinite(fragment)):
                    result.add_warning(f"{prefix}: chunk {ck} fragment[{vi}] NaN/Inf")
                total_verts += len(fragment)

            # Spot-check bin bounds for point clouds only
            if check_bin_layout and has_bins and chunks_checked_for_bin_bounds < 3:
                from zarr_vectors.spatial.chunking import fragment_index_to_bin
                chunks_checked_for_bin_bounds += 1
                for vi, fragment in enumerate(groups):
                    if len(fragment) == 0:
                        continue
                    try:
                        bin_coords = fragment_index_to_bin(vi, ck, level_bins_per_chunk)
                    except Exception:
                        continue
                    bin_lo = np.array(
                        [bc * bs for bc, bs in zip(bin_coords, level_bin_shape)],
                        dtype=np.float64,
                    )
                    bin_hi = bin_lo + np.array(level_bin_shape, dtype=np.float64)
                    # Allow small tolerance for float rounding
                    tol = 1e-4
                    out_of_bin = np.any(
                        (fragment < bin_lo - tol) | (fragment >= bin_hi + tol),
                        axis=1,
                    )
                    n_out = int(np.sum(out_of_bin))
                    if n_out > 0:
                        result.add_warning(
                            f"{prefix}: chunk {ck} fragment[{vi}] has {n_out} points "
                            f"outside bin {bin_coords} bounds"
                        )

        result.add_pass(
            f"{prefix}: {len(chunk_keys)} chunks decoded, {total_verts} vertices"
        )

        try:
            la = lg.attrs
            evc = la.get("vertex_count")
            if evc is not None:
                if total_verts != evc:
                    result.add_error(f"{prefix}: metadata vertex_count={evc}, actual={total_verts}")
                else:
                    result.add_pass(f"{prefix}: vertex_count matches")
        except Exception:
            pass

        try:
            manifests = read_all_object_manifests(lg)
            for oid, mf in enumerate(manifests):
                for cc, fragment_index in mf:
                    if cc not in chunk_fragment_counts:
                        result.add_error(f"{prefix}: obj {oid} refs non-existent chunk {cc}")
                    elif fragment_index >= chunk_fragment_counts[cc]:
                        result.add_error(f"{prefix}: obj {oid} refs fragment_idx={fragment_index} >= {chunk_fragment_counts[cc]}")
            result.add_pass(f"{prefix}: object_index validated ({len(manifests)} objects)")
        except Exception:
            pass

        # Walk every cross_chunk_links/<delta>/ family.  Two passes:
        #
        # (a) Structural pass — for each cell key, parse the dotted
        #     canonical chunk-tuple and check the cell-key arity
        #     (sid_ndim * link_width) and the canonical-sort
        #     invariant (chunks non-decreasing in lex order).
        # (b) Endpoint-presence pass — read records in input order
        #     via read_cross_chunk_links so endpoint 0 is the
        #     owning-level source.  For delta=0 every endpoint must
        #     exist in this level's chunk grid; for delta != 0 only
        #     endpoint 0 is constrained here.
        from zarr_vectors.core.arrays import list_cross_link_deltas
        from zarr_vectors.core.paths import (
            cross_chunk_links_path, parse_cell_key,
        )
        for d in list_cross_link_deltas(lg):
            family = cross_chunk_links_path(d)
            try:
                family_meta = lg.read_array_meta(family) or {}
                link_width = int(family_meta.get("link_width", 2))
                ccl_sid_ndim = int(family_meta.get("sid_ndim", 0))
                directed = bool(family_meta.get("directed", False))
                store = str(family_meta.get("store", "canonical"))
            except Exception:
                continue
            if ccl_sid_ndim == 0:
                continue

            # The canonical-sort invariant (chunks non-decreasing in the
            # cell key) only holds for an undirected, single-cell family.
            # Directed cells key on input endpoint order, and duplicate
            # cells lead with each incident chunk, so both legitimately
            # break lex ordering — skip the check for them.
            enforce_canonical = not directed and store == "canonical"
            cell_keys = sorted(lg.list_chunks(family))
            for ckey in cell_keys:
                try:
                    canonical_chunks = parse_cell_key(
                        ckey, sid_ndim=ccl_sid_ndim, link_width=link_width,
                    )
                except ValueError as exc:
                    result.add_error(
                        f"{prefix}: ccl[delta={d}] cell {ckey!r} "
                        f"malformed: {exc}"
                    )
                    continue
                if enforce_canonical:
                    for i in range(1, link_width):
                        if canonical_chunks[i] < canonical_chunks[i - 1]:
                            result.add_error(
                                f"{prefix}: ccl[delta={d}] cell {ckey!r} "
                                f"violates canonical-sort invariant"
                            )
                            break

            try:
                ccl = read_cross_chunk_links(lg, delta=d)
            except Exception:
                ccl = []

            # read_cross_chunk_links returns one row per on-disk record
            # (physical, so duplicated copies are counted); it must match
            # the family's recorded num_physical_records.
            if "num_physical_records" in family_meta:
                expected_phys = int(family_meta["num_physical_records"])
                if expected_phys != len(ccl):
                    result.add_error(
                        f"{prefix}: ccl[delta={d}] num_physical_records="
                        f"{expected_phys} != {len(ccl)} rows on disk"
                    )
            for record in ccl:
                # record is a tuple of (chunk_coords, vi) endpoints
                # in input order; endpoint 0 is the owning-level
                # source side.
                src_chunk = record[0][0]
                if src_chunk not in chunk_fragment_counts:
                    result.add_error(
                        f"{prefix}: ccl[delta={d}] refs non-existent "
                        f"source chunk {src_chunk}"
                    )
                if d == 0:
                    for ep_chunk, _ in record[1:]:
                        if ep_chunk not in chunk_fragment_counts:
                            result.add_error(
                                f"{prefix}: ccl[delta=0] refs "
                                f"non-existent chunk {ep_chunk}"
                            )
            result.add_pass(
                f"{prefix}: ccl[delta={d}] validated "
                f"({len(ccl)} links across {len(cell_keys)} cells)"
            )

    return result
