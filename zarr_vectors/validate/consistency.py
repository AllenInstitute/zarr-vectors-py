"""Level 3 consistency validation — verify data arrays are internally consistent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.core.arrays import (
    _check_not_legacy_ccl_blob,
    _decode_ccl_cell_payload,
    _iter_populated_cells_in_shard,
    _list_kN_arrays,
    _walk_populated_shards,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    read_cross_chunk_links,
)
from zarr_vectors.core.paths import (
    cross_chunk_links_path,
    cross_chunk_link_attributes_path,
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

        # Walk every cross_chunk_links/<delta>/ group and verify the
        # v0.8 kN-array layout's per-cell invariants:
        #  - no legacy monolithic ``data`` blob present
        #  - cell-payload byte length % (9 * link_width) == 0
        #  - ci_i ∈ [0, K-1] and coverage set(ci) == {0..K-1}
        #  - canonical ci = [0, 1] for delta=0 link_width=2 (undirected groups only)
        #  - cell-coord K chunk segments are in strict lex order
        #  - chunk existence at the relevant level
        #  - same-chunk warning for populated k1 cells
        from zarr_vectors.core.arrays import (
            list_cross_chunk_link_attribute_deltas,
            list_cross_link_deltas,
        )
        for d in list_cross_link_deltas(lg):
            parent_name = cross_chunk_links_path(d)
            ccl_meta = lg.read_array_meta(parent_name)
            if not ccl_meta:
                result.add_error(
                    f"{prefix}: ccl[delta={d}] missing parent group metadata"
                )
                continue
            try:
                _check_not_legacy_ccl_blob(lg, full_name=parent_name)
            except Exception as e:
                result.add_error(f"{prefix}: {e}")
                continue
            link_width = int(ccl_meta.get("link_width", 0))
            sid_ndim_meta = int(ccl_meta.get("sid_ndim", 0)) or ndim
            ccl_directed = bool(ccl_meta.get("directed", False))
            if link_width <= 0:
                # Group exists with no records yet (writer stamps
                # link_width when first cell is written).  Nothing to
                # validate.
                continue

            ccl_cell_count = 0
            ccl_record_count = 0
            per_cell_counts: dict[tuple[int, tuple[int, ...]], int] = {}
            for K, arr in _list_kN_arrays(
                lg, delta=d, link_width=link_width,
            ):
                shard_shape = tuple(int(s) for s in (arr.shards or arr.chunks))
                origin_attr = tuple(
                    int(x)
                    for x in arr.attrs.get(
                        "chunk_origin", (0,) * sid_ndim_meta,
                    )
                )
                for shard_coord in _walk_populated_shards(arr):
                    shard_origin = tuple(
                        shard_coord[i] * shard_shape[i] for i in range(arr.ndim)
                    )
                    for cell_idx, payload in _iter_populated_cells_in_shard(
                        arr, shard_origin, shard_shape,
                    ):
                        ccl_cell_count += 1
                        per_cell_counts[(K, tuple(int(x) for x in cell_idx))] = 0
                        if not isinstance(payload, (bytes, bytearray)):
                            payload = bytes(payload)
                        rec_size = 9 * link_width
                        if len(payload) % rec_size != 0:
                            result.add_error(
                                f"{prefix}: ccl[delta={d}] k{K} cell {tuple(cell_idx)} "
                                f"byte length {len(payload)} not multiple of {rec_size}"
                            )
                            continue
                        # Reconstruct sorted chunks from cell coord.
                        sorted_chunks = tuple(
                            tuple(
                                int(cell_idx[k * sid_ndim_meta + a])
                                + origin_attr[a]
                                for a in range(sid_ndim_meta)
                            )
                            for k in range(K)
                        )
                        if K >= 2:
                            ordered = all(
                                sorted_chunks[i] < sorted_chunks[i + 1]
                                for i in range(K - 1)
                            )
                            if not ordered:
                                result.add_error(
                                    f"{prefix}: ccl[delta={d}] k{K} cell "
                                    f"{tuple(cell_idx)} chunk segments not "
                                    f"strictly lex-sorted: {sorted_chunks}"
                                )
                        if K == 1:
                            result.add_warning(
                                f"{prefix}: ccl[delta={d}] k1 cell {tuple(cell_idx)} "
                                f"is legal but consider using "
                                f"links/{d}/<chunk> for intra-chunk edges"
                            )
                        # Decode records and check ci coverage / canonical
                        # form / chunk existence.
                        records = _decode_ccl_cell_payload(
                            bytes(payload),
                            sorted_chunks,
                            link_width=link_width,
                        )
                        ccl_record_count += len(records)
                        per_cell_counts[(K, tuple(int(x) for x in cell_idx))] = len(
                            records
                        )
                        for rec in records:
                            ci_seen = set()
                            for endpoint_i, (chunk, _vi) in enumerate(rec):
                                # Recover ci by reverse-lookup in sorted_chunks.
                                try:
                                    ci = sorted_chunks.index(chunk)
                                except ValueError:
                                    result.add_error(
                                        f"{prefix}: ccl[delta={d}] k{K} record "
                                        f"endpoint chunk {chunk} not in "
                                        f"cell sorted_chunks {sorted_chunks}"
                                    )
                                    continue
                                if not 0 <= ci < K:
                                    result.add_error(
                                        f"{prefix}: ccl[delta={d}] k{K} ci={ci} "
                                        f"out of range [0,{K - 1}]"
                                    )
                                ci_seen.add(ci)
                                # Chunk existence: endpoint 0 lives at owning
                                # level, others at owning + delta.
                                if endpoint_i == 0 or d == 0:
                                    if chunk not in chunk_fragment_counts:
                                        result.add_error(
                                            f"{prefix}: ccl[delta={d}] k{K} "
                                            f"endpoint {endpoint_i} refs "
                                            f"non-existent chunk {chunk}"
                                        )
                            if ci_seen != set(range(K)):
                                result.add_error(
                                    f"{prefix}: ccl[delta={d}] k{K} record ci set "
                                    f"{sorted(ci_seen)} does not cover {{0..{K - 1}}}"
                                )
                            if d == 0 and link_width == 2 and K == 2 and not ccl_directed:
                                # Canonical ci = [0, 1]: endpoint 0 must be at
                                # smaller chunk, endpoint 1 at larger.  Skipped
                                # for directed (walk-order) groups, which
                                # intentionally preserve predecessor/successor
                                # endpoint order instead.
                                if rec[0][0] != sorted_chunks[0] or rec[1][0] != sorted_chunks[1]:
                                    result.add_error(
                                        f"{prefix}: ccl[delta=0] k2 record not "
                                        f"canonical ci=[0,1]: {rec}"
                                    )

            result.add_pass(
                f"{prefix}: ccl[delta={d}] validated "
                f"({ccl_cell_count} cells, {ccl_record_count} records)"
            )

            # Per-cell attribute parity: every cross_chunk_link_attributes/
            # <name>/<delta>/kK cell at the same cell coord has matching
            # row count.
            try:
                attr_root = lg.zarr_group["cross_chunk_link_attributes"]
            except Exception:
                attr_root = None
            if attr_root is not None:
                for attr_name in list(attr_root):
                    delta_seg = f"{d:+d}" if d != 0 else "0"
                    try:
                        attr_delta_group = attr_root[attr_name][delta_seg]
                    except Exception:
                        continue
                    # Walk kK sub-arrays of attribute group.
                    for child_name in list(attr_delta_group):
                        if not child_name.startswith("k"):
                            continue
                        try:
                            arr = attr_delta_group[child_name]
                        except Exception:
                            continue
                        try:
                            K = int(child_name[1:])
                        except ValueError:
                            continue
                        shard_shape = tuple(
                            int(s) for s in (arr.shards or arr.chunks)
                        )
                        for shard_coord in _walk_populated_shards(arr):
                            shard_origin = tuple(
                                shard_coord[i] * shard_shape[i]
                                for i in range(arr.ndim)
                            )
                            for cell_idx, payload in _iter_populated_cells_in_shard(
                                arr, shard_origin, shard_shape,
                            ):
                                cell_key = (K, tuple(int(x) for x in cell_idx))
                                link_count = per_cell_counts.get(cell_key)
                                if link_count is None:
                                    result.add_error(
                                        f"{prefix}: ccl_attr[{attr_name}/delta={d}] "
                                        f"k{K} cell {tuple(cell_idx)} has no "
                                        f"matching link cell"
                                    )
                                    continue
                                # Row count is determined by attribute dtype; we
                                # cannot recompute without dtype metadata.  At
                                # least sanity-check payload presence.
                                if link_count > 0 and not payload:
                                    result.add_error(
                                        f"{prefix}: ccl_attr[{attr_name}/delta={d}] "
                                        f"k{K} cell {tuple(cell_idx)} empty but "
                                        f"link cell has {link_count} records"
                                    )

    return result
