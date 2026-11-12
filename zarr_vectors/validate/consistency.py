"""Level 3 consistency validation — verify data arrays are internally consistent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_vectors.core.arrays import (
    chunk_fragments_tile,
    chunk_link_fragments_aligned,
    list_chunk_keys,
    read_chunk_vertex_buffer,
    read_chunk_vertices,
    read_object_manifest_rows,
)
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.typing import ChunkCoords
from zarr_vectors.validate.structure import ValidationResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group


def _lex_sign(offset: ChunkCoords) -> int:
    """Sign of a chunk offset under lexicographic order.

    ``+1`` when the first non-zero component is positive, ``-1`` when it
    is negative, ``0`` for the all-zero offset.  This is the ordering the
    canonical sort induces: sorting endpoints by ``(chunk, vi)`` puts the
    lex-smallest chunk first, so every stored offset in a canonical
    intra-level family is lex-non-negative.
    """
    for c in offset:
        if int(c) > 0:
            return 1
        if int(c) < 0:
            return -1
    return 0


def validate_consistency(store_path: str | Path | Group) -> ValidationResult:
    """Level 3: verify internal data consistency."""
    result = ValidationResult(level=3)

    try:
        root = open_store(store_path)
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

        from zarr_vectors.octree_chunks.guard import is_octree_level

        if is_octree_level(lg):
            # Its chunks are leaves, not uniform cells: every check below
            # would misread them.  It has checks of its own.
            _validate_octree_level(lg, ndim, prefix, result)
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

        # A level claiming ``fragments_tile`` is telling readers they may
        # return a chunk's whole buffer without consulting its fragment
        # index.  A claim that does not hold is a silent wrong answer --
        # rows nothing references come back -- so it is worth confirming
        # against what is actually stored.
        claims_tiling = bool(
            getattr(level_meta_obj, "fragments_tile", False),
        )
        # Likewise ``link_fragments_aligned``: an object read takes only
        # its own link fragments, so a misfiled record is a face that
        # silently goes missing.
        claims_alignment = bool(
            getattr(level_meta_obj, "link_fragments_aligned", False),
        )

        for ck in chunk_keys:
            try:
                groups = read_chunk_vertices(lg, ck, ndim=ndim)
            except Exception as e:
                result.add_error(f"{prefix}: chunk {ck} decode failed: {e}")
                continue

            chunk_fragment_counts[ck] = len(groups)

            if claims_tiling:
                # Against the BUFFER's row count, not the fragments' total.
                # Those two differ exactly when rows go unreferenced, which
                # is the case the claim must not be allowed to hide.
                buf = read_chunk_vertex_buffer(lg, ck, ndim=ndim, default=None)
                n_rows = 0 if buf is None else len(buf)
                if not chunk_fragments_tile(lg, ck, n_rows):
                    result.add_error(
                        f"{prefix}: fragments_tile is set but chunk {ck} "
                        f"does not tile its buffer"
                    )
                    claims_tiling = False   # one report per level is enough

            if claims_alignment and not chunk_link_fragments_aligned(lg, ck):
                result.add_error(
                    f"{prefix}: link_fragments_aligned is set but chunk {ck}'s "
                    f"intra-chunk links are not filed by vertex fragment"
                )
                claims_alignment = False

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

        for problem in _dense_index_problems(lg):
            result.add_error(f"{prefix}: {problem}")

        try:
            ids, manifests = read_object_manifest_rows(lg)
            for oid, mf in zip(ids.tolist(), manifests):
                for cc, fragment_index in mf:
                    if cc not in chunk_fragment_counts:
                        result.add_error(f"{prefix}: obj {oid} refs non-existent chunk {cc}")
                    elif fragment_index >= chunk_fragment_counts[cc]:
                        result.add_error(
                            f"{prefix}: obj {oid} refs "
                            f"fragment_idx={fragment_index} >= "
                            f"{chunk_fragment_counts[cc]}"
                        )
            result.add_pass(f"{prefix}: object_index validated ({len(manifests)} objects)")
        except Exception:
            pass

        # Walk every links/<delta>/ family.  Two passes:
        #
        # (a) Structural pass — for each <offsets> segment, parse the
        #     dotted relative-offset tuples and check the canonical-sort
        #     invariant (offsets non-negative and non-decreasing in lex
        #     order) on the intra-level, undirected, single-copy families
        #     where it applies.
        # (b) Endpoint-presence pass — read records in input order
        #     via read_links so endpoint 0 is the owning-level
        #     source.  For delta=0 every endpoint must exist in this
        #     level's chunk grid; for delta != 0 only endpoint 0 is
        #     constrained here.
        from zarr_vectors.core.arrays import (
            list_link_deltas,
            list_link_offsets,
            read_links,
        )
        from zarr_vectors.core.paths import links_group_path, parse_offsets
        for d in list_link_deltas(lg):
            family = links_group_path(d)
            try:
                family_meta = lg.read_array_meta(family) or {}
                link_width = int(family_meta.get("link_width", 2))
                link_sid_ndim = int(family_meta.get("sid_ndim", 0))
                directed = bool(family_meta.get("directed", False))
                store = str(family_meta.get("store", "canonical"))
            except Exception:
                continue
            if link_sid_ndim == 0:
                continue

            # Under the offset layout the canonical-sort invariant is a
            # property of the *directory name*, so it costs one parse per
            # offset array rather than one per cell.  It only holds for an
            # undirected, single-copy, intra-level family: directed
            # segments key on input endpoint order, duplicate families lead
            # with each incident chunk, and cross-level records are never
            # sorted (their source is always input endpoint 0) — all three
            # legitimately carry lex-negative offsets.
            enforce_canonical = (
                not directed and store == "canonical" and d == 0
            )
            segments = list_link_offsets(lg, d)
            for seg in segments:
                try:
                    offsets = parse_offsets(
                        seg, sid_ndim=link_sid_ndim, link_width=link_width,
                    )
                except ValueError as exc:
                    result.add_error(
                        f"{prefix}: links[delta={d}] offsets segment "
                        f"{seg!r} malformed: {exc}"
                    )
                    continue
                if not enforce_canonical:
                    continue
                # All-zero offsets are the intra-chunk array — legal, and
                # deduped by the vertex-index tie-break rather than by the
                # offset sign.
                for i, off in enumerate(offsets):
                    if _lex_sign(off) < 0:
                        result.add_error(
                            f"{prefix}: links[delta={d}] segment {seg!r} "
                            f"offset {i + 1} is lexicographically negative; "
                            f"a canonical family stores each record once, "
                            f"under the positive offset"
                        )
                    if i and offsets[i] < offsets[i - 1]:
                        result.add_error(
                            f"{prefix}: links[delta={d}] segment {seg!r} "
                            f"offsets are not non-decreasing; violates the "
                            f"canonical-sort invariant"
                        )

            try:
                links = read_links(lg, delta=d)
            except Exception:
                links = []

            # read_links returns one row per on-disk record (physical, so
            # duplicated copies are counted); it must match the family's
            # recorded num_physical_records.
            if "num_physical_records" in family_meta:
                expected_phys = int(family_meta["num_physical_records"])
                if expected_phys != len(links):
                    result.add_error(
                        f"{prefix}: links[delta={d}] num_physical_records="
                        f"{expected_phys} != {len(links)} rows on disk"
                    )
            for record in links:
                # record is a tuple of (chunk_coords, vi) endpoints
                # in input order; endpoint 0 is the owning-level
                # source side.
                src_chunk = record[0][0]
                if src_chunk not in chunk_fragment_counts:
                    result.add_error(
                        f"{prefix}: links[delta={d}] refs non-existent "
                        f"source chunk {src_chunk}"
                    )
                if d == 0:
                    for ep_chunk, _ in record[1:]:
                        if ep_chunk not in chunk_fragment_counts:
                            result.add_error(
                                f"{prefix}: links[delta=0] refs "
                                f"non-existent chunk {ep_chunk}"
                            )
            result.add_pass(
                f"{prefix}: links[delta={d}] validated "
                f"({len(links)} links across {len(segments)} offset arrays)"
            )

    return result


def _validate_octree_level(lg: Any, ndim: int, prefix: str, result: ValidationResult) -> None:
    """Level 3 checks for a level with octree chunks (:mod:`zarr_vectors.octree_chunks`).

    Each leaf is one chunk at its anchor cell, so the uniform checks do
    not apply.  These do: the leaf and cell tables read; every leaf, and
    nothing else, has a vertex chunk at its anchor, holding the vertices
    and fragments both tables say; the level's ``vertex_count``; object
    manifests naming ``(anchor, fragment)`` of a leaf; every link joining
    stored leaves at rows they hold, as many as its family records; no
    links to other levels; the marker codec that makes readers without
    octree chunks refuse the level; and, on a sharded level, every leaf
    inside one shard (:func:`_check_octree_shards`).
    """
    from zarr_vectors.constants import VERTEX_FRAGMENTS, VERTICES
    from zarr_vectors.core.arrays import list_link_deltas
    from zarr_vectors.encoding.octree_marker import OCTREE_MARKER_CODEC, is_marker
    from zarr_vectors.octree_chunks.convert import (
        cell_table,
        leaf_table,
        octree_meta,
        read_octree_link_cells,
    )

    block = octree_meta(lg) or {}
    if "converting" in block:
        result.add_error(
            f"{prefix}: part way through {block['converting']} (interrupted?); run "
            f"zarr_vectors.octree_chunks.{block['converting']} on it again to finish"
        )
        return
    try:
        table = leaf_table(lg)
        cells = cell_table(lg)
    except Exception as e:
        result.add_error(f"{prefix}: octree chunks: cannot read the leaf and cell tables: {e}")
        return
    _check_octree_shards(lg, table, prefix, result)
    n_leaves = len(table)
    anchors = [tuple(a) for a in table.anchors().tolist()]
    leaf_at = {a: n for n, a in enumerate(anchors)}
    stored = {tuple(int(c) for c in ck) for ck in list_chunk_keys(lg)}
    for what, bad in (("leaf anchors with no vertex chunk", [a for a in anchors if a not in stored]),
                      ("vertex chunks at no leaf's anchor", sorted(stored - set(anchors)))):
        if bad:
            result.add_error(f"{prefix}: octree chunks: {len(bad)} {what}, e.g. {bad[0]}")

    leaf_col = cells[:, 3] if len(cells) else np.zeros(0, np.int64)
    if len(cells) and ((leaf_col < 0) | (leaf_col >= n_leaves)).any():
        result.add_error(f"{prefix}: octree chunks: the cell table names leaves that do not exist")
        return
    cell_rows = np.bincount(leaf_col, weights=cells[:, 4], minlength=n_leaves).astype(np.int64)
    cell_frags = np.bincount(leaf_col, weights=cells[:, 5], minlength=n_leaves).astype(np.int64)
    rows = np.zeros(n_leaves, dtype=np.int64)
    frags = np.zeros(n_leaves, dtype=np.int64)
    keys = [".".join(str(c) for c in a) for a in anchors if a in stored]
    with lg.batched_reads([(VERTICES, keys), (VERTEX_FRAGMENTS, keys)]):
        for n, a in enumerate(anchors):
            if a not in stored:
                continue
            try:
                groups = read_chunk_vertices(lg, a, ndim=ndim, _leaf=True)
            except Exception as e:
                result.add_error(f"{prefix}: leaf {n} (chunk {a}) decode failed: {e}")
                continue
            rows[n] = sum(len(g) for g in groups)
            frags[n] = len(groups)
    weight = np.asarray(table.weight, dtype=np.int64)
    for what, other in (("leaf table weight", weight), ("cell table vertices", cell_rows)):
        bad = np.flatnonzero(rows != other)
        if len(bad):
            n = int(bad[0])
            result.add_error(
                f"{prefix}: octree chunks: {len(bad)} leaves hold a different vertex count "
                f"than the {what}, e.g. leaf {n}: {int(rows[n])} vs {int(other[n])}"
            )
    bad = np.flatnonzero(frags != cell_frags)
    if len(bad):
        n = int(bad[0])
        result.add_error(
            f"{prefix}: octree chunks: {len(bad)} leaves hold a different fragment count "
            f"than the cell table, e.g. leaf {n}: {int(frags[n])} vs {int(cell_frags[n])}"
        )
    total = int(rows.sum())
    evc = (lg.attrs.get("zarr_vectors_level") or {}).get("vertex_count")
    if evc is not None and int(evc) != total:
        result.add_error(f"{prefix}: metadata vertex_count={evc}, actual={total}")

    try:
        ids, manifests = read_object_manifest_rows(lg)
    except Exception:
        ids, manifests = np.zeros(0, np.int64), []
    for oid, mf in zip(ids.tolist(), manifests):
        for cc, f in mf:
            n = leaf_at.get(tuple(int(c) for c in cc))
            if n is None:
                result.add_error(f"{prefix}: obj {oid} refs {cc}, which is no leaf's anchor")
            elif not 0 <= int(f) < frags[n]:
                result.add_error(f"{prefix}: obj {oid} refs fragment_idx={f} of leaf {n}, "
                                 f"which has {int(frags[n])}")

    n_links = 0
    for d in list_link_deltas(lg):
        if d != 0:
            result.add_error(f"{prefix}: links[delta={d}] on a level with octree chunks "
                             f"(links between levels are not supported with them)")
    try:
        src, r_src, oth, r_oth = read_octree_link_cells(lg, table)
    except Exception as e:
        result.add_error(f"{prefix}: octree links unreadable: {e}")
    else:
        n_links = len(src)
        if (oth < 0).any():
            result.add_error(f"{prefix}: octree links: {int((oth < 0).sum())} name a leaf "
                             f"that is not stored")
        ok = oth >= 0
        for end, (leaf, row) in enumerate(((src, r_src), (oth, r_oth))):
            bad = ok & ((row < 0) | (row >= rows[np.where(ok, leaf, 0)]))
            if bad.any():
                result.add_error(f"{prefix}: octree links: {int(bad.sum())} name a row "
                                 f"past the end of their leaf (endpoint {end})")
        family = (lg.read_array_meta("links/0") or {}) if lg.array_exists("links/0") else {}
        if "num_physical_records" in family and int(family["num_physical_records"]) != n_links:
            result.add_error(f"{prefix}: octree links: num_physical_records="
                             f"{family['num_physical_records']} != {n_links} stored")

    try:
        codecs = lg.zarr_group[VERTICES].metadata.codecs if stored else []
        if codecs and type(codecs[0]).__name__ in ("ShardingCodec", "OctreeShardingCodec"):
            codecs = codecs[0].codecs
        if stored and not any(is_marker(c) for c in codecs):
            result.add_warning(
                f"{prefix}: octree chunks, but vertices lacks the {OCTREE_MARKER_CODEC} "
                f"codec: a reader without octree chunks would read its leaves as cells"
            )
    except Exception:
        pass
    result.add_pass(
        f"{prefix}: octree chunks: {n_leaves} leaves, {total} vertices, {n_links} links validated"
    )


def _check_octree_shards(lg: Any, table: Any, prefix: str, result: ValidationResult) -> None:
    """No leaf wider than the level's recorded cap, nor -- the level being
    sharded -- outside one shard of its arrays' current shard grid.

    ``to_octree`` fits the leaves to the shards, but the shard repackers
    (``shard_store``, ``reshard``) move cells unchanged and may be given
    a shard shape that splits leaves.  On a level converted with a cap
    (its octree block's ``max_leaf_cells``) that is an error; on one
    converted without (unsharded then, or by a ``to_octree`` before the
    cap) a warning.  Either way ``to_uniform`` then ``to_octree`` fits
    the leaves to the shards again.
    """
    from zarr_vectors.octree_chunks.convert import array_shard_grid

    if not len(table):
        return
    cap = table.max_leaf_cells
    sides = table.sides()
    if cap is not None and int(sides.max()) > int(cap):
        n = int(np.argmax(sides))
        result.add_error(
            f"{prefix}: octree chunks: {int((sides > cap).sum())} leaves are wider than the "
            f"level's max_leaf_cells={cap}, e.g. leaf {table.key(n)} ({int(sides[n])} cells per side)"
        )
    try:
        grid = array_shard_grid(lg)
    except Exception:
        grid = None
    if grid is None or grid[0] is None or len(grid[0]) != 3:
        return
    shard, origin = grid
    bad = table.straddling(shard, origin)
    if not len(bad):
        return
    n = int(bad[0])
    msg = (
        f"{prefix}: octree chunks: {len(bad)} leaves are not inside one shard of "
        f"{list(shard)} cells, e.g. leaf {table.key(n)} ({int(sides[n])} cells per side "
        f"at cell {tuple(table.anchors()[n].tolist())}), so its data sits in its anchor's "
        f"shard and per-shard writes and reads miss it; run zarr_vectors.octree_chunks."
        f"to_uniform, then to_octree, to fit the leaves to these shards"
    )
    if cap is not None:
        fitted = (f"shards of {list(table.shard_shape)} cells" if table.shard_shape is not None
                  else "no shards")
        result.add_error(f"{msg} (it was converted with leaves capped at {cap} cells for "
                         f"{fitted} and has been resharded since)")
    else:
        result.add_warning(f"{msg} (it was converted without a leaf cap)")


def _dense_index_problems(lg: Any) -> list[str]:
    """What is wrong with a dense object index's arrays (spec L1/L2)."""
    from zarr_vectors.core import dense_manifests as dense

    try:
        if not dense.is_dense(lg):
            return []
        meta = lg.read_array_meta("object_index") or {}
    except Exception:
        return []
    problems = []
    for path in (dense.SPANS_PATH, dense.BLOCKS_PATH, "object_index/object_ids"):
        if not lg.array_exists(path):
            problems.append(f"dense object_index is missing {path}")
    if lg.array_exists("object_index/manifests"):
        problems.append("dense object_index also has a vlen manifests array")
    if problems:
        return problems
    spans = np.asarray(lg.read_array(dense.SPANS_PATH))
    blocks_node = lg.zarr_group[dense.BLOCKS_PATH]
    blocks_shape = blocks_node.shape
    blocks_dtype = np.dtype(blocks_node.dtype)
    if spans.dtype != np.int64:
        problems.append(f"manifest_spans is {spans.dtype}; expected int64")
    if blocks_dtype.name not in dense.BLOCK_DTYPES:
        problems.append(
            f"manifest_blocks is {blocks_dtype}; expected one of "
            f"{', '.join(dense.BLOCK_DTYPES)}"
        )
    stamped = meta.get(dense.BLOCK_DTYPE_ATTR)
    if stamped is not None and stamped != blocks_dtype.name:
        problems.append(
            f"object_index says {dense.BLOCK_DTYPE_ATTR}={stamped!r}, but "
            f"manifest_blocks is {blocks_dtype}"
        )
    n, sid = int(meta.get("num_objects", spans.shape[0])), int(meta.get("sid_ndim", 0))
    if spans.ndim != 2 or spans.shape[1] != 2 or spans.shape[0] < n:
        problems.append(f"manifest_spans has shape {spans.shape}; expected ({n}, 2)")
        return problems
    if sid and blocks_shape[1:] != (sid + 1,):
        problems.append(
            f"manifest_blocks has shape {blocks_shape}; expected (n, {sid + 1})"
        )
    used = spans[:n]
    if (used[:, 1] < 0).any():
        problems.append("manifest_spans has a negative count")
    live = used[used[:, 1] > 0]
    if live.size and (
        (live[:, 0] < 0).any() or (live[:, 0] + live[:, 1] > blocks_shape[0]).any()
    ):
        problems.append("manifest_spans reaches past the end of manifest_blocks")
    return problems
