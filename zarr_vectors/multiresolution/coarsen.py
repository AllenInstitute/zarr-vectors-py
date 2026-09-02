"""Multi-resolution pyramid construction orchestrator.

Two entry points (use one):

* ``build_pyramid(store, factors=[(cf_1, sf_1), ...])`` builds every
  coarser level in sequence, optionally emitting cross-level link
  arrays (``cross_level_storage="implicit"`` or ``"explicit"``).
* ``coarsen_level(store, source, target, coarsen_factor=..., sparsity_factor=...)``
  writes a single coarser level for callers that want manual control.

Both use the per-object pyramid: each surviving object's vertices are
aggregated into bin centroids (metavertices) that may be shared
between objects, and per-object OIDs are preserved across levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_MULTISCALE_LINKS,
    CAP_PRESERVED_OBJECT_IDS,
    CAP_SHARED_FRAGMENTS,
    COARSEN_PER_OBJECT,
    DEFAULT_CROSS_LEVEL_DEPTH,
    DEFAULT_CROSS_LEVEL_STORAGE,
    OBJECT_ATTRIBUTES,
    VERTICES,
    XLEVEL_EXPLICIT,
    XLEVEL_NONE,
    VALID_XLEVEL_STORAGE,
)
from zarr_vectors.core.arrays import (
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_all_object_manifests,
    read_chunk_vertices,
    read_links,
    read_object_attributes,
    write_chunk_vertices,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.metadata import (
    LevelMetadata,
    get_level_chunk_shape,
)
from zarr_vectors.core.store import (
    create_resolution_level,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import ArrayError, CoarseningError
from zarr_vectors.multiresolution.object_selection import apply_sparsity
from zarr_vectors.spatial.boundary import build_vertex_chunk_mapping
from zarr_vectors.spatial.chunking import assign_chunks
from zarr_vectors.typing import ChunkCoords


# ===================================================================
# Single-level coarsening
# ===================================================================

def coarsen_level(
    store_path: str | Path,
    source_level: int,
    target_level: int,
    *,
    coarsen_factor: float = 1.0,
    sparsity_factor: float = 1.0,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_storage: str = XLEVEL_NONE,
    method: str = COARSEN_PER_OBJECT,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Coarsen a single level and write it to the store.

    Per-object vertex aggregation with stable OIDs across levels.  A
    metavertex's source vertices may come from multiple source objects;
    the resulting metavertex appears in each of those objects' manifests
    at the coarser level.

    Args:
        store_path: Path to the zarr vectors store.
        source_level: Level to read from.
        target_level: Level to write to (must not exist).
        coarsen_factor: Per-object vertex aggregation factor (>= 1),
            expressed as a **ratio against the source level's bin**, not the
            root's: the target bin is ``source_level.bin_shape *
            coarsen_factor``, seeded from the root's effective bin at level 0.
            Factors therefore compound down a pyramid.  ``1.0`` is the identity
            (no aggregation).
        sparsity_factor: Object-dropping factor (≥ 1).  Survivors keep
            their OIDs; dropped objects leave empty manifest slots.
            ``1.0`` is the identity (no drop).
        chunk_scale_factor: Per-axis multiplier applied to the source
            level's ``chunk_shape`` to derive the target level's
            ``chunk_shape``.  ``1`` (default) keeps the chunk grid
            unchanged.  Scalar values apply uniformly to every axis;
            tuples set per-axis multipliers.  Each multiplier must be a
            positive integer (nested chunk grids).  When the resulting
            target chunk_shape differs from the root chunk_shape it is
            stamped on the target level's ``LevelMetadata.chunk_shape``;
            otherwise the target inherits from root.
        sparsity_strategy: Object selection strategy — ``"random"`` (core)
            or a name registered by ``zarr-vectors-tools``.
        sparsity_seed: Random seed.
        cross_level_storage: When called via ``build_pyramid`` this is
            threaded through to enable inline ``±1`` cross-level link
            emission.  Standalone callers should leave it at the
            ``"none"`` default.
        method: Coarsening method.  ``"per_object"`` (core default) uses the
            metavertex-binning pyramid; any other name is dispatched to a
            method registered by ``zarr-vectors-tools`` via
            :mod:`zarr_vectors.multiresolution.registry` (raises if that
            package is not installed).
        options: Extra keywords for a registered ``method``, forwarded
            verbatim.  A strategy's own knobs (a polyline coarsener's
            ``coarsen_mode``, a codec, a worker pool) were unreachable
            through this entry point, so a caller who wanted one had to
            bypass the dispatch and import the strategy directly —
            defeating the registry.  Ignored by the built-in
            ``"per_object"`` method, which takes no extra keywords; passing
            options with it raises rather than silently dropping them.

    Returns:
        Summary dict.  Always includes ``method``,
        ``preserves_object_ids``, ``vertex_count``.
    """
    kwargs = dict(
        store_path=store_path,
        source_level=source_level,
        target_level=target_level,
        coarsen_factor=coarsen_factor,
        sparsity_factor=sparsity_factor,
        chunk_scale_factor=chunk_scale_factor,
        sparsity_strategy=sparsity_strategy,
        sparsity_seed=sparsity_seed,
        cross_level_storage=cross_level_storage,
    )
    if method == COARSEN_PER_OBJECT:
        if options:
            raise CoarseningError(
                f"method='{COARSEN_PER_OBJECT}' takes no extra options; got "
                f"{sorted(options)}. Those keywords belong to a registered "
                f"strategy — pass method= as well."
            )
        return _per_object_coarsen(**kwargs)
    # Advanced coarsening methods live in zarr-vectors-tools.
    from zarr_vectors.multiresolution.registry import require_coarsen_strategy
    return require_coarsen_strategy(method)(**kwargs, **(options or {}))


def _per_object_coarsen(
    *,
    store_path: str | Path,
    source_level: int,
    target_level: int,
    coarsen_factor: float,
    sparsity_factor: float,
    chunk_scale_factor: int | tuple[int, ...] = 1,
    sparsity_strategy: str,
    sparsity_seed: int | None,
    cross_level_storage: str = XLEVEL_NONE,
) -> dict[str, Any]:
    """Per-object pyramid: aggregate within-bin source vertices into
    shared metavertices, preserving each surviving object's OID and
    its trajectory through the new metavertices.

    See the 12-step implementation sketch in the plan file
    ``Provenance-preserving pyramid: shared metavertices + ID-stable
    objects`` (`schema/zarr_vectors.linkml.yaml` schema captures the
    persistent metadata side).
    """
    root = open_store(str(store_path), mode="r+")
    root_meta = read_root_metadata(root)
    ndim = root_meta.sid_ndim

    # Source level's chunk_shape — may itself be a per-level override.
    try:
        src_level_meta = read_level_metadata(root, source_level)
    except Exception:
        src_level_meta = None
    source_chunk_shape = get_level_chunk_shape(root_meta, src_level_meta)

    # The bin ``coarsen_factor`` multiplies is the SOURCE LEVEL's, not the
    # root's, so factors compose per level: ``[2, 2, 2]`` bins at 2x, 4x and 8x
    # the root bin rather than 2x three times over.  Level 0 carries no
    # bin_shape of its own, so the root's effective bin seeds the chain.
    _src_bin = getattr(src_level_meta, "bin_shape", None) if src_level_meta else None
    base_bin = (
        tuple(float(b) for b in _src_bin) if _src_bin
        else root_meta.effective_bin_shape
    )

    # Target level's chunk_shape = source × chunk_scale_factor (per-axis).
    if isinstance(chunk_scale_factor, (tuple, list)):
        if len(chunk_scale_factor) != ndim:
            raise CoarseningError(
                f"chunk_scale_factor rank {len(chunk_scale_factor)} "
                f"!= sid_ndim {ndim}"
            )
        chunk_scale = tuple(int(r) for r in chunk_scale_factor)
    else:
        chunk_scale = tuple(int(chunk_scale_factor) for _ in range(ndim))
    if any(r < 1 for r in chunk_scale):
        raise CoarseningError(
            f"chunk_scale_factor must be positive integers per axis, "
            f"got {chunk_scale}"
        )
    target_chunk_shape = tuple(
        float(s) * int(r) for s, r in zip(source_chunk_shape, chunk_scale)
    )
    # The on-disk per-level chunk_shape field is omitted when the
    # target equals root (the implicit default).  Compare via float.
    target_chunk_shape_override: tuple[float, ...] | None
    if all(
        abs(t - r) < 1e-9
        for t, r in zip(target_chunk_shape, root_meta.chunk_shape)
    ):
        target_chunk_shape_override = None
    else:
        target_chunk_shape_override = target_chunk_shape
    chunk_shape = target_chunk_shape  # used for assign_chunks below

    src_group = get_resolution_level(root, source_level)

    # --- Step 0: read source manifests + vertex positions ----------------
    # Read source vertex positions, indexed by (chunk_coords, fragment_idx).
    src_fragment_positions: dict[tuple[ChunkCoords, int], npt.NDArray] = {}
    for cc in list_chunk_keys(src_group, VERTICES):
        try:
            fragments = read_chunk_vertices(src_group, cc, ndim=ndim)
        except ArrayError:
            continue
        for fragment_idx, fragment in enumerate(fragments):
            src_fragment_positions[(cc, fragment_idx)] = fragment

    src_has_objects = "object_index" in src_group
    if src_has_objects:
        src_manifests = read_all_object_manifests(src_group)
    else:
        # No object_index — treat the level as one implicit object whose
        # manifest enumerates every fragment in chunk-major order.
        implicit: list[tuple[ChunkCoords, int]] = []
        for cc in list_chunk_keys(src_group, VERTICES):
            fragment_idx = 0
            while (cc, fragment_idx) in src_fragment_positions:
                implicit.append((cc, fragment_idx))
                fragment_idx += 1
        src_manifests = [implicit] if implicit else []
    n_src_objects = len(src_manifests)
    if n_src_objects == 0:
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": 0,
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
        }

    # --- Step 1: drop a fraction of source objects ----------------------
    keep_oids: list[int]
    if sparsity_factor > 1.0 and n_src_objects > 1:
        keep_frac = 1.0 / sparsity_factor
        kept = apply_sparsity(
            n_src_objects, keep_frac, sparsity_strategy,
            seed=sparsity_seed,
            representative_points=None,
            bin_shape=base_bin,
        )
        keep_oids = sorted(int(o) for o in kept)
    else:
        keep_oids = list(range(n_src_objects))
    keep_set = set(keep_oids)

    # --- Step 2-3: build (source vertex → bin → metavertex) map ---------
    # Per-object ordered source-vertex positions (with their global index
    # in the flat source-vertex array).
    per_object_positions: dict[int, np.ndarray] = {}
    flat_positions: list[np.ndarray] = []
    flat_oid_of_v: list[int] = []
    next_global = 0
    for oid in keep_oids:
        manifest = src_manifests[oid]
        parts: list[np.ndarray] = []
        for cc, fragment_idx in manifest:
            fragment = src_fragment_positions.get((cc, fragment_idx))
            if fragment is None or len(fragment) == 0:
                continue
            parts.append(np.asarray(fragment, dtype=np.float32))
        if not parts:
            per_object_positions[oid] = np.zeros((0, ndim), dtype=np.float32)
            continue
        obj_positions = np.concatenate(parts, axis=0)
        per_object_positions[oid] = obj_positions
        flat_positions.append(obj_positions)
        flat_oid_of_v.extend([oid] * obj_positions.shape[0])
        next_global += obj_positions.shape[0]

    if not flat_positions:
        # Surviving objects had no vertices.  Write an empty level.
        _write_empty_preserve_level(
            root, source_level, target_level,
            base_bin=base_bin,
            root_bin=root_meta.effective_bin_shape,
            coarsen_factor=coarsen_factor,
            sparsity_factor=sparsity_factor,
            inherited_num_objects=n_src_objects,
        )
        return {
            "vertex_count": 0,
            "object_count": 0,
            "objects_kept": len(keep_oids),
            "method": COARSEN_PER_OBJECT,
            "preserves_object_ids": True,
            "shared_fragments": True,
        }

    all_pos = np.concatenate(flat_positions, axis=0)

    # Target bin shape: source bin_shape × coarsen_factor.
    # Target bin shape: the SOURCE level's bin_shape x coarsen_factor, so the
    # factor is a per-level ratio and successive levels compound.
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)

    # Compute per-vertex bin coords: (N, ndim) int64.
    bin_shape_arr = np.asarray(target_bin_shape, dtype=np.float64)
    bin_coords = np.floor(all_pos / bin_shape_arr).astype(np.int64)
    # Combine each bin coord tuple into a single sort-key for np.unique.
    bin_keys = np.ascontiguousarray(bin_coords).view(
        np.dtype((np.void, bin_coords.dtype.itemsize * bin_coords.shape[1]))
    ).ravel()
    _, inverse = np.unique(bin_keys, return_inverse=True)
    inverse = inverse.astype(np.int64, copy=False)
    n_metavertices = int(inverse.max()) + 1 if inverse.size > 0 else 0

    # --- Step 3 (continued): centroid per bin --------------------------
    meta_positions = np.zeros((n_metavertices, ndim), dtype=np.float32)
    bin_counts = np.zeros(n_metavertices, dtype=np.int64)
    np.add.at(meta_positions, inverse, all_pos)
    np.add.at(bin_counts, inverse, 1)
    meta_positions /= bin_counts[:, None]

    # --- Step 4: chunk-assign metavertices ------------------------------
    chunk_assignments = assign_chunks(meta_positions, chunk_shape)

    # --- Step 5: per-chunk fragment layout (one fragment per metavertex) ------------
    metavertex_to_ref: dict[int, tuple[ChunkCoords, int]] = {}
    per_chunk_groups: dict[ChunkCoords, list[np.ndarray]] = {}
    for cc, indices in sorted(chunk_assignments.items()):
        # ``indices`` are metavertex indices that fell in this chunk.
        for fragment_idx, mv_idx in enumerate(indices.tolist()):
            metavertex_to_ref[int(mv_idx)] = (cc, fragment_idx)
            per_chunk_groups.setdefault(cc, []).append(
                meta_positions[mv_idx:mv_idx + 1]
            )

    # --- Step 6: write per-chunk fragments --------------------------
    arrays_present = [VERTICES, "object_index"] if src_has_objects else [VERTICES]
    level_meta_initial = LevelMetadata(
        level=target_level,
        vertex_count=int(n_metavertices),
        arrays_present=arrays_present,
        bin_shape=target_bin_shape,
        # Fold-change relative to LEVEL 0, not to the source level: this is
        # what becomes the NGFF ``scale`` transform. With per-level coarsen
        # factors the two differ — [2, 2] is ratio 2 then 4 — so it has to be
        # derived from the bin shapes rather than echoing coarsen_factor.
        bin_ratio=tuple(
            max(1, int(round(float(t) / float(r))))
            for t, r in zip(target_bin_shape, root_meta.effective_bin_shape)
        ),
        chunk_shape=target_chunk_shape_override,
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=src_has_objects,
        inherited_num_objects=n_src_objects if src_has_objects else 0,
        shared_fragments=True,
    )
    level_group = create_resolution_level(root, target_level, level_meta_initial)
    create_vertices_array(level_group, dtype="float32")
    if src_has_objects:
        create_object_index_array(level_group)

    for cc, groups in sorted(per_chunk_groups.items()):
        write_chunk_vertices(level_group, cc, groups, dtype=np.float32)

    # --- Step 7: emit per-object manifests ------------------------------
    # We need to map each source vertex back to its metavertex_index.
    # Walk per-object slices of the flat ``inverse`` array.
    cursor = 0
    new_manifests: dict[int, list[tuple[ChunkCoords, int]]] = {}
    for oid in keep_oids:
        n = per_object_positions[oid].shape[0]
        if n == 0:
            cursor += 0
            new_manifests[oid] = []
            continue
        mv_seq = inverse[cursor:cursor + n].tolist()
        cursor += n
        # Deduplicate consecutive duplicates while preserving order.
        manifest: list[tuple[ChunkCoords, int]] = []
        prev = -1
        for mv_idx in mv_seq:
            if mv_idx == prev:
                continue
            prev = mv_idx
            manifest.append(metavertex_to_ref[int(mv_idx)])
        new_manifests[oid] = manifest

    # --- Step 9: emit object_index (gap-fill for dropped OIDs) ----------
    if src_has_objects:
        write_object_index(
            level_group, new_manifests, sid_ndim=ndim,
            total_objects=n_src_objects,
        )

    # --- Step 10: per-object attributes with present_mask ---------------
    src_obj_attr_group_name = f"{OBJECT_ATTRIBUTES}"
    if src_obj_attr_group_name in src_group:
        src_obj_attr_group = src_group[src_obj_attr_group_name]
        # children() covers both layouts: legacy per-chunk-array groups
        # under group_keys() and standalone arrays under array_keys().
        attr_names = src_obj_attr_group.children()
    else:
        attr_names = []
    for attr_name in attr_names:
        try:
            src_data = read_object_attributes(src_group, attr_name)
        except ArrayError:
            continue
        # Dense (O, C) or (O,) padded to the inherited OID space, with
        # rows for survivors copied over.  Layout matches the source's
        # OID space (which already equals n_src_objects).
        out_data = np.zeros_like(src_data)
        for oid in keep_oids:
            if oid < len(src_data):
                out_data[oid] = src_data[oid]
        mask = np.zeros(n_src_objects, dtype=np.uint8)
        for oid in keep_oids:
            mask[oid] = 1
        create_object_attributes_array(level_group, attr_name)
        write_object_attributes(level_group, attr_name, out_data, present_mask=mask)

    # --- Step 12: stamp root capability tokens --------------------------
    if src_has_objects:
        _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)
    _stamp_root_capability(root, CAP_SHARED_FRAGMENTS)

    # --- Step 13: emit inline ±1 cross-level link arrays ----------------
    # Must stay after step 6's create_resolution_level: the anchor's
    # scale factors are read off the target level's metadata.
    if cross_level_storage != XLEVEL_NONE and n_metavertices > 0:
        _emit_inline_cross_level_links(
            root,
            src_group=src_group,
            level_group=level_group,
            source_level=source_level,
            ndim=ndim,
            bin_shape_arr=bin_shape_arr,
            bin_keys=bin_keys,
            coarse_chunk_assignments_mv=chunk_assignments,
            storage=cross_level_storage,
        )

    return {
        "vertex_count": int(n_metavertices),
        "object_count": len(keep_oids),
        "objects_kept": len(keep_oids),
        "source_objects": n_src_objects,
        "method": COARSEN_PER_OBJECT,
        "preserves_object_ids": True,
        "shared_fragments": True,
    }


def _emit_inline_cross_level_links(
    root,
    *,
    src_group,
    level_group,
    source_level: int,
    ndim: int,
    bin_shape_arr: npt.NDArray[np.float64],
    bin_keys: npt.NDArray,
    coarse_chunk_assignments_mv: dict[ChunkCoords, npt.NDArray[np.int64]],
    storage: str,
) -> None:
    """Emit the ``±1`` link families for one coarsen step.

    Re-walks the source level in chunk-major order, re-bins each
    vertex against ``bin_shape_arr``, and looks up the matching
    metavertex via the ``bin_key`` ↔ ``mv_idx`` map implicit in
    ``np.unique(bin_keys, return_inverse=inverse)``.  Translates
    metavertex IDs to chunk-major-flat coarse indices via the
    just-written coarse-level chunks, then dispatches to
    :func:`_write_cross_level_edges`.

    ORDERING CONSTRAINT: the target level must already exist on disk
    with its ``LevelMetadata`` — including any ``chunk_shape`` override
    — before this runs.  ``write_links`` derives ``(r_src, r_trg)`` via
    ``arrays._derive_level_scales``, which falls back to ALL-ONES when it
    cannot read the target level.  All-ones is right only for a pyramid
    built at the default ``chunk_scale_factor=1``; on a scaled pyramid it
    silently mis-anchors every cross-level record.  Callers must keep
    this after the ``create_resolution_level`` for ``source_level + 1``
    (step 6 of :func:`_per_object_coarsen`).
    """
    # bin_key_bytes → mv_idx (bin-key-ordered, matches np.unique output).
    unique_keys = np.unique(bin_keys)
    bin_key_to_mv: dict[bytes, int] = {
        bytes(k): i for i, k in enumerate(unique_keys)
    }

    # mv_idx → chunk-major-flat coarse index.
    coarse_chunk_assignments, n_coarse = _reconstruct_chunk_assignments(
        level_group, ndim,
    )
    mv_to_coarse_global: dict[int, int] = {}
    for cc, mv_indices_for_chunk in sorted(coarse_chunk_assignments_mv.items()):
        for local_vg, mv_idx in enumerate(mv_indices_for_chunk.tolist()):
            mv_to_coarse_global[int(mv_idx)] = int(
                coarse_chunk_assignments[cc][local_vg]
            )

    # Build fine→coarse parent[] by re-walking source in chunk-major order.
    fine_chunk_assignments, n_fine = _reconstruct_chunk_assignments(
        src_group, ndim,
    )
    parent = np.full(n_fine, -1, dtype=np.int64)
    cursor = 0
    key_dtype = np.dtype((
        np.void, int(bin_shape_arr.shape[0]) * np.dtype(np.int64).itemsize,
    ))
    for cc in list_chunk_keys(src_group, VERTICES):
        try:
            fragments = read_chunk_vertices(
                src_group, cc, dtype=np.float32, ndim=ndim,
            )
        except ArrayError:
            continue
        for fragment in fragments:
            n_local = int(fragment.shape[0])
            if n_local == 0:
                continue
            local_bins = np.floor(
                np.asarray(fragment, dtype=np.float32) / bin_shape_arr,
            ).astype(np.int64)
            local_keys = np.ascontiguousarray(local_bins).view(key_dtype).ravel()
            for j in range(n_local):
                mv = bin_key_to_mv.get(bytes(local_keys[j]))
                if mv is not None:
                    parent[cursor + j] = mv_to_coarse_global[int(mv)]
            cursor += n_local

    _write_cross_level_edges(
        root,
        fine_level=source_level,
        delta=1,
        fine_chunk_assignments=fine_chunk_assignments,
        coarse_chunk_assignments=coarse_chunk_assignments,
        n_fine=n_fine,
        n_coarse=n_coarse,
        parent=parent,
        sid_ndim=ndim,
        storage=storage,
    )


def _write_empty_preserve_level(
    root,
    source_level: int,
    target_level: int,
    *,
    base_bin: tuple[float, ...],
    root_bin: tuple[float, ...],
    coarsen_factor: float,
    sparsity_factor: float,
    inherited_num_objects: int,
) -> None:
    """Write an empty ID-preserving level when no surviving object has vertices.

    ``base_bin`` is the SOURCE level's bin (what ``coarsen_factor`` multiplies);
    ``root_bin`` is level 0's, needed for the level-0-relative ``bin_ratio``.
    """
    ndim = len(base_bin)
    target_bin_shape = tuple(float(b) * float(coarsen_factor) for b in base_bin)
    level_meta = LevelMetadata(
        level=target_level,
        vertex_count=0,
        arrays_present=[VERTICES, "object_index"],
        bin_shape=target_bin_shape,
        # Level-0-relative, as above — it is the NGFF scale, not the per-level
        # factor.
        bin_ratio=tuple(
            max(1, int(round(float(t) / float(r))))
            for t, r in zip(target_bin_shape, root_bin)
        ),
        object_sparsity=(1.0 / sparsity_factor),
        coarsening_method=COARSEN_PER_OBJECT,
        parent_level=source_level,
        preserves_object_ids=True,
        inherited_num_objects=inherited_num_objects,
        shared_fragments=True,
    )
    level_group = create_resolution_level(root, target_level, level_meta)
    create_vertices_array(level_group, dtype="float32")
    create_object_index_array(level_group)
    # Empty object_index with the inherited size — all manifests are [].
    write_object_index(
        level_group, {}, sid_ndim=ndim,
        total_objects=inherited_num_objects,
    )
    _stamp_root_capability(root, CAP_PRESERVED_OBJECT_IDS)


def _stamp_root_capability(root_group, cap: str) -> None:
    """Add ``cap`` to root metadata's ``format_capabilities`` (idempotent)."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    caps = list(zv.get("format_capabilities", []))
    if cap not in caps:
        caps.append(cap)
        zv["format_capabilities"] = caps
        root_group.attrs.update({"zarr_vectors": zv})


def _stamp_root_cross_level(
    root_group, *, depth: int, storage: str,
) -> None:
    """Persist cross_level_depth/cross_level_storage on root metadata."""
    attrs = root_group.attrs.to_dict()
    zv = attrs.get("zarr_vectors", {})
    zv["cross_level_depth"] = int(depth)
    zv["cross_level_storage"] = storage
    root_group.attrs.update({"zarr_vectors": zv})


def _reconstruct_chunk_assignments(
    level_group, ndim: int,
) -> tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]:
    """Rebuild ``{chunk_coords: vertex_indices}`` from on-disk vertex chunks.

    The "vertex index" assigned to each vertex is the position it would
    occupy in a flat enumeration that walks chunks in
    ``list_chunk_keys`` order and concatenates each chunk's vertex
    groups in order.  This matches the convention used by
    ``build_vertex_chunk_mapping`` for in-memory edge partitioning.

    Returns the assignments dict and the total vertex count.
    """
    chunk_keys = list_chunk_keys(level_group, VERTICES)
    assignments: dict[ChunkCoords, npt.NDArray[np.int64]] = {}
    cursor = 0
    for cc in chunk_keys:
        try:
            fragments = read_chunk_vertices(level_group, cc, ndim=ndim)
        except ArrayError:
            continue
        n = sum(int(fragment.shape[0]) for fragment in fragments)
        if n == 0:
            continue
        assignments[cc] = np.arange(cursor, cursor + n, dtype=np.int64)
        cursor += n
    return assignments, cursor


def _decode_parent_from_plus_one(
    fine_lg,
    *,
    fine_assn: dict[ChunkCoords, npt.NDArray[np.int64]],
    coarse_assn: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_fine: int,
) -> npt.NDArray[np.int64] | None:
    """Decode a fine→coarse ``parent`` array from the already-written ``+1`` family.

    Reads every ``links/<+1>/<offsets>/`` record at the fine level and
    converts each ``(chunk, local_idx)`` endpoint to a global flat index
    via the supplied chunk-assignment dicts.  Endpoint 0 is the fine side
    and endpoint 1 the coarse side — :func:`write_links` keeps that order
    for ``delta != 0``.  Intra- and cross-chunk records arrive together:
    they differ only by which offsets array holds them, and ``read_links``
    already re-anchored each one back to absolute chunk coords.  Returns
    ``None`` when the family is absent or empty.
    """
    parent = np.full(n_fine, -1, dtype=np.int64)
    found_any = False

    try:
        records = read_links(fine_lg, delta=1)
    except (ArrayError, KeyError):
        records = []
    for (cc_s, vi_s), (cc_t, vi_t) in records:
        parent[int(fine_assn[cc_s][vi_s])] = int(coarse_assn[cc_t][vi_t])
        found_any = True

    return parent if found_any else None


def _finalize_cross_level_for_store(
    store_path: str | Path,
    *,
    cross_level_depth: int,
    cross_level_storage: str,
) -> None:
    """Persist root cross-level metadata and emit ``±N`` (N ≥ 2) link arrays.

    Adjacent ``±1`` arrays are emitted inline during coarsening (see
    :func:`_emit_inline_cross_level_links`).  This finalize pass walks
    every adjacent (fine, coarse) level pair, decodes the on-disk
    ``+1`` parent map back into a flat fine→coarse array, then composes
    step-by-step to produce ``+N``/``-N`` link arrays for N ≥ 2 up to
    ``cross_level_depth``.

    ``cross_level_depth=-1`` means "walk all available level pairs".
    """
    root = open_store(str(store_path), mode="r+")
    _stamp_root_cross_level(
        root, depth=cross_level_depth, storage=cross_level_storage,
    )
    if cross_level_storage == XLEVEL_NONE or cross_level_depth == 0:
        return

    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    levels = sorted(list_resolution_levels(root))
    if len(levels) < 2:
        return

    _stamp_root_capability(root, CAP_MULTISCALE_LINKS)

    # Build per-level chunk_assignments + total counts once.
    per_level: dict[int, tuple[dict[ChunkCoords, npt.NDArray[np.int64]], int]] = {}
    for lvl in levels:
        lg = get_resolution_level(root, lvl)
        per_level[lvl] = _reconstruct_chunk_assignments(lg, ndim)

    max_delta = (
        max(levels) - min(levels)
        if cross_level_depth == -1
        else int(cross_level_depth)
    )
    if max_delta < 2:
        return  # +1/-1 was already emitted inline

    # Cache each adjacent (fine_level, fine_level+1) parent array.
    adjacent_parent: dict[int, npt.NDArray[np.int64]] = {}
    for fine_level in levels[:-1]:
        coarse_level = fine_level + 1
        if coarse_level not in per_level:
            continue
        fine_assn, n_fine = per_level[fine_level]
        coarse_assn, _ = per_level[coarse_level]
        if n_fine == 0:
            continue
        fine_lg = get_resolution_level(root, fine_level)
        parent = _decode_parent_from_plus_one(
            fine_lg,
            fine_assn=fine_assn,
            coarse_assn=coarse_assn,
            n_fine=n_fine,
        )
        if parent is not None:
            adjacent_parent[fine_level] = parent

    # Compose deeper-delta parents and emit.
    for fine_level in levels[:-1]:
        if fine_level not in adjacent_parent:
            continue
        fine_assn, n_fine = per_level[fine_level]
        parent = adjacent_parent[fine_level].copy()
        for step in range(2, max_delta + 1):
            coarse_level = fine_level + step
            if coarse_level not in per_level:
                break
            inter_level = coarse_level - 1
            if inter_level not in adjacent_parent:
                break
            inter_parent = adjacent_parent[inter_level]
            coarse_assn, n_coarse = per_level[coarse_level]
            if n_coarse == 0:
                break

            composed = np.full(n_fine, -1, dtype=np.int64)
            valid = parent >= 0
            composed[valid] = inter_parent[parent[valid]]
            parent = composed
            if not np.any(parent >= 0):
                break

            _write_cross_level_edges(
                root,
                fine_level=fine_level,
                delta=step,
                fine_chunk_assignments=fine_assn,
                coarse_chunk_assignments=coarse_assn,
                n_fine=n_fine,
                n_coarse=n_coarse,
                parent=parent,
                sid_ndim=ndim,
                storage=cross_level_storage,
            )


def _write_cross_level_edges(
    root_group,
    *,
    fine_level: int,
    delta: int,
    fine_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    coarse_chunk_assignments: dict[ChunkCoords, npt.NDArray[np.int64]],
    n_fine: int,
    n_coarse: int,
    parent: npt.NDArray[np.int64],
    sid_ndim: int,
    storage: str,
) -> None:
    """Materialize ``delta``-step cross-level edges between two adjacent levels.

    ``parent[i]`` is the metanode index in the coarser level that fine
    vertex ``i`` belongs to.  The cross-level edges are trivially
    ``(i, parent[i])`` for each fine vertex.

    Writes the ``+delta`` family under the fine level.  When
    ``storage='explicit'`` also writes the matching ``-delta`` family
    under the coarse level by swapping endpoint roles.

    Records go to :func:`write_links` in global ``(chunk_coords,
    vertex_idx)`` form; it routes each one to the offsets array for the
    gap between its endpoints, so intra- and cross-chunk edges are one
    call.  Crucially it anchors that gap through
    :func:`~zarr_vectors.spatial.boundary.anchor_chunk` using the two
    levels' real ``chunk_shape`` scales, which is the only correct way
    to difference chunk coords when ``chunk_scale_factor > 1`` gives the
    levels different chunk grids.  Both calls therefore require the
    level they reference across to already carry its metadata — see the
    ordering note on :func:`_emit_inline_cross_level_links`.
    """
    if storage == XLEVEL_NONE or delta == 0:
        return

    # Drop orphaned fine vertices (parent < 0) before building edges.
    valid_mask = parent >= 0
    if not np.any(valid_mask):
        return
    fine_global = np.flatnonzero(valid_mask).astype(np.int64)
    parent_valid = parent[valid_mask].astype(np.int64)

    # Build chunk-mapping tables for both levels.
    fine_vchunks, fine_vlocal, fine_chunk_list = build_vertex_chunk_mapping(
        fine_chunk_assignments, n_fine, sorted(fine_chunk_assignments.keys()),
    )
    coarse_vchunks, coarse_vlocal, coarse_chunk_list = build_vertex_chunk_mapping(
        coarse_chunk_assignments, n_coarse, sorted(coarse_chunk_assignments.keys()),
    )

    fine_eps = [
        (fine_chunk_list[int(fine_vchunks[i])], int(fine_vlocal[i]))
        for i in fine_global.tolist()
    ]
    coarse_eps = [
        (coarse_chunk_list[int(coarse_vchunks[i])], int(coarse_vlocal[i]))
        for i in parent_valid.tolist()
    ]

    # Endpoint 0 leads and stays at the owning level, so each call's
    # anchor uses that level's scale as ``r_src``.  ``directed=True``:
    # fine→coarse parenthood is data, not an undirected pair.
    fine_lg = get_resolution_level(root_group, fine_level)
    write_links(
        fine_lg,
        [[f, c] for f, c in zip(fine_eps, coarse_eps)],
        sid_ndim,
        delta=delta,
        link_width=2,
        directed=True,
    )

    if storage == XLEVEL_EXPLICIT:
        # Mirror at the coarse level under -delta: the coarse endpoint
        # leads, so the anchor's scales swap with it.  No re-partitioning
        # by hand — write_links re-derives the offsets from the coarse
        # grid, which is what the fine-side split cannot speak to.
        coarse_lg = get_resolution_level(root_group, fine_level + delta)
        write_links(
            coarse_lg,
            [[c, f] for f, c in zip(fine_eps, coarse_eps)],
            sid_ndim,
            delta=-delta,
            link_width=2,
            directed=True,
        )


# ===================================================================
# Full pyramid builder
# ===================================================================

def build_pyramid(
    store_path: str | Path,
    *,
    factors: list[tuple[float, float]],
    chunk_scale_factors: list[int | tuple[int, ...]] | None = None,
    sparsity_strategy: str = "random",
    sparsity_seed: int | None = None,
    cross_level_depth: int = DEFAULT_CROSS_LEVEL_DEPTH,
    cross_level_storage: str = DEFAULT_CROSS_LEVEL_STORAGE,
    method: str = COARSEN_PER_OBJECT,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a multi-resolution pyramid for an existing store.

    Pass ``factors=[(coarsen_2, sparsity_3), ...]`` where ``factors[i]``
    is applied to produce level ``i+1`` from level ``i``.  Either factor
    at ``1.0`` opts out of that axis.  Uses the per-object pyramid:
    each surviving object's vertices are aggregated into bin centroids
    (metavertices); metavertices may be shared between objects and OIDs
    are preserved across levels.

    Args:
        store_path: Path to the store with level 0.
        factors: List of ``(coarsen_factor, sparsity_factor)`` tuples,
            one per coarser level.  ``coarsen_factor`` is a **per-level ratio
            against the level below**, so factors compound: ``[(2, 1), (2, 1),
            (2, 1)]`` bins at 2x, 4x and 8x the root bin.
        chunk_scale_factors: Optional per-level multipliers applied to
            the source level's ``chunk_shape`` to derive each target
            level's ``chunk_shape``.  Aligned with ``factors`` (same
            length).  Each entry is either a scalar int (uniform per
            axis) or a per-axis tuple.  ``None`` (default) means
            all-ones: every level inherits root ``chunk_shape``.
        sparsity_strategy: Object selection strategy — ``"random"`` (core)
            or a name registered by ``zarr-vectors-tools``.
        sparsity_seed: Random seed.
        method: Coarsening method; ``"per_object"`` (core default) or a
            ``zarr-vectors-tools``-registered method (see
            :func:`coarsen_level`).
        cross_level_depth: Maximum absolute level delta for materialized
            cross-pyramid-level link arrays.  ``0`` = none, ``N`` = up
            to ``±N`` per pair (or ``+N`` only when
            ``cross_level_storage='implicit'``), ``-1`` = walk all
            available level pairs.  Default ``1``.
        cross_level_storage: ``"none"`` / ``"implicit"`` / ``"explicit"``.
            ``"explicit"`` materializes both ``+N`` (at the finer level)
            and ``-N`` (at the coarser level); ``"implicit"`` writes
            only ``+N``.  Default ``"explicit"``.

    Returns:
        Summary dict.
    """
    if cross_level_storage not in VALID_XLEVEL_STORAGE:
        raise ValueError(
            f"cross_level_storage={cross_level_storage!r} not in "
            f"{sorted(VALID_XLEVEL_STORAGE)}"
        )
    if cross_level_depth < -1:
        raise ValueError(
            f"cross_level_depth must be ≥ -1 (got {cross_level_depth})"
        )
    if chunk_scale_factors is not None and len(chunk_scale_factors) != len(factors):
        raise ValueError(
            f"chunk_scale_factors length {len(chunk_scale_factors)} != "
            f"factors length {len(factors)}",
        )

    summaries: list[dict[str, Any]] = []
    for i, fac in enumerate(factors):
        if isinstance(fac, (tuple, list)) and len(fac) == 2:
            cf, sf = float(fac[0]), float(fac[1])
        else:
            raise ValueError(
                f"factors[{i}] must be a (coarsen_factor, sparsity_factor) "
                f"tuple; got {fac!r}"
            )
        chunk_scale = (
            chunk_scale_factors[i] if chunk_scale_factors is not None else 1
        )
        summaries.append(coarsen_level(
            store_path,
            source_level=i,
            target_level=i + 1,
            coarsen_factor=cf,
            sparsity_factor=sf,
            chunk_scale_factor=chunk_scale,
            sparsity_strategy=sparsity_strategy,
            sparsity_seed=sparsity_seed,
            cross_level_storage=cross_level_storage,
            method=method,
            options=options,
        ))

    # Compose deeper-delta cross-level links from the inline-emitted +1
    # arrays.  Also stamps root cross-level metadata + the multiscale
    # links capability.
    _finalize_cross_level_for_store(
        store_path,
        cross_level_depth=cross_level_depth,
        cross_level_storage=cross_level_storage,
    )

    return {
        "levels_created": len(summaries),
        "level_specs": summaries,
        # The method actually used, not the built-in: this reported
        # "per_object" even when every level had been produced by a
        # registered strategy, so the one field naming what built the
        # pyramid was wrong exactly when it mattered.
        "method": method,
        "cross_level_depth": cross_level_depth,
        "cross_level_storage": cross_level_storage,
    }
