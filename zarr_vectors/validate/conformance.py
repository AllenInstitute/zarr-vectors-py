"""Level 4-5 conformance validation — geometry rules and multi-resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from zarr_vectors.constants import (
    GEOM_GRAPH,
    GEOM_LINE,
    GEOM_MESH,
    GEOM_POINT_CLOUD,
    GEOM_POLYLINE,
    GEOM_SKELETON,
    GEOM_STREAMLINE,
    LINKS_EXPLICIT,
    LINKS_IMPLICIT_BRANCHES,
    LINKS_IMPLICIT_SEQUENTIAL,
)
from zarr_vectors.core.arrays import list_chunk_keys, read_chunk_vertices
from zarr_vectors.core.store import (
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
)
from zarr_vectors.validate.structure import ValidationResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group

GEOMETRY_LINK_REQ: dict[str, set[str]] = {
    GEOM_POINT_CLOUD: set(),
    GEOM_LINE: {LINKS_IMPLICIT_SEQUENTIAL},
    GEOM_POLYLINE: {LINKS_IMPLICIT_SEQUENTIAL},
    GEOM_STREAMLINE: {LINKS_IMPLICIT_SEQUENTIAL},
    GEOM_GRAPH: {LINKS_EXPLICIT},
    GEOM_SKELETON: {LINKS_IMPLICIT_BRANCHES, LINKS_EXPLICIT},
    GEOM_MESH: {LINKS_EXPLICIT},
}


def validate_conformance(store_path: str | Path | Group) -> ValidationResult:
    """Level 4: verify geometry-specific conformance."""
    result = ValidationResult(level=4)

    try:
        root = open_store(store_path)
        meta = read_root_metadata(root)
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result

    geom_types = meta.geometry_types or []
    lc = meta.links_convention

    for gt in geom_types:
        req = GEOMETRY_LINK_REQ.get(gt)
        if req is None:
            result.add_warning(f"Unknown geometry type: '{gt}'")
            continue
        if not req:
            result.add_pass(f"'{gt}': no links convention required")
            continue
        if lc not in req:
            # Check if this is a composite store with a geometry index
            # (each type has its own namespaced link arrays, so the root
            # links_convention only applies to the primary geometry)
            levels = list_resolution_levels(root)
            is_composite = False
            if 0 in levels:
                try:
                    lg_check = get_resolution_level(root, 0)
                    if "geometry_index" in lg_check:
                        is_composite = True
                except Exception:
                    pass
            if is_composite:
                result.add_pass(
                    f"'{gt}': composite store — namespaced links (convention N/A)"
                )
            else:
                result.add_error(f"'{gt}' requires links in {req}, got '{lc}'")
        else:
            result.add_pass(f"'{gt}': links_convention '{lc}' valid")

    levels = list_resolution_levels(root)
    if 0 not in levels:
        result.add_warning("No level 0 for geometry checks")
        return result

    lg = get_resolution_level(root, 0)

    if GEOM_MESH in geom_types:
        # Check mesh link_width — in composite stores, look at links_mesh/
        found_mesh_links = False
        try:
            lmeta = lg.read_array_meta("links/0")
            lw = lmeta.get("link_width", 0)
            if lw >= 3:
                result.add_pass(f"Mesh link_width={lw}")
                found_mesh_links = True
            elif lw > 0:
                result.add_error(f"Mesh link_width={lw}, must be >= 3")
                found_mesh_links = True
        except Exception:
            pass
        if not found_mesh_links:
            # Try namespaced links_mesh/ (composite store)
            try:
                mesh_links_group = lg["links_mesh"]
                lw = mesh_links_group.attrs.to_dict().get("link_width", 0)
                if lw >= 3:
                    result.add_pass(f"Mesh link_width={lw} (namespaced)")
                elif lw > 0:
                    result.add_error(f"Mesh link_width={lw}, must be >= 3")
                else:
                    result.add_pass("Mesh links_mesh exists (composite)")
            except Exception:
                result.add_warning("Mesh geometry but no links metadata")

    if GEOM_POINT_CLOUD in geom_types and not any(
        gt in geom_types for gt in [GEOM_GRAPH, GEOM_SKELETON, GEOM_MESH]
    ):
        try:
            lg.read_array_meta("links/0")
            result.add_warning("Point cloud but links array exists")
        except Exception:
            result.add_pass("Point cloud: no links (correct)")

    return result


def validate_multiresolution(store_path: str | Path | Group) -> ValidationResult:
    """Level 5: verify multi-resolution pyramid conformance."""
    result = ValidationResult(level=5)

    try:
        root = open_store(store_path)
        meta = read_root_metadata(root)
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result

    ndim = meta.sid_ndim
    levels = sorted(list_resolution_levels(root))

    if len(levels) <= 1:
        result.add_pass("Single resolution level — no pyramid to validate")
        return result

    expected = list(range(len(levels)))
    if levels != expected:
        result.add_error(f"Levels {levels}, expected {expected}")
    else:
        result.add_pass(f"Levels {levels} contiguous")

    # Counts are of each level's COMPLETE content: on an additive level
    # (refinement "add") that is the sum over its chain, so the
    # non-increasing rule compares what a reader of each level sees.
    from zarr_vectors.core.refinement import level_chain, stored_object_ids

    own_counts: dict[int, int] = {}
    own_objects: dict[int, object] = {}
    for li in levels:
        try:
            lg = get_resolution_level(root, li)
            vc = lg.attrs.get("vertex_count")
            if vc is None:
                vc = 0
                for ck in list_chunk_keys(lg):
                    try:
                        gs = read_chunk_vertices(lg, ck, ndim=ndim)
                        vc += sum(len(g) for g in gs)
                    except Exception:
                        pass
            own_counts[li] = int(vc)
            own_objects[li] = stored_object_ids(lg)
        except Exception as e:
            result.add_error(f"resolution_{li}: {e}")

    prev_count: int | None = None
    prev_objects: int | None = None
    prev_ratio_product: int = 1
    for li in levels:
        if li not in own_counts:
            continue
        try:
            lg = get_resolution_level(root, li)
            attrs = lg.attrs
            chain = level_chain(root, li)
            vc = sum(own_counts.get(lv, 0) for lv in chain)
            id_sets = [own_objects.get(lv) for lv in chain]
            n_objects = (
                None if any(ids is None for ids in id_sets)
                else int(np.unique(np.concatenate(id_sets)).size)
                if id_sets else 0
            )
            if prev_objects is not None and n_objects is not None:
                if n_objects > prev_objects:
                    result.add_error(
                        f"resolution_{li}: {n_objects} objects > "
                        f"resolution_{li-1} ({prev_objects})"
                    )
            if n_objects is not None:
                prev_objects = n_objects
            if prev_count is not None:
                if vc > prev_count:
                    result.add_error(f"resolution_{li}: {vc} > resolution_{li-1} ({prev_count})")
                else:
                    result.add_pass(
                        f"resolution_{li}: {vc} verts "
                        f"({prev_count / max(vc, 1):.1f}x reduction)"
                    )
            prev_count = vc

            # Check bin_ratio is non-decreasing (in volume) across levels
            bin_ratio = attrs.get("bin_ratio")
            if bin_ratio is not None:
                ratio_product = 1
                for r in bin_ratio:
                    ratio_product *= int(r)
                if li > 0 and ratio_product < prev_ratio_product:
                    result.add_error(
                        f"resolution_{li}: bin_ratio volume {ratio_product} "
                        f"< previous level {prev_ratio_product}"
                    )
                prev_ratio_product = ratio_product

            # Check object_sparsity is in (0, 1]
            sparsity = attrs.get("object_sparsity", 1.0)
            if not (0.0 < sparsity <= 1.0):
                result.add_error(
                    f"resolution_{li}: object_sparsity={sparsity} out of range"
                )
        except Exception as e:
            result.add_error(f"resolution_{li}: {e}")

    return result
