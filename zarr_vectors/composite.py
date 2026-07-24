"""Composite stores — multiple geometry types in one zarr vectors store.

A single ``.zarrvectors`` store can hold points, streamlines, meshes,
and graphs simultaneously. Each geometry type's data is stored under
a named namespace within the resolution level, tracked by a
``geometry_index`` that maps geometry types to their array ranges.

Usage::

    from zarr_vectors.composite import add_geometry, read_composite

    # Start with a point cloud
    write_points("brain.zarrvectors", positions, chunk_shape=(100, 100, 100))

    # Add a graph to the same store
    add_geometry("brain.zarrvectors", "graph", positions=nodes, edges=edges)

    # Read all geometries
    composite = read_composite("brain.zarrvectors")
    print(composite.keys())  # ['point_cloud', 'graph']
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    GEOM_GRAPH,
    GEOM_LINE,
    GEOM_MESH,
    GEOM_POINT_CLOUD,
    GEOM_SKELETON,
    GEOM_STREAMLINE,
    RESOLUTION_PREFIX,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_object_index_array,
    create_vertices_array,
    list_chunk_keys,
    read_chunk_vertices,
    write_chunk_vertices,
    write_object_index,
)
from zarr_vectors.core.metadata import LevelMetadata, RootMetadata
from zarr_vectors.core.store import (
    FsGroup,
    _create_or_open_store,
    _ensure_root_metadata_for_write,
    create_resolution_level,
    get_resolution_level,
    open_store,
    read_root_metadata,
)
from zarr_vectors.spatial.chunking import assign_chunks, compute_bounds
from zarr_vectors.typing import ChunkCoords, ObjectManifest

if TYPE_CHECKING:
    from zarr_vectors.core.store import ReadSource


# ===================================================================
# Geometry index
# ===================================================================

def _read_geometry_index(level_group: FsGroup) -> dict[str, dict[str, Any]]:
    """Read the geometry index from a level group.

    Returns:
        ``{geometry_type: {"vertex_range": [start, end], ...}}``
    """
    try:
        gi_group = level_group["geometry_index"]
        return gi_group.attrs.to_dict()
    except Exception:
        return {}


def _write_geometry_index(
    level_group: FsGroup,
    index: dict[str, dict[str, Any]],
) -> None:
    """Write/update the geometry index on a level group."""
    gi_group = level_group.require_group("geometry_index")
    gi_group.attrs.update(index)


def _update_root_geometry_types(
    root: FsGroup,
    new_type: str,
) -> None:
    """Add a geometry type to the root metadata if not already present."""
    attrs = root.attrs.to_dict()
    zv = attrs.get("zarr_vectors", attrs)
    current_types = zv.get("geometry_types", [])
    if new_type not in current_types:
        current_types.append(new_type)
        zv["geometry_types"] = current_types
        if "zarr_vectors" in attrs:
            attrs["zarr_vectors"] = zv
        root.attrs.update(attrs)


# ===================================================================
# Add geometry to existing store
# ===================================================================

def add_geometry(
    store_path: str | Path,
    geometry_type: str,
    *,
    positions: npt.NDArray[np.floating] | None = None,
    edges: npt.NDArray[np.integer] | None = None,
    faces: npt.NDArray[np.integer] | None = None,
    polylines: list[npt.NDArray[np.floating]] | None = None,
    vertex_attributes: dict[str, npt.NDArray] | None = None,
    level: int = 0,
    # Deprecated alias:
    attributes: dict[str, npt.NDArray] | None = None,
) -> dict[str, Any]:
    """Add a geometry type to an existing zarr vectors store.

    The new geometry's vertex data is written to a namespaced array
    ``vertices_<geometry_type>/`` within the same resolution level.
    A geometry index tracks which arrays belong to which type.

    Args:
        store_path: Path to an existing store.
        geometry_type: One of ``"point_cloud"``, ``"graph"``,
            ``"skeleton"``, ``"mesh"``, ``"streamline"``, ``"line"``.
        positions: ``(N, D)`` vertex positions (for points, graphs, meshes).
        edges: ``(E, 2)`` edge array (for graphs/skeletons).
        faces: ``(F, L)`` face array (for meshes).
        polylines: List of ``(N_k, D)`` arrays (for streamlines).
        vertex_attributes: Per-vertex attributes ``{name: array}``.
            (Spec name; replaces ``attributes``.)
        level: Resolution level to add to (default 0).

    Returns:
        Summary dict with ``vertex_count``, ``geometry_type``.

    Raises:
        ValueError: If the geometry type requires data not provided.
    """
    # Back-compat: accept the legacy `attributes` kwarg.
    if attributes is not None:
        if vertex_attributes is not None:
            raise TypeError(
                "got both `attributes` and `vertex_attributes`; "
                "pass only `vertex_attributes`."
            )
        import warnings
        warnings.warn(
            "`attributes` is deprecated; use `vertex_attributes`.",
            DeprecationWarning, stacklevel=2,
        )
        vertex_attributes = attributes

    store_path = Path(store_path)
    sample_points: npt.NDArray | None = None
    if positions is not None:
        sample_points = np.asarray(positions)
    elif polylines:
        sample_points = np.concatenate([np.asarray(p) for p in polylines], axis=0)

    create_kwargs: dict[str, Any] = {}
    if sample_points is not None and sample_points.size > 0:
        inferred_bounds = compute_bounds(sample_points)
        create_kwargs["bounds"] = (
            inferred_bounds[0].tolist(),
            inferred_bounds[1].tolist(),
        )
        create_kwargs["ndim"] = int(sample_points.shape[1])

    root = _create_or_open_store(str(store_path), **create_kwargs)

    if sample_points is None or sample_points.size == 0:
        meta = read_root_metadata(root)
    else:
        meta = _ensure_root_metadata_for_write(
            root,
            inferred_ndim=int(sample_points.shape[1]),
            geometry_type=geometry_type,
        )
    ndim = meta.sid_ndim
    chunk_shape = meta.chunk_shape

    # Get or create the level group
    try:
        level_group = get_resolution_level(root, level)
    except Exception:
        level_meta = LevelMetadata(
            level=level, vertex_count=0, arrays_present=[VERTICES],
        )
        level_group = create_resolution_level(root, level, level_meta)

    # Determine the namespaced array name
    array_namespace = f"vertices_{geometry_type}"

    # Read existing geometry index
    geom_index = _read_geometry_index(level_group)

    vertex_count = 0

    if geometry_type in (GEOM_POINT_CLOUD,):
        if positions is None:
            raise ValueError("Point cloud requires 'positions'")
        vertex_count = _write_namespaced_vertices(
            level_group, array_namespace, positions, chunk_shape, ndim,
        )
        geom_index[geometry_type] = {
            "array": array_namespace,
            "vertex_count": vertex_count,
            "has_links": False,
        }

    elif geometry_type in (GEOM_GRAPH, GEOM_SKELETON):
        if positions is None:
            raise ValueError(f"{geometry_type} requires 'positions'")
        vertex_count = _write_namespaced_vertices(
            level_group, array_namespace, positions, chunk_shape, ndim,
        )
        links_ns = f"links_{geometry_type}"
        if edges is not None:
            _write_namespaced_links(level_group, links_ns, edges)
        geom_index[geometry_type] = {
            "array": array_namespace,
            "vertex_count": vertex_count,
            "links_array": links_ns if edges is not None else None,
            "edge_count": len(edges) if edges is not None else 0,
            "has_links": edges is not None,
        }

    elif geometry_type == GEOM_MESH:
        if positions is None:
            raise ValueError("Mesh requires 'positions'")
        vertex_count = _write_namespaced_vertices(
            level_group, array_namespace, positions, chunk_shape, ndim,
        )
        links_ns = f"links_{geometry_type}"
        if faces is not None:
            _write_namespaced_links(level_group, links_ns, faces)
        geom_index[geometry_type] = {
            "array": array_namespace,
            "vertex_count": vertex_count,
            "links_array": links_ns if faces is not None else None,
            "face_count": len(faces) if faces is not None else 0,
            "has_links": faces is not None,
        }

    elif geometry_type in (GEOM_STREAMLINE, "polyline"):
        if polylines is None:
            raise ValueError("Streamlines require 'polylines'")
        # Concatenate all polyline vertices and write
        all_verts = np.concatenate(polylines, axis=0).astype(np.float32)
        vertex_count = _write_namespaced_vertices(
            level_group, array_namespace, all_verts, chunk_shape, ndim,
        )
        geom_index[geometry_type] = {
            "array": array_namespace,
            "vertex_count": vertex_count,
            "polyline_count": len(polylines),
            "has_links": False,
        }

    elif geometry_type == GEOM_LINE:
        if positions is None:
            raise ValueError("Lines require 'positions' (endpoints)")
        vertex_count = _write_namespaced_vertices(
            level_group, array_namespace, positions, chunk_shape, ndim,
        )
        geom_index[geometry_type] = {
            "array": array_namespace,
            "vertex_count": vertex_count,
            "has_links": False,
        }

    else:
        raise ValueError(f"Unknown geometry type: '{geometry_type}'")

    # Update geometry index
    _write_geometry_index(level_group, geom_index)

    # Update root metadata geometry_types
    _update_root_geometry_types(root, geometry_type)

    return {
        "geometry_type": geometry_type,
        "vertex_count": vertex_count,
        "array_namespace": array_namespace,
    }


# ===================================================================
# Read composite
# ===================================================================

def read_composite(
    store_path: ReadSource,
    level: int = 0,
) -> dict[str, dict[str, Any]]:
    """Read all geometry types from a composite store.

    Returns:
        ``{geometry_type: {"positions": ..., "vertex_count": ..., ...}}``
        for each geometry type present in the store.
    """
    root = open_store(store_path)
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim

    level_group = get_resolution_level(root, level)
    geom_index = _read_geometry_index(level_group)

    result: dict[str, dict[str, Any]] = {}

    if geom_index:
        # Composite store with geometry index — read namespaced arrays
        for geom_type, info in geom_index.items():
            array_name = info.get("array", f"vertices_{geom_type}")
            try:
                positions = _read_namespaced_vertices(
                    level_group, array_name, ndim,
                )
                entry: dict[str, Any] = {
                    "positions": positions,
                    "vertex_count": len(positions),
                }
                links_array = info.get("links_array")
                if links_array and info.get("has_links"):
                    try:
                        links = _read_namespaced_links(level_group, links_array)
                        entry["links"] = links
                        if "edge_count" in info:
                            entry["edge_count"] = info["edge_count"]
                        if "face_count" in info:
                            entry["face_count"] = info["face_count"]
                    except Exception:
                        pass
                if "polyline_count" in info:
                    entry["polyline_count"] = info["polyline_count"]
                result[geom_type] = entry
            except Exception:
                result[geom_type] = {"vertex_count": 0, "error": "unreadable"}

        # Also read standard vertices/ for geometry types NOT in the index
        # (the original geometry written via write_points/write_graph etc.)
        indexed_types = set(geom_index.keys())
        for geom_type in meta.geometry_types:
            if geom_type not in indexed_types and geom_type not in result:
                try:
                    chunk_keys = list_chunk_keys(level_group)
                    all_verts: list[npt.NDArray] = []
                    for ck in chunk_keys:
                        groups = read_chunk_vertices(
                            level_group, ck, dtype=np.float32, ndim=ndim,
                        )
                        for g in groups:
                            if len(g) > 0:
                                all_verts.append(g)
                    if all_verts:
                        positions = np.concatenate(all_verts, axis=0)
                    else:
                        positions = np.zeros((0, ndim), dtype=np.float32)
                    result[geom_type] = {
                        "positions": positions,
                        "vertex_count": len(positions),
                    }
                except Exception:
                    result[geom_type] = {"vertex_count": 0, "error": "unreadable"}
    else:
        # Single-geometry store (no geometry index) — read via standard path
        for geom_type in meta.geometry_types:
            try:
                chunk_keys = list_chunk_keys(level_group)
                all_verts: list[npt.NDArray] = []
                for ck in chunk_keys:
                    groups = read_chunk_vertices(
                        level_group, ck, dtype=np.float32, ndim=ndim,
                    )
                    for g in groups:
                        if len(g) > 0:
                            all_verts.append(g)
                if all_verts:
                    positions = np.concatenate(all_verts, axis=0)
                else:
                    positions = np.zeros((0, ndim), dtype=np.float32)
                result[geom_type] = {
                    "positions": positions,
                    "vertex_count": len(positions),
                }
            except Exception:
                result[geom_type] = {"vertex_count": 0, "error": "unreadable"}

    return result


# ===================================================================
# Helpers
# ===================================================================

def _write_namespaced_vertices(
    level_group: FsGroup,
    array_name: str,
    positions: npt.NDArray[np.floating],
    chunk_shape: tuple[float, ...],
    ndim: int,
) -> int:
    """Write vertices to a namespaced array within a level."""
    positions = np.asarray(positions, dtype=np.float32)
    n_verts = len(positions)

    arr_group = level_group.require_group(array_name)
    arr_group.attrs.update({
        "dtype": "float32",
        "ndim": ndim,
        "vertex_count": n_verts,
    })

    chunk_assignments = assign_chunks(positions, chunk_shape)

    for chunk_coords, global_indices in sorted(chunk_assignments.items()):
        chunk_verts = positions[global_indices]
        key = ".".join(str(c) for c in chunk_coords)

        raw = chunk_verts.astype(np.float32).tobytes()
        level_group.write_bytes(array_name, key, raw)

    return n_verts


def _read_namespaced_vertices(
    level_group: FsGroup,
    array_name: str,
    ndim: int,
) -> npt.NDArray[np.float32]:
    """Read all vertices from a namespaced array."""
    if not level_group.array_exists(array_name):
        return np.zeros((0, ndim), dtype=np.float32)

    parts: list[npt.NDArray] = []
    for chunk_key in level_group.list_chunks(array_name):
        raw = level_group.read_bytes(array_name, chunk_key)
        if raw:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, ndim)
            parts.append(arr)

    if not parts:
        return np.zeros((0, ndim), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _write_namespaced_links(
    level_group: FsGroup,
    array_name: str,
    links: npt.NDArray[np.integer],
) -> None:
    """Write links (edges/faces) as a single chunked Zarr v3 array.

    The array lives at ``<array_name>`` as a 2D int64 array with shape
    ``(N, link_width)``.  Per-array attrs carry ``link_count`` and
    ``link_width`` for reader convenience (both also derivable from the
    array's own shape).
    """
    links = np.asarray(links, dtype=np.int64)
    link_width = int(links.shape[1]) if links.ndim == 2 else 2
    if links.size == 0:
        # Reader returns an empty array when the path is missing; mirror
        # that by deleting any prior array at the same path.
        level_group.delete_subtree(array_name)
        return
    level_group.write_array(
        array_name, links,
        attributes={
            "link_count": int(links.shape[0]),
            "link_width": link_width,
        },
    )


def _read_namespaced_links(
    level_group: FsGroup,
    array_name: str,
) -> npt.NDArray[np.int64]:
    """Read links from a namespaced array."""
    if not level_group.standalone_array_exists(array_name):
        return np.zeros((0, 2), dtype=np.int64)
    arr = level_group.read_array(array_name)
    if arr.ndim == 1:
        attrs = level_group.read_array_attrs(array_name)
        link_width = int(attrs.get("link_width", 2))
        return arr.reshape(-1, link_width).astype(np.int64, copy=False)
    return arr.astype(np.int64, copy=False)
