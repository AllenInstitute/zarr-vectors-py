"""Rechunk engine — reads a source store and writes a rechunked copy.

The rechunked store has an extra prefix dimension on its chunk keys:
``(prefix_bin, z, y, x)`` instead of ``(z, y, x)``.  All objects in
the same prefix bin are physically contiguous, enabling O(1) group
or attribute-based filtering.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    GEOM_LINE,
    GEOM_POLYLINE,
    GEOM_STREAMLINE,
    GROUP_ATTRIBUTES,
    GROUPS,
    LINK_FRAGMENTS,
    LINKS,
    LINKS_IMPLICIT_SEQUENTIAL,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.arrays import (
    create_attribute_array,
    create_groupings_attributes_array,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_link_deltas,
    read_all_object_manifests,
    read_chunk_attributes,
    read_groupings_attributes,
    read_object_attributes,
    read_object_vertices,
    vertices_dtype,
    write_chunk_attributes,
    write_chunk_vertices,
    write_groupings_attributes,
    write_links,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    FsGroup,
    create_resolution_level,
    create_store,
    get_resolution_level,
    list_resolution_levels,
    open_store,
    read_root_metadata,
    update_level_metadata,
)
from zarr_vectors.rechunk.spatial import _ATTR_META_OWNED
from zarr_vectors.rechunk.spec import DimensionMapper, RechunkSpec
from zarr_vectors.typing import ChunkCoords, ObjectManifest


#: Level-0 families :func:`rechunk` carries into the output.
_CARRIED = {
    VERTICES, VERTEX_FRAGMENTS, OBJECT_INDEX, VERTEX_ATTRIBUTES,
    OBJECT_ATTRIBUTES, GROUPS, GROUP_ATTRIBUTES, LINKS, LINK_FRAGMENTS,
}


def _check_carried(src_root: FsGroup, src_meta: Any) -> None:
    """Refuse a source holding data :func:`rechunk` would not write out.

    ``rechunk`` regroups objects under a bin prefix.  It writes vertices,
    manifests, vertex / object / group attributes, groupings, and -- for
    an ``implicit_sequential`` store -- the bridge links between an
    object's consecutive fragments, which follow from the manifests.
    Anything else would be silently left behind, so it is refused.
    """
    levels = list_resolution_levels(src_root)
    problems: list[str] = []
    if [lv for lv in levels if lv != 0]:
        problems.append(
            f"resolution levels {[lv for lv in levels if lv != 0]} (rechunk "
            f"writes level 0 only; remove them and run build_pyramid on the "
            f"output)")
    lg = get_resolution_level(src_root, 0)
    extra = sorted(
        n for n in lg.children()
        if n not in _CARRIED and not n.startswith((".", "_"))
    )
    if extra:
        problems.append(f"level-0 families {extra}")
    if list_link_deltas(lg) and (
        src_meta.links_convention != LINKS_IMPLICIT_SEQUENTIAL
        or list_link_deltas(lg) != [0]
    ):
        problems.append(
            f"explicit links (links_convention="
            f"{src_meta.links_convention!r}, deltas {list_link_deltas(lg)}); "
            f"use rechunk_spatial to change the chunk grid of a store with "
            f"topology")
    if problems:
        raise NotImplementedError(
            "rechunk() cannot carry " + "; ".join(problems)
            + ". Refusing rather than writing a store without them."
        )


def rechunk(
    store_path: str | Path,
    spec: RechunkSpec,
    output: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rechunk a store along a non-spatial dimension.

    Reads object data from the source store, assigns each object to
    a rechunk bin via ``DimensionMapper``, and writes the result to
    an output store where chunk keys have a prefix dimension
    ``(bin, z, y, x)``.

    Carried: vertices, manifests (objects that have vertices are
    renumbered densely in bin order), vertex attributes, object
    attributes (remapped to the new ids), groupings and group attributes,
    and the bridge links of an ``implicit_sequential`` store.  A source
    holding anything else -- levels above 0, explicit links, fragment or
    link attributes -- is refused with ``NotImplementedError`` before
    anything is written.

    ``RechunkSpec(by="spatial")`` has no bins to prefix, so it is handed
    to :func:`~zarr_vectors.rechunk.spatial.rechunk_spatial`, which
    changes the chunk grid and carries links and every level.

    Args:
        store_path: Source store path.
        spec: Rechunk specification.
        output: Output store path. If None, rechunks in-place by
            writing to a temporary store then replacing the source.
        overwrite: Replace an existing ``output``.  Without it an
            existing output is refused rather than deleted.

    Returns:
        Summary dict with ``objects_rechunked``, ``bins_created``,
        ``output_path``.
    """
    store_path = Path(store_path)

    # Determine output path
    in_place = output is None
    if in_place:
        output_path = store_path.parent / (store_path.name + ".rechunked")
        for scratch in (output_path,
                        store_path.parent / (store_path.name + ".backup")):
            if scratch.exists():
                raise FileExistsError(
                    f"{scratch} exists; an in-place rechunk needs that name "
                    f"for scratch space and will not delete it")
    else:
        output_path = Path(output)
        if output_path.resolve() == store_path.resolve():
            raise ValueError("output must differ from the source; pass "
                             "output=None to rechunk in place")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} exists; pass overwrite=True to replace it")

    if spec.by == "spatial":
        return _rechunk_spatial(store_path, spec, output_path, in_place)

    # Read source
    src_root = open_store(str(store_path))
    src_meta = read_root_metadata(src_root)
    ndim = src_meta.sid_ndim
    _check_carried(src_root, src_meta)

    if output_path.exists():
        shutil.rmtree(output_path)

    chunk_shape = spec.spatial_chunk_shape or src_meta.chunk_shape

    # Read level 0 data
    src_level = get_resolution_level(src_root, 0)

    # Read object manifests
    try:
        manifests = read_all_object_manifests(src_level)
        n_objects = len(manifests)
    except Exception:
        manifests = []
        n_objects = 0

    # Read groupings
    groupings: list[list[int]] | None = None
    try:
        from zarr_vectors.core.arrays import read_all_groupings
        groupings = read_all_groupings(src_level)
    except Exception:
        groupings = None

    # Read object attributes (for attribute-based rechunking)
    object_attributes: dict[str, npt.NDArray] | None = None
    if spec.by.startswith("attribute:"):
        # Try to read the attribute as per-object data
        attr_name = spec.by.split(":", 1)[1]
        object_attributes = {}
        try:
            obj_attr_data = read_object_attributes(src_level, attr_name)
            object_attributes[attr_name] = obj_attr_data
        except Exception:
            # Attribute might need to be computed (e.g. length for polylines)
            if attr_name == "length" and n_objects > 0:
                lengths = _compute_object_lengths(src_level, n_objects, ndim)
                object_attributes[attr_name] = lengths
            else:
                raise ValueError(
                    f"Cannot read or compute attribute '{attr_name}'"
                )

    # Map objects to rechunk bins
    mapper = DimensionMapper(spec)
    if n_objects > 0:
        raw_bins = mapper.map_objects(
            n_objects=n_objects,
            groupings=groupings,
            object_attributes=object_attributes,
        )
    else:
        # No objects — rechunk spatially only
        raw_bins = {}

    # Renumber the bins densely before anything consumes them.  The
    # mapper's indices can have holes -- edges no object falls between,
    # an empty group -- and a reader resolves a value to a bin by its
    # position in chunk_attribute_values.  With holes, the key prefixes
    # and the grid kept the raw index while that list was compacted, so a
    # query came back empty, or with another bin's data under its label.
    # Ungrouped objects (-1) go last, so group g stays ahead of them.
    used = sorted(set(raw_bins.values()) - {-1})
    if -1 in raw_bins.values():
        used.append(-1)
    renumber = {old: new for new, old in enumerate(used)}
    obj_to_bin = {oid: renumber[b] for oid, b in raw_bins.items()}
    unique_bins = list(range(len(used))) or [0]
    by_bin: dict[int, list[int]] = {b: [] for b in unique_bins}
    for oid in sorted(obj_to_bin):
        by_bin[obj_to_bin[oid]].append(oid)

    # Create output store
    spatial_dim_names = [
        a.get("name", f"dim{i}")
        for i, a in enumerate(src_meta.spatial_index_dims)
    ]
    rechunk_dims = [spec.dimension_name, *spatial_dim_names]

    out_root = create_store(
        str(output_path),
        axes=src_meta.spatial_index_dims,
        chunk_shape=chunk_shape,
        bounds=src_meta.bounds,
        geometry_types=src_meta.geometry_types,
        # Packing is a property of the store, so a rechunk of a sharded
        # store produces a sharded store. Dropping it here would have
        # quietly unsharded every rechunk output.
        shard_shape=src_meta.shard_shape,
        links_convention=src_meta.links_convention,
        object_index_convention=src_meta.object_index_convention,
        cross_chunk_strategy=src_meta.cross_chunk_strategy,
        base_bin_shape=src_meta.base_bin_shape,
    )

    # One label per bin, in bin order: what a reader's attribute_filter
    # names to select it.  Recorded for group and object-id rechunks too,
    # under the dimension's name: renumbering is otherwise the one step
    # that loses which bin holds which group.
    chunk_attribute_name: str | None = None
    chunk_attribute_values: list[Any] | None = None
    if obj_to_bin and spec.by != "spatial":
        chunk_attribute_name = (
            spec.by.split(":", 1)[1] if spec.by.startswith("attribute:")
            else spec.dimension_name
        )
        chunk_attribute_values = [mapper.labels[old] for old in used]

    # Create level 0
    level_meta = LevelMetadata(
        level=0,
        vertex_count=0,  # updated below
        arrays_present=[VERTICES, "object_index"],
        chunk_dims=rechunk_dims,
        chunk_attribute_name=chunk_attribute_name,
        chunk_attribute_values=chunk_attribute_values,
    )
    out_level = create_resolution_level(out_root, 0, level_meta)
    # Rechunk prefixes every chunk key with a leading bin index, so the
    # single vlen arrays need that extra axis on their grid.  Activate
    # the single-array layout with the right rank for the write span.
    from zarr_vectors.core.arrays import level_grid_layout
    _spatial_origin, _spatial_grid = level_grid_layout(
        src_meta.bounds, chunk_shape,
    )
    # The output grid carries a leading attribute-bin axis. A per-axis
    # declaration gets 1 prepended for it, so a shard never straddles
    # bins -- they are queried independently, and packing two into one
    # object drags each into the other's reads. A scalar broadcasts,
    # which is what a scalar means.
    from zarr_vectors.core.metadata import normalise_shard_shape

    _rechunk_grid = (len(unique_bins), *_spatial_grid)
    _rechunk_session = out_level.native_sharded_arrays(
        normalise_shard_shape(
            src_meta.shard_shape, len(_rechunk_grid), bin_axis=True,
        ),
        _rechunk_grid,
        origin=(0, *_spatial_origin),
    )

    # Read at the dtype the source declares: a float64 cell read as
    # float32 decodes to garbage at twice the row count.
    vdtype = vertices_dtype(src_level)
    cs = np.asarray(chunk_shape, dtype=np.float64)

    vattrs = {
        name: dict(src_level.read_array_meta(f"{VERTEX_ATTRIBUTES}/{name}") or {})
        for name in _names(src_level, VERTEX_ATTRIBUTES)
    }
    # Bridges are what connect a path across its fragments, so a path store
    # gets them whether or not the source happened to need any: a smaller
    # spatial_chunk_shape can split a path that used to sit in one chunk.
    write_bridges = bool(list_link_deltas(src_level)) or (
        src_meta.links_convention == LINKS_IMPLICIT_SEQUENTIAL
        and bool(set(src_meta.geometry_types)
                 & {GEOM_LINE, GEOM_POLYLINE, GEOM_STREAMLINE})
    )
    chunk_cache: dict[tuple, dict[str, list]] = {}

    def _attr_fragments(cc: tuple, f: int) -> dict[str, npt.NDArray]:
        got = chunk_cache.get(cc)
        if got is None:
            got = {}
            for name, meta in vattrs.items():
                adt = np.dtype(meta.get("dtype", "float32"))
                rs = meta.get("row_shape") or []
                ncols = int(rs[0]) if rs else max(
                    1, len(meta.get("channel_names") or []))
                got[name] = read_chunk_attributes(
                    src_level, name, cc, dtype=adt, ncols=ncols)
            chunk_cache.clear()
            chunk_cache[cc] = got
        return {name: np.asarray(groups[f]) for name, groups in got.items()}

    total_vertices = 0
    object_manifests_out: dict[int, ObjectManifest] = {}
    # Source id -> output id, for objects that were written.  An object
    # with no vertices gets no manifest, so it gets no id either.
    old_to_new: dict[int, int] = {}

    with _rechunk_session:
        create_vertices_array(out_level, dtype=vdtype.name)
        create_object_index_array(out_level)
        for name, meta in vattrs.items():
            rs = meta.get("row_shape") or []
            extra = {k: v for k, v in meta.items()
                     if k not in _ATTR_META_OWNED}
            create_attribute_array(
                out_level, name, dtype=str(meta.get("dtype", "float32")),
                channel_names=meta.get("channel_names"),
                ncols=int(rs[0]) if rs else None, extra_meta=extra or None,
            )
        bridge_chunks: list = []
        bridge_rows: list = []

        for bin_idx in unique_bins:
            # Each object keeps its own fragments: one per run of its
            # vertices inside one cell, as the type writers lay them out.
            # Merging a cell into one fragment made every manifest name
            # fragment 0, so reading one object returned the whole cell.
            cells: dict[ChunkCoords, list[npt.NDArray]] = {}
            cell_attrs: dict[str, dict[ChunkCoords, list]] = {
                name: {} for name in vattrs}
            for oid in by_bin[bin_idx]:
                try:
                    fragments = read_object_vertices(
                        src_level, oid, dtype=vdtype, ndim=ndim,
                    )
                except Exception:
                    fragments = []
                src_manifest = manifests[oid] if vattrs and fragments else []
                manifest: ObjectManifest = []
                for i, fragment in enumerate(fragments):
                    attrs = (_attr_fragments(tuple(src_manifest[i][0]),
                                             int(src_manifest[i][1]))
                             if vattrs else {})
                    for cell, a, b in _runs_by_cell(fragment, cs):
                        key: ChunkCoords = (bin_idx, *cell)
                        runs = cells.setdefault(key, [])
                        manifest.append((key, len(runs)))
                        runs.append(fragment[a:b])
                        for name, arr in attrs.items():
                            cell_attrs[name].setdefault(key, []).append(arr[a:b])
                if not manifest:
                    continue
                old_to_new[oid] = len(object_manifests_out)
                object_manifests_out[old_to_new[oid]] = manifest

            starts: dict[ChunkCoords, list[int]] = {}
            for key in sorted(cells):
                write_chunk_vertices(out_level, key, cells[key], dtype=vdtype)
                for name in vattrs:
                    write_chunk_attributes(
                        out_level, name, key, cell_attrs[name][key],
                        dtype=np.dtype(vattrs[name].get("dtype", "float32")),
                    )
                lens = [len(r) for r in cells[key]]
                starts[key] = np.concatenate([[0], np.cumsum(lens)]).tolist()
                total_vertices += sum(lens)

            # implicit_sequential: a fragment's vertices are connected in
            # order, and each step from one fragment of an object to the
            # next is a bridge link -- last vertex of one, first of the
            # next -- exactly as write_polylines / write_lines lay them out.
            if write_bridges:
                for oid in by_bin[bin_idx]:
                    man = object_manifests_out.get(old_to_new.get(oid, -1), [])
                    for (ka, fa), (kb, fb) in zip(man, man[1:]):
                        bridge_chunks.append((ka, kb))
                        bridge_rows.append((starts[ka][fa + 1] - 1,
                                            starts[kb][fb]))

        if bridge_rows:
            write_links(
                out_level, [], ndim + 1, delta=0, link_width=2,
                _arrays=(np.asarray(bridge_chunks, np.int64),
                         np.asarray(bridge_rows, np.int64)),
            )

        # The manifests use (prefix, z, y, x) coords — ndim+1 dimensions
        if object_manifests_out:
            write_object_index(
                out_level, object_manifests_out, sid_ndim=ndim + 1,
            )

    update_level_metadata(out_level, vertex_count=int(total_vertices))
    present = [VERTICES, OBJECT_INDEX]
    if vattrs:
        present.append(VERTEX_ATTRIBUTES)
    if out_level.array_exists(LINKS):
        present.append(LINKS)
    update_level_metadata(out_level, arrays_present=present)

    # Object attributes follow their objects to the new ids.
    new_to_old = np.array(
        [old for old, _new in sorted(old_to_new.items(), key=lambda t: t[1])],
        dtype=np.int64,
    )
    for name in _names(src_level, OBJECT_ATTRIBUTES):
        data = np.asarray(read_object_attributes(src_level, name))
        create_object_attributes_array(out_level, name, dtype=str(data.dtype))
        write_object_attributes(out_level, name, data[new_to_old])

    # Group attributes are per group, and group ids are unchanged.
    for name in _names(src_level, GROUP_ATTRIBUTES):
        data = np.asarray(read_groupings_attributes(src_level, name))
        create_groupings_attributes_array(out_level, name, dtype=str(data.dtype))
        write_groupings_attributes(out_level, name, data)

    # Groupings ride along whatever the rechunk was by, remapped to the
    # new ids (they used to be written for by="group" only).
    if groupings:
        from zarr_vectors.core.arrays import (
            create_groupings_array,
            write_groupings,
        )

        new_groupings: dict[int, list[int]] = {}
        for gid, members in enumerate(groupings):
            new_members = [old_to_new[m] for m in members if m in old_to_new]
            if new_members:
                new_groupings[gid] = new_members

        if new_groupings:
            create_groupings_array(out_level)
            write_groupings(out_level, new_groupings)

    # In-place: replace source with output
    if in_place:
        backup = store_path.parent / (store_path.name + ".backup")
        store_path.rename(backup)
        output_path.rename(store_path)
        shutil.rmtree(backup)
        output_path = store_path

    return {
        "objects_rechunked": len(object_manifests_out),
        "bins_created": len(unique_bins),
        "total_vertices": total_vertices,
        "rechunk_dims": rechunk_dims,
        "output_path": str(output_path),
    }


def _names(level_group: FsGroup, family: str) -> list[str]:
    """Member names of a named-array family (``vertex_attributes`` ...)."""
    if family not in level_group.children():
        return []
    return sorted(level_group[family].children())


def _rechunk_spatial(
    store_path: Path, spec: RechunkSpec, output_path: Path, in_place: bool,
) -> dict[str, Any]:
    """``by="spatial"``: change the chunk grid only, through rechunk_spatial."""
    from zarr_vectors.rechunk.spatial import rechunk_spatial

    src_meta = read_root_metadata(open_store(str(store_path)))
    chunk_shape = spec.spatial_chunk_shape or src_meta.chunk_shape
    summary = rechunk_spatial(store_path, output_path, tuple(chunk_shape),
                              force=output_path.exists())
    if in_place:
        backup = store_path.parent / (store_path.name + ".backup")
        store_path.rename(backup)
        output_path.rename(store_path)
        shutil.rmtree(backup)
        output_path = store_path
    level0 = next((r for r in summary["levels"] if r["level"] == 0), {})
    return {
        "objects_rechunked": level0.get("objects", 0),
        "bins_created": 1,
        "total_vertices": level0.get("vertices", 0),
        "rechunk_dims": [
            a.get("name", f"dim{i}")
            for i, a in enumerate(src_meta.spatial_index_dims)
        ],
        "output_path": str(output_path),
        "levels": summary["levels"],
    }


def _runs_by_cell(
    positions: npt.NDArray, cs: npt.NDArray[np.float64],
) -> list[tuple[tuple[int, ...], int, int]]:
    """Split ``positions`` into maximal runs of vertices sharing a cell.

    A fragment is one contiguous run of an object's vertices inside one
    cell; a polyline that leaves a cell and comes back is two fragments
    there, not one with a segment invented across the gap.  Cells are
    ``floor(position / chunk_shape)``, as
    :func:`~zarr_vectors.spatial.chunking.assign_chunks` computes them.
    """
    if len(positions) == 0:
        return []
    cells = np.floor(positions / cs).astype(np.int64)
    breaks = (np.flatnonzero(np.any(cells[1:] != cells[:-1], axis=1)) + 1).tolist()
    starts = [0, *breaks]
    ends = [*breaks, len(positions)]
    return [
        (tuple(cells[a].tolist()), a, b)
        for a, b in zip(starts, ends)
    ]


def _compute_object_lengths(
    level_group: FsGroup,
    n_objects: int,
    ndim: int,
) -> npt.NDArray[np.float64]:
    """Compute path length for each object (for polyline-like data)."""
    lengths = np.zeros(n_objects, dtype=np.float64)
    vdtype = vertices_dtype(level_group)
    for oid in range(n_objects):
        try:
            verts_list = read_object_vertices(
                level_group, oid, dtype=vdtype, ndim=ndim,
            )
            all_verts = np.concatenate(
                [v for v in verts_list if len(v) > 0], axis=0,
            )
            if len(all_verts) >= 2:
                diffs = np.diff(all_verts, axis=0)
                lengths[oid] = float(
                    np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
                )
        except Exception:
            pass
    return lengths
