"""Rechunk engine — reads a source store and writes a rechunked copy.

The rechunked store has an extra prefix dimension on its chunk keys:
``(prefix_bin, z, y, x)`` instead of ``(z, y, x)``.  All objects in
the same prefix bin are physically contiguous, enabling O(1) group
or attribute-based filtering.  Object ids, attributes and links are
carried (see :func:`rechunk`).
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
    create_groupings_attributes_array,
    create_links_family,
    create_object_attributes_array,
    create_object_index_array,
    create_vertices_array,
    list_link_deltas,
    read_groupings_attributes,
    read_object_attributes,
    read_object_vertices,
    vertices_dtype,
    write_groupings_attributes,
    write_links,
    write_object_attributes,
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
from zarr_vectors.rechunk.spatial import _map_links, _read_level, _write_level
from zarr_vectors.rechunk.spec import DimensionMapper, RechunkSpec

#: Level-0 families :func:`rechunk` carries into the output.
_CARRIED = {
    VERTICES, VERTEX_FRAGMENTS, OBJECT_INDEX, VERTEX_ATTRIBUTES,
    OBJECT_ATTRIBUTES, GROUPS, GROUP_ATTRIBUTES, LINKS, LINK_FRAGMENTS,
}


def _check_carried(src_root: FsGroup, src_meta: Any) -> None:
    """Refuse a source holding data :func:`rechunk` would not write out."""
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
    stale = [d for d in list_link_deltas(lg) if d != 0]
    if stale:
        problems.append(
            f"cross-level link families {stale} on level 0 with no level "
            f"to point at (remove them, or rebuild the pyramid)")
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

    Object ids are preserved, so object attributes, groupings and group
    attributes copy verbatim.  Vertex attributes follow their vertices,
    shared fragments stay shared, and ``links/0`` is re-addressed onto the
    new keys whatever its ``link_width`` (a link may join two bins).  Under
    ``implicit_sequential``, a fragment the new grid cuts gets a bridge
    link across each cut.  Levels above 0, link/fragment attributes, a
    fragment shared by objects in different bins, and vertices no object
    names raise ``NotImplementedError`` before anything is written.
    ``by="spatial"`` is handed to
    :func:`~zarr_vectors.rechunk.spatial.rechunk_spatial`.

    Args:
        store_path: Source store path.
        spec: Rechunk specification.
        output: Output store path. If None, rechunks in-place by
            writing to a temporary store then replacing the source.
        overwrite: Replace an existing ``output`` instead of raising.

    Returns:
        Summary dict with ``objects_rechunked``, ``bins_created``,
        ``total_vertices``, ``rechunk_dims``, ``output_path``, ``links``
        (records written to ``links/0``) and ``dropped_links`` (source
        records naming no vertex, which cannot be carried).
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
    # Before _check_carried, which would report octree_chunks as a family.
    from zarr_vectors.octree_chunks.guard import refuse_octree_read

    refuse_octree_read(get_resolution_level(src_root, 0), "rechunk")
    _check_carried(src_root, src_meta)

    chunk_shape = spec.spatial_chunk_shape or src_meta.chunk_shape

    # Level 0 as a vertex table with its fragments and manifests, as
    # rechunk_spatial reads it: links address rows of that table, so they
    # can be re-addressed once the rows have new homes.  Keys of any rank:
    # an attribute-chunked store can be re-binned.
    src_level = get_resolution_level(src_root, 0)
    lv = _read_level(src_level, ndim, any_rank=True)
    ids = lv.object_ids
    present_ids = [int(o) for o, m in zip(ids.tolist(), lv.manifests) if m]
    n_objects = int(ids.max()) + 1 if ids.size else 0

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
        all_bins = mapper.map_objects(
            n_objects=n_objects,
            groupings=groupings,
            object_attributes=object_attributes,
        )
        # Only objects with vertices take a bin: an empty bin is never
        # allocated, nor labelled.
        raw_bins = {oid: all_bins[oid] for oid in present_ids}
    else:
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

    # Each vertex row goes to its object's bin.  Object ids are kept, so
    # nothing else (attributes, groupings, links) needs renumbering.
    cells = _row_cells(lv, obj_to_bin, np.asarray(chunk_shape, np.float64))

    # Create output store
    spatial_dim_names = [
        a.get("name", f"dim{i}")
        for i, a in enumerate(src_meta.spatial_index_dims)
    ]
    rechunk_dims = [spec.dimension_name, *spatial_dim_names]

    from zarr_vectors.constants import CAP_PRESERVED_OBJECT_IDS, CAP_SHARED_FRAGMENTS

    # Everything that can refuse has run; only now replace an output.
    if output_path.exists():
        shutil.rmtree(output_path)

    caps = [c for c in (src_meta.format_capabilities or [])
            if c in (CAP_PRESERVED_OBJECT_IDS, CAP_SHARED_FRAGMENTS)]
    out_root = create_store(
        str(output_path),
        axes=src_meta.spatial_index_dims,
        chunk_shape=chunk_shape,
        bounds=src_meta.bounds,
        geometry_types=src_meta.geometry_types,
        # Packing is a property of the store, so a rechunk of a sharded
        # store produces a sharded store. Dropping it here would have
        # quietly unsharded every rechunk output.  The same for the
        # octree marker: an octree-sharded store's output is octree-sharded.
        shard_shape=src_meta.shard_shape,
        octree_sharding=src_meta.octree_sharding,
        links_convention=src_meta.links_convention,
        object_index_convention=src_meta.object_index_convention,
        cross_chunk_strategy=src_meta.cross_chunk_strategy,
        base_bin_shape=src_meta.base_bin_shape,
        format_capabilities=caps or None,
    )

    # One label per bin, in bin order: what a reader's attribute_filter
    # names to select it.  Recorded for group and object-id rechunks too,
    # under the dimension's name: renumbering is otherwise the one step
    # that loses which bin holds which group.
    chunk_attribute_name: str | None = None
    chunk_attribute_values: list[Any] | None = None
    if obj_to_bin and spec.by != "spatial":
        # The same name as chunk_dims[0]: the leading axis is the bin
        # axis, and a reader's attribute_filter keys on this name.
        chunk_attribute_name = spec.dimension_name
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

    # A path store always gets bridges: a cell boundary that cuts one of
    # the source's fragments makes two, joined by nothing else.
    bridge_splits = src_meta.links_convention == LINKS_IMPLICIT_SEQUENTIAL and (
        bool(list_link_deltas(src_level))
        or bool(set(src_meta.geometry_types)
                & {GEOM_LINE, GEOM_POLYLINE, GEOM_STREAMLINE})
    )
    n_links = dropped_links = 0

    with _rechunk_session:
        create_vertices_array(out_level, dtype=lv.dtype.name)
        create_object_index_array(out_level)
        # Each object keeps its own fragments, one per run of a source
        # fragment inside one output cell, never merged with a neighbour:
        # the source's links (and bridges) address those runs' ends.
        # Shared fragments stay shared.
        new_cc, new_local, total_vertices = _write_level(
            out_level, lv, chunk_shape, ndim,
            n_objects if present_ids else 0,
            cells=cells, merge_private=False,
        )
        maps = (new_cc, new_local)

        pol, link_c, link_v, dropped_links = _map_links(
            src_level, 0, lv, lv, maps, maps)
        if bridge_splits:
            bc, bv = _split_bridges(lv, maps)
            if len(bv):
                if pol is not None and pol[0] != 2:
                    raise NotImplementedError(
                        f"rechunk: links/0 has link_width {pol[0]}, but a "
                        f"cell boundary splitting an implicit_sequential "
                        f"fragment needs a 2-wide bridge")
                link_c = np.concatenate([link_c, bc])
                link_v = np.concatenate([link_v, bv])
        if pol is not None or len(link_v):
            link_width, _sid, directed, store = pol or (2, None, None, None)
            kw = {k: v for k, v in (("directed", directed), ("store", store))
                  if v is not None}
            if pol is not None:
                create_links_family(out_level, delta=0, link_width=link_width,
                                    sid_ndim=ndim + 1, **kw)
            if len(link_v):
                write_links(out_level, [], ndim + 1, delta=0,
                            link_width=link_width, _arrays=(link_c, link_v),
                            **kw)
            n_links = int(len(link_v))

    update_level_metadata(out_level, vertex_count=int(total_vertices))
    present = [VERTICES, OBJECT_INDEX]
    if lv.vattr_meta:
        present.append(VERTEX_ATTRIBUTES)
    if out_level.array_exists(LINKS):
        present.append(LINKS)
    update_level_metadata(out_level, arrays_present=present)

    # Ids are preserved, so per-object and per-group data copy verbatim.
    for name in _names(src_level, OBJECT_ATTRIBUTES):
        data = np.asarray(read_object_attributes(src_level, name))
        create_object_attributes_array(out_level, name, dtype=str(data.dtype))
        write_object_attributes(out_level, name, data)

    for name in _names(src_level, GROUP_ATTRIBUTES):
        data = np.asarray(read_groupings_attributes(src_level, name))
        create_groupings_attributes_array(out_level, name, dtype=str(data.dtype))
        write_groupings_attributes(out_level, name, data)

    if groupings:
        from zarr_vectors.core.arrays import (
            create_groupings_array,
            write_groupings,
        )

        create_groupings_array(out_level)
        write_groupings(out_level, {g: list(m) for g, m in enumerate(groupings)})

    # In-place: replace source with output
    if in_place:
        backup = store_path.parent / (store_path.name + ".backup")
        store_path.rename(backup)
        output_path.rename(store_path)
        shutil.rmtree(backup)
        output_path = store_path

    return {
        "objects_rechunked": len(present_ids),
        "bins_created": len(unique_bins),
        "total_vertices": int(total_vertices),
        "rechunk_dims": rechunk_dims,
        "output_path": str(output_path),
        "links": n_links,
        "dropped_links": int(dropped_links),
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


def _row_cells(
    lv: Any, obj_to_bin: dict[int, int], cs: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Each row's output key ``(bin, *floor(position / chunk_shape))``.

    A row's bin is its object's.  Raises ``NotImplementedError`` for a
    fragment shared by objects in different bins (a vertex has one home,
    and a bin read must find all of its objects' vertices) and for
    vertices no object names (they have no bin).
    """
    n = int(lv.pos.shape[0])
    row_bin = np.full(n, -1, np.int64)
    for oid, seq in zip(lv.object_ids.tolist(), lv.manifests):
        b = obj_to_bin.get(int(oid))
        if b is None:
            continue
        for k in seq:
            rows = lv.frags[k]
            held = row_bin[rows]
            if np.any((held >= 0) & (held != b)):
                raise NotImplementedError(
                    f"rechunk: object {oid} shares a fragment with an object "
                    f"in another bin; a shared vertex can be stored in one "
                    f"bin only. Use rechunk_spatial, or bin so that objects "
                    f"sharing fragments fall together.")
            row_bin[rows] = b
    named = np.zeros(n, bool)
    for k, rows in enumerate(lv.frags):
        named[rows] = True
    orphan = named & (row_bin < 0)
    if orphan.any():
        raise NotImplementedError(
            f"rechunk: level 0 holds {int(orphan.sum())} vertices in fragments "
            f"no object names, so they have no bin; use rechunk_spatial.")
    spatial = (np.floor(lv.pos / cs).astype(np.int64) if n
               else np.zeros((0, len(cs)), np.int64))
    return np.concatenate([row_bin[:, None], spatial], axis=1)


def _split_bridges(lv: Any, maps: tuple) -> tuple[npt.NDArray, npt.NDArray]:
    """Bridges across the cuts the new grid makes inside source fragments.

    Under ``implicit_sequential`` consecutive rows of a fragment are
    joined implicitly; once a cell boundary splits a fragment into runs,
    the step between runs needs an explicit link: last row of one run to
    the first of the next.  Joins the source already had between its own
    fragments come across with its links.
    """
    new_cc, new_local = maps
    rank = int(new_cc.shape[1])
    a_rows: list[npt.NDArray] = []
    b_rows: list[npt.NDArray] = []
    for rows in lv.frags:
        if rows.size < 2:
            continue
        c = new_cc[rows]
        cut = np.flatnonzero(np.any(c[1:] != c[:-1], axis=1))
        if cut.size:
            a_rows.append(rows[cut])
            b_rows.append(rows[cut + 1])
    if not a_rows:
        return np.zeros((0, 2, rank), np.int64), np.zeros((0, 2), np.int64)
    a = np.concatenate(a_rows)
    b = np.concatenate(b_rows)
    chunks = np.stack([new_cc[a], new_cc[b]], axis=1)
    vi = np.stack([new_local[a], new_local[b]], axis=1)
    return chunks, vi


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
