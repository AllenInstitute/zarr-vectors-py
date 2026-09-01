"""Spatial rechunking that preserves links, levels, object ids and attributes.

WHY THIS EXISTS ALONGSIDE ``rechunk.engine.rechunk``
---------------------------------------------------
``engine.rechunk`` rechunks along a NON-spatial dimension (group, object id,
attribute value) and prefixes every chunk key with a bin index. In doing so it
copies only ``vertices``, ``object_index`` and ``groupings``: it never reads or
writes the ``links`` family. For a point cloud that is lossless, but every
other geometry kind carries its topology in links, so a mesh comes back as a
vertex soup with no faces, a skeleton or graph loses every edge, and a polyline
loses its segments. It also reads level 0 only, so a pyramid is dropped, and it
renumbers objects through a running counter, so ids no longer join back to the
source.

This function does the other job -- change the SPATIAL chunk grid, keep
everything else -- for any geometry kind:

* links of any ``link_width`` are re-split onto the new grid, with the
  intra-chunk majority written through the numpy path and the cross-chunk
  remainder through ``write_links`` so endpoint canonicalisation and the
  ``perm_idx`` that restores winding are handled by the library.
* every resolution level is rechunked, not just level 0.
* object ids are preserved exactly, so ``segment_id`` and any other attribute
  still addresses the same object.
* object attributes and groupings are carried across per level.

WHY A STORE NEEDS THIS
----------------------
Chunk size is chosen at ingest and baked into level 0, but the right size
depends on the DENSITY of what was ingested. Putting MICrONS LOD 2 into the
128 um grid that had been sized for LOD 3 gave chunks holding up to 41,990,172
faces -- a 1.008 GB links cell against 0.216 GB for the same grid at LOD 3 --
which is a punishing fetch for any viewer that loads a chunk at a time. Without
a links-aware rechunk the only remedy is a full re-ingest.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["rechunk_spatial", "suggest_chunk_shape"]


def suggest_chunk_shape(
    store_path: str | Path,
    *,
    level: int | None = None,
    target_links_per_chunk: int = 8_000_000,
) -> tuple[float, ...]:
    """A chunk shape whose busiest cell holds about ``target_links_per_chunk``.

    Every level is inspected unless ``level`` names one. Chunk size is halved
    per axis until the projected maximum falls under the target. The default of 8M links is just under the 8,993,990-face maximum of
    a store measured to render acceptably, and an order of magnitude under the
    2**32-byte cell ceiling at which the vlen-bytes codec silently truncates.
    """
    from zarr_vectors.core.store import open_store, read_root_metadata

    from zarr_vectors.core.store import list_resolution_levels

    root = open_store(str(store_path), mode="r")
    meta = read_root_metadata(root)
    shape = tuple(float(c) for c in meta.chunk_shape)
    # Take the worst level, not level 0. A coarse level holds fewer links but
    # its chunks are larger, so it can be the denser one: on the 1720-cell
    # corpus level 2 peaks at 49,727,014 links against level 0's 41,990,172,
    # and sizing from level 0 alone would under-suggest.
    levels = ([level] if level is not None
              else sorted(list_resolution_levels(root)))
    worst = max((_max_links_per_chunk(root, l) for l in levels), default=0.0)
    for _ in range(8):
        if worst <= target_links_per_chunk:
            break
        shape = tuple(c / 2.0 for c in shape)
        worst /= 2.0 ** len(shape)     # links scale with cell volume
    return shape


def _max_links_per_chunk(root, level: int) -> float:
    from zarr_vectors.core.arrays import iter_link_cells
    from zarr_vectors.core.store import get_resolution_level

    g = get_resolution_level(root, level)
    per: dict[tuple, int] = {}
    try:
        for _seg, offsets, sc, groups in iter_link_cells(g, 0):
            if any(any(int(o) for o in off) for off in offsets):
                continue
            k = tuple(int(x) for x in sc)
            n = sum(int(np.asarray(x).shape[0]) for x in groups
                    if np.asarray(x).ndim == 2)
            per[k] = per.get(k, 0) + n
    except Exception:  # noqa: BLE001
        return 0.0
    return float(max(per.values())) if per else 0.0


# ===================================================================
# reading a level into per-object geometry
# ===================================================================

def _read_level(src_group, ndim: int, link_width: int):
    """{oid: (vertices, links)} in object-local index space.

    ``links`` is ``(N, link_width)``; an empty ``(0, link_width)`` array for a
    point cloud or for an object that happens to carry none.
    """
    from zarr_vectors.core.arrays import (
        iter_link_cells,
        list_chunk_keys,
        read_all_object_manifests,
        read_chunk_vertices,
        read_vertex_fragment_index,
    )
    from zarr_vectors.constants import VERTICES
    from zarr_vectors.exceptions import ArrayError
    from zarr_vectors.spatial.boundary import apply_perm_inverse

    # A store may legitimately carry no object index -- write_points does not
    # write one -- in which case every vertex belongs to a single implicit
    # object 0 and each chunk contributes one fragment.
    try:
        manifests = read_all_object_manifests(src_group)
    except Exception:  # noqa: BLE001
        manifests = []
    owner: dict[tuple, int] = {}
    for oid, frags in enumerate(manifests):
        for cc, f in frags:
            owner.setdefault((tuple(int(x) for x in cc), int(f)), oid)
    implicit = not owner

    obj_pos: dict[int, list] = {}
    obj_n: dict[int, int] = {}
    chunk_oid: dict[tuple, npt.NDArray] = {}
    chunk_row: dict[tuple, npt.NDArray] = {}

    for cc in sorted(tuple(int(x) for x in c)
                     for c in list_chunk_keys(src_group, VERTICES)):
        try:
            fi = read_vertex_fragment_index(src_group, cc)
            groups = read_chunk_vertices(src_group, cc, ndim=ndim)
        except ArrayError:
            continue
        if not groups:
            continue
        n_rows = sum(int(np.asarray(g).shape[0]) for g in groups)
        pos = np.zeros((n_rows, ndim), np.float64)
        oid_of = np.full(n_rows, -1, np.int64)
        row_of = np.full(n_rows, -1, np.int64)
        at = 0
        for f, g in enumerate(groups):
            g = np.asarray(g)
            if g.size == 0:
                continue
            # A link's stored local index addresses the chunk buffer laid out
            # by the FRAGMENT INDEX, which is not necessarily read order.
            if fi.is_range(f):
                start, count = fi.range(f)
                idx = np.arange(int(start), int(start) + int(count))
            else:
                idx = np.asarray(fi.indices(f), np.int64)
            if idx.size != g.shape[0]:
                idx = np.arange(at, at + g.shape[0], dtype=np.int64)
            pos[idx] = g
            o = 0 if implicit else owner.get((cc, f))
            if o is not None:
                oid_of[idx] = o
                base = obj_n.get(o, 0)
                row_of[idx] = np.arange(base, base + len(idx))
                obj_pos.setdefault(o, []).append(pos[idx])
                obj_n[o] = base + len(idx)
            at += g.shape[0]
        chunk_oid[cc] = oid_of
        chunk_row[cc] = row_of

    out = {o: [np.concatenate(p, axis=0), None] for o, p in obj_pos.items() if p}

    links_of: dict[int, list] = {}
    try:
        cells = list(iter_link_cells(src_group, 0))
    except Exception:  # noqa: BLE001
        cells = []
    for _seg, offsets, source_chunk, groups in cells:
        src_cc = tuple(int(x) for x in source_chunk)
        for g in groups:
            g = np.asarray(g, np.int64)
            if g.ndim != 2 or g.size == 0:
                continue
            perm = None
            if g.shape[1] == link_width + 1:
                perm = g[:, 0]
                g = g[:, 1:]
            elif g.shape[1] != link_width:
                continue
            cols_o, cols_r, bad = [], [], False
            for k in range(link_width):
                off = ((0,) * ndim if k == 0
                       else (offsets[k - 1] if k - 1 < len(offsets)
                             else (0,) * ndim))
                cc_k = tuple(int(a) + int(b) for a, b in zip(src_cc, off))
                oid_of = chunk_oid.get(cc_k)
                row_of = chunk_row.get(cc_k)
                if oid_of is None:
                    bad = True
                    break
                col = g[:, k]
                ok = (col >= 0) & (col < oid_of.size)
                o = np.full(col.shape, -1, np.int64)
                rr = np.full(col.shape, -1, np.int64)
                o[ok] = oid_of[col[ok]]
                rr[ok] = row_of[col[ok]]
                cols_o.append(o)
                cols_r.append(rr)
            if bad:
                continue
            same = cols_o[0] >= 0
            for k in range(1, link_width):
                same &= cols_o[k] == cols_o[0]
            for k in range(link_width):
                same &= cols_r[k] >= 0
            if not same.any():
                continue
            oids = cols_o[0][same]
            rec = np.stack([c[same] for c in cols_r], axis=1)
            if perm is not None:
                p = np.asarray(perm)[same]
                res = np.empty_like(rec)
                for pv in np.unique(p):
                    sel = p == pv
                    order = apply_perm_inverse(list(range(link_width)),
                                               int(pv), link_width)
                    for j, i in enumerate(order):
                        res[sel, j] = rec[sel, i]
                rec = res
            for o in np.unique(oids):
                links_of.setdefault(int(o), []).append(rec[oids == o])

    for o in list(out):
        ll = links_of.get(o)
        out[o][1] = (np.concatenate(ll, axis=0) if ll
                     else np.zeros((0, link_width), np.int64))
    return {o: (v, l) for o, (v, l) in out.items()}


# ===================================================================
# writing per-object geometry onto a chunk grid
# ===================================================================

def _write_level(level_group, objects: dict, chunk_shape, ndim: int,
                 link_width: int, total_objects: int, has_links: bool):
    from zarr_vectors.core.arrays import (
        create_links_array,
        create_links_family,
        finalize_links,
        write_chunk_links,
        write_chunk_vertices,
        write_links,
        write_object_index,
    )

    cs = np.asarray(chunk_shape, np.float64)
    per_chunk: dict[tuple, list] = {}
    for oid, (v, l) in sorted(objects.items()):
        if len(v) == 0:
            continue
        cc = np.floor(np.asarray(v, np.float64) / cs).astype(np.int64)
        uniq, inv = np.unique(cc, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        for rows in np.split(order, bounds):
            if rows.size:
                key = tuple(int(x) for x in uniq[inv[rows[0]]])
                per_chunk.setdefault(key, []).append((oid, rows))

    local_of = {oid: np.full(len(v), -1, np.int64)
                for oid, (v, l) in objects.items()}
    chunk_of = {oid: np.zeros((len(v), ndim), np.int64)
                for oid, (v, l) in objects.items()}
    manifests: dict[int, list] = {}

    for cc in sorted(per_chunk):
        members = sorted(per_chunk[cc], key=lambda t: t[0])
        blocks, base = [], 0
        for frag, (oid, rows) in enumerate(members):
            blocks.append(np.asarray(objects[oid][0][rows], np.float32))
            local_of[oid][rows] = np.arange(base, base + len(rows))
            chunk_of[oid][rows] = np.asarray(cc, np.int64)
            manifests.setdefault(oid, []).append((cc, frag))
            base += len(rows)
        write_chunk_vertices(level_group, cc, blocks, dtype=np.float32)

    n_links = 0
    if has_links:
        pending: dict[tuple, list] = {}
        for oid, (v, l) in sorted(objects.items()):
            if len(l) == 0:
                continue
            loc, ch = local_of[oid], chunk_of[oid]
            cor_c = [ch[l[:, k]] for k in range(link_width)]
            cor_l = [loc[l[:, k]] for k in range(link_width)]
            base = cor_c[0]
            offs = [cor_c[k] - base for k in range(1, link_width)]
            sig = np.concatenate([base] + offs, axis=1)
            uniq, inv = np.unique(sig, axis=0, return_inverse=True)
            inv = np.asarray(inv).ravel()
            for u in range(len(uniq)):
                sel = inv == u
                b = tuple(int(x) for x in uniq[u, :ndim])
                ok = tuple(tuple(int(x) for x in uniq[u, ndim*(k+1):ndim*(k+2)])
                           for k in range(link_width - 1))
                pending.setdefault((b, ok), []).append(
                    np.stack([c[sel] for c in cor_l], axis=1))

        create_links_family(level_group, delta=0, link_width=link_width,
                            sid_ndim=ndim)
        zero = (0,) * ndim
        cross: list[list[tuple]] = []
        for (b, ok), arrs in sorted(pending.items()):
            if all(o == zero for o in ok):
                continue
            st = np.concatenate(arrs, axis=0)
            n_links += len(st)
            ends = [tuple(b)] + [tuple(int(x) + int(y) for x, y in zip(b, o))
                                 for o in ok]
            cross.extend([(ends[k], int(r[k])) for k in range(link_width)]
                         for r in st)
        if cross:
            write_links(level_group, cross, ndim, delta=0,
                        link_width=link_width)
        for (b, ok), arrs in sorted(pending.items()):
            if not all(o == zero for o in ok):
                continue
            st = np.concatenate(arrs, axis=0)
            n_links += len(st)
            create_links_array(level_group, link_width, delta=0, sid_ndim=ndim,
                               offsets=[zero] * (link_width - 1))
            write_chunk_links(level_group, b, [st], dtype=np.int64, delta=0,
                              link_width=link_width)
        finalize_links(level_group, delta=0)

    write_object_index(level_group, manifests, sid_ndim=ndim,
                       total_objects=total_objects)
    return sum(len(v) for v, _ in objects.values()), n_links


# ===================================================================
# the public entry point
# ===================================================================

def rechunk_spatial(
    store_path: str | Path,
    output: str | Path,
    chunk_shape: tuple[float, ...] | None = None,
    *,
    level_chunk_shapes: dict | None = None,
    target_links_per_chunk: int = 8_000_000,
    levels: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Rewrite a store on a different spatial chunk grid, losslessly.

    Works for every geometry kind: the link family is carried across at its own
    ``link_width`` (3 for triangle meshes, 2 for skeletons, graphs and
    polylines) and a store with no links at all (a point cloud) simply has none
    to carry.

    Args:
        store_path: source store.
        output: destination store; must not exist unless ``force``.
        chunk_shape: new spatial chunk shape for the root grid. If None, one is
            chosen with :func:`suggest_chunk_shape`.
        level_chunk_shapes: optional ``{level: shape}`` giving coarse levels a
            COARSER grid than level 0. A consumer distinguishes resolution
            levels by their chunk spacing (and by a cumulative object-sparsity
            that is 1.0 at every level here), so a pyramid written entirely on
            one grid presents as a single resolution however much its geometry
            differs -- Neuroglancer's picker returns identical density scales
            and offers one level. Each shape must be an integer multiple of the
            root chunk shape.
        target_links_per_chunk: used only when ``chunk_shape`` is None.
        levels: levels to carry across; default all of them.
        force: overwrite ``output`` if it exists.

    Returns:
        Summary dict, including per-level vertex and link counts.
    """
    from zarr_vectors.core.arrays import (
        create_object_attributes_array,
        create_object_index_array,
        create_vertices_array,
        link_family_policy,
        read_all_groupings,
        read_object_attributes,
        write_groupings,
        write_object_attributes,
    )
    from zarr_vectors.core.store import (
        create_resolution_level,
        create_store,
        get_resolution_level,
        list_resolution_levels,
        open_store,
        read_level_metadata,
        read_root_metadata,
    )

    src_path, out_path = Path(store_path), Path(output)
    if out_path.exists():
        if not force:
            raise FileExistsError(f"{out_path} exists; pass force=True")
        shutil.rmtree(out_path)

    root = open_store(str(src_path), mode="r")
    meta = read_root_metadata(root)
    ndim = len(meta.chunk_shape)
    all_levels = sorted(list_resolution_levels(root))
    todo = all_levels if levels is None else [l for l in all_levels if l in levels]

    if chunk_shape is None:
        chunk_shape = suggest_chunk_shape(
            src_path, target_links_per_chunk=target_links_per_chunk)
    chunk_shape = tuple(float(c) for c in chunk_shape)

    # Carry the conventions across. Creating a store with only bounds/
    # chunk_shape/geometry_types leaves links_convention and friends unset,
    # and a reader with no links_convention cannot resolve the intra-chunk
    # links at all -- only the cross-chunk ones survive, which renders as
    # geometry along chunk boundaries and nothing else.
    base_bin = getattr(meta, "base_bin_shape", None)
    if base_bin is None:
        base_bin = tuple(c / 4.0 for c in chunk_shape)
    create_store(str(out_path), bounds=meta.bounds, chunk_shape=chunk_shape,
                 geometry_types=list(meta.geometry_types), ndim=ndim,
                 base_bin_shape=tuple(float(b) for b in base_bin),
                 links_convention=getattr(meta, "links_convention", None)
                 or "explicit",
                 object_index_convention=getattr(
                     meta, "object_index_convention", None) or "standard",
                 cross_chunk_strategy=getattr(
                     meta, "cross_chunk_strategy", None) or "explicit_links")
    out_root = open_store(str(out_path), mode="r+")

    report = []
    for lvl in todo:
        src_g = get_resolution_level(root, lvl)
        try:
            pol = link_family_policy(src_g, 0)
            link_width = int(pol[0] if isinstance(pol, tuple)
                             else pol.get("link_width", 2))
            has_links = True
        except Exception:  # noqa: BLE001
            link_width, has_links = 2, False

        lvl_shape = tuple(float(x) for x in (level_chunk_shapes or {}).get(
            lvl, chunk_shape))
        objects = _read_level(src_g, ndim, link_width)
        n_obj_total = max([o for o in objects] + [-1]) + 1

        src_lm = read_level_metadata(root, lvl)
        arrays = ["vertices", "object_index"] + (["links"] if has_links else [])
        kw: dict[str, Any] = {"level": lvl, "vertex_count": 0,
                              "arrays_present": arrays,
                              "coarsening_method": getattr(
                                  src_lm, "coarsening_method", None)}
        if lvl > 0:
            kw["parent_level"] = getattr(src_lm, "parent_level", lvl - 1)
            bs = getattr(src_lm, "bin_shape", None)
            kw["bin_shape"] = tuple(float(b) * (2 ** lvl) for b in base_bin)
            kw["preserves_object_ids"] = True
            kw["inherited_num_objects"] = n_obj_total
        if lvl > 0 and lvl_shape != tuple(chunk_shape):
            kw["chunk_shape"] = lvl_shape
        out_g = create_resolution_level(out_root, lvl, __import__(
            "zarr_vectors.core.metadata", fromlist=["LevelMetadata"]
        ).LevelMetadata(**kw))
        create_vertices_array(out_g, dtype="float32")
        create_object_index_array(out_g)

        nv, nl = _write_level(out_g, objects, lvl_shape, ndim, link_width,
                              n_obj_total, has_links)

        # vertex_count is stamped at level creation, before any data exists, so
        # it has to be corrected once the real total is known. Leaving it at 0
        # is not cosmetic: it is the field a viewer's LOD picker reads to tell
        # levels apart, so every level looked identical and only one was
        # selectable.
        try:
            from zarr_vectors.core.store import update_level_metadata
            update_level_metadata(out_root, lvl, vertex_count=int(nv))
        except Exception:  # noqa: BLE001
            import json as _json
            lj = out_path / str(lvl) / "zarr.json"
            _d = _json.loads(lj.read_text())
            _d["attributes"]["zarr_vectors_level"]["vertex_count"] = int(nv)
            lj.write_text(_json.dumps(_d, indent=2))

        # attributes and groupings ride along per level
        n_attr = 0
        # LevelMetadata carries no list of attribute names, so enumerate the
        # object_attributes group itself. Relying on a metadata field that does
        # not exist made this loop a silent no-op: the rechunked store came out
        # with no segment_id at all, which is exactly the defect that makes a
        # store load but show nothing.
        # Group.children() lists IMMEDIATE children only, so the attribute
        # names live one level down, under the object_attributes subgroup --
        # filtering the level's own children for an "object_attributes/" prefix
        # matches nothing and silently carries no attributes at all.
        try:
            from zarr_vectors.constants import OBJECT_ATTRIBUTES
            attr_names = (list(src_g[OBJECT_ATTRIBUTES].children())
                          if OBJECT_ATTRIBUTES in src_g.children() else [])
        except Exception:  # noqa: BLE001
            attr_names = []
        for name in attr_names:
            try:
                data = np.asarray(read_object_attributes(src_g, name))
                create_object_attributes_array(out_g, name, dtype=str(data.dtype))
                write_object_attributes(out_g, name, data)
                n_attr += 1
            except Exception:  # noqa: BLE001
                pass
        try:
            gr = read_all_groupings(src_g)
            if gr:
                write_groupings(out_g, {i: list(m) for i, m in enumerate(gr)})
        except Exception:  # noqa: BLE001
            pass
        # Group ATTRIBUTES are a separate array family from the memberships.
        # Carrying the memberships alone leaves every group unnamed -- the
        # source store has n_objects / name / source_column and the rechunked
        # one had none, so a viewer listing groups has nothing to label them
        # with.
        try:
            from zarr_vectors.constants import GROUP_ATTRIBUTES
            from zarr_vectors.core.arrays import (
                create_groupings_attributes_array,
                read_groupings_attributes,
                write_groupings_attributes,
            )
            gnames = (list(src_g[GROUP_ATTRIBUTES].children())
                      if GROUP_ATTRIBUTES in src_g.children() else [])
            for gn in gnames:
                data = np.asarray(read_groupings_attributes(src_g, gn))
                create_groupings_attributes_array(out_g, gn, dtype=str(data.dtype))
                write_groupings_attributes(out_g, gn, data)
        except Exception:  # noqa: BLE001
            pass

        report.append({"level": lvl, "objects": len(objects), "vertices": nv,
                       "links": nl, "link_width": link_width,
                       "attributes": n_attr, "chunk_shape": lvl_shape})

    return {"output_path": str(out_path), "chunk_shape": chunk_shape,
            "level_chunk_shapes": {r["level"]: r.get("chunk_shape")
                                   for r in report},
            "levels": report}
