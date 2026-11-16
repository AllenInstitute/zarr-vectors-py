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
  records written through ``write_links`` so endpoint canonicalisation and
  the ``perm_idx`` that restores winding are handled by the library.
* every resolution level is rechunked, not just level 0, with its own
  ``bin_shape`` and its cross-level ``links/<+-k>`` families.
* fragments shared between objects (coarse-level metavertices) stay shared,
  and each object's manifest keeps its order.
* object ids are preserved exactly, so ``segment_id`` and any other attribute
  still addresses the same object.
* vertex attributes, object attributes and groupings are carried across per
  level; a level with fragment or link attributes is refused.

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

#: Vertex-attribute array keys the writer owns: the core fields
#: create_attribute_array sets, and the presence bookkeeping that names
#: the SOURCE grid's chunks.  Everything else (dictionary encoding,
#: user extensions) is copied to the output array.
_ATTR_META_OWNED = frozenset({
    "zv_array", "name", "dtype", "channel_names", "row_shape",
    "nonempty_chunks", "zarr_vectors_presence",
})


def suggest_chunk_shape(
    store_path: str | Path,
    *,
    level: int | None = None,
    target_links_per_chunk: int = 8_000_000,
) -> tuple[float, ...]:
    """A chunk shape whose busiest cell holds about ``target_links_per_chunk``.

    Every level is inspected unless ``level`` names one. Chunk size is halved
    per axis until the projected maximum falls under the target. The
    default of 8M links is just under the 8,993,990-face maximum of
    a store measured to render acceptably, and an order of magnitude under the
    2**32-byte cell ceiling at which the vlen-bytes codec silently truncates.
    """
    from zarr_vectors.core.store import list_resolution_levels, open_store, read_root_metadata

    root = open_store(str(store_path), mode="r")
    meta = read_root_metadata(root)
    shape = tuple(float(c) for c in meta.chunk_shape)
    # Take the worst level, not level 0. A coarse level holds fewer links but
    # its chunks are larger, so it can be the denser one: on the 1720-cell
    # corpus level 2 peaks at 49,727,014 links against level 0's 41,990,172,
    # and sizing from level 0 alone would under-suggest.
    levels = ([level] if level is not None
              else sorted(list_resolution_levels(root)))
    worst = max((_max_links_per_chunk(root, lo) for lo in levels), default=0.0)
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
# reading a level as a vertex table plus its fragments
# ===================================================================
#
# A level is read as ONE table of vertices plus the fragments that slice
# it, not as per-object geometry.  Coarse pyramid levels share fragments
# between objects -- a metavertex sits in the manifest of every object it
# aggregates -- and an object's manifest is an ORDERED walk that may visit
# the same fragment more than once.  A per-object read either has to pick
# one owner per fragment (and every other object loses that geometry) or
# copy the vertex once per owner (and the level grows).  Keeping fragments
# as the unit keeps each vertex once and every manifest's order intact.

class _Level:
    """One level read off disk.

    ``pos``: ``(N, ndim)`` vertex positions, source chunks back to back.
    ``frags``: per source fragment, its table rows in fragment order.
    ``manifests``: per object row, the ordered list of fragment numbers.
    ``chunk_base`` / ``chunk_size``: where each source chunk's buffer
    starts in the table, for turning a stored ``(chunk, local index)``
    link endpoint into a table row.
    ``vattrs``: ``{name: (N, ...) array}`` per-vertex attributes.
    """

    def __init__(self):
        self.pos = None
        self.frags: list[npt.NDArray[np.int64]] = []
        self.manifests: list[list[int]] = []
        self.object_ids = np.zeros(0, np.int64)
        self.chunk_base: dict[tuple, int] = {}
        self.chunk_size: dict[tuple, int] = {}
        self.vattrs: dict[str, Any] = {}
        self.vattr_meta: dict[str, dict] = {}
        self.dtype = np.dtype(np.float32)


def _list_names(group, family: str) -> list[str]:
    try:
        return (sorted(group[family].children())
                if family in group.children() else [])
    except Exception:  # noqa: BLE001
        return []


def _read_level(src_group, ndim: int) -> _Level:
    from zarr_vectors.constants import VERTEX_ATTRIBUTES, VERTICES
    from zarr_vectors.core.arrays import (
        list_chunk_keys,
        read_chunk_attributes,
        read_chunk_vertices,
        read_object_manifest_rows,
        read_vertex_fragment_index,
    )
    from zarr_vectors.exceptions import ArrayError

    lv = _Level()
    attr_names = _list_names(src_group, VERTEX_ATTRIBUTES)
    for name in attr_names:
        lv.vattr_meta[name] = dict(
            src_group.read_array_meta(f"{VERTEX_ATTRIBUTES}/{name}") or {})

    frag_no: dict[tuple, int] = {}
    pos_parts: list = []
    attr_parts: dict[str, list] = {n: [] for n in attr_names}
    base = 0
    dtype = None
    for cc in sorted(tuple(int(x) for x in c)
                     for c in list_chunk_keys(src_group, VERTICES)):
        if len(cc) != ndim:
            raise ValueError(
                f"rechunk_spatial: chunk key {cc} has {len(cc)} axes but the "
                f"store has {ndim} spatial axes; a store chunked by an "
                f"attribute or by source cannot be rechunked spatially")
        try:
            fi = read_vertex_fragment_index(src_group, cc)
            groups = read_chunk_vertices(src_group, cc, ndim=ndim)
        except ArrayError:
            continue
        if not groups:
            continue
        idxs = []
        n_rows = 0
        at = 0
        for f, g in enumerate(groups):
            g = np.asarray(g)
            # A link's stored local index addresses the chunk buffer laid out
            # by the FRAGMENT INDEX, which is not necessarily read order.
            if fi.is_range(f):
                start, count = fi.range(f)
                idx = np.arange(int(start), int(start) + int(count))
            else:
                idx = np.asarray(fi.indices(f), np.int64)
            if idx.size != g.shape[0]:
                idx = np.arange(at, at + g.shape[0], dtype=np.int64)
            at += g.shape[0]
            idxs.append(idx)
            if idx.size:
                n_rows = max(n_rows, int(idx.max()) + 1)
            if dtype is None and g.size:
                dtype = g.dtype
        pos = np.zeros((n_rows, ndim), np.float64)
        for f, (g, idx) in enumerate(zip(groups, idxs)):
            if idx.size:
                pos[idx] = np.asarray(g, np.float64).reshape(len(idx), ndim)
            frag_no[(cc, f)] = len(lv.frags)
            lv.frags.append(base + idx.astype(np.int64))
        for name in attr_names:
            meta = lv.vattr_meta[name]
            adt = np.dtype(meta.get("dtype", "float32"))
            rs = meta.get("row_shape") or []
            ncols = int(rs[0]) if rs else max(1, len(meta.get("channel_names") or []))
            agroups = read_chunk_attributes(src_group, name, cc, dtype=adt,
                                            ncols=ncols)
            buf = np.zeros((n_rows,) + ((ncols,) if ncols > 1 else ()), adt)
            for a, idx in zip(agroups, idxs):
                if idx.size:
                    buf[idx] = np.asarray(a, adt).reshape(buf[idx].shape)
            attr_parts[name].append(buf)
        lv.chunk_base[cc] = base
        lv.chunk_size[cc] = n_rows
        base += n_rows
        pos_parts.append(pos)

    lv.pos = (np.concatenate(pos_parts) if pos_parts
              else np.zeros((0, ndim), np.float64))
    lv.vattrs = {n: np.concatenate(p) for n, p in attr_parts.items() if p}
    lv.dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)

    # A store may legitimately carry no object index -- write_points does not
    # write one -- in which case every fragment belongs to one implicit
    # object 0.
    try:
        ids, manifests = read_object_manifest_rows(src_group)
    except Exception:  # noqa: BLE001
        ids, manifests = np.zeros(0, dtype=np.int64), []
    if len(manifests):
        lv.object_ids = np.asarray(ids, np.int64)
        for man in manifests:
            seq = []
            for cc, f in man:
                k = frag_no.get((tuple(int(x) for x in cc), int(f)))
                if k is None:
                    raise ValueError(
                        f"rechunk_spatial: a manifest names fragment {f} of "
                        f"chunk {tuple(cc)}, which the level does not hold")
                seq.append(k)
            lv.manifests.append(seq)
    elif lv.frags:
        lv.object_ids = np.zeros(1, np.int64)
        lv.manifests = [list(range(len(lv.frags)))]
    return lv


def _endpoint_rows(lv: _Level, chunks, vi) -> npt.NDArray[np.int64]:
    """Table rows for ``(chunk, local index)`` endpoints; -1 when unknown."""
    out = np.full(vi.shape, -1, np.int64)
    flat_c = chunks.reshape(-1, chunks.shape[-1])
    res = out.reshape(-1)
    for i, (c, v) in enumerate(zip(flat_c.tolist(), vi.reshape(-1).tolist())):
        k = tuple(c)
        b = lv.chunk_base.get(k)
        if b is not None and 0 <= v < lv.chunk_size[k]:
            res[i] = b + v
    return out


# ===================================================================
# writing a level onto a chunk grid
# ===================================================================

def _write_level(level_group, lv: _Level, chunk_shape, ndim: int,
                 total_objects: int):
    """Write vertices, fragments, manifests and vertex attributes.

    Every source fragment is cut into runs of consecutive rows that fall
    in one new chunk; each run becomes a new fragment, so a manifest that
    walked fragments ``A, B`` now walks ``A's runs, B's runs`` and the
    vertex order of every object is unchanged.  A fragment several objects
    share (or one object visits twice) is written once and referenced from
    each visit.  A fragment referenced exactly once is private to that
    visit, so its runs are merged with the neighbouring private runs of the
    same walk that land in the same chunk -- otherwise level 0, where every
    fragment is private, would keep one fragment per OLD chunk crossing.

    Returns ``(new_chunk, new_local)`` per table row, for re-addressing
    links, and the vertex count.
    """
    from zarr_vectors.core.arrays import (
        create_attribute_array,
        write_chunk_attributes,
        write_chunk_vertices,
        write_object_index,
    )

    n = int(lv.pos.shape[0])
    cs = np.asarray(chunk_shape, np.float64)
    new_cc = (np.floor(lv.pos / cs).astype(np.int64) if n
              else np.zeros((0, ndim), np.int64))
    new_local = np.full(n, -1, np.int64)

    refs = np.zeros(len(lv.frags), np.int64)
    for seq in lv.manifests:
        for k in seq:
            refs[k] += 1

    def runs(k: int):
        rows = lv.frags[k]
        if rows.size == 0:
            return []
        c = new_cc[rows]
        # A row may only join the run before it if it lands in the same chunk.
        cut = np.flatnonzero(np.any(c[1:] != c[:-1], axis=1)) + 1
        return [(tuple(int(x) for x in new_cc[r[0]]), r)
                for r in np.split(rows, cut)]

    # chunk -> list of row arrays (one per new fragment), in creation order
    chunk_frags: dict[tuple, list] = {}

    def new_frag(cc: tuple, rows) -> tuple:
        lst = chunk_frags.setdefault(cc, [])
        lst.append(rows)
        return (cc, len(lst) - 1)

    shared_refs: dict[int, list] = {}
    manifests: dict[int, list] = {}
    for oid, seq in zip(lv.object_ids.tolist(), lv.manifests):
        out: list = []
        open_private: tuple | None = None   # (cc, fragment) still growing
        for k in seq:
            if refs[k] > 1:
                if k not in shared_refs:
                    shared_refs[k] = [new_frag(cc, r) for cc, r in runs(k)]
                out.extend(shared_refs[k])
                open_private = None
                continue
            for cc, r in runs(k):
                if open_private is not None and open_private[0] == cc:
                    lst = chunk_frags[cc]
                    lst[open_private[1]] = np.concatenate(
                        [lst[open_private[1]], r])
                else:
                    open_private = new_frag(cc, r)
                    out.append(open_private)
        if out:
            manifests[int(oid)] = out
    # Fragments no object names still hold vertices; keep them.
    for k in np.flatnonzero(refs == 0).tolist():
        for cc, r in runs(k):
            new_frag(cc, r)

    for name, meta in lv.vattr_meta.items():
        rs = meta.get("row_shape") or []
        ncols = int(rs[0]) if rs else None
        extra = {k: v for k, v in meta.items()
                 if k not in _ATTR_META_OWNED}
        create_attribute_array(level_group, name,
                               dtype=str(meta.get("dtype", "float32")),
                               channel_names=meta.get("channel_names"),
                               ncols=ncols, extra_meta=extra or None)

    written = 0
    for cc in sorted(chunk_frags):
        frs = chunk_frags[cc]
        at = 0
        for r in frs:
            new_local[r] = np.arange(at, at + len(r))
            at += len(r)
        written += at
        write_chunk_vertices(level_group, cc,
                             [np.asarray(lv.pos[r], lv.dtype) for r in frs],
                             dtype=lv.dtype)
        for name, data in lv.vattrs.items():
            write_chunk_attributes(level_group, name, cc,
                                   [data[r] for r in frs], dtype=data.dtype)

    write_object_index(level_group, manifests, sid_ndim=ndim,
                       total_objects=total_objects)
    return new_cc, new_local, written


def _carry_links(out_group, src_group, delta: int, src_lv: _Level,
                 dst_lv: _Level, src_map, dst_map, ndim: int) -> tuple[int, int]:
    """Re-address one ``links/<delta>`` family onto the new grids.

    Endpoint 0 lives at the owning level, endpoints k > 0 at
    ``level + delta``.  Returns ``(written, dropped)``; a record is dropped
    only when an endpoint did not resolve to a vertex in the SOURCE store.
    """
    from zarr_vectors.core.arrays import (
        create_links_family,
        link_family_policy,
        read_link_arrays,
        write_links,
    )

    pol = link_family_policy(src_group, delta)
    if pol is None:
        return 0, 0
    link_width, _sid, directed, store = pol
    create_links_family(out_group, delta=delta, link_width=link_width,
                        sid_ndim=ndim, directed=directed, store=store)
    chunks, vi = read_link_arrays(src_group, delta=delta)
    if vi.shape[0] == 0:
        return 0, 0
    rows = np.empty_like(vi)
    rows[:, :1] = _endpoint_rows(src_lv, chunks[:, :1], vi[:, :1])
    rows[:, 1:] = _endpoint_rows(dst_lv, chunks[:, 1:], vi[:, 1:])
    ok = np.all(rows >= 0, axis=1)
    rows = rows[ok]
    if store == "duplicate":
        # read_link_arrays returns one copy per incident chunk; write_links
        # makes those copies again.
        _, first = np.unique(rows, axis=0, return_index=True)
        rows = rows[np.sort(first)]
    new_c = np.empty(rows.shape + (ndim,), np.int64)
    new_v = np.empty(rows.shape, np.int64)
    new_c[:, 0] = src_map[0][rows[:, 0]]
    new_v[:, 0] = src_map[1][rows[:, 0]]
    if link_width > 1:
        new_c[:, 1:] = dst_map[0][rows[:, 1:]]
        new_v[:, 1:] = dst_map[1][rows[:, 1:]]
    if len(rows):
        write_links(out_group, [], ndim, delta=delta, link_width=link_width,
                    directed=directed, store=store, _arrays=(new_c, new_v))
    return int(len(rows)), int((~ok).sum())


# ===================================================================
# the public entry point
# ===================================================================

#: Level families this function carries.  Anything else present with
#: content makes it refuse rather than write a store that silently lacks it.
_CARRIED = {"vertices", "vertex_fragments", "object_index", "links",
            "link_fragments", "vertex_attributes", "object_attributes",
            "groups", "group_attributes"}


def _refuse_uncarried(src_group, lvl: int) -> None:
    extra = []
    for name in src_group.children():
        if name in _CARRIED or name.startswith("_") or name.startswith("."):
            continue
        extra.append(name)
    if extra:
        raise NotImplementedError(
            f"rechunk_spatial: level {lvl} carries {sorted(extra)}, which "
            f"spatial rechunking does not re-address (fragment and link "
            f"attributes are tied to the source chunking); refusing rather "
            f"than writing a store without them")


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

    Works for every geometry kind: every ``links/<delta>`` family -- the
    intra-level one and the cross-level ``±k`` ones of a pyramid -- is
    carried across at its own ``link_width`` (3 for triangle meshes, 2 for
    skeletons, graphs and polylines), and a store with no links at all (a
    point cloud) simply has none to carry.

    Each vertex is written once.  A fragment shared between objects (the
    metavertices of a coarse pyramid level) stays shared: vertices are
    grouped on the new grid by the set of objects that own them, and each
    group is listed in every owner's manifest.  Object ids, per-vertex,
    per-object and per-group attributes, groupings, each level's
    ``bin_shape`` / ``bin_ratio`` and the root's base bin are carried
    unchanged.  Fragment and link attributes are not: they are tied to the
    source's fragments and link cells, so a level carrying them is refused.

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
        levels: levels to carry across; default all of them.  Cross-level
            links whose other end is a level left out are not carried.
        force: overwrite ``output`` if it exists.

    Returns:
        Summary dict, including per-level vertex and link counts.
    """
    from zarr_vectors.constants import (
        CAP_MULTISCALE_LINKS,
        CAP_PRESERVED_OBJECT_IDS,
        CAP_SHARED_FRAGMENTS,
        GROUP_ATTRIBUTES,
        GROUPS,
        LINKS,
        OBJECT_ATTRIBUTES,
        VERTEX_ATTRIBUTES,
    )
    from zarr_vectors.core.arrays import (
        create_groupings_attributes_array,
        create_object_attributes_array,
        create_object_index_array,
        create_vertices_array,
        list_link_deltas,
        read_all_groupings,
        read_groupings_attributes,
        read_object_attributes,
        write_groupings,
        write_groupings_attributes,
        write_object_attributes,
    )
    from zarr_vectors.core.metadata import LevelMetadata
    from zarr_vectors.core.store import (
        create_resolution_level,
        create_store,
        get_resolution_level,
        list_resolution_levels,
        open_store,
        read_level_metadata,
        read_root_metadata,
        update_level_metadata,
    )

    src_path, out_path = Path(store_path), Path(output)
    if out_path.resolve() == src_path.resolve():
        raise ValueError("rechunk_spatial: output must differ from the source")
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists; pass force=True")

    root = open_store(str(src_path), mode="r")
    meta = read_root_metadata(root)
    ndim = len(meta.chunk_shape)
    all_levels = sorted(list_resolution_levels(root))
    todo = all_levels if levels is None else [lo for lo in all_levels if lo in levels]

    # Read and check everything BEFORE touching the output, so a refusal
    # never costs the caller an existing directory.  Each level is read as
    # uniform cells, which a level with octree chunks does not have (its
    # links are filed by leaf).
    from zarr_vectors.octree_chunks.guard import refuse_octree_levels

    refuse_octree_levels(root, todo, "rechunk_spatial", read=True)
    for lvl in todo:
        _refuse_uncarried(get_resolution_level(root, lvl), lvl)

    if chunk_shape is None:
        chunk_shape = suggest_chunk_shape(
            src_path, target_links_per_chunk=target_links_per_chunk)
    chunk_shape = tuple(float(c) for c in chunk_shape)

    if out_path.exists():
        shutil.rmtree(out_path)

    # Carry the conventions across. Creating a store with only bounds/
    # chunk_shape/geometry_types leaves links_convention and friends unset,
    # and a reader with no links_convention cannot resolve the intra-chunk
    # links at all -- only the cross-chunk ones survive, which renders as
    # geometry along chunk boundaries and nothing else.
    # The base bin is the SOURCE's effective one: it is the level-0
    # resolution, and every coarse level's bin_ratio is relative to it.
    base_bin = tuple(float(b) for b in meta.effective_bin_shape)
    caps = [c for c in (meta.format_capabilities or [])
            if c in (CAP_PRESERVED_OBJECT_IDS, CAP_SHARED_FRAGMENTS,
                     CAP_MULTISCALE_LINKS)]
    create_store(str(out_path), bounds=meta.bounds, chunk_shape=chunk_shape,
                 geometry_types=list(meta.geometry_types), ndim=ndim,
                 base_bin_shape=base_bin,
                 shard_shape=getattr(meta, "shard_shape", None),
                 # Packing is the store's: an octree-sharded store (the
                 # zarr_octree_sharding codec) rechunks to one, as a
                 # sharded store rechunks to a sharded one.
                 octree_sharding=getattr(meta, "octree_sharding", None),
                 links_convention=getattr(meta, "links_convention", None)
                 or "explicit",
                 object_index_convention=getattr(
                     meta, "object_index_convention", None) or "standard",
                 cross_chunk_strategy=getattr(
                     meta, "cross_chunk_strategy", None) or "explicit_links",
                 cross_level_depth=getattr(meta, "cross_level_depth", None),
                 cross_level_storage=getattr(meta, "cross_level_storage", None),
                 format_capabilities=caps or None)
    out_root = open_store(str(out_path), mode="r+")

    read: dict[int, _Level] = {}
    maps: dict[int, tuple] = {}
    report = []
    for lvl in todo:
        src_g = get_resolution_level(root, lvl)
        lvl_shape = tuple(float(x) for x in (level_chunk_shapes or {}).get(
            lvl, chunk_shape))
        lv = _read_level(src_g, ndim)
        read[lvl] = lv
        ids = lv.object_ids
        n_obj_total = int(ids.max()) + 1 if ids.size else (1 if lv.pos.shape[0] else 0)

        src_lm = read_level_metadata(root, lvl)
        has_links = bool(list_link_deltas(src_g))
        arrays = ["vertices", "object_index"] + (["links"] if has_links else [])
        if lv.vattr_meta:
            arrays.append(VERTEX_ATTRIBUTES)
        kw: dict[str, Any] = {"level": lvl, "vertex_count": 0,
                              "arrays_present": arrays,
                              "object_sparsity": src_lm.object_sparsity,
                              "coarsening_method": src_lm.coarsening_method}
        if lvl > 0:
            kw["parent_level"] = (src_lm.parent_level
                                  if src_lm.parent_level is not None else lvl - 1)
            # The source level's own bin, not one re-derived from the level
            # index: a pyramid need not coarsen by 2 per level.
            kw["bin_shape"] = (tuple(src_lm.bin_shape) if src_lm.bin_shape
                               else tuple(b * (2 ** lvl) for b in base_bin))
            if src_lm.bin_ratio is not None:
                kw["bin_ratio"] = tuple(src_lm.bin_ratio)
            kw["preserves_object_ids"] = getattr(
                src_lm, "preserves_object_ids", True)
            kw["inherited_num_objects"] = n_obj_total
        if lvl > 0 and lvl_shape != tuple(chunk_shape):
            kw["chunk_shape"] = lvl_shape
        out_g = create_resolution_level(out_root, lvl, LevelMetadata(**kw))
        create_vertices_array(out_g, dtype=str(lv.dtype))
        create_object_index_array(out_g)

        new_cc, new_local, nv = _write_level(out_g, lv, lvl_shape, ndim,
                                             n_obj_total)
        maps[lvl] = (new_cc, new_local)
        update_level_metadata(out_g, vertex_count=int(nv))

        # attributes and groupings ride along per level
        n_attr = 0
        for name in _list_names(src_g, OBJECT_ATTRIBUTES):
            data = np.asarray(read_object_attributes(src_g, name))
            create_object_attributes_array(out_g, name, dtype=str(data.dtype))
            write_object_attributes(out_g, name, data)
            n_attr += 1
        gr = (read_all_groupings(src_g)
              if GROUPS in src_g.children() else None)
        if gr:
            write_groupings(out_g, {i: list(m) for i, m in enumerate(gr)})
        # Group ATTRIBUTES are a separate array family from the memberships.
        for gn in _list_names(src_g, GROUP_ATTRIBUTES):
            data = np.asarray(read_groupings_attributes(src_g, gn))
            create_groupings_attributes_array(out_g, gn, dtype=str(data.dtype))
            write_groupings_attributes(out_g, gn, data)

        report.append({"level": lvl,
                       "objects": sum(1 for m in lv.manifests if m),
                       "vertices": nv, "links": 0, "cross_level_links": 0,
                       "dropped_links": 0, "vertex_attributes": len(lv.vattrs),
                       "attributes": n_attr, "chunk_shape": lvl_shape})

    # Links last: a cross-level family needs both levels' metadata (their
    # chunk grids anchor its offsets) and both levels' new addresses.
    by_level = {r["level"]: r for r in report}
    for lvl in todo:
        src_g = get_resolution_level(root, lvl)
        out_g = get_resolution_level(out_root, lvl)
        for delta in list_link_deltas(src_g):
            other = lvl + delta
            if other not in read:
                continue
            w, dropped = _carry_links(out_g, src_g, delta, read[lvl],
                                      read[other], maps[lvl], maps[other], ndim)
            key = "links" if delta == 0 else "cross_level_links"
            by_level[lvl][key] += w
            by_level[lvl]["dropped_links"] += dropped
        if out_g.array_exists(LINKS):
            update_level_metadata(out_g, add_arrays_present=LINKS)

    return {"output_path": str(out_path), "chunk_shape": chunk_shape,
            "level_chunk_shapes": {r["level"]: r.get("chunk_shape")
                                   for r in report},
            "levels": report}
