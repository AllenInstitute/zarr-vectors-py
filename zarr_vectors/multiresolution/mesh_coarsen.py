"""A mesh's faces or cells at a coarse level of the per-object pyramid.

The per-object coarsener merges each bin's vertices into one metavertex.
This module gives the coarse level the mesh's connectivity, planned
before the level's vertices are written (a volume mesh's boundary
metavertices move) and written after:

* **Volume meshes** (a ``links/0`` family with ``element``): cells on
  the bin lattice (:mod:`zarr_vectors.multiresolution.volume`), with the
  boundary metavertices snapped out to the fine surface -- each moves to
  the centroid of the fine boundary vertices in its bin -- which undoes
  most of the lattice's half-bin shrink.  Cells inverted by the move get
  their vertices' centroids back; any still inverted are dropped.
* **Surface meshes** (polygon faces, no ``element``): vertex clustering.
  Each fine face is re-pointed at the metavertices of its vertices' bins;
  a face whose vertices fall in fewer distinct bins than it has vertices
  collapses and is dropped, and faces over the same metavertices are kept
  once, with the first one's winding.
* **Per-face / per-cell attributes** (``link_attributes/<name>/0``) come
  down too.  Each coarse face or cell takes the fine ones it stands for:
  for a surface, the fine faces that clustered onto it; for a volume, the
  fine cells whose centroid lies in its lattice block.  Floating-point
  values are averaged, weighted by area or volume; integer and boolean
  values (labels, material ids) take the value with the most area or
  volume.  A coarse cell no fine cell maps to takes the level-wide mean
  or majority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import LINK_ATTRIBUTES
from zarr_vectors.core.elements import MIXED, boundary_faces, family_element, signed_volumes
from zarr_vectors.core.paths import links_group_path
from zarr_vectors.multiresolution.volume import coarse_element, lattice_cells


@dataclass
class CoarseMeshPlan:
    """What :func:`plan_coarse_mesh` decided; :func:`write_coarse_mesh`
    writes it once the metavertices have chunk addresses."""

    cells: npt.NDArray[np.int64]
    element: str | None
    attributes: dict[str, npt.NDArray] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fine records
# ---------------------------------------------------------------------------


def _fine_records(src_group, ndim: int):
    """The source level's ``links/0`` records as global vertex ids, with
    every referenced chunk's positions (in the dtype stored, so bins
    computed from them match the coarsener's)."""
    from zarr_vectors.core.arrays import read_chunk_vertices, read_link_arrays

    chunks, vi = read_link_arrays(src_group, delta=0)
    if vi.size == 0:
        return None
    flat = chunks.reshape(-1, ndim)
    uc, inv = np.unique(flat, axis=0, return_inverse=True)
    parts, offs = [], [0]
    for c in uc.tolist():
        frags = read_chunk_vertices(src_group, tuple(c), ndim=ndim)
        v = (np.concatenate([np.asarray(f).reshape(-1, ndim) for f in frags])
             if frags else np.zeros((0, ndim), np.float32))
        parts.append(v)
        offs.append(offs[-1] + len(v))
    positions = np.concatenate(parts)
    gid = (np.asarray(offs[:-1])[inv.ravel()] + vi.ravel()).reshape(vi.shape)
    return gid, positions


def _bin_lookup(
    mv_bins: npt.NDArray[np.int64], query: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Index of the metavertex whose bin is each ``query`` row; -1 if none."""
    out = np.full(query.shape[0], -1, dtype=np.int64)
    if mv_bins.shape[0] == 0 or query.shape[0] == 0:
        return out
    lo = np.minimum(mv_bins.min(0), query.min(0))
    dims = np.maximum(mv_bins.max(0), query.max(0)) - lo + 1
    lin = np.ravel_multi_index((mv_bins - lo).T, dims)
    order = np.argsort(lin)
    srt = lin[order]
    q = np.ravel_multi_index((query - lo).T, dims)
    pos = np.minimum(np.searchsorted(srt, q), srt.size - 1)
    hit = srt[pos] == q
    out[hit] = order[pos[hit]]
    return out


def _measures(positions: npt.NDArray, records: npt.NDArray, element: str | None) -> npt.NDArray:
    """Volume of each cell, or area of each polygon face (fanned)."""
    pos = np.asarray(positions, dtype=np.float64)
    if element is not None:
        return np.abs(signed_volumes(pos, records, element))
    area = np.zeros(records.shape[0])
    a = pos[records[:, 0]]
    for i in range(1, records.shape[1] - 1):
        b, c = pos[records[:, i]], pos[records[:, i + 1]]
        if pos.shape[1] == 3:
            area += 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        else:
            area += 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    return area


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def _link_attribute_names(src_group) -> list[str]:
    from zarr_vectors.core.arrays import list_link_attribute_deltas

    if not src_group.array_exists(LINK_ATTRIBUTES):
        return []
    try:
        names = sorted(src_group[LINK_ATTRIBUTES].child_names())
    except Exception:
        return []
    return [n for n in names if 0 in list_link_attribute_deltas(src_group, n)]


def aggregate(
    values: npt.NDArray, groups: npt.NDArray[np.int64], weights: npt.NDArray,
    n_groups: int,
) -> npt.NDArray:
    """One value per group: weighted mean for floats, the value with the
    most weight for integers and booleans.  ``groups`` < 0 are ignored;
    an empty group gets the overall mean / majority."""
    values = np.asarray(values)
    flat = values.reshape(values.shape[0], -1)
    keep = groups >= 0
    g, w, v = groups[keep], np.asarray(weights, np.float64)[keep], flat[keep]
    out = np.empty((n_groups, flat.shape[1]), dtype=values.dtype)
    for c in range(flat.shape[1]):
        col = v[:, c]
        if np.issubdtype(values.dtype, np.floating):
            tot = np.bincount(g, weights=w, minlength=n_groups)
            s = np.bincount(g, weights=w * col.astype(np.float64), minlength=n_groups)
            fallback = (np.sum(w * col) / np.sum(w)) if w.sum() > 0 else np.nan
            out[:, c] = np.where(tot > 0, s / np.where(tot > 0, tot, 1), fallback)
        else:
            vals, vinv = np.unique(col, return_inverse=True)
            pair = g * len(vals) + vinv.ravel()
            wsum = np.bincount(pair, weights=w, minlength=n_groups * len(vals))
            table = wsum.reshape(n_groups, len(vals)) if len(vals) else np.zeros((n_groups, 1))
            overall = vals[np.argmax(table.sum(0))] if len(vals) else 0
            best = vals[np.argmax(table, axis=1)] if len(vals) else np.zeros(n_groups)
            out[:, c] = np.where(table.sum(1) > 0, best, overall)
    return out.reshape((n_groups, *values.shape[1:]))


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_coarse_mesh(
    src_group,
    *,
    ndim: int,
    mv_bins: npt.NDArray[np.int64],
    bin_shape: npt.NDArray[np.float64],
    meta_positions: npt.NDArray[np.floating],
    object_mvs: list[npt.NDArray[np.int64]],
) -> CoarseMeshPlan | None:
    """Plan the coarse level's faces or cells; may move ``meta_positions``
    (a volume mesh's boundary metavertices) in place.  None when the
    source level has no polygon faces or volume cells."""
    from zarr_vectors.core.arrays import link_family_policy, read_link_attributes

    policy = link_family_policy(src_group, 0)
    if policy is None:
        return None
    link_width = policy[0]
    element = family_element(src_group.read_array_meta(links_group_path(0)) or {})
    if element is None and link_width < 3:
        return None  # edges: not a mesh
    fine = _fine_records(src_group, ndim)
    if fine is None:
        return None
    gid, fpos = fine
    fine_mv = _bin_lookup(mv_bins, np.floor(fpos / bin_shape).astype(np.int64))
    rec_mv = fine_mv[gid]
    weights = _measures(fpos, gid, element)
    names = _link_attribute_names(src_group)
    fine_attrs = {n: read_link_attributes(src_group, n, delta=0) for n in names}

    if element is not None:
        if ndim != 3:
            return None
        return _plan_volume(
            element, gid, fpos, fine_mv, rec_mv, weights, fine_attrs,
            mv_bins=mv_bins, bin_shape=bin_shape,
            meta_positions=meta_positions, object_mvs=object_mvs,
        )
    return _plan_surface(rec_mv, weights, fine_attrs)


def _plan_surface(rec_mv, weights, fine_attrs) -> CoarseMeshPlan:
    width = rec_mv.shape[1]
    srt = np.sort(rec_mv, axis=1)
    ok = (rec_mv >= 0).all(axis=1) & (srt[:, 1:] != srt[:, :-1]).all(axis=1)
    keep_idx = np.flatnonzero(ok)
    if keep_idx.size == 0:
        return CoarseMeshPlan(np.zeros((0, width), np.int64), None,
                              summary={"face_count": 0})
    _, first, group = np.unique(srt[keep_idx], axis=0, return_index=True, return_inverse=True)
    group = group.ravel()
    order = np.argsort(first)  # coarse faces in the order their first fine face came
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    cells = rec_mv[keep_idx[first[order]]]
    groups = np.full(rec_mv.shape[0], -1, dtype=np.int64)
    groups[keep_idx] = rank[group]
    attrs = {n: aggregate(v, groups, weights, cells.shape[0]) for n, v in fine_attrs.items()}
    return CoarseMeshPlan(cells, None, attrs, {"face_count": int(cells.shape[0])})


def _plan_volume(
    element, gid, fpos, fine_mv, rec_mv, weights, fine_attrs, *,
    mv_bins, bin_shape, meta_positions, object_mvs,
) -> CoarseMeshPlan:
    out_element = coarse_element(element if element != MIXED else "tetra")
    cells, dropped, blocks = lattice_cells(
        mv_bins, meta_positions, object_mvs, out_element, return_blocks=True,
    )
    summary: dict[str, Any] = {"element": out_element}
    moved = 0
    if cells.shape[0]:
        moved, cells, blocks, dropped2 = _snap_boundary(
            cells, blocks, out_element, gid, fpos, fine_mv, meta_positions, element,
        )
        dropped += dropped2
    # Each fine cell stands for the lattice block holding its centroid:
    # blocks join bin centres, so a point's block starts at bin floor(p/b - 1/2).
    attrs: dict[str, npt.NDArray] = {}
    if fine_attrs and cells.shape[0]:
        centroid = fpos[gid].astype(np.float64).mean(axis=1)
        fine_block = np.floor(centroid / bin_shape - 0.5).astype(np.int64)
        ublocks, cell_group = np.unique(blocks, axis=0, return_inverse=True)
        fine_group = _bin_lookup(ublocks, fine_block)
        per_block = {
            n: aggregate(v, fine_group, weights, ublocks.shape[0])
            for n, v in fine_attrs.items()
        }
        attrs = {n: v[cell_group.ravel()] for n, v in per_block.items()}
    summary.update(cell_count=int(cells.shape[0]), cells_dropped_inverted=int(dropped),
                   boundary_vertices_snapped=int(moved))
    return CoarseMeshPlan(cells, out_element, attrs, summary)


def _snap_boundary(cells, blocks, element, gid, fpos, fine_mv, meta_positions, fine_element):
    """Move each coarse boundary metavertex to the centroid of the fine
    boundary vertices in its bin; undo the move where it inverts a cell,
    and drop cells still inverted after that."""
    width = cells.shape[1]
    coarse_boundary = np.unique(np.concatenate([f.ravel() for f in boundary_faces(cells, element)]))
    fine_boundary = np.unique(np.concatenate(
        [f.ravel() for f in boundary_faces(gid, fine_element)]
    ))
    mv = fine_mv[fine_boundary]
    keep = mv >= 0
    mv, fb = mv[keep], fine_boundary[keep]
    n = meta_positions.shape[0]
    count = np.bincount(mv, minlength=n)
    target = np.zeros((n, meta_positions.shape[1]))
    for a in range(meta_positions.shape[1]):
        target[:, a] = np.bincount(mv, weights=fpos[fb, a].astype(np.float64), minlength=n)
    movable = np.zeros(n, dtype=bool)
    movable[coarse_boundary] = True
    movable &= count > 0
    original = np.asarray(meta_positions, dtype=np.float64).copy()
    snapped = original.copy()
    snapped[movable] = target[movable] / count[movable, None]
    for _ in range(4):
        bad = signed_volumes(snapped, cells, element) <= 0
        if not bad.any():
            break
        revert = np.unique(cells[bad])
        revert = revert[movable[revert]]
        if revert.size == 0:
            break
        snapped[revert] = original[revert]
        movable[revert] = False
    ok = signed_volumes(snapped, cells, element) > 0
    meta_positions[:] = snapped.astype(meta_positions.dtype)
    return int(movable.sum()), cells[ok].reshape(-1, width), blocks[ok], int((~ok).sum())


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_coarse_mesh(
    level_group, plan: CoarseMeshPlan, *, ndim: int,
    chunk_slots, mv_chunk_slot, mv_fragment_idx,
) -> dict[str, Any]:
    """Write a planned coarse level's records and attributes."""
    from zarr_vectors.core.arrays import (
        stamp_link_element,
        write_link_attributes,
        write_links,
    )

    cells = plan.cells
    if cells.shape[0] == 0:
        return plan.summary
    slot_chunks = np.asarray(chunk_slots, dtype=np.int64).reshape(-1, ndim)
    cc = slot_chunks[mv_chunk_slot[cells]]
    vi = mv_fragment_idx[cells]  # one metavertex per fragment
    partition = write_links(
        level_group, [], ndim, delta=0, link_width=cells.shape[1], _arrays=(cc, vi),
    )
    for name, values in plan.attributes.items():
        write_link_attributes(
            level_group, name, values, num_links=cells.shape[0], partition=partition,
        )
    if plan.element is not None:
        stamp_link_element(level_group, plan.element, _records_are_new=True)
    return plan.summary
