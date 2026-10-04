"""Convert a finished replacement pyramid into an additive one, in place.

Every level but the coarsest keeps only what the next coarser level does
not have, and is marked ``refinement: "add"``; its complete content is then
the union over :func:`~zarr_vectors.core.refinement.level_chain`, and that
union is exactly what the level held before.  Levels are converted finest
first, so level ``L + 1`` is still complete when level ``L`` is cut against
it.

What "the same thing" means at the next level:

* **Objects**, on a level with an object index: an object whose manifest
  at ``L + 1`` is non-empty is dropped from ``L`` -- every fragment only it
  references, and every vertex only those fragments hold.  Objects keep
  their ids; a dropped one keeps its (now empty) manifest row, exactly as a
  sparsified level does.
* **Points**, on an object-less point cloud: the multiset of exact vertex
  coordinates (bytes) present at ``L + 1`` is removed from ``L``.

The cut works on stored cells, not through a reader and writer: vertices,
``vertex_fragments``, per-vertex and per-fragment attributes, links of every
offsets array at delta 0 with their attributes, and the object index are
rewritten row-for-row, so what stays is byte-identical, keeps its chunk,
codec and sharding, and nothing has to be re-encoded.  Cross-level link
families (``delta != 0``) are removed from every level: a cross-level link
joins one object's vertices at two levels, and an object of an additive
pyramid lives at one.

This does not decide whether additive refinement *makes sense* for a
pyramid -- that a coarse level's objects are worth drawing at full
resolution beside a finer level's.  Builders warn about that
(``zarr_vectors_tools``); this module only performs the conversion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import (
    CAP_ADDITIVE_LEVELS,
    CAP_MULTISCALE_LINKS,
    ENCODING_DRACO,
    FRAGMENT_ATTRIBUTES,
    LINK_ATTRIBUTES,
    LINKS,
    OBJECT_INDEX,
    REFINEMENT_ADD,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
    XLEVEL_NONE,
)
from zarr_vectors.core.group import Group
from zarr_vectors.exceptions import CoarseningError

_WINDOW = 64


def make_levels_additive(
    store: Any,
    levels: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Make ``levels`` (default: every level but the coarsest) additive.

    Level 0 is rewritten in place: afterwards it holds only the objects or
    points no coarser level has, and level 0 *alone* is no longer the whole
    dataset -- a reader must follow the chain (every reader in this package
    does by default; ``own_level_only=True`` sees the stored data).

    The root gains ``CAP_ADDITIVE_LEVELS`` in ``required_capabilities``
    before anything else changes, and each level is marked additive before
    its data is cut, so an interrupted conversion reads back with some
    objects twice, never with any missing.

    Returns:
        One report per converted level: ``level``, ``mode`` (``"objects"``
        or ``"points"``), the vertices and objects before and after.

    Raises:
        CoarseningError: If the pyramid cannot be converted (levels not
            contiguous, a level already additive, Draco-encoded vertices,
            a level whose objects cannot be matched to the next).
    """
    from zarr_vectors.core.refinement import (
        declare_additive_levels,
        declare_required_capability,
        level_refinement,
    )
    from zarr_vectors.core.store import (
        get_resolution_level,
        list_resolution_levels,
        open_store,
    )

    root = store if isinstance(store, Group) else open_store(str(store), mode="r+")
    present = list_resolution_levels(root)
    if present != list(range(len(present))):
        raise CoarseningError(f"levels {present} are not contiguous from 0")
    if len(present) < 2:
        raise CoarseningError("a single-level store has nothing to make additive")
    targets = sorted(present[:-1] if levels is None else {int(lv) for lv in levels})
    for lv in targets:
        if lv not in present or lv + 1 not in present:
            raise CoarseningError(
                f"level {lv} cannot be additive: it needs a level {lv + 1}"
            )
    for lv in present:
        if level_refinement(get_resolution_level(root, lv)) == REFINEMENT_ADD:
            raise CoarseningError(f"level {lv} is already additive")
    for lv in present:
        vmeta = get_resolution_level(root, lv).read_array_meta(VERTICES) or {}
        if vmeta.get("encoding") == ENCODING_DRACO:
            raise CoarseningError(
                f"level {lv} stores Draco-encoded vertices, which cannot be "
                f"cut row by row"
            )

    declare_required_capability(root, CAP_ADDITIVE_LEVELS)
    _drop_cross_level_links(root, present)

    reports = []
    # Finest first: level L is cut against level L + 1 while that level is
    # still complete.
    for lv in targets:
        declare_additive_levels(root, [lv])
        reports.append(_cut_level(root, lv))
    return reports


# ------------------------------------------------------------------ levels


def _drop_cross_level_links(root: Group, levels: Sequence[int]) -> None:
    """Remove every ``links/<delta>`` and ``link_attributes/*/<delta>``
    family with ``delta != 0``, and say so on the root."""
    from zarr_vectors.core.paths import parse_delta
    from zarr_vectors.core.store import (
        get_resolution_level,
        read_root_metadata,
        update_root_metadata,
    )

    removed = False
    for lv in levels:
        lg = get_resolution_level(root, lv)
        families = []
        if lg.array_exists(LINKS):
            families.append(LINKS)
        if lg.array_exists(LINK_ATTRIBUTES):
            families.extend(
                f"{LINK_ATTRIBUTES}/{name}"
                for name in lg[LINK_ATTRIBUTES].children()
            )
        for family in families:
            try:
                children = list(lg[family].children())
            except Exception:  # noqa: BLE001
                continue
            for name in children:
                try:
                    delta = parse_delta(name)
                except ValueError:
                    continue
                if delta != 0:
                    lg.delete_subtree(f"{family}/{name}")
                    removed = True
    meta = read_root_metadata(root)
    caps = [c for c in meta.format_capabilities if c != CAP_MULTISCALE_LINKS]
    if removed or meta.cross_level_depth or meta.cross_level_storage != XLEVEL_NONE:
        update_root_metadata(
            root, cross_level_depth=0, cross_level_storage=XLEVEL_NONE,
            format_capabilities=caps,
        )


def _cut_level(root: Group, level: int) -> dict[str, Any]:
    from zarr_vectors.core.refinement import stored_object_ids
    from zarr_vectors.core.store import get_resolution_level, read_root_metadata

    lg = get_resolution_level(root, level)
    nxt = get_resolution_level(root, level + 1)
    ndim = read_root_metadata(root).sid_ndim
    own_ids = stored_object_ids(lg)
    if own_ids is not None:
        next_ids = stored_object_ids(nxt)
        if next_ids is None:
            raise CoarseningError(
                f"level {level} tracks objects but level {level + 1} does not, "
                f"so its objects cannot be matched"
            )
        keep = np.setdiff1d(own_ids, next_ids)
        report = _subset_level(root, lg, ndim, keep_objects=keep)
        report.update(
            mode="objects", objects_before=int(own_ids.size),
            objects_after=int(keep.size),
        )
    else:
        drop = _points_in(nxt, ndim)
        report = _subset_level(root, lg, ndim, drop_points=drop)
        report.update(mode="points")
    report["level"] = level
    return report


def _points_in(level_group: Group, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Every vertex of a level as raw row bytes: ``(unique rows, counts)``."""
    from zarr_vectors.core.arrays import list_chunk_keys, vertices_dtype

    row_bytes = vertices_dtype(level_group).itemsize * ndim
    rows = []
    for cc in list_chunk_keys(level_group):
        raw = level_group.read_bytes(VERTICES, _key(cc))
        if raw:
            rows.append(_as_rows(raw, row_bytes))
    if not rows:
        empty = np.zeros(0, dtype=np.dtype((np.void, row_bytes)))
        return empty, np.zeros(0, dtype=np.int64)
    uniq, counts = np.unique(np.concatenate(rows), return_counts=True)
    return uniq, counts.astype(np.int64)


# ------------------------------------------------------------------- cells


def _key(cc: Sequence[int]) -> str:
    return ".".join(str(int(c)) for c in cc)


def _as_rows(raw: bytes, row_bytes: int) -> np.ndarray:
    """``raw`` as a 1-D array of fixed-width opaque rows."""
    return np.frombuffer(raw, dtype=np.dtype((np.void, row_bytes)))


def _subset_level(
    root: Group,
    lg: Group,
    ndim: int,
    *,
    keep_objects: npt.NDArray[np.int64] | None = None,
    drop_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Rewrite a level keeping only ``keep_objects`` (or dropping the
    multiset ``drop_points``)."""
    from zarr_vectors.core.arrays import (
        _maybe_batched_reads,
        finalize_links,
        list_chunk_keys,
        read_object_manifest_rows,
        read_vertex_fragment_index,
        stamp_fragments_tile,
        vertices_dtype,
        write_object_index,
    )
    from zarr_vectors.core.link_groups import (
        FRAGMENT_LINK_GROUPS,
        stamp_fragment_link_groups,
    )
    from zarr_vectors.core.store import update_level_metadata

    claims = lg.level_claims()
    row_bytes = vertices_dtype(lg).itemsize * ndim
    keys = [tuple(int(c) for c in cc) for cc in list_chunk_keys(lg)]
    vertices_before = 0

    # ---- pass 1: which rows (and fragments) of each chunk stay ----------
    kept_frags: dict[tuple[int, ...], list[int]] = {}
    masks: dict[tuple[int, ...], np.ndarray] = {}
    manifests_new: dict[int, list] | None = None
    oi_meta: dict[str, Any] = {}
    if keep_objects is not None:
        ids, manifests = read_object_manifest_rows(lg)
        keep_set = set(int(i) for i in keep_objects)
        wanted: dict[tuple[int, ...], set[int]] = {}
        for oid, manifest in zip(ids.tolist(), manifests):
            if oid not in keep_set:
                continue
            for cc, frag in manifest:
                wanted.setdefault(tuple(int(c) for c in cc), set()).add(int(frag))
        oi_meta = lg.read_array_meta(OBJECT_INDEX) or {}
    remaining = None
    if drop_points is not None:
        remaining = drop_points[1].copy()

    key_strs = [_key(cc) for cc in keys]
    with _maybe_batched_reads(lg, [(VERTEX_FRAGMENTS, key_strs)]):
        fragment_indexes = {
            cc: read_vertex_fragment_index(lg, cc) for cc in keys
        }
    for cc in keys:
        fi = fragment_indexes[cc]
        raw = lg.read_bytes(VERTICES, _key(cc)) if drop_points is not None else None
        n_rows = (
            len(raw) // row_bytes if raw is not None else _extent(fi)
        )
        if keep_objects is not None:
            frags = sorted(f for f in wanted.get(cc, ()) if f < fi.num_fragments)
            mask = np.zeros(max(n_rows, _extent(fi)), dtype=bool)
            for f in frags:
                mask[_fragment_rows(fi, f)] = True
            kept_frags[cc] = frags
        else:
            mask = np.ones(n_rows, dtype=bool)
            uniq, _counts = drop_points
            if raw and uniq.size:
                rows = _as_rows(raw, row_bytes)
                pos = np.searchsorted(uniq, rows)
                pos_c = np.minimum(pos, uniq.size - 1)
                hit = (pos < uniq.size) & (uniq[pos_c] == rows)
                for i in np.flatnonzero(hit):
                    j = pos_c[i]
                    if remaining[j] > 0:
                        remaining[j] -= 1
                        mask[i] = False
        masks[cc] = mask

    # ---- pass 2: rewrite every cell of every chunk ----------------------
    vertex_attrs = _children(lg, VERTEX_ATTRIBUTES)
    fragment_attrs = _children(lg, FRAGMENT_ATTRIBUTES)
    link_arrays = _link_arrays(lg)
    vertices_after = 0
    for start in range(0, len(keys), _WINDOW):
        window = keys[start:start + _WINDOW]
        with lg.batched_writes():
            for cc in window:
                fi = fragment_indexes[cc]
                mask = masks[cc]
                kept, before = _rewrite_chunk(
                    lg, cc, fi, mask, kept_frags.get(cc),
                    row_bytes=row_bytes, vertex_attrs=vertex_attrs,
                    fragment_attrs=fragment_attrs,
                )
                vertices_before += before
                vertices_after += kept
                for array in link_arrays:
                    _rewrite_links(lg, array, cc, masks, kept_frags, fi)

    if keep_objects is not None:
        manifests_new = {}
        remaps = {
            cc: {old: new for new, old in enumerate(frags)}
            for cc, frags in kept_frags.items()
        }
        keep_set = set(int(i) for i in keep_objects)
        for oid, manifest in zip(ids.tolist(), manifests):
            if oid in keep_set:
                manifests_new[oid] = [
                    (tuple(cc), remaps[tuple(int(c) for c in cc)][int(f)])
                    for cc, f in manifest
                ]
            else:
                manifests_new[oid] = []
        write_object_index(
            lg, manifests_new, int(oi_meta.get("sid_ndim", ndim)),
        )

    if link_arrays:
        finalize_links(lg, delta=0)
    update_level_metadata(lg, vertex_count=int(vertices_after))
    # The rewrite withdrew the level's claims; the ones that still hold are
    # stamped again, verified against what is now stored.
    if claims.get("fragments_tile"):
        stamp_fragments_tile(lg, ndim)
    if claims.get(FRAGMENT_LINK_GROUPS):
        stamp_fragment_link_groups(lg, root)
    return {
        "vertices_before": int(vertices_before),
        "vertices_after": int(vertices_after),
    }


def _extent(fi: Any) -> int:
    return int(fi.vertex_extent) if fi.num_fragments else 0


def _fragment_rows(fi: Any, f: int) -> np.ndarray:
    if fi.is_range(f):
        start, count = fi.range(f)
        return np.arange(start, start + count, dtype=np.int64)
    return np.asarray(fi.indices(f), dtype=np.int64)


def _children(lg: Group, family: str) -> list[str]:
    if not lg.array_exists(family):
        return []
    try:
        return sorted(lg[family].children())
    except Exception:  # noqa: BLE001
        return []


def _rewrite_chunk(
    lg: Group,
    cc: tuple[int, ...],
    fi: Any,
    mask: np.ndarray,
    frags: list[int] | None,
    *,
    row_bytes: int,
    vertex_attrs: list[str],
    fragment_attrs: list[str],
) -> tuple[int, int]:
    """Rewrite one chunk's vertices, fragments and their attributes.
    Returns ``(rows kept, rows before)``."""
    from zarr_vectors.encoding.fragments import encode_fragments

    key = _key(cc)
    raw = lg.read_bytes(VERTICES, key)
    n_rows = len(raw) // row_bytes if raw else 0
    if mask.size < n_rows:
        mask = np.concatenate([mask, np.zeros(n_rows - mask.size, dtype=bool)])
    mask = mask[:n_rows]
    kept = int(mask.sum())
    if kept == n_rows and frags is None:
        return kept, n_rows
    new_index = np.cumsum(mask) - 1  # old row -> new row, where kept

    if kept == 0:
        _blank_chunk(lg, key, vertex_attrs, fragment_attrs)
        return 0, n_rows

    lg.write_bytes(VERTICES, key, _as_rows(raw, row_bytes)[mask].tobytes())

    order = range(fi.num_fragments) if frags is None else frags
    fragments: list[Any] = []
    for f in order:
        rows = _fragment_rows(fi, f)
        rows = rows[(rows < n_rows)]
        rows = rows[mask[rows]] if rows.size else rows
        if fi.is_range(f):
            start = fi.range(f)[0]
            new_start = int(np.count_nonzero(mask[:start]))
            fragments.append((new_start, int(rows.size)))
        elif rows.size:
            fragments.append(np.asarray(new_index[rows], dtype=np.int64))
        else:
            fragments.append((0, 0))
    lg.write_bytes(VERTEX_FRAGMENTS, key, encode_fragments(fragments))

    for name in vertex_attrs:
        path = f"{VERTEX_ATTRIBUTES}/{name}"
        cell = lg.read_bytes(path, key)
        if not cell:
            continue
        if len(cell) % n_rows:
            raise CoarseningError(
                f"{path} cell {key} does not hold one row per vertex"
            )
        lg.write_bytes(path, key, _as_rows(cell, len(cell) // n_rows)[mask].tobytes())

    if frags is not None:
        for name in fragment_attrs:
            path = f"{FRAGMENT_ATTRIBUTES}/{name}"
            cell = lg.read_bytes(path, key)
            if not cell or not fi.num_fragments:
                continue
            if len(cell) % fi.num_fragments:
                raise CoarseningError(
                    f"{path} cell {key} does not hold one row per fragment"
                )
            rows = _as_rows(cell, len(cell) // fi.num_fragments)
            lg.write_bytes(path, key, rows[np.asarray(frags, dtype=np.int64)].tobytes())
    return kept, n_rows


def _blank_chunk(
    lg: Group, key: str, vertex_attrs: list[str], fragment_attrs: list[str],
) -> None:
    """Empty every per-vertex cell of a chunk that keeps nothing."""
    lg.write_bytes(VERTICES, key, b"")
    lg.write_bytes(VERTEX_FRAGMENTS, key, b"")
    for family, names in ((VERTEX_ATTRIBUTES, vertex_attrs),
                          (FRAGMENT_ATTRIBUTES, fragment_attrs)):
        for name in names:
            path = f"{family}/{name}"
            if lg.chunk_exists(path, key):
                lg.write_bytes(path, key, b"")


# ------------------------------------------------------------------- links


def _link_arrays(lg: Group) -> list[dict[str, Any]]:
    """Every delta-0 link array of the level, with what decoding it takes."""
    from zarr_vectors.core.arrays import (
        link_family_policy,
        links_has_perm,
        list_link_attribute_offsets,
        list_link_offsets,
    )
    from zarr_vectors.core.paths import is_intra, links_group_path, parse_offsets

    policy = link_family_policy(lg, 0)
    if policy is None:
        return []
    link_width, sid_ndim, directed, store = policy
    attr_names = _children(lg, LINK_ATTRIBUTES)
    out = []
    for seg in list_link_offsets(lg, 0):
        name = f"{links_group_path(0)}/{seg}"
        meta = lg.read_array_meta(name) or {}
        raw_offsets = meta.get("offsets")
        if raw_offsets is not None:
            offsets = tuple(tuple(int(c) for c in o) for o in raw_offsets)
        else:
            offsets = parse_offsets(seg, sid_ndim=sid_ndim, link_width=link_width)
        has_perm = bool(meta.get("has_perm", links_has_perm(
            offsets, delta=0, directed=directed, store=store,
        )))
        out.append({
            "name": name,
            "segment": seg,
            "offsets": offsets,
            "intra": is_intra(offsets),
            "dtype": np.dtype(meta.get("dtype", "int64")),
            "width": link_width + (1 if has_perm else 0),
            "perm": 1 if has_perm else 0,
            "link_width": link_width,
            "attributes": [
                f"{LINK_ATTRIBUTES}/{a}/0/{seg}" for a in attr_names
                if seg in list_link_attribute_offsets(lg, a, 0)
            ],
        })
    return out


def _rewrite_links(
    lg: Group,
    array: dict[str, Any],
    cc: tuple[int, ...],
    masks: dict[tuple[int, ...], np.ndarray],
    kept_frags: dict[tuple[int, ...], list[int]],
    fi: Any,
) -> None:
    """Rewrite one link cell (and its attribute cells): keep the links
    whose every endpoint stays, renumbered into the kept rows."""
    from zarr_vectors.core.arrays import read_link_fragment_index, write_chunk_links
    from zarr_vectors.core.link_groups import _expand_rows
    from zarr_vectors.encoding.ragged import decode_ragged_blob, encode_ragged_blob

    key = _key(cc)
    name = array["name"]
    if not lg.chunk_exists(name, key):
        return
    raw = lg.read_bytes(name, key)
    if not raw:
        return
    dtype, width, perm = array["dtype"], array["width"], array["perm"]
    if array["intra"]:
        full = np.frombuffer(raw, dtype=dtype).reshape(-1, width)
        lf = read_link_fragment_index(lg, cc)
        group_rows, group_ids = _expand_rows(lf)
        if group_rows.size != len(full) or np.unique(group_rows).size != len(full):
            raise CoarseningError(
                f"{name} cell {key}: its link groups do not hold every link "
                f"exactly once, so it cannot be cut"
            )
        groups_physical = [
            group_rows[group_ids == g] for g in range(lf.num_fragments)
        ]
    else:
        blocks = decode_ragged_blob(raw, dtype, ncols=width)
        sizes = [len(b) for b in blocks]
        full = (
            np.concatenate(blocks, axis=0) if blocks
            else np.zeros((0, width), dtype=dtype)
        )
        bounds = np.cumsum([0] + sizes)
        groups_physical = [
            np.arange(bounds[i], bounds[i + 1], dtype=np.int64)
            for i in range(len(sizes))
        ]

    # Which links survive, and their endpoints renumbered.
    endpoints = full[:, perm:].astype(np.int64)
    keep = np.ones(len(full), dtype=bool)
    renumbered = endpoints.copy()
    for k in range(endpoints.shape[1]):
        target = cc if k == 0 else tuple(
            int(a) + int(b) for a, b in zip(cc, array["offsets"][k - 1])
        )
        mask = masks.get(target)
        col = endpoints[:, k]
        if mask is None:
            keep &= False
            continue
        ok = (col >= 0) & (col < mask.size)
        ok[ok] = mask[col[ok]]
        keep &= ok
        index = np.cumsum(mask) - 1
        renumbered[ok, k] = index[col[ok]]
    rows = full.copy()
    rows[:, perm:] = renumbered.astype(dtype)

    groups_kept = [g[keep[g]] for g in groups_physical]
    if array["intra"]:
        frags = kept_frags.get(cc)
        if (
            frags is not None
            and len(groups_kept) == fi.num_fragments
            and all(not groups_kept[f].size for f in set(range(fi.num_fragments)) - set(frags))
        ):
            # Groups that followed the fragments keep following them.
            groups_kept = [groups_kept[f] for f in frags]
    order = (
        np.concatenate(groups_kept) if groups_kept else np.zeros(0, dtype=np.int64)
    )
    if array["intra"]:
        write_chunk_links(
            lg, cc, [rows[g] for g in groups_kept], dtype=dtype, delta=0,
            offsets=array["offsets"], link_width=array["link_width"],
        )
    else:
        nonempty = [rows[g] for g in groups_kept if g.size]
        lg.write_bytes(
            name, key, encode_ragged_blob(nonempty, dtype) if nonempty else b"",
        )
    for path in array["attributes"]:
        cell = lg.read_bytes(path, key) if lg.chunk_exists(path, key) else b""
        if not cell or not len(full):
            continue
        if len(cell) % len(full):
            raise CoarseningError(f"{path} cell {key} does not align with its links")
        lg.write_bytes(
            path, key, _as_rows(cell, len(cell) // len(full))[order].tobytes()
            if order.size else b"",
        )


__all__ = ["make_levels_additive"]
