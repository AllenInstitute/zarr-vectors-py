"""The dense object-index layout: manifests as fixed-width integer arrays.

The vlen layout stores one ``bytes`` blob per object, so a million
objects is a million Python objects to write and a million to decode, on
the host, whatever the caller held them as. The dense layout stores the
same manifests as two numeric arrays::

    object_index/manifest_spans    int64 (n_objects, 2)   start, count
    object_index/manifest_blocks   int64 (n_blocks, sid_ndim + 1)
                                          chunk coords..., fragment index
    object_index/object_ids        int64 (n_objects,)     as in V2

Object row ``o`` owns blocks ``start .. start + count``, one row per
fragment it references, in manifest order. An object with no fragments
has ``count == 0``. Range and explicit-list blocks of the vlen encoding
are expanded to one row per fragment, so every block is fixed width and
a whole-index read is a gather, never a decode.

Spans rather than offsets, so that writes stay proportional to what they
change: an append adds span rows and block rows; a patch appends the
replacement blocks and rewrites only the patched objects' span rows,
leaving the blocks they used to own unreferenced until the next full
rewrite (``vacuum``, or any ``write_object_index``) compacts them.

Selected per store by the root's ``manifest_layout`` (``"dense"``), or
per index by the first write that creates it. An index keeps its layout:
appends and patches never convert one layout into the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.exceptions import ArrayError

if TYPE_CHECKING:  # pragma: no cover
    from zarr_vectors.core.group import Group

#: ``object_index`` ``layout`` value for this layout.
OBJECT_INDEX_LAYOUT_DENSE = "dense_manifests_v1"

SPANS_ARRAY = "manifest_spans"
BLOCKS_ARRAY = "manifest_blocks"

#: Rows per zarr chunk, for spans and blocks alike: the same bucket the
#: vlen manifests use, for the same reason (a fixed chunk, set by the
#: creating write, never clamped to that write's size).
_BUCKET = 16_384

SPANS_PATH = f"{OBJECT_INDEX}/{SPANS_ARRAY}"
BLOCKS_PATH = f"{OBJECT_INDEX}/{BLOCKS_ARRAY}"


def is_dense(level_group: Group, meta: dict[str, Any] | None = None) -> bool:
    """Whether ``level_group``'s object index uses this layout.

    The ``layout`` field decides; failing that, the arrays do -- a caller
    that stamps its own ``object_index`` metadata after an append may
    carry a vlen ``layout`` over from habit, and the arrays on disk are
    the truth about what was written.
    """
    if meta is None:
        try:
            meta = level_group.read_array_meta(OBJECT_INDEX) or {}
        except Exception:
            meta = {}
    if meta.get("layout") == OBJECT_INDEX_LAYOUT_DENSE:
        return True
    # ``manifests`` first: on a vlen index it is the node a read has
    # already resolved, so the common case costs no extra lookup.
    if level_group.array_exists(f"{OBJECT_INDEX}/manifests"):
        return False
    return level_group.array_exists(SPANS_PATH)


def store_default(level_group: Group) -> str:
    """The store's ``manifest_layout`` (``"vlen"`` unless it says ``"dense"``)."""
    from zarr_vectors.core.group import Group

    try:
        root = Group._from_backend(level_group._backend, "")
        zv = root.attrs.to_dict().get("zarr_vectors") or {}
    except Exception:
        return "vlen"
    return "dense" if zv.get("manifest_layout") == "dense" else "vlen"


# --------------------------------------------------------------------
# Converting between forms


def csr_from_manifests(
    manifests: list[Any], sid_ndim: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(offsets, coords, frags)`` for manifests given as ``(coords, frag)`` lists."""
    counts = np.fromiter((len(m or ()) for m in manifests), dtype=np.int64, count=len(manifests))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    flat = [pair for m in manifests for pair in (m or ())]
    if total:
        coords = np.asarray([tuple(cc) for cc, _ in flat], dtype=np.int64).reshape(total, sid_ndim)
        frags = np.fromiter((int(fi) for _, fi in flat), dtype=np.int64, count=total)
    else:
        coords = np.empty((0, sid_ndim), dtype=np.int64)
        frags = np.empty(0, dtype=np.int64)
    return offsets, coords, frags


def manifests_from_csr(
    offsets: npt.NDArray[np.int64], coords: npt.NDArray[np.int64], frags: npt.NDArray[np.int64],
) -> list[list[tuple[tuple[int, ...], int]]]:
    """The ``[(chunk_coords, fragment_index), ...]`` lists the manifest API returns."""
    pairs = list(zip(map(tuple, coords.tolist()), frags.tolist()))
    bounds = offsets.tolist()
    return [pairs[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def _gather(spans: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(offsets, block_rows)``: CSR offsets over ``spans`` and the block
    row each output row comes from."""
    counts = spans[:, 1]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(offsets[-1])
    if total == 0:
        return offsets, np.empty(0, dtype=np.int64)
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets[:-1], counts)
    return offsets, np.repeat(spans[:, 0], counts) + within


# --------------------------------------------------------------------
# Reading


def _node(level_group: Group, path: str) -> Any:
    return level_group._require_array_node(path)


#: An index array is read as slices of its runs of consecutive rows when
#: there are at most this many runs, or when they average at least a zarr
#: chunk; otherwise by one orthogonal selection. zarr's orthogonal indexer
#: costs per index (6.3 s for the 147.6M block rows of a 100 um export),
#: but a slice decodes every chunk it touches, so short runs decode the
#: same chunk again and again: 75k runs of ~200 rows took 115 s as slices
#: against 1.0 s through the indexer.
_SLICE_RUNS_MAX = 64


def _rows(level_group: Group, path: str, rows: npt.NDArray[np.int64] | slice) -> np.ndarray:
    """Rows of a numeric array, in the order asked, through the offline
    snapshot when one is active."""
    if level_group._offline is not None:
        return np.asarray(level_group.read_array(path))[rows]
    node = _node(level_group, path)
    if isinstance(rows, slice):
        return np.asarray(node[rows])
    if rows.size == 0:
        return np.empty((0, *node.shape[1:]), dtype=node.dtype)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if int(rows.min()) < 0 or int(rows.max()) >= int(node.shape[0]):
        # zarr's bounds checks and negative indices, not a slice's clipping.
        return np.asarray(node.get_orthogonal_selection((rows,)))
    if rows.size > 1 and not bool(np.all(rows[1:] > rows[:-1])):
        uniq, inverse = np.unique(rows, return_inverse=True)
        return _rows_increasing(node, uniq)[inverse.reshape(-1)]
    return _rows_increasing(node, rows)


def _rows_increasing(node: Any, rows: npt.NDArray[np.int64]) -> np.ndarray:
    """``node[rows]`` for strictly increasing, in-bounds ``rows``: each run
    of consecutive rows read as a slice, all runs in one gather."""
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    n_runs = int(breaks.size) + 1
    if n_runs > _SLICE_RUNS_MAX and rows.size < int(node.chunks[0]) * n_runs:
        return np.asarray(node.get_orthogonal_selection((rows,)))
    starts = rows[np.concatenate([[0], breaks])]
    stops = rows[np.concatenate([breaks - 1, [rows.size - 1]])] + 1
    if n_runs == 1:
        return np.asarray(node[int(starts[0]):int(stops[0])])
    import asyncio

    from zarr.core.sync import sync

    async def _all() -> list[Any]:
        return await asyncio.gather(*(
            node._async_array.getitem(slice(int(a), int(b)))
            for a, b in zip(starts, stops)
        ))

    return np.concatenate([np.asarray(part) for part in sync(_all())], axis=0)


def num_rows(level_group: Group) -> int:
    if level_group._offline is not None:
        return int(np.asarray(level_group.read_array(SPANS_PATH)).shape[0])
    return int(_node(level_group, SPANS_PATH).shape[0])


def read_csr(
    level_group: Group, rows: npt.NDArray[np.int64] | None = None, *, stop: int | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(offsets, coords, frags)`` for ``rows`` (all rows ``[0, stop)`` when None).

    Ascending object rows written in order -- what ascending ids give --
    own ascending block rows, so those are read as they are: no
    ``unique``, no ``searchsorted``, and a run of them as a slice. Rows
    in any other order cost one ``unique``.
    """
    sel: Any = slice(0, stop) if rows is None else np.asarray(rows, dtype=np.int64)
    spans = _rows(level_group, SPANS_PATH, sel).astype(np.int64, copy=False).reshape(-1, 2)
    offsets, block_rows = _gather(spans)
    if block_rows.size == 0:
        sid = _sid(level_group)
        return offsets, np.empty((0, sid), np.int64), np.empty(0, np.int64)
    n_blocks = num_blocks(level_group)
    increasing = bool(np.all(block_rows[1:] > block_rows[:-1]))
    lo, hi = (
        (int(block_rows[0]), int(block_rows[-1])) if increasing
        else (int(block_rows.min()), int(block_rows.max()))
    )
    if lo < 0 or hi >= n_blocks:
        raise ArrayError(
            f"object_index/{SPANS_ARRAY} refers to block rows outside "
            f"{BLOCKS_ARRAY} ({n_blocks} rows)"
        )
    if increasing and hi - lo + 1 == block_rows.size:
        blocks = _rows(level_group, BLOCKS_PATH, slice(lo, hi + 1))
    elif increasing:
        blocks = _rows(level_group, BLOCKS_PATH, block_rows)
    else:
        uniq, inverse = np.unique(block_rows, return_inverse=True)
        blocks = np.asarray(_rows(level_group, BLOCKS_PATH, uniq))[inverse.reshape(-1)]
    blocks = np.asarray(blocks, dtype=np.int64)
    return offsets, blocks[:, :-1], blocks[:, -1]


def num_blocks(level_group: Group) -> int:
    if level_group._offline is not None:
        return int(np.asarray(level_group.read_array(BLOCKS_PATH)).shape[0])
    return int(_node(level_group, BLOCKS_PATH).shape[0])


def _sid(level_group: Group) -> int:
    meta = level_group.read_array_meta(OBJECT_INDEX) or {}
    return int(meta.get("sid_ndim", 3))


def present_mask(level_group: Group, stop: int | None = None) -> npt.NDArray[np.bool_]:
    """Which object rows hold at least one fragment."""
    spans = _rows(level_group, SPANS_PATH, slice(0, stop)).reshape(-1, 2)
    return np.asarray(spans[:, 1] > 0)


# --------------------------------------------------------------------
# Writing


def _create(level_group: Group, path: str, data: npt.NDArray[np.int64]) -> None:
    from zarr_vectors.core.arrays import _object_shards

    chunks = (_BUCKET, *data.shape[1:])
    level_group.write_array(
        path, data, chunks=chunks, shards=_object_shards(level_group, chunks),
    )


def _truncate(level_group: Group, path: str, n: int) -> None:
    """Shrink ``path`` to its first ``n`` rows, storing what writing only them would.

    ``resize`` drops the storage objects wholly past the new end, but the
    object holding row ``n`` keeps the old rows after it. They are cleared
    to the fill value first -- up to the end of that object (the shard when
    sharded) -- so its bytes are a fresh write's, and inner chunks left
    holding only fill are not stored.
    """
    node = _node(level_group, path)
    n0 = int(node.shape[0])
    if n >= n0:
        return
    outer = int((node.shards or node.chunks)[0])
    end = min(n0, -(-n // outer) * outer)
    if end > n:
        node[n:end] = node.fill_value
    if "shape" in node.attrs:
        node.metadata.attributes["shape"] = [n, *node.shape[1:]]
    node.resize((n, *node.shape[1:]))
    level_group._invalidate_node(path)


def write(
    level_group: Group,
    offsets: npt.NDArray[np.int64],
    coords: npt.NDArray[np.int64],
    frags: npt.NDArray[np.int64],
    *,
    sid_ndim: int,
    mode: str = "replace",
    at: int | None = None,
) -> int:
    """Write manifests given as CSR arrays; returns the row they start at.

    ``replace`` rewrites spans and blocks. ``append`` adds rows at ``at``
    (default: the end), padding a gap with empty objects; an ``at`` before
    the end (a torn flush's residue) truncates the spans there first. The
    object id table and the index metadata are the caller's, as they are
    for the vlen writer.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, sid_ndim)
    frags = np.asarray(frags, dtype=np.int64).reshape(-1)
    counts = np.diff(offsets)
    n = int(counts.size)
    if coords.shape[0] != frags.size or int(offsets[-1]) != frags.size:
        raise ArrayError(
            f"{frags.size} fragments, {coords.shape[0]} coordinate rows and "
            f"offsets ending at {int(offsets[-1])} disagree"
        )
    if frags.size and int(frags.min()) < 0:
        raise ArrayError(f"fragment_index must be >= 0, got {int(frags.min())}")
    new_blocks = np.concatenate([coords, frags[:, None]], axis=1)
    exists = level_group.array_exists(SPANS_PATH) and level_group.array_exists(BLOCKS_PATH)

    if mode == "replace" or not exists:
        start = 0 if mode == "replace" else int(at or 0)
        spans = np.empty((start + n, 2), dtype=np.int64)
        spans[:start] = (0, 0)
        spans[start:, 0] = offsets[:-1]
        spans[start:, 1] = counts
        _create(level_group, SPANS_PATH, spans)
        _create(level_group, BLOCKS_PATH, new_blocks)
        if not exists:
            _declare_capability(level_group)
        return start

    n0 = num_rows(level_group)
    b0 = num_blocks(level_group)
    start = n0 if at is None else int(at)
    if start < 0:
        raise ArrayError(f"Append index {at} is negative")
    if start < n0:
        # Residue past the commit point: keep the rows before ``start`` and
        # the blocks they use, and drop the rest of both -- the blocks too,
        # or every rewrite from ``start`` leaves the old rows' blocks behind,
        # unreferenced, and appends after them. Rows' blocks are laid out in
        # the order written, so the kept rows use blocks below the highest
        # end any of them names (a patch that moved a row's blocks past
        # residue keeps that residue too: never less than is referenced).
        keep = _kept_block_end(level_group, start, n0, b0)
        if keep < b0:
            _truncate(level_group, BLOCKS_PATH, keep)
            b0 = keep
        # The kept rows stay where they are: cut the spans at ``start`` --
        # clearing the rest of the object holding the new last row, as a
        # fresh write stores it -- and append, never read them back to
        # rewrite them.
        _truncate(level_group, SPANS_PATH, start)
    tail = np.empty((max(start - n0, 0) + n, 2), dtype=np.int64)
    pad = max(start - n0, 0)
    tail[:pad] = (b0, 0)
    tail[pad:, 0] = b0 + offsets[:-1]
    tail[pad:, 1] = counts
    if new_blocks.size:
        level_group.extend_array(BLOCKS_PATH, new_blocks)
    if tail.size:
        level_group.extend_array(SPANS_PATH, tail)
    return start


#: Span rows read per piece when a resume has to scan the kept rows.
_SCAN_ROWS = 1 << 20


def _span_pieces(level_group: Group, lo: int, hi: int) -> Any:
    """Span rows ``[lo, hi)`` as ``(n, 2)`` int64 pieces of bounded size."""
    for a in range(lo, hi, _SCAN_ROWS):
        b = min(a + _SCAN_ROWS, hi)
        yield _rows(level_group, SPANS_PATH, slice(a, b)).astype(
            np.int64, copy=False,
        ).reshape(-1, 2)


def _kept_block_end(level_group: Group, start: int, n0: int, b0: int) -> int:
    """The highest block end any of rows ``[0, start)`` names (0 for none).

    What an append at ``start < n0`` truncates ``manifest_blocks`` to.
    Usually proved from the residue alone, without reading the rows
    kept: when every row from ``start - 1`` on begins where the row
    before it ended and the last ends at the block count -- the layout
    appends leave -- the residue's blocks are exactly ``[E, b0)``, with
    ``E`` the end of row ``start - 1``, and no kept row's blocks can lie
    in them (no two rows share blocks; every writer gives each row its
    own), so the answer is ``E``. That reads ``n0 - start + 1`` rows, the
    ones being replaced.

    Otherwise -- a patch or a placed range moved blocks out of row order,
    or reserved blocks sit past the last row -- the kept rows are scanned,
    in pieces of :data:`_SCAN_ROWS`, so memory stays bounded.

    The proof does not see a kept row a patch emptied after the residue
    was written: such a row's (empty) span starts at the block count of
    that time, which the scan counts, keeping the residue's blocks
    unreferenced behind it; the proof drops them. Every row reads the
    same either way.
    """
    if start <= 0:
        return 0
    consecutive = True
    prev_end: int | None = None
    first_end = 0
    for piece in _span_pieces(level_group, start - 1, n0):
        ends = piece[:, 0] + piece[:, 1]
        if prev_end is None:
            first_end = int(ends[0])
        elif int(piece[0, 0]) != prev_end:
            consecutive = False
            break
        if not np.array_equal(piece[1:, 0], ends[:-1]):
            consecutive = False
            break
        prev_end = int(ends[-1])
    if consecutive and prev_end == b0:
        return first_end
    keep = 0
    for piece in _span_pieces(level_group, 0, start):
        keep = max(keep, int((piece[:, 0] + piece[:, 1]).max()))
    return keep


def _declare_capability(level_group: Group) -> None:
    """Add ``dense_manifests`` to the root's ``format_capabilities``.

    ``create_store(manifest_layout="dense")`` stamps it, but an index can
    also become dense later -- ``write_object_manifests(layout="dense")``
    on a store created without the choice -- and the store should say so
    either way. A root without a zarr-vectors block (a bare group in a
    test) is left alone.
    """
    import zarr

    from zarr_vectors.constants import CAP_DENSE_MANIFESTS
    from zarr_vectors.core.store import update_root_metadata

    try:
        root_zarr = zarr.open_group(
            level_group._zarr.store, path="/", mode="r+", zarr_format=3,
        )
    except Exception:  # noqa: BLE001 - nowhere to declare it
        return
    block = root_zarr.attrs.get("zarr_vectors")
    if not block or CAP_DENSE_MANIFESTS in (block.get("format_capabilities") or []):
        return
    update_root_metadata(
        type(level_group)._from_zarr(root_zarr), add_capabilities=[CAP_DENSE_MANIFESTS],
    )


def patch(
    level_group: Group,
    rows: npt.NDArray[np.int64],
    offsets: npt.NDArray[np.int64],
    coords: npt.NDArray[np.int64],
    frags: npt.NDArray[np.int64],
    *,
    sid_ndim: int,
) -> npt.NDArray[np.bool_]:
    """Point ``rows`` at new manifests; returns which of them held one before.

    The new blocks are appended and only the named span rows rewritten;
    a row at or past the end extends the spans (a gap is padded empty).
    """
    rows = np.asarray(rows, dtype=np.int64)
    n0 = num_rows(level_group)
    held = np.zeros(rows.size, dtype=bool)
    old = rows < n0
    if old.any():
        held[old] = _rows(level_group, SPANS_PATH, rows[old]).reshape(-1, 2)[:, 1] > 0
    b0 = num_blocks(level_group)
    new_blocks = np.concatenate(
        [np.asarray(coords, np.int64).reshape(-1, sid_ndim), np.asarray(frags, np.int64)[:, None]],
        axis=1,
    )
    if new_blocks.size:
        level_group.extend_array(BLOCKS_PATH, new_blocks)
    counts = np.diff(np.asarray(offsets, np.int64))
    spans = np.stack([b0 + np.asarray(offsets[:-1], np.int64), counts], axis=1)
    size = max(n0, int(rows.max()) + 1) if rows.size else n0
    if size > n0:
        pad = np.empty((size - n0, 2), dtype=np.int64)
        pad[:] = (b0, 0)
        level_group.extend_array(SPANS_PATH, pad)
    node = _node(level_group, SPANS_PATH)
    order = np.argsort(rows, kind="stable")
    node.set_orthogonal_selection((rows[order], slice(None)), spans[order])
    level_group._invalidate_node(SPANS_PATH)
    return held


def remove(level_group: Group) -> None:
    """Drop the dense arrays (before a vlen rewrite of the same index)."""
    oi = level_group.zarr_group.require_group(OBJECT_INDEX)
    for name in (SPANS_ARRAY, BLOCKS_ARRAY):
        if name in oi:
            del oi[name]
        level_group._invalidate_node(f"{OBJECT_INDEX}/{name}")
