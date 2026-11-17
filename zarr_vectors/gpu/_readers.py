"""Device versions of the per-chunk readers that take ``device=``.

Each function here takes the same arguments as the host reader of the same
name in :mod:`zarr_vectors.core.arrays` and returns the same array on the
device, read through :func:`~zarr_vectors.core.cells.read_cells`' device
path (decoded on the device when the array's codecs allow, else on the host
and copied up once). Where its answer could differ from the host reader's
-- another dtype than the stored one, a cell off the grid, a fragment index
that is empty or points outside the buffer, any read error -- it returns
``NotImplemented`` and the host reader runs instead, so results and errors
are the host's.
"""

from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np

from zarr_vectors.constants import FRAGMENT_ATTRIBUTES, VERTEX_FRAGMENTS, VERTICES


def served_by_store(level_group: Any, keys: list[str]) -> bool:
    """Whether the cells ``keys`` are what the store holds.

    Not inside a shard transaction (its staged cells), an offline session
    (its snapshot), or for a composite level's cells of another store:
    only ``Group.read_bytes`` serves those, so they are read on the host.
    """
    if level_group._shard_txn is not None or level_group._offline is not None:
        return False
    return all(level_group._source_cell(k) is None for k in dict.fromkeys(keys))


def device_readable(level_group: Any, name: str, keys: list[str]) -> bool:
    """:func:`served_by_store`, and every cell on ``name``'s grid (the host
    readers raise for a cell off it, where ``read_cells`` reads it empty)."""
    if not served_by_store(level_group, keys):
        return False
    bounds = level_group.chunk_grid_bounds(name)
    if bounds is None:
        return False
    origin, shape = bounds
    origin = origin or (0,) * len(shape)
    for key in keys:
        coords = [int(c) for c in key.split(".")]
        if len(coords) != len(shape) or any(
            not 0 <= c - o < s for c, o, s in zip(coords, origin, shape)
        ):
            return False
    return True


def _key(chunk_coords: Any) -> str:
    return ".".join(str(int(c)) for c in chunk_coords)


def _cells(chunk_coords: Any) -> np.ndarray:
    return np.asarray([tuple(int(c) for c in chunk_coords)], dtype=np.int64)


def read_chunk_vertex_buffer(
    level_group: Any, chunk_coords: Any, dtype: Any = None, ndim: int = 3, **_: Any,
) -> Any:
    from zarr_vectors.core.arrays import vertices_dtype
    from zarr_vectors.core.cells import read_cells

    if dtype is not None and np.dtype(dtype) != vertices_dtype(level_group):
        return NotImplemented
    if not device_readable(level_group, VERTICES, [_key(chunk_coords)]):
        return NotImplemented
    batch = read_cells(
        level_group, _cells(chunk_coords), [VERTICES], ndim=ndim,
        device="cuda", on_error="raise",
    )
    return batch[VERTICES].data


def fragment_row_index(fragments: Any, n_rows: int) -> Any:
    """The buffer rows every fragment of a one-cell :class:`FragmentColumn`
    covers, in fragment order; ``None`` unless every range lies inside
    ``[0, n_rows)`` with a non-negative count and every explicit index does
    too (the host reader slices or wraps those, and keeps that case)."""
    starts, counts = fragments.starts, fragments.counts
    is_range = starts >= 0
    ends = starts + counts
    ok = bool(cp.all(~is_range | ((counts >= 0) & (ends <= n_rows))))
    idx = fragments.indices
    ok = ok and bool(cp.all((idx >= 0) & (idx < n_rows))) if idx.size else ok
    if not ok:
        return None
    first = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(counts)])
    total = int(first[-1])
    j = cp.arange(total, dtype=cp.int64)
    f = cp.searchsorted(first, j, side="right") - 1
    local = j - first[f]
    padded = cp.concatenate([idx, cp.zeros(1, dtype=cp.int64)])
    at = cp.minimum(fragments.index_offsets[f] + local, padded.size - 1)
    return cp.where(is_range[f], starts[f] + local, padded[at])


def read_chunk_vertex_rows(
    level_group: Any, chunk_coords: Any, dtype: Any = None, ndim: int = 3, **_: Any,
) -> Any:
    from zarr_vectors.core.arrays import vertices_dtype
    from zarr_vectors.core.cells import read_cells

    if dtype is not None and np.dtype(dtype) != vertices_dtype(level_group):
        return NotImplemented
    keys = [_key(chunk_coords)]
    if not (device_readable(level_group, VERTICES, keys)
            and device_readable(level_group, VERTEX_FRAGMENTS, keys)):
        return NotImplemented
    batch = read_cells(
        level_group, _cells(chunk_coords), [VERTICES], ndim=ndim,
        device="cuda", on_error="raise", fragments=True,
    )
    # No fragments: the host reader's empty array, or its error for a
    # cell with no index at all.
    if int(batch.fragments.cell_offsets[-1]) == 0:
        return NotImplemented
    data = batch[VERTICES].data
    rows = fragment_row_index(batch.fragments, int(data.shape[0]))
    if rows is None:
        return NotImplemented
    return data[rows]


def read_chunk_fragment_attributes(
    level_group: Any, attr_name: str, chunk_coords: Any, dtype: Any = None,
    ncols: int | None = None, *, default: Any = None, **_: Any,
) -> Any:
    from zarr_vectors.core.arrays import _resolve_attribute_layout
    from zarr_vectors.core.cells import read_cells

    name = f"{FRAGMENT_ATTRIBUTES}/{attr_name}"
    key = _key(chunk_coords)
    # The layout read_cells will use must be the one asked for.
    if _resolve_attribute_layout(level_group, name, dtype, ncols) != (
        _resolve_attribute_layout(level_group, name, None, None)
    ):
        return NotImplemented
    if not device_readable(level_group, name, [key]):
        return NotImplemented
    if not level_group.chunk_exists(name, key):
        return NotImplemented
    batch = read_cells(
        level_group, _cells(chunk_coords), [name], device="cuda", on_error="raise",
    )
    return batch[name].data


def read_link_attributes(
    level_group: Any, attr_name: str, dtype: Any = None, *, delta: int = 0, **_: Any,
) -> Any:
    from zarr_vectors.core.arrays import list_link_attribute_offsets
    from zarr_vectors.core.cells import read_cells
    from zarr_vectors.core.paths import link_attributes_group_path

    group_path = link_attributes_group_path(attr_name, delta)
    if not level_group.array_exists(group_path):
        return NotImplemented
    out_dtype, row_shape = np.dtype(np.float32), ()
    blocks = []
    for seg in list_link_attribute_offsets(level_group, attr_name, delta):
        name = f"{group_path}/{seg}"
        keys = sorted(level_group.list_chunks(name))
        meta = level_group.read_array_meta(name) or {}
        out_dtype = np.dtype(meta["dtype"])
        row_shape = tuple(int(x) for x in meta.get("row_shape", ()))
        if dtype is not None and np.dtype(dtype) != out_dtype:
            return NotImplemented
        if not keys:
            continue
        if not device_readable(level_group, name, keys):
            return NotImplemented
        cells = np.asarray([[int(c) for c in k.split(".")] for k in keys], dtype=np.int64)
        data = read_cells(level_group, cells, [name], device="cuda", on_error="raise")[name].data
        if data.shape[0]:
            blocks.append(data)
    if not blocks:
        return cp.empty((0, *row_shape), dtype=out_dtype)
    return cp.concatenate(blocks, axis=0)
