"""The whole-array encoders, run on the device.

A writer handed device arrays used to copy each one to the host and
encode there. These are the same encoders run where the arrays already
are, so that what crosses to the host is the *encoded* form, which is
often much smaller: a fragment that is a run of consecutive vertex
indices becomes 16 bytes of range table however long it is.

Each function here mirrors its numpy original in
:mod:`zarr_vectors.encoding.fragments` step for step, raises the same
errors with the same messages, and returns the same host values; the
tests compare the bytes. Only the arithmetic moves: every byte is still
framed, compressed (by libzstd) and written on the host, which is what
keeps a store written from device arrays identical to one written from
numpy.
"""

from __future__ import annotations

from typing import Any

import cupy as cp
import numpy as np

from zarr_vectors import _xp
from zarr_vectors.exceptions import ArrayError


def _int64(a: Any, what: str) -> Any:
    arr = cp.asarray(a)
    if arr.ndim != 1:
        raise ArrayError(f"{what} must be 1-D, got shape {arr.shape}")
    if arr.size and arr.dtype.kind not in "iu":
        raise ArrayError(f"{what} must be integers, got {arr.dtype}")
    return arr.astype(cp.int64, copy=False)


def _owner(offsets: Any, n: int) -> Any:
    """The CSR row every element ``0..n-1`` belongs to (``np.repeat`` of rows)."""
    return cp.searchsorted(offsets[1:], cp.arange(n, dtype=cp.int64), side="right")


def _scalars(*values: Any) -> list[int]:
    """Several device scalars in one download."""
    if not values:
        return []
    return [int(v) for v in _xp.to_host(cp.stack([cp.asarray(v, dtype=cp.int64) for v in values]))]


def classify_fragments_csr(indices: Any, offsets: Any, *, force_explicit: bool) -> Any:
    """:func:`~zarr_vectors.encoding.fragments.classify_fragments_csr` on the device.

    Returns host :class:`FragmentSections`: four downloads, of the
    sections rather than of the indices.
    """
    from zarr_vectors.encoding.fragments import FragmentSections

    idx = _int64(indices, "indices")
    off = _int64(offsets, "offsets")
    if off.size == 0:
        raise ArrayError(
            f"offsets must start at 0 and end at len(indices)={idx.size}; got []..[]"
        )
    counts = cp.diff(off)
    first, last, min_count, min_idx = _scalars(
        off[0], off[-1],
        counts.min() if counts.size else 0,
        idx.min() if idx.size else 0,
    )
    if first != 0 or last != idx.size:
        raise ArrayError(
            f"offsets must start at 0 and end at len(indices)={idx.size}; "
            f"got [{first}]..[{last}]"
        )
    if counts.size and min_count < 0:
        raise ArrayError("offsets must be non-decreasing")
    if idx.size and min_idx < 0:
        raise ArrayError("Explicit fragment indices must be non-negative")

    num = counts.size
    starts = off[:-1]
    if force_explicit or idx.size == 0:
        is_range = cp.zeros(num, dtype=bool)
    else:
        breaks = cp.concatenate((cp.zeros(1, cp.int64), cp.cumsum(cp.diff(idx) != 1)))
        nonempty = counts > 0
        last_of = cp.maximum(off[1:] - 1, 0)
        is_range = nonempty & (breaks[last_of] == breaks[cp.minimum(starts, max(idx.size - 1, 0))])
    range_table = cp.stack([idx[starts[is_range]], counts[is_range]], axis=1).reshape(-1, 2)
    explicit = ~is_range
    explicit_offsets = cp.concatenate(
        (cp.zeros(1, cp.int64), cp.cumsum(counts[explicit])),
    ).astype(cp.int64)
    explicit_indices = idx[explicit[_owner(off, idx.size)]] if idx.size else idx
    return FragmentSections(
        is_range=_xp.to_host(is_range),
        range_table=_xp.to_host(range_table),
        explicit_offsets=_xp.to_host(explicit_offsets),
        explicit_indices=_xp.to_host(explicit_indices),
    )


def encode_object_manifests_csr(
    chunk_coords: Any,
    fragment_idx: Any,
    manifest_offsets: Any,
    *,
    sid_ndim: int | None,
    mode_single: int,
) -> np.ndarray:
    """:func:`~zarr_vectors.encoding.fragments.encode_object_manifests_csr`
    on the device: the block table is built there, and one download per
    distinct block count brings the finished blobs back."""
    cc = cp.asarray(chunk_coords)
    idx = cp.asarray(fragment_idx)
    if idx.ndim != 1:
        raise ArrayError(f"fragment_idx must be 1-D, got shape {idx.shape}")
    total = int(idx.size)
    if cc.size == 0:
        cc = cc.reshape(0, sid_ndim if sid_ndim is not None else 0)
    if cc.ndim != 2 or cc.shape[0] != total:
        raise ArrayError(
            f"chunk_coords must be ({total}, sid_ndim); got shape {cc.shape}"
        )
    if sid_ndim is None:
        sid_ndim = int(cc.shape[1])
    elif total and cc.shape[1] != sid_ndim:
        raise ArrayError(
            f"chunk_coords have rank {cc.shape[1]}, expected sid_ndim={sid_ndim}"
        )
    for name, arr in (("chunk_coords", cc), ("fragment_idx", idx)):
        if arr.size and arr.dtype.kind not in "iu":
            raise ArrayError(f"{name} must be integers, got {arr.dtype}")
    idx = idx.astype(cp.int64, copy=False)
    if manifest_offsets is None:
        off = cp.arange(total + 1, dtype=cp.int64)
    else:
        off = _int64(manifest_offsets, "manifest_offsets")
    counts = cp.diff(off)
    min_idx, first, last, min_count = _scalars(
        idx.min() if total else 0,
        off[0] if off.size else -1, off[-1] if off.size else -1,
        counts.min() if counts.size else 0,
    )
    if total and min_idx < 0:
        raise ArrayError(f"fragment_index must be >= 0, got {min_idx}")
    if manifest_offsets is not None and (off.size == 0 or first != 0 or last != total):
        raise ArrayError(f"manifest_offsets must start at 0 and end at {total}")
    if counts.size and min_count < 0:
        raise ArrayError("manifest_offsets must be non-decreasing")

    width = sid_ndim * 8 + 1 + 8
    table = cp.empty((total, width), dtype=cp.uint8)
    table[:, : sid_ndim * 8] = (
        cp.ascontiguousarray(cc.astype(cp.int64)).view(cp.uint8).reshape(total, sid_ndim * 8)
    )
    table[:, sid_ndim * 8] = mode_single
    table[:, sid_ndim * 8 + 1:] = cp.ascontiguousarray(idx).view(cp.uint8).reshape(total, 8)

    out = np.empty(counts.size, dtype=object)
    host_counts = _xp.to_host(counts)
    for k in np.unique(host_counts).tolist():
        rows = np.flatnonzero(host_counts == k)
        blob_width = 4 + k * width
        blobs = cp.empty((rows.size, blob_width), dtype=cp.uint8)
        blobs[:, :4] = cp.asarray(np.array([k], dtype="<u4").view(np.uint8))
        if k:
            gather = off[cp.asarray(rows)][:, None] + cp.arange(k)
            blobs[:, 4:] = table[gather].reshape(rows.size, k * width)
        out[rows] = _xp.to_host(blobs).view(f"V{blob_width}").ravel().astype(object)
    return out
