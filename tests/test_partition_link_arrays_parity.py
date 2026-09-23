"""The array link partitioner files every record where the record one does.

``partition_link_arrays`` already routes ``write_links``' canonical path,
and the array-form link writers route through it too. Nothing compared it
with ``partition_records_by_offset``, the definition it replaced, so a
divergence would move links to other cells without any error. This pins
the two together over the shapes a store can hold: link widths, spatial
ranks, negative chunk coords, direction, cross-level records and chunk
scales.

Bucket order differs by design (sorted here, first-seen there), and no
cell's bytes depend on it, so the comparison is per key.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from zarr_vectors.core.arrays import links_has_perm
from zarr_vectors.core.paths import parse_offsets
from zarr_vectors.spatial.boundary import (
    partition_link_arrays,
    partition_records_by_offset,
)


def _records(rng, n, L, sid, *, spread=2, intra_share=0.3):
    """Random records near a few centres, some wholly inside one chunk."""
    centres = rng.integers(-3, 4, (4, sid))
    chunks = np.empty((n, L, sid), dtype=np.int64)
    for r in range(n):
        base = centres[rng.integers(len(centres))]
        if rng.random() < intra_share:
            chunks[r] = base
        else:
            chunks[r] = base + rng.integers(-spread + 1, spread, (L, sid))
    vi = rng.integers(0, 40, (n, L)).astype(np.int64)
    # Ties on (chunk, vi) inside a record exercise the stable sort.
    if n and L > 1:
        chunks[0, 1] = chunks[0, 0]
        vi[0, 1] = vi[0, 0]
    return chunks, vi


def _as_records(chunks, vi):
    return [
        [(tuple(int(c) for c in chunks[r, k]), int(vi[r, k]))
         for k in range(chunks.shape[1])]
        for r in range(chunks.shape[0])
    ]


def _expected_rows(entries, segment, sid, *, delta, directed):
    L = len(entries[0][0])
    offsets = parse_offsets(segment, sid_ndim=sid, link_width=L)
    has_perm = links_has_perm(
        offsets, delta=delta, directed=directed, store="canonical",
    )
    rows = [
        ([perm] if has_perm else []) + list(vi_in_src)
        for vi_in_src, perm, _ in entries
    ]
    width = (1 if has_perm else 0) + len(entries[0][0])
    return np.asarray(rows, dtype=np.int64).reshape(len(entries), width)


_CASES = [
    # (L, sid, directed, cross_level, scale_src, scale_trg)
    *[(L, sid, directed, False, None, None)
      for L, sid, directed in itertools.product((1, 2, 3, 4), (3, 4), (False, True))],
    (2, 3, False, True, (1, 1, 1), (1, 1, 1)),
    (2, 3, False, True, (2, 2, 2), (1, 1, 1)),
    (2, 3, False, True, (1, 1, 1), (2, 2, 2)),
    (2, 4, False, True, (1, 1, 1, 1), (1, 2, 2, 2)),
]


@pytest.mark.parametrize("L,sid,directed,cross_level,scale_src,scale_trg", _CASES)
def test_array_partition_matches_record_partition(
    L, sid, directed, cross_level, scale_src, scale_trg,
):
    rng = np.random.default_rng(1000 * L + 10 * sid + int(directed) + 7 * int(cross_level))
    chunks, vi = _records(rng, 300, L, sid)
    ones = (1,) * sid
    scale_src = scale_src or ones
    scale_trg = scale_trg or ones
    delta = 1 if cross_level else 0

    by_records = partition_records_by_offset(
        _as_records(chunks, vi), L, sid,
        scale_src=scale_src, scale_trg=scale_trg,
        directed=directed, store="canonical", cross_level=cross_level,
    )
    by_arrays = partition_link_arrays(
        chunks, vi, link_width=L, sid_ndim=sid,
        scale_src=scale_src, scale_trg=scale_trg,
        directed=directed, cross_level=cross_level,
    )

    assert set(by_arrays) == set(by_records)
    for key, entries in by_records.items():
        rows, input_idx = by_arrays[key]
        np.testing.assert_array_equal(
            rows,
            _expected_rows(entries, key[0], sid, delta=delta, directed=directed),
            err_msg=f"rows differ for {key}",
        )
        np.testing.assert_array_equal(
            input_idx, [idx for _, _, idx in entries],
            err_msg=f"input order differs for {key}",
        )


def test_no_records_partition_to_nothing():
    chunks = np.empty((0, 2, 3), dtype=np.int64)
    vi = np.empty((0, 2), dtype=np.int64)
    assert partition_link_arrays(
        chunks, vi, link_width=2, sid_ndim=3,
        scale_src=(1, 1, 1), scale_trg=(1, 1, 1),
    ) == {}
    assert partition_records_by_offset(
        [], 2, 3, scale_src=(1, 1, 1), scale_trg=(1, 1, 1),
    ) == {}
