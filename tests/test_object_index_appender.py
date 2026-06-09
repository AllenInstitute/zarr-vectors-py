"""Tests for the streaming :class:`ObjectIndexAppender`.

Covers:

* Parity — streaming fragments-then-paths produces a store that reads back
  identically to the full-rebuild ``write_object_index`` +
  ``write_object_attributes`` + ``write_groupings`` path.
* ``base_oid`` truncation — opening at ``base_oid`` keeps fragment blobs,
  drops stale prior-run path blobs, and appends new paths contiguously.
* Chunk-boundary correctness — appends straddling
  ``OBJECT_INDEX_MANIFEST_BUCKET`` boundaries read back exactly.
* Empty — zero appended paths leaves only group 0 and ``num_objects ==
  base_oid``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_MANIFEST_BUCKET,
    read_group_object_ids,
    read_object_attributes,
    read_object_manifest,
    write_groupings,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.core.store import create_store
from zarr_vectors.core.streaming import ObjectIndexAppender

SID_NDIM = 3


def _manifest(oid: int):
    """Deterministic small manifest for object ``oid``.

    One or two mode-0 single-fragment ``(chunk_coords, fragment_index)``
    references derived from ``oid`` so every object is distinguishable.
    """
    c0 = (oid % 5, (oid // 5) % 5, (oid // 25) % 5)
    refs = [(c0, oid % 3)]
    if oid % 2 == 0:
        c1 = ((oid + 1) % 5, oid % 5, (oid // 7) % 5)
        refs.append((c1, (oid // 2) % 4))
    return refs


def test_parity_with_full_rebuild(tmp_path: Path):
    n_frag, n_path = 7, 20
    total = n_frag + n_path
    manifests = {oid: _manifest(oid) for oid in range(total)}
    lengths = np.arange(10, 10 + total, dtype=np.int32)
    frag_oids = list(range(n_frag))
    path_oids = list(range(n_frag, total))

    # Reference: full rebuild.
    ref = create_store(str(tmp_path / "ref.zarrvectors"))
    write_object_index(ref, manifests, SID_NDIM, total_objects=total)
    write_object_attributes(ref, "length", lengths, mode="replace")
    write_groupings(ref, {0: frag_oids, 1: path_oids})

    # Streaming: fragments first, then append paths in several batches.
    stream = create_store(str(tmp_path / "stream.zarrvectors"))
    write_object_index(
        stream, {oid: manifests[oid] for oid in frag_oids}, SID_NDIM,
        total_objects=n_frag,
    )
    write_object_attributes(stream, "length", lengths[:n_frag], mode="replace")

    app = ObjectIndexAppender(stream, n_frag, SID_NDIM, frag_oids)
    for start in range(n_frag, total, 6):
        end = min(start + 6, total)
        app.append(
            [manifests[oid] for oid in range(start, end)],
            lengths[start:end],
        )
    assert app.close() == total

    # Manifests read back identically per object.
    for oid in range(total):
        assert read_object_manifest(stream, oid) == read_object_manifest(ref, oid)

    # Length attribute identical.
    np.testing.assert_array_equal(
        read_object_attributes(stream, "length"),
        read_object_attributes(ref, "length"),
    )

    # Groupings identical (group 1 stored implicitly as a range on the
    # streaming side but reads back to the same members).
    assert list(read_group_object_ids(stream, 0)) == list(read_group_object_ids(ref, 0))
    assert list(read_group_object_ids(stream, 1)) == list(read_group_object_ids(ref, 1))
    assert list(read_group_object_ids(stream, 1)) == path_oids


def test_base_oid_truncation(tmp_path: Path):
    n, stale, k = 5, 8, 6  # fragments [0,5), stale paths [5,13)
    m = n + stale
    pre = {oid: _manifest(oid) for oid in range(m)}
    pre_lengths = np.arange(100, 100 + m, dtype=np.int32)

    store = create_store(str(tmp_path / "trunc.zarrvectors"))
    write_object_index(store, pre, SID_NDIM, total_objects=m)
    write_object_attributes(store, "length", pre_lengths, mode="replace")

    # Capture fragment blobs to prove they are byte-unchanged after append.
    arr = store.zarr_group[OBJECT_INDEX]["manifests"]
    frag_blobs_before = [bytes(arr[i:i + 1][0]) for i in range(n)]

    new_manifests = [_manifest(1000 + i) for i in range(k)]
    new_lengths = np.arange(500, 500 + k, dtype=np.int32)
    app = ObjectIndexAppender(store, n, SID_NDIM, list(range(n)))
    app.append(new_manifests, new_lengths)
    assert app.close() == n + k

    meta = store.read_array_meta(OBJECT_INDEX)
    assert meta["num_objects"] == n + k

    arr = store.zarr_group[OBJECT_INDEX]["manifests"]
    assert arr.shape[0] == n + k  # stale [n, m) gone
    for i in range(n):
        assert bytes(arr[i:i + 1][0]) == frag_blobs_before[i]
    for i in range(k):
        assert read_object_manifest(store, n + i) == [
            (tuple(c), fi) for c, fi in new_manifests[i]
        ]

    # Fragment lengths preserved, new path lengths appended.
    out = read_object_attributes(store, "length")
    np.testing.assert_array_equal(out[:n], pre_lengths[:n])
    np.testing.assert_array_equal(out[n:], new_lengths)


def test_chunk_boundary_correctness(tmp_path: Path):
    bucket = OBJECT_INDEX_MANIFEST_BUCKET
    base = bucket - 4  # start just below a bucket boundary
    # Batches straddling the boundary: small, exactly one bucket, small.
    batch_sizes = [10, bucket, 7]

    store = create_store(str(tmp_path / "boundary.zarrvectors"))
    write_object_index(
        store, {oid: _manifest(oid) for oid in range(base)}, SID_NDIM,
        total_objects=base,
    )
    write_object_attributes(
        store, "length", np.zeros(base, dtype=np.int32), mode="replace",
    )

    app = ObjectIndexAppender(store, base, SID_NDIM, list(range(base)))
    oid = base
    expected: dict[int, list] = {}
    for size in batch_sizes:
        manifests = []
        for _ in range(size):
            manifests.append(_manifest(2_000_000 + oid))
            expected[oid] = [(tuple(c), fi) for c, fi in manifests[-1]]
            oid += 1
        app.append(manifests, np.full(size, oid, dtype=np.int32))
    total = base + sum(batch_sizes)
    assert app.close() == total

    # Every appended object reads back exactly, including across boundaries.
    for o, refs in expected.items():
        assert read_object_manifest(store, o) == refs
    # Spot-check a fragment near the boundary is intact.
    assert read_object_manifest(store, base - 1) == [
        (tuple(c), fi) for c, fi in _manifest(base - 1)
    ]


def test_empty_append(tmp_path: Path):
    n = 4
    store = create_store(str(tmp_path / "empty.zarrvectors"))
    write_object_index(
        store, {oid: _manifest(oid) for oid in range(n)}, SID_NDIM,
        total_objects=n,
    )
    write_object_attributes(store, "length", np.arange(n, dtype=np.int32))

    with ObjectIndexAppender(store, n, SID_NDIM, list(range(n))) as app:
        pass  # no append → context manager calls close()

    meta = store.read_array_meta(OBJECT_INDEX)
    assert meta["num_objects"] == n
    assert list(read_group_object_ids(store, 0)) == list(range(n))
    # Group 1 is the empty range [n, n).
    assert list(read_group_object_ids(store, 1)) == []
