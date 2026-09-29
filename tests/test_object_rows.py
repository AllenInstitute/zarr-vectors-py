"""Filling the object layer from several processes at once (A7).

``at=`` appends are a resume cursor: a write below the current length
truncates, one above it pads, and every call resizes. So disjoint ranges
written concurrently used to lose rows. Now a coordinator reserves the
rows once and each worker places its own aligned range; the store it
commits must read back exactly as one written serially.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.core.arrays import read_object_attributes
from zarr_vectors.exceptions import ArrayError

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves

ALIGN = zb.concurrency_contract()["at_alignment"]  # 65,536 unsharded


def _level(path, layout):
    root = zb.create_store(
        str(path), bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout=layout,
    )
    return zb.get_resolution_level(root, 0)


def _manifests(lo, hi):
    """Objects ``lo..hi``: object o names ``o % 3`` fragments, deterministically."""
    oids = np.arange(lo, hi, dtype=np.int64)
    counts = oids % 3
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    owner = np.repeat(oids, counts)
    j = np.arange(offsets[-1]) - np.repeat(offsets[:-1], counts)
    coords = np.stack([owner % 2, (owner // 2) % 2, j % 2], axis=1).astype(np.int64)
    frags = owner * 10 + j
    return offsets, coords, frags


def _lengths(lo, hi):
    return (np.arange(lo, hi) % 97).astype(np.float32)


def _place(path, lo, hi, block_at):
    """One worker: its rows of manifests and one attribute column, in place."""
    level = zb.get_resolution_level(zb.open_store(str(path), mode="r+"), 0)
    offsets, coords, frags = _manifests(lo, hi)
    zb.write_object_manifests(
        level, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
        mode="place", at=lo, block_at=block_at,
    )
    zb.write_object_attribute_columns(level, {"length": _lengths(lo, hi)}, mode="place", at=lo)


def _reference(path, layout, n):
    level = _level(path, layout)
    offsets, coords, frags = _manifests(0, n)
    zb.write_object_manifests(
        level, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
        mode="append", at=0,
    )
    zb.write_object_attribute_columns(level, {"length": _lengths(0, n)}, at=0)
    zb.commit_object_index(level, n, sid_ndim=3)
    return level


def _same_reads(a, b):
    ca, cb = zb.read_all_object_manifests_csr(a), zb.read_all_object_manifests_csr(b)
    for x, y in zip(ca, cb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    ids = [0, 5, ALIGN - 1, ALIGN, 2 * ALIGN + 7]
    assert zb.read_object_manifests(a, ids=ids) == zb.read_object_manifests(b, ids=ids)
    np.testing.assert_array_equal(
        read_object_attributes(a, "length"), read_object_attributes(b, "length"),
    )
    assert zb.object_count(a) == zb.object_count(b)
    from zarr_vectors.core.arrays import object_present_count

    assert object_present_count(a) == object_present_count(b)


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_workers_placing_disjoint_ranges_leave_the_serial_store(tmp_path, layout):
    n = 2 * ALIGN + 1000  # three workers; the last range ends at n
    ranges = [(0, ALIGN), (ALIGN, 2 * ALIGN), (2 * ALIGN, n)]
    ref = _reference(tmp_path / "ref.zv", layout, n)

    path = tmp_path / "par.zv"
    level = _level(path, layout)
    block_starts = [None] * len(ranges)
    n_blocks = None
    if layout == "dense":
        counts = [int(_manifests(lo, hi)[0][-1]) for lo, hi in ranges]
        starts = zb.aligned_regions(counts, zb.concurrency_contract()["block_alignment"])
        block_starts, n_blocks = starts[:-1].tolist(), int(starts[-1])
    reserved = zb.reserve_object_rows(
        level, n, sid_ndim=3, n_blocks=n_blocks,
        columns={"length": ("float32", (), float("nan"))},
    )
    assert reserved["at_alignment"] == ALIGN
    assert zb.object_count(level) == 0  # reserving commits nothing

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_place, args=(path, lo, hi, b))
        for (lo, hi), b in zip(ranges, block_starts)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0

    level = zb.get_resolution_level(zb.open_store(str(path), mode="r+"), 0)
    zb.commit_object_index(level, n)
    _same_reads(level, ref)


def test_a_dense_row_nobody_wrote_is_refused_at_commit(tmp_path):
    level = _level(tmp_path / "s.zv", "dense")
    zb.reserve_object_rows(level, 10, sid_ndim=3, n_blocks=16)
    offsets, coords, frags = _manifests(0, 6)
    zb.write_object_manifests(
        level, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
        mode="place", at=0, block_at=0,
    )
    with pytest.raises(ArrayError, match="4 of the 10 rows have no object id"):
        zb.commit_object_index(level, 10)
    zb.commit_object_index(level, 6)  # the written prefix commits
    assert zb.object_count(level) == 6


def test_vlen_rows_nobody_wrote(tmp_path):
    level = _level(tmp_path / "s.zv", "vlen")
    zb.reserve_object_rows(level, 5, sid_ndim=3)
    zb.commit_object_index(level, 5)  # a bucket nobody wrote: empty objects
    assert zb.read_all_object_manifests(level) == [[]] * 5

    # Rows 0, 3, 4 share a bucket with the rows written: zarr leaves them b"".
    offsets, coords, frags = _manifests(1, 3)
    zb.write_object_manifests(
        level, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
        mode="place", at=1,
    )
    with pytest.raises(ArrayError, match="3 of the 5 rows hold no manifest"):
        zb.commit_object_index(level, 5)
    for at in (0, 3, 4):
        zb.write_object_manifests(
            level, chunk_coords=np.empty((0, 3), np.int64), fragment_idx=np.empty(0, np.int64),
            manifest_offsets=np.zeros(2, np.int64), mode="place", at=at,
        )
    zb.commit_object_index(level, 5)
    got = zb.read_all_object_manifests(level)
    assert got[0] == [] and got[3] == [] and len(got[2]) == 2


def test_placing_outside_the_reservation_is_refused(tmp_path):
    level = _level(tmp_path / "s.zv", "vlen")
    zb.reserve_object_rows(level, 4, sid_ndim=3, columns={"length": "float32"})
    offsets, coords, frags = _manifests(0, 5)
    with pytest.raises(ArrayError, match="reserve"):
        zb.write_object_manifests(
            level, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
            mode="place", at=0,
        )
    with pytest.raises(ArrayError, match="reserve"):
        zb.write_object_attribute_columns(
            level, {"length": np.zeros(3, np.float32)}, mode="place", at=2,
        )
    with pytest.raises(ArrayError, match="not reserved"):
        zb.write_object_attribute_columns(
            level, {"other": np.zeros(1, np.float32)}, mode="place", at=0,
        )
    with pytest.raises(ArrayError, match="needs at="):
        zb.write_object_attribute_columns(level, {"length": np.zeros(1)}, mode="place")


def test_a_reservation_grows_but_never_shrinks_and_keeps_rows(tmp_path):
    level = _level(tmp_path / "s.zv", "vlen")
    offsets, coords, frags = _manifests(0, 3)
    zb.write_object_manifests(
        level, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
        mode="append", at=0,
    )
    zb.write_object_attributes(level, "length", _lengths(0, 3))
    zb.reserve_object_rows(level, 6, columns={"length": "float32"})
    zb.reserve_object_rows(level, 2, columns={"length": "float32"})  # no shrink
    zb.write_object_attributes(level, "length", np.array([1.0, 2.0], np.float32),
                               present_mask=np.array([1, 0]), mode="place", at=4)
    zb.commit_object_index(level, 6, sid_ndim=3)
    got = zb.read_all_object_manifests(level)
    assert len(got) == 6 and got[5] == [] and len(got[2]) == 2
    col = read_object_attributes(level, "length")
    assert col.shape == (6,) and col[4] == 1.0 and np.isnan(col[5]) and np.isnan(col[3])


def test_a_sharded_reservation_aligns_to_the_shard(tmp_path):
    level = _level(tmp_path / "s.zv", "dense")
    out = zb.reserve_object_rows(
        level, 100, sid_ndim=3, n_blocks=100, shard_rows=4 * ALIGN,
        columns={"length": "float32"},
    )
    assert out["at_alignment"] == out["block_alignment"] == 4 * ALIGN
    assert zb.concurrency_contract(level)["at_alignment"] == 4 * ALIGN
    assert zb.object_shard_rows(level) == 4 * ALIGN
    with pytest.raises(ArrayError, match="repack it with shard_object_layer"):
        zb.reserve_object_rows(level, 200, shard_rows=ALIGN)


def test_the_contract_says_what_is_safe():
    import zarr_vectors as zv

    c = zv.concurrency_contract()
    assert c["shard_is_unit"] and c["concurrent_at_ranges"]
    assert c["at_alignment"] == 65_536 and c["block_alignment"] == 16_384
    assert zv.runtime_capabilities()["concurrency_contract"]


def test_aligned_regions():
    starts = zb.aligned_regions([10, 0, 70_000, 5], 65_536)
    assert starts.tolist() == [0, 65_536, 65_536, 196_608, 262_144]
