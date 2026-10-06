"""A dense append below the row count, without reading back the rows it keeps.

A resume (``at=`` the committed count, residue past it) used to read
every span row before ``at`` to find where the kept rows' blocks end, and
then rewrite the whole spans array: O(rows) on every resume. It now
proves the end from the residue in the common case, scans in bounded
pieces otherwise, and cuts the spans at ``at`` instead of rewriting them.

The reference is the implementation it replaced (zarr-vectors 218c633),
reproduced below: every file of the store must come out byte for byte
the same, over appends, torn flushes, pads, patches, placed ranges and a
sharded object layer.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core.arrays import patch_object_manifests

from ._store_compare import assert_stores_identical

pytestmark = pytest.mark.vlen_only  # dense throughout, chosen here


def _old_write(level_group, offsets, coords, frags, *, sid_ndim, mode="replace", at=None):
    """``dense_manifests.write`` as of 218c633 (read back, rewrite)."""
    offsets = np.asarray(offsets, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, sid_ndim)
    frags = np.asarray(frags, dtype=np.int64).reshape(-1)
    counts = np.diff(offsets)
    n = int(counts.size)
    new_blocks = np.concatenate([coords, frags[:, None]], axis=1)
    exists = level_group.array_exists(dense.SPANS_PATH) and level_group.array_exists(dense.BLOCKS_PATH)
    if mode == "replace" or not exists:
        start = 0 if mode == "replace" else int(at or 0)
        spans = np.empty((start + n, 2), dtype=np.int64)
        spans[:start] = (0, 0)
        spans[start:, 0] = offsets[:-1]
        spans[start:, 1] = counts
        dense._create(level_group, dense.SPANS_PATH, spans)
        dense._create(level_group, dense.BLOCKS_PATH, new_blocks)
        if not exists:
            dense._declare_capability(level_group)
        return start
    n0 = dense.num_rows(level_group)
    b0 = dense.num_blocks(level_group)
    start = n0 if at is None else int(at)
    head = None
    if start < n0:
        head = dense._rows(level_group, dense.SPANS_PATH, slice(0, start)).reshape(-1, 2)
        keep = int((head[:, 0] + head[:, 1]).max()) if start else 0
        if keep < b0:
            dense._truncate(level_group, dense.BLOCKS_PATH, keep)
            b0 = keep
    tail = np.empty((max(start - n0, 0) + n, 2), dtype=np.int64)
    pad = max(start - n0, 0)
    tail[:pad] = (b0, 0)
    tail[pad:, 0] = b0 + offsets[:-1]
    tail[pad:, 1] = counts
    if new_blocks.size:
        level_group.extend_array(dense.BLOCKS_PATH, new_blocks)
    if head is not None:
        dense._create(level_group, dense.SPANS_PATH, np.concatenate([head, tail]).astype(np.int64))
    elif tail.size:
        level_group.extend_array(dense.SPANS_PATH, tail)
    return start


def _new_store(root, name="s.zv"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    zb.create_store(
        str(path), bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout="dense",
    )
    return path


def _level(path):
    return zb.get_resolution_level(zb.open_store(str(path), mode="r+"), 0)


def _batch(rng, n, *, tag=0, empty_share=0.25):
    counts = np.where(rng.random(n) < empty_share, 0, rng.integers(1, 5, n))
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    m = int(off[-1])
    return dict(
        chunk_coords=rng.integers(0, 2, (m, 3)),
        fragment_idx=rng.integers(0, 1000, m) * 4 + tag,
        manifest_offsets=off,
    )


def _append(level, batch, at):
    zb.write_object_manifests(level, **batch, mode="append", at=at)


def _twins(tmp_path, build):
    """Two copies of a store ``build`` made: one to resume old, one new."""
    a = _new_store(tmp_path / "old")
    build(_level(a))
    b = tmp_path / "new" / a.name
    shutil.copytree(a, b)
    return a, b


def _resume_both(monkeypatch, a, b, batch, at):
    with monkeypatch.context() as m:
        m.setattr(dense, "write", _old_write)
        _append(_level(a), batch, at)
    _append(_level(b), batch, at)
    assert_stores_identical(a, b)


SCENARIOS = ["torn", "torn-at-zero", "torn-across-buckets", "pad", "patched", "placed"]


def _build(scenario, rng):
    def build(level):
        if scenario == "torn":
            _append(level, _batch(rng, 300), 0)
            _append(level, _batch(rng, 200), 300)
            zb.commit_object_index(level, 500, sid_ndim=3)
            _append(level, _batch(rng, 77, tag=1), 500)        # residue
        elif scenario == "torn-at-zero":
            _append(level, _batch(rng, 50, tag=1), 0)
        elif scenario == "torn-across-buckets":
            _append(level, _batch(rng, 17_000), 0)
            zb.commit_object_index(level, 17_000, sid_ndim=3)
            _append(level, _batch(rng, 20_000, tag=1), 17_000)  # residue
        elif scenario == "pad":
            _append(level, _batch(rng, 40), 0)
        elif scenario == "patched":
            _append(level, _batch(rng, 400), 0)
            _append(level, _batch(rng, 50, tag=1), 400)         # residue
            # A kept row's blocks moved past the residue: the scan path.
            patch_object_manifests(level, {7: [((1, 1, 1), 9), ((0, 1, 0), 3)]}, 3)
        elif scenario == "placed":
            _append(level, _batch(rng, 100), 0)
            zb.commit_object_index(level, 100, sid_ndim=3)
            zb.reserve_object_rows(level, 300, sid_ndim=3, n_blocks=2_000)
            b = _batch(rng, 200, tag=1)
            zb.write_object_manifests(level, **b, mode="place", at=100, block_at=1_000)
            # As BRIDGE does: the placed ids are their rows.
            zb.commit_object_index(level, 100, sid_ndim=3, object_ids_sorted=True)
        return level
    return build


AT = {"torn": 500, "torn-at-zero": 0, "torn-across-buckets": 17_000,
      "pad": 64, "patched": 400, "placed": 150}


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("sharded", [False, True])
def test_a_resume_stores_what_the_read_back_did(tmp_path, monkeypatch, scenario, sharded):
    rng = np.random.default_rng(len(scenario))
    build = _build(scenario, rng)

    def built(level):
        build(level)
        if sharded:
            zb.shard_object_layer(level, zb.OBJECT_SHARD_ROW_MULTIPLE)

    a, b = _twins(tmp_path, built)
    _resume_both(monkeypatch, a, b, _batch(np.random.default_rng(99), 123, tag=2), AT[scenario])
    # And an empty resume, which still truncates.
    a2, b2 = _twins(tmp_path / "empty", built)
    _resume_both(monkeypatch, a2, b2, _batch(np.random.default_rng(5), 0), AT[scenario])


def test_a_resume_after_a_torn_flush_stores_what_one_write_does(tmp_path):
    rng = np.random.default_rng(3)
    first, second, residue = _batch(rng, 900), _batch(rng, 300, tag=2), _batch(rng, 40, tag=1)
    torn = _new_store(tmp_path / "torn")
    lg = _level(torn)
    _append(lg, first, 0)
    _append(lg, residue, 900)
    _append(lg, second, 900)
    once = _new_store(tmp_path / "once")
    lg = _level(once)
    _append(lg, first, 0)
    _append(lg, second, 900)
    assert_stores_identical(torn, once)


def test_the_common_resume_reads_only_the_rows_it_replaces(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    path = _new_store(tmp_path / "s")
    lg = _level(path)
    _append(lg, _batch(rng, 5_000), 0)
    _append(lg, _batch(rng, 60, tag=1), 5_000)  # residue
    seen = []
    real = dense._rows

    def spy(level_group, p, rows):
        if p == dense.SPANS_PATH:
            seen.append(rows)
        return real(level_group, p, rows)

    monkeypatch.setattr(dense, "_rows", spy)
    _append(_level(path), _batch(rng, 10, tag=2), 5_000)
    assert seen and all(isinstance(s, slice) and s.start >= 4_999 for s in seen), seen


def test_the_scan_is_read_in_bounded_pieces(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    path = _new_store(tmp_path / "s")
    lg = _level(path)
    _append(lg, _batch(rng, 3_000), 0)
    _append(lg, _batch(rng, 30, tag=1), 3_000)
    patch_object_manifests(lg, {5: [((1, 1, 1), 1)]}, 3)  # forces the scan
    monkeypatch.setattr(dense, "_SCAN_ROWS", 512)
    seen = []
    real = dense._rows

    def spy(level_group, p, rows):
        if p == dense.SPANS_PATH and isinstance(rows, slice):
            seen.append(rows.stop - rows.start)
        return real(level_group, p, rows)

    monkeypatch.setattr(dense, "_rows", spy)
    _append(_level(path), _batch(rng, 10, tag=2), 3_000)
    assert seen and max(seen) <= 512


def test_a_row_tombstoned_after_the_residue_is_the_one_difference(tmp_path, monkeypatch):
    """The documented case the proof does not see: same reads, fewer blocks."""
    rng = np.random.default_rng(8)

    def build(level):
        _append(level, _batch(rng, 200, empty_share=0.0), 0)
        _append(level, _batch(rng, 20, tag=1, empty_share=0.0), 200)  # residue
        patch_object_manifests(level, {3: []}, 3)  # a tombstone, after it

    a, b = _twins(tmp_path, build)
    batch = _batch(np.random.default_rng(1), 30, tag=2)
    with monkeypatch.context() as m:
        m.setattr(dense, "write", _old_write)
        _append(_level(a), batch, 200)
    _append(_level(b), batch, 200)
    la, lb = _level(a), _level(b)
    ca, cb = dense.read_csr(la, stop=230), dense.read_csr(lb, stop=230)
    for x, y in zip(ca, cb):
        np.testing.assert_array_equal(x, y)
    assert dense.num_blocks(lb) < dense.num_blocks(la)
