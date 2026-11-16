"""A new dense index chunks ``manifest_blocks`` at 65,536 rows; an old one keeps 16,384.

Spans and the id table keep the 16,384-row bucket. Everything that depends
on the blocks' chunk reads it from the array, so the place protocol's
``block_alignment``, the streaming writer's block flushes, a torn flush's
truncation, a patch and a shard repack all work on either chunking -- and
an index created before keeps its own through every append.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import zarr_vectors as zv
import zarr_vectors.building as zb
from zarr_vectors.core import dense_manifests as dense
from zarr_vectors.core.arrays import OBJECT_ATTRIBUTE_ROW_BUCKET
from zarr_vectors.core.object_rows import row_alignment
from zarr_vectors.core.streaming import ObjectIndexAppender

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves

B = dense.BLOCKS_BUCKET


def _level(tmp_path, name="s"):
    root = zb.create_store(
        tmp_path / f"{name}.zarrvectors", bounds=([0.0] * 3, [100.0] * 3),
        chunk_shape=(50.0,) * 3, geometry_types=["polyline"], manifest_layout="dense",
    )
    return zb.get_resolution_level(root, 0)


def _node(lg, path):
    return lg.zarr_group[path]


def _csr(rng, n, max_k=6):
    counts = rng.integers(0, max_k + 1, n)
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    cc = rng.integers(0, 2, (int(off[-1]), 3)).astype(np.int64)
    fi = rng.integers(0, 1 << 20, int(off[-1])).astype(np.int64)
    return off, cc, fi


def _expected(off, cc, fi):
    return dense.manifests_from_csr(off, cc, fi)


def _write(lg, off, cc, fi, **kw):
    return zb.write_object_manifests(
        lg, chunk_coords=cc, fragment_idx=fi, manifest_offsets=off, **kw,
    )


def _chunks(lg):
    return {
        name: int(_node(lg, f"object_index/{name}").chunks[0])
        for name in ("manifest_spans", "manifest_blocks", "object_ids")
    }


WANT = {"manifest_spans": 16_384, "manifest_blocks": 65_536, "object_ids": 16_384}


def test_the_bucket():
    assert B == 65_536 == OBJECT_ATTRIBUTE_ROW_BUCKET
    assert dense.bucket(dense.BLOCKS_PATH) == B
    assert dense.bucket(dense.SPANS_PATH) == zb.OBJECT_INDEX_MANIFEST_BUCKET == 16_384


@pytest.mark.parametrize("writer", ["index", "append", "reserve", "stream", "appender"])
def test_every_creating_writer_chunks_blocks_at_65536(tmp_path, writer):
    lg = _level(tmp_path)
    off, cc, fi = _csr(np.random.default_rng(1), 100)
    if writer == "index":
        zb.write_object_index(lg, dict(enumerate(_expected(off, cc, fi))), 3)
    elif writer == "append":
        _write(lg, off, cc, fi, mode="append", at=0)
    elif writer == "reserve":
        zb.reserve_object_rows(lg, 100, sid_ndim=3, n_blocks=int(off[-1]))
    elif writer == "stream":
        with zb.object_manifest_writer(lg, at=0) as w:
            w.write(chunk_coords=cc, fragment_idx=fi, manifest_offsets=off)
    else:
        with ObjectIndexAppender(lg, 0, 3, []) as app:
            app.append(_expected(off, cc, fi), [1] * 100)
    assert _chunks(lg) == WANT


def test_a_sharded_layer_tiles_the_shard(tmp_path):
    lg = _level(tmp_path)
    zb.reserve_object_rows(lg, 0, sid_ndim=3, n_blocks=0, shard_rows=1 << 20)
    blocks = _node(lg, dense.BLOCKS_PATH)
    assert blocks.chunks[0] == B and blocks.shards[0] == 1 << 20
    assert _node(lg, dense.SPANS_PATH).chunks[0] == 16_384
    assert zv.concurrency_contract(lg)["block_alignment"] == 1 << 20


def _old_index(lg, monkeypatch, off, cc, fi):
    """An index created before: blocks chunked at 16,384."""
    monkeypatch.setattr(dense, "BLOCKS_BUCKET", 16_384)
    _write(lg, off, cc, fi, mode="append", at=0)
    monkeypatch.undo()
    assert _node(lg, dense.BLOCKS_PATH).chunks[0] == 16_384


def test_an_old_index_keeps_its_chunk_and_its_alignment(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    lg = _level(tmp_path)
    off, cc, fi = _csr(rng, 50)
    _old_index(lg, monkeypatch, off, cc, fi)
    assert row_alignment(lg) == (65_536, 16_384)
    assert zv.concurrency_contract(lg)["block_alignment"] == 16_384
    off2, cc2, fi2 = _csr(rng, 50)
    _write(lg, off2, cc2, fi2, mode="append", at=50)
    zb.reserve_object_rows(lg, 400, n_blocks=40_000)
    assert _node(lg, dense.BLOCKS_PATH).chunks[0] == 16_384
    # A new index, and the contract without a level, say 65,536.
    assert row_alignment(_level(tmp_path, "t")) == (65_536, 65_536)
    assert zv.concurrency_contract()["block_alignment"] == 65_536


def test_placed_ranges_aligned_by_the_contract(tmp_path):
    """Several writers place disjoint ranges at once, blocks aligned at 65,536."""
    rng = np.random.default_rng(3)
    lg = _level(tmp_path)
    contract = zv.concurrency_contract(lg)
    rows_align, blocks_align = contract["at_alignment"], contract["block_alignment"]
    assert blocks_align == B
    parts = [_csr(rng, n, max_k=3) for n in (rows_align, rows_align, 1000)]
    row_starts = zb.aligned_regions([p[0].size - 1 for p in parts], rows_align)
    block_starts = zb.aligned_regions([int(p[0][-1]) for p in parts], blocks_align)
    n = int(row_starts[-2]) + parts[-1][0].size - 1
    zb.reserve_object_rows(lg, n, sid_ndim=3, n_blocks=int(block_starts[-1]))
    assert _node(lg, dense.BLOCKS_PATH).chunks[0] == B

    def place(i):
        off, cc, fi = parts[i]
        _write(lg, off, cc, fi, mode="place", at=int(row_starts[i]),
               block_at=int(block_starts[i]))

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(place, range(3)))
    zb.commit_object_index(lg, n, sid_ndim=3)
    want = [m for p in parts for m in _expected(*p)]
    assert zb.read_all_object_manifests(lg) == want


def test_a_torn_flush_truncates_across_block_chunks(tmp_path):
    """Residue reaching several 65,536-row chunks is cut where the kept rows end."""
    rng = np.random.default_rng(4)
    lg = _level(tmp_path)
    a = _csr(rng, 30_000)          # ~90k blocks: two chunks
    b = _csr(rng, 30_000)
    _write(lg, *a, mode="append", at=0)
    _write(lg, *b, mode="append", at=30_000)
    assert _node(lg, dense.BLOCKS_PATH).shape[0] > 2 * B
    c = _csr(rng, 10)
    _write(lg, *c, mode="append", at=30_000)        # the second batch was residue
    assert _node(lg, dense.BLOCKS_PATH).shape[0] == int(a[0][-1] + c[0][-1])
    zb.commit_object_index(lg, 30_010, sid_ndim=3)
    assert zb.read_all_object_manifests(lg) == _expected(*a) + _expected(*c)
    # The cut chunk holds what a fresh write of the kept rows stores.
    fresh = _level(tmp_path, "fresh")
    _write(fresh, *a, mode="append", at=0)
    _write(fresh, *c, mode="append", at=30_000)
    np.testing.assert_array_equal(
        _node(lg, dense.BLOCKS_PATH)[:], _node(fresh, dense.BLOCKS_PATH)[:],
    )


def test_patch_and_streamed_writes(tmp_path):
    rng = np.random.default_rng(5)
    lg = _level(tmp_path)
    off, cc, fi = _csr(rng, 40_000)
    with zb.object_manifest_writer(lg, at=0) as w:
        for lo in range(0, 40_000, 7_000):
            hi = min(lo + 7_000, 40_000)
            w.write(chunk_coords=cc[off[lo]:off[hi]], fragment_idx=fi[off[lo]:off[hi]],
                    manifest_offsets=off[lo:hi + 1] - off[lo])
    zb.commit_object_index(lg, 40_000, sid_ndim=3)
    want = _expected(off, cc, fi)
    assert zb.read_all_object_manifests(lg) == want
    zb.patch_object_manifests(lg, {7: [((1, 1, 1), 3)], 39_999: []}, 3)
    want[7], want[39_999] = [((1, 1, 1), 3)], []
    assert zb.read_all_object_manifests(lg) == want
    assert _chunks(lg) == WANT


def test_a_shard_repack_keeps_the_chunk(tmp_path):
    rng = np.random.default_rng(6)
    lg = _level(tmp_path)
    off, cc, fi = _csr(rng, 30_000)
    _write(lg, off, cc, fi, mode="append", at=0)
    zb.commit_object_index(lg, 30_000, sid_ndim=3)
    want = zb.read_all_object_manifests(lg)
    assert want == _expected(off, cc, fi)
    zb.shard_object_layer(lg, 1 << 20)
    blocks = _node(lg, dense.BLOCKS_PATH)
    assert blocks.chunks[0] == B and blocks.shards[0] == 1 << 20
    assert zb.read_all_object_manifests(lg) == want
    zb.shard_object_layer(lg, None)
    assert _node(lg, dense.BLOCKS_PATH).shards is None
    assert _chunks(lg) == WANT
    assert zb.read_all_object_manifests(lg) == want


def test_the_feature_is_advertised():
    assert "manifest-blocks-64k" in zv.FEATURES
