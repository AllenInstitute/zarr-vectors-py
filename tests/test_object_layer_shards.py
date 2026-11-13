"""Sharding the object layer (A6), with no zarr-vectors format change.

The object layer is 1-D along objects, one storage object per row bucket:
about 70,000 objects per column at 10^9 objects. Sharded, it is packed
``shard_rows // bucket`` buckets per object -- by zarr's own
``sharding_indexed``, recorded in each array's ``zarr.json``, so nothing
new for a reader to learn. The tests pin that sharding changes no value
any reader returns, that writers keep a sharded layer sharded, and that a
plain zarr read agrees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from zarr_vectors import building as zb
from zarr_vectors.constants import OBJECT_ATTRIBUTES, OBJECT_INDEX
from zarr_vectors.core.arrays import (
    object_layer_arrays,
    read_object_attributes,
    write_object_attributes,
    write_object_index,
)
from zarr_vectors.exceptions import ArrayError

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves

ROWS = zb.OBJECT_SHARD_ROW_MULTIPLE  # 65,536


def _level(tmp_path, layout, name="s.zv"):
    path = tmp_path / layout / name
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zb.create_store(
        str(path), bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout=layout,
    )
    return path, zb.get_resolution_level(root, 0)


def _fill(level, n, rng):
    manifests = {
        oid: [
            (tuple(int(c) for c in rng.integers(0, 2, 3)), int(rng.integers(0, 50)))
            for _ in range(int(rng.integers(0, 3)))
        ]
        for oid in range(n)
    }
    write_object_index(level, manifests, 3)
    write_object_attributes(level, "length", rng.uniform(0, 1, n).astype(np.float32))
    write_object_attributes(level, "rgb", rng.integers(0, 255, (n, 3)).astype(np.uint8))
    return manifests


def _snapshot(level):
    csr = zb.read_all_object_manifests_csr(level)
    return {
        "offsets": np.asarray(csr.offsets), "coords": np.asarray(csr.chunk_coords),
        "frags": np.asarray(csr.fragment_idx),
        "ids": None if csr.object_ids is None else np.asarray(csr.object_ids),
        "some": zb.read_object_manifests(level, ids=[5, 70_000, 3]),
        "length": read_object_attributes(level, "length"),
        "rgb": read_object_attributes(level, "rgb"),
        "count": zb.object_count(level),
    }


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], np.ndarray):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        else:
            assert a[k] == b[k], k


def _files(path: Path, sub: str) -> int:
    return sum(1 for p in (path / "0" / sub).rglob("*") if p.is_file() and p.name != "zarr.json")


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_sharding_changes_no_value_and_packs_the_buckets(tmp_path, layout):
    path, level = _level(tmp_path, layout)
    n = 2 * ROWS + 123  # three shards, the last partial
    _fill(level, n, np.random.default_rng(0))
    before = _snapshot(level)
    files_before = _files(path, OBJECT_ATTRIBUTES)

    report = zb.shard_object_layer(level, ROWS)
    assert sorted(report["arrays_repacked"]) == sorted(object_layer_arrays(level))
    for name in object_layer_arrays(level):
        arr = level.zarr_group[name]
        assert arr.shards[0] == ROWS, name
    _same(before, _snapshot(level))
    assert zb.object_shard_rows(level) == ROWS
    assert zb.store_layout(level).object_shard_rows == ROWS
    # 65,536-row attribute buckets: one file per column per shard now.
    assert _files(path, OBJECT_ATTRIBUTES) <= files_before
    assert not list(path.rglob(f"*{'__repacking'}*"))

    # A plain zarr read of a sharded column is the same data.
    plain = zarr.open_array(str(path / "0" / OBJECT_ATTRIBUTES / "length"), mode="r")
    np.testing.assert_array_equal(plain[:], before["length"])

    # And back: unsharding changes nothing either.
    zb.shard_object_layer(level, None)
    assert zb.object_shard_rows(level) is None
    _same(before, _snapshot(level))


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_writers_keep_a_sharded_layer_sharded(tmp_path, layout):
    _, level = _level(tmp_path, layout)
    rng = np.random.default_rng(1)
    _fill(level, 1000, rng)
    zb.shard_object_layer(level, 2 * ROWS)

    # A new column, an append to the index, a full rewrite of the index.
    write_object_attributes(level, "late", np.arange(1000, dtype=np.int32))
    zb.write_object_manifests(
        level, chunk_coords=np.zeros((10, 3), np.int64), fragment_idx=np.arange(10),
        mode="append", at=1000,
    )
    zb.commit_object_index(level, 1010, sid_ndim=3)
    write_object_attributes(level, "late", np.arange(10, dtype=np.int32), mode="append", at=1000)
    for name in object_layer_arrays(level):
        assert level.zarr_group[name].shards[0] == 2 * ROWS, name
    assert read_object_attributes(level, "late").tolist() == list(range(1000)) + list(range(10))
    assert zb.read_object_manifests(level, ids=[1009])[1009] == [((0, 0, 0), 9)]

    write_object_index(level, {i: [((0, 0, 0), i)] for i in range(50)}, 3)
    for name in (f"{OBJECT_INDEX}/object_ids", *object_layer_arrays(level)[:1]):
        assert level.zarr_group[name].shards[0] == 2 * ROWS, name
    assert zb.object_count(level) == 50


def test_shard_rows_must_tile_both_buckets(tmp_path):
    _, level = _level(tmp_path, "vlen")
    _fill(level, 10, np.random.default_rng(2))
    for bad in (0, 16_384, 100_000, -ROWS, "x"):
        with pytest.raises(ArrayError, match="multiple of 65536"):
            zb.shard_object_layer(level, bad)


def test_shard_store_covers_the_object_layer_when_asked(tmp_path):
    path, level = _level(tmp_path, "dense")
    _fill(level, 300, np.random.default_rng(3))
    before = _snapshot(level)
    out = zb.shard_store(str(path), shard_shape=2, object_shard_rows=ROWS)
    assert out["object_arrays_repacked"] == len(object_layer_arrays(level))
    level = zb.get_resolution_level(zb.open_store(str(path)), 0)
    assert zb.object_shard_rows(level) == ROWS
    _same(before, _snapshot(level))
