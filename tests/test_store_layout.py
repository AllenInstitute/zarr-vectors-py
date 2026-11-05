"""``building.store_layout``: a store's layout, resolved from what it holds.

A store records its layout structure by structure, some of it after it
was created, and ``zv_version`` only names the build that created it. The
resolver reads each structure and sums them up as ``min_reader``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors import building as zb
from zarr_vectors.constants import CAP_DENSE_MANIFESTS, FORMAT_VERSION
from zarr_vectors.core.arrays import write_object_index

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves


def _store(tmp_path, name="s.zv", **kw):
    root = zb.create_store(
        str(tmp_path / name), bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3, **kw,
    )
    return root, zb.get_resolution_level(root, 0)


def test_a_new_store(tmp_path):
    root, level = _store(tmp_path)
    got = zb.store_layout(root)
    assert got == zb.store_layout(level)
    assert got.created_by == FORMAT_VERSION
    assert got.object_index is None and got.link_policy == {}
    assert not got.presence_deferred and got.min_reader == "0.9.0"


def test_each_structure_moves_the_oldest_reader(tmp_path):
    root, level = _store(tmp_path, shard_shape=2)
    assert zb.store_layout(root).shard_shape == 2
    zb.create_link_segments(level, sid_ndim=3)
    zb.write_object_manifests(
        level, chunk_coords=np.zeros((1, 3), np.int64), fragment_idx=np.arange(1),
        layout="vlen",
    )
    zb.commit_object_index(level, 1, sid_ndim=3)
    got = zb.store_layout(level)
    assert got.object_index == zb.OBJECT_INDEX_LAYOUT_V1
    assert got.link_policy == {0: (False, "canonical")}
    assert got.min_reader == "0.9.0"

    zb.defer_presence(level)
    assert zb.store_layout(level).min_reader == "0.9.3"


def test_an_id_table_needs_0_9_2(tmp_path):
    _, level = _store(tmp_path)
    # write_object_index stores its ids (V2) whatever they are.
    write_object_index(level, {3: [((0, 0, 0), 1)], 7: [((0, 0, 0), 2)]}, 3)
    got = zb.store_layout(level)
    assert got.object_index == zb.OBJECT_INDEX_LAYOUT_V2 and got.min_reader == "0.9.2"


def test_a_dense_index_made_after_creation_is_declared(tmp_path):
    root, level = _store(tmp_path)  # created without choosing a layout
    assert CAP_DENSE_MANIFESTS not in zb.store_layout(root).format_capabilities
    zb.write_object_manifests(
        level, chunk_coords=np.zeros((2, 3), np.int64), fragment_idx=np.arange(2),
        layout="dense",
    )
    got = zb.store_layout(zb.open_store(str(tmp_path / "s.zv")))
    assert got.object_index == zb.OBJECT_INDEX_LAYOUT_DENSE
    assert CAP_DENSE_MANIFESTS in got.format_capabilities
    assert got.min_reader == "0.9.4"


def test_a_root_asks_about_the_level_named(tmp_path):
    root, _ = _store(tmp_path)
    assert zb.store_layout(root, level=3).object_index is None  # no such level
