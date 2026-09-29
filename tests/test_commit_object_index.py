"""``commit_object_index``: the commit ``write_object_manifests`` leaves to the caller.

Every caller of the array-form manifest writer committed ``object_index``'s
metadata by hand, each a little differently; only zv's own appender
stamped ``num_present`` and ``object_ids_sorted``. The commit has to leave
an index every reader accepts, with counts that match what is stored --
including after a tombstone followed by an append.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import (
    commit_object_index,
    create_store,
    get_resolution_level,
    object_count,
    patch_object_manifests,
    read_all_object_manifests,
    read_all_object_manifests_csr,
    read_object_manifests,
    write_object_manifests,
)
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import (
    OBJECT_IDS_SORTED_ATTR,
    object_present_count,
    object_present_mask,
    write_object_index,
)
from zarr_vectors.exceptions import ArrayError

pytestmark = pytest.mark.vlen_only  # these pick their layouts themselves


def _level(tmp_path, layout):
    path = tmp_path / layout / "s.zarrvectors"
    path.parent.mkdir()
    root = create_store(
        path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        geometry_types=["polyline"], manifest_layout=layout,
    )
    return get_resolution_level(root, 0)


def _write(lg, first, n, *, empty=()):
    """Objects ``first..first+n``, each one block; ``empty`` ones none."""
    counts = np.array([0 if first + o in empty else 1 for o in range(n)])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    k = int(offsets[-1])
    coords = np.tile(np.array([[0, 1, 0]], dtype=np.int64), (k, 1))
    frags = np.arange(k, dtype=np.int64) + first
    return write_object_manifests(
        lg, chunk_coords=coords, fragment_idx=frags, manifest_offsets=offsets,
        mode="append", at=first,
    )


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_a_new_index_is_readable_once_committed(tmp_path, layout):
    lg = _level(tmp_path, layout)
    _write(lg, 0, 6, empty={2})
    meta = commit_object_index(lg, 6, sid_ndim=3)
    assert meta["num_objects"] == 6 and meta["num_present"] == 5
    assert meta["layout"] == ("dense_manifests_v1" if layout == "dense" else "vlen_manifests_v1")
    got = read_all_object_manifests(lg)
    # Object 2 holds nothing, so object 5 names fragment 4.
    assert len(got) == 6 and got[2] == [] and got[5] == [((0, 1, 0), 4)]
    assert object_count(lg) == 6 and object_present_count(lg) == 5


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_a_tombstone_then_an_append_leaves_the_counts_exact(tmp_path, layout):
    lg = _level(tmp_path, layout)
    _write(lg, 0, 4)
    commit_object_index(lg, 4, sid_ndim=3)
    patch_object_manifests(lg, {1: []}, 3)  # a removal
    _write(lg, 4, 3)
    commit_object_index(lg, 7)
    assert object_present_count(lg) == int(object_present_mask(lg).sum()) == 6


@pytest.mark.parametrize("layout", ["vlen", "dense"])
def test_rows_past_the_commit_are_residue(tmp_path, layout):
    lg = _level(tmp_path, layout)
    _write(lg, 0, 5)
    commit_object_index(lg, 3, sid_ndim=3)
    assert object_count(lg) == 3 and object_present_count(lg) == 3
    assert len(read_all_object_manifests_csr(lg).offsets) == 4
    with pytest.raises(ArrayError, match="exceeds the 5 manifest rows"):
        commit_object_index(lg, 6)


def test_a_count_the_caller_already_has_is_taken(tmp_path, monkeypatch):
    lg = _level(tmp_path, "vlen")
    _write(lg, 0, 4, empty={0})
    monkeypatch.setattr(type(lg), "read_vlen_array_raw", None)  # must not be read
    assert commit_object_index(lg, 4, sid_ndim=3, num_present=3)["num_present"] == 3
    with pytest.raises(ArrayError, match="outside"):
        commit_object_index(lg, 4, num_present=5)


def test_the_commit_merges_into_the_metadata(tmp_path):
    lg = _level(tmp_path, "dense")
    _write(lg, 0, 2)
    lg.write_array_meta(OBJECT_INDEX, {"bridge_note": "kept"})
    commit_object_index(lg, 2, sid_ndim=3)
    assert lg.read_array_meta(OBJECT_INDEX)["bridge_note"] == "kept"


@pytest.mark.parametrize("ids,sorted_", [([3, 7, 9], True), ([9, 3, 7], False)])
def test_ids_sorted_is_stamped_from_the_ids(tmp_path, ids, sorted_):
    lg = _level(tmp_path, "vlen")
    write_object_index(lg, {i: [((0, 0, 0), i)] for i in ids}, 3)
    assert lg.read_array_meta(OBJECT_INDEX)["layout"] == "vlen_manifests_v2"
    # Rewrite the table in the given order, as an id-carrying writer would.
    lg.write_array("object_index/object_ids", np.asarray(ids, dtype=np.int64))
    meta = commit_object_index(lg, 3)
    assert meta[OBJECT_IDS_SORTED_ATTR] is sorted_
    assert sorted(read_object_manifests(lg, ids=ids)) == sorted(ids)
