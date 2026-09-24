"""``patch_object_manifests`` from the supported surface.

It was reachable only from ``core.arrays``, so a consumer blanking the
manifests of objects it was about to rewrite either reached past the
contract or paid :func:`write_object_manifests`' whole-index rewrite for
every resume.  The property that makes it worth promoting is that it
touches only the zarr chunks the named rows fall in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zarr_vectors.building import (
    OBJECT_INDEX_MANIFEST_BUCKET,
    patch_object_manifests,
    read_object_manifests,
)
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core.arrays import write_object_index
from zarr_vectors.core.store import create_store

SID_NDIM = 3
B = OBJECT_INDEX_MANIFEST_BUCKET


def _manifest(oid: int):
    return [((oid % 5, (oid // 5) % 5, (oid // 25) % 5), oid % 3)]


def _bucket_files(store: Path) -> dict[str, tuple[int, bytes]]:
    chunk_dir = store / OBJECT_INDEX / "manifests" / "c"
    return {
        p.name: (p.stat().st_mtime_ns, p.read_bytes())
        for p in chunk_dir.iterdir()
    }


@pytest.mark.vlen_only
def test_blanking_touches_only_the_buckets_holding_the_ids(tmp_path):
    n = 3 * B + 10
    path = tmp_path / "s.zarrvectors"
    root = create_store(str(path))
    write_object_index(
        root, {oid: _manifest(oid) for oid in range(n)}, SID_NDIM,
        total_objects=n,
    )
    before = _bucket_files(path)
    assert len(before) == 4, sorted(before)

    blanked = [3, 17, 2 * B + 5, 2 * B + 999]
    patch_object_manifests(root, {oid: [] for oid in blanked}, SID_NDIM)

    after = _bucket_files(path)
    changed = sorted(name for name in before if after[name] != before[name])
    assert changed == ["0", "2"]

    got = read_object_manifests(root, ids=[2, 3, 17, 2 * B + 5, 2 * B + 6])
    assert got[3] == [] and got[17] == [] and got[2 * B + 5] == []
    assert got[2] == [((2, 0, 0), 2)]
    assert got[2 * B + 6] == _manifest(2 * B + 6)

    meta = root.read_array_meta(OBJECT_INDEX)
    assert meta["num_present"] == n - len(blanked)
    assert meta["num_objects"] == n


def test_num_present_stays_exact_across_blank_and_refill(tmp_path):
    root = create_store(str(tmp_path / "s.zarrvectors"))
    n = 40
    write_object_index(
        root, {oid: _manifest(oid) for oid in range(n)}, SID_NDIM,
        total_objects=n,
    )
    patch_object_manifests(root, {5: [], 9: []}, SID_NDIM)
    # Refill one, blank another, and re-blank one already blank: only the
    # net change may move the count.
    patch_object_manifests(
        root, {5: _manifest(5), 11: [], 9: []}, SID_NDIM,
    )
    meta = root.read_array_meta(OBJECT_INDEX)
    assert meta["num_present"] == n - 2
