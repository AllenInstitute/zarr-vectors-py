"""The reverse index: which object owns each fragment.

Without it, "which objects reference this fragment" can only be answered
by decoding every manifest in the level.  Four places in this package
each rebuild that inversion in memory, and ``vacuum(drop_empty_fragments
=True)`` raised ``NotImplementedError`` for want of it.

The format already reserves the slot: ``fragment_attributes`` is
documented as carrying "parent-IDs (e.g. an ``object_id`` fragment
attribute carrying the OID that owns each fragment)".
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import build_fragment_owner_index
from zarr_vectors.core.arrays import (
    FRAGMENT_OWNER_NONE,
    FRAGMENT_OWNER_SHARED,
    build_fragment_owner_column,
    patch_object_manifests,
    read_fragment_owners,
    read_object_manifest_rows,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.ops import vacuum
from zarr_vectors.types.points import write_points


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "owners.zv"
    n = 600
    write_points(
        path,
        rng.uniform(0, 400, (n, 3)).astype(np.float32),
        chunk_shape=(100.0, 100.0, 100.0),
        # Sparse, non-zero-based ids, so a row is never mistaken for one.
        object_ids=np.repeat(np.arange(30) * 7 + 3, n // 30),
    )
    return path


def test_the_column_agrees_with_the_manifests(store):
    build_fragment_owner_index(store)
    level = get_resolution_level(open_store(store), 0)
    ids, manifests = read_object_manifest_rows(level)

    truth: dict[tuple, set[int]] = {}
    for oid, manifest in zip(ids.tolist(), manifests):
        for cc, fi in manifest:
            truth.setdefault(
                (tuple(int(c) for c in cc), int(fi)), set()
            ).add(int(oid))
    assert truth, "fixture produced no fragments"

    for (cc, fi), owners in truth.items():
        got = read_fragment_owners(level, cc, fi)
        if len(owners) == 1:
            assert got == [next(iter(owners))]
        else:
            # A jointly-owned fragment reports None so the caller falls
            # back to the manifests rather than trusting one of several.
            assert got is None


def test_a_shared_fragment_is_marked_not_guessed(store):
    level = get_resolution_level(open_store(store, mode="r+"), 0)
    ids, manifests = read_object_manifest_rows(level)
    donor = next(
        (int(o), m) for o, m in zip(ids.tolist(), manifests) if m
    )
    other = next(
        int(o) for o, m in zip(ids.tolist(), manifests)
        if int(o) != donor[0] and m
    )
    # Point a second object at the first one's fragment.
    patch_object_manifests(level, {other: list(donor[1])}, 3)

    owners = build_fragment_owner_column(level)
    cc, fi = donor[1][0]
    assert owners[tuple(int(c) for c in cc)][int(fi)] == FRAGMENT_OWNER_SHARED


def test_an_unreferenced_fragment_reads_as_unowned(store):
    level = get_resolution_level(open_store(store, mode="r+"), 0)
    ids, manifests = read_object_manifest_rows(level)
    victim, manifest = next(
        (int(o), m) for o, m in zip(ids.tolist(), manifests) if m
    )
    patch_object_manifests(level, {victim: []}, 3)

    owners = build_fragment_owner_column(level)
    cc, fi = manifest[0]
    assert owners[tuple(int(c) for c in cc)][int(fi)] == FRAGMENT_OWNER_NONE


def test_vacuum_reports_unreferenced_fragments(store):
    level = get_resolution_level(open_store(store, mode="r+"), 0)
    ids, manifests = read_object_manifest_rows(level)
    victim = next(int(o) for o, m in zip(ids.tolist(), manifests) if m)
    patch_object_manifests(level, {victim: []}, 3)

    report = vacuum(open_store(store, mode="r+"), drop_empty_fragments=True)
    assert report.dropped_fragments_per_chunk


def test_an_edit_session_agrees_with_the_scan(store):
    """The column is an optimisation, so it must give the same answer."""
    from zarr_vectors.ops import EditSession

    build_fragment_owner_index(store)
    level = get_resolution_level(open_store(store), 0)
    ids, manifests = read_object_manifest_rows(level)
    oid, manifest = next(
        (int(o), m) for o, m in zip(ids.tolist(), manifests) if m
    )
    cc, fi = manifest[0]

    with EditSession(open_store(store, mode="r+")) as session:
        from_column = session._oids_referencing(0, cc, int(fi))
        # Force the scan and compare.
        session._fragment_owners = None
        session._build_fragment_owners(0)
        from_scan = session._oids_referencing(0, cc, int(fi))
    assert from_column == from_scan == [oid]
