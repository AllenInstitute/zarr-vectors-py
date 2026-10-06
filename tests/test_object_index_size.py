"""The object index bounds padding, not objects.

``OBJECT_INDEX_MAX_ROWS`` was a ceiling on the index's length, from when
an id was its manifest's row. Ids are stored now and rows are dense, so
that ceiling had become one on the object count: a whole-brain build
(~1.5e9 objects) was refused by every writer, a tombstone patch of one
row included. What still deserves a bound is the one place an id sets
the row count, ``write_object_index(total_objects=)``, which declares
empty rows; that bound is checked before the range is built.

The ceiling is lowered here, so "past it" needs dozens of rows, not 67M.
"""

from __future__ import annotations

import time

import pytest

from zarr_vectors.building import patch_object_manifests, read_object_manifests
from zarr_vectors.constants import OBJECT_INDEX
from zarr_vectors.core import arrays
from zarr_vectors.core.arrays import object_count, write_object_index
from zarr_vectors.core.store import create_store
from zarr_vectors.exceptions import ArrayError

SID_NDIM = 3
CEILING = 8


def _manifest(oid: int):
    return [((oid % 5, (oid // 5) % 5, 0), oid % 3)]


@pytest.fixture
def low_ceiling(monkeypatch):
    monkeypatch.setattr(arrays, "OBJECT_INDEX_MAX_ROWS", CEILING)


@pytest.fixture(params=["vlen", "dense"])
def layout(request):
    return request.param


@pytest.fixture
def root(tmp_path):
    return create_store(str(tmp_path / "s.zarrvectors"))


def test_more_objects_than_the_ceiling_write(root, low_ceiling, layout):
    n = 4 * CEILING
    write_object_index(
        root, {o: _manifest(o) for o in range(n)}, SID_NDIM, layout=layout,
    )
    assert object_count(root) == n
    assert read_object_manifests(root, ids=[n - 1])[n - 1] == _manifest(n - 1)


def test_a_declared_space_the_objects_fill_writes(root, low_ceiling, layout):
    n = 4 * CEILING
    write_object_index(
        root, {o: _manifest(o) for o in range(n)}, SID_NDIM,
        total_objects=n, layout=layout,
    )
    assert object_count(root) == n


def test_a_patch_of_an_index_past_the_ceiling_writes(root, low_ceiling, layout):
    # BRIDGE's tombstone: blank a few rows of a large index, and add one.
    n = 4 * CEILING
    write_object_index(
        root, {o: _manifest(o) for o in range(n)}, SID_NDIM, layout=layout,
    )
    patch_object_manifests(root, {3: [], 17: [], n: _manifest(n)}, SID_NDIM)

    got = read_object_manifests(root, ids=[2, 3, 17, n])
    assert got[3] == [] and got[17] == []
    assert got[2] == _manifest(2) and got[n] == _manifest(n)
    meta = root.read_array_meta(OBJECT_INDEX)
    assert meta["num_objects"] == n + 1
    assert meta["num_present"] == n + 1 - 2


def test_declared_padding_past_the_ceiling_is_refused(root, low_ceiling, layout):
    with pytest.raises(ArrayError, match="padding"):
        write_object_index(
            root, {0: _manifest(0), 1: _manifest(1)}, SID_NDIM,
            total_objects=CEILING + 3, layout=layout,
        )


def test_padding_at_the_ceiling_writes(root, low_ceiling, layout):
    write_object_index(
        root, {0: _manifest(0), 1: _manifest(1)}, SID_NDIM,
        total_objects=CEILING + 2, layout=layout,
    )
    assert object_count(root) == CEILING + 2


def test_an_address_as_a_slot_count_fails_before_building_the_range(root):
    # At the real ceiling: set(range(2**40)) would never finish, so the
    # refusal has to come before it, not after.
    t0 = time.perf_counter()
    with pytest.raises(ArrayError, match="padding"):
        write_object_index(root, {7: _manifest(7)}, SID_NDIM, total_objects=2**40)
    assert time.perf_counter() - t0 < 5.0
