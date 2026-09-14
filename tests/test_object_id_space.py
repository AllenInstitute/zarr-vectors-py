"""Object ids are stored, not implied by row position.

Under the original layout an object id WAS its row index, so the index
was as long as the largest id plus one: one object with id 20,000,000
allocated twenty million rows, and a 64-bit segmentation id could not be
stored at all.  The ids now live in a sibling ``object_ids`` array and
rows are dense, so an id may be arbitrary, sparse and huge.

A store written by the older layout is simply one whose id table is the
identity, so both read through the same path and nothing has to be
rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.core.arrays import (
    OBJECT_INDEX_LAYOUT_V1,
    object_count,
    object_ids_for_rows,
    object_rows_for_ids,
    read_object_id_table,
    read_object_manifest,
    read_object_manifests,
)
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import read_points, write_points

# Sparse, non-zero-based, and past what a row-indexed layout could hold.
WILD_IDS = np.array([5, 12, 900_000, 2**40, 2**52 + 7], dtype=np.int64)


@pytest.fixture
def wild_store(tmp_path):
    positions = np.array(
        [[float(i * 3), 1.0, 2.0] for i in range(len(WILD_IDS))],
        dtype=np.float32,
    )
    store = tmp_path / "wild.zv"
    write_points(
        store, positions, chunk_shape=(50.0, 50.0, 50.0), object_ids=WILD_IDS,
    )
    return store


def test_rows_are_dense_however_large_the_ids(wild_store):
    level = get_resolution_level(open_store(wild_store), 0)
    # Five objects, five rows -- not 2**52 + 8 of them.
    assert object_count(level) == len(WILD_IDS)
    assert read_object_id_table(level).tolist() == sorted(WILD_IDS.tolist())


def test_every_id_round_trips(wild_store):
    level = get_resolution_level(open_store(wild_store), 0)
    for oid in WILD_IDS.tolist():
        assert read_object_manifest(level, oid), f"no manifest for {oid}"


def test_plural_read_is_keyed_by_id_not_row(wild_store):
    level = get_resolution_level(open_store(wild_store), 0)
    wanted = [int(WILD_IDS[-1]), int(WILD_IDS[0])]
    got = read_object_manifests(level, ids=wanted)
    assert sorted(got) == sorted(wanted)


def test_an_absent_id_resolves_to_nothing(wild_store):
    level = get_resolution_level(open_store(wild_store), 0)
    found, rows = object_rows_for_ids(level, [999, int(WILD_IDS[0])])
    assert found.tolist() == [int(WILD_IDS[0])]
    assert rows.size == 1


def test_readers_select_by_id(wild_store):
    result = read_points(wild_store, object_ids=[int(WILD_IDS[3])])
    assert len(result["positions"]) == 1


def test_the_facade_speaks_ids(wild_store):
    level = zv.open(wild_store).level(0)
    assert sorted(level.objects.ids().tolist()) == sorted(WILD_IDS.tolist())
    assert level.objects[int(WILD_IDS[4])].vertex_count == 1
    assert int(WILD_IDS[2]) in level.objects
    assert 999 not in level.objects


def test_identity_is_the_answer_for_a_row_indexed_store(tmp_path):
    """A store with no id table reads as though its ids were its rows."""
    store = tmp_path / "dense.zv"
    write_points(
        store,
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=np.array([0, 1]),
    )
    level = get_resolution_level(open_store(store, mode="r+"), 0)

    # Force the older layout: drop the table and relabel.
    meta = dict(level.read_array_meta("object_index"))
    meta["layout"] = OBJECT_INDEX_LAYOUT_V1
    del level.zarr_group["object_index"]["object_ids"]
    level.write_array_meta("object_index", meta)
    level._object_id_lookup_cache = None

    assert read_object_id_table(level) is None
    assert object_ids_for_rows(level).tolist() == [0, 1]
    found, rows = object_rows_for_ids(level, [1])
    assert found.tolist() == [1] and rows.tolist() == [1]
    assert read_object_manifest(level, 1)


def test_dense_ids_are_still_written_in_order(tmp_path):
    store = tmp_path / "ordered.zv"
    write_points(
        store,
        np.random.default_rng(0).uniform(0, 40, (30, 3)).astype(np.float32),
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=np.arange(30),
    )
    level = get_resolution_level(open_store(store), 0)
    table = read_object_id_table(level)
    assert table.tolist() == list(range(30))
    # Ascending ids let a lookup binary-search without sorting first.
    assert level.read_array_meta("object_index").get("object_ids_sorted")
