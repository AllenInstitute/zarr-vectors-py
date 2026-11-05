"""Rebuilding ``nonempty_chunks`` on a natively sharded array.

Presence used to be derivable only from a per-cell store listing, which a
sharded array does not have: zarr builds its chunk grid from the SHARD
shape, so a key under ``<array>/c/`` names a shard.  That is why sharding
had to be the last coordinator pass — an array stayed flat until every
worker had finished and the manifest had been rebuilt.

Deriving from the shards instead removes that ordering constraint, which
is what lets a store be born sharded.  The cost has to stay one read per
shard that exists; a per-cell fallback would be *correct* and therefore
invisible, so it is asserted directly.

Covers:

* Every cell recovered, including from partial edge shards.
* Empty payloads excluded; a non-zero grid origin honoured.
* One store read per present shard, not per cell.
* ``"skip"`` and ``"raise"`` still mean what they meant.
* The level-wide rebuild reaches sharded arrays.
"""

from __future__ import annotations

import pytest

from zarr_vectors.building import rebuild_presence
from zarr_vectors.core.arrays import create_vertices_array
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.exceptions import ShardedPresenceError


def _sharded(root, name="sh", grid_shape=(5, 5, 5), shard_shape=(2, 2, 2)):
    """A sharded cell array whose grid does NOT divide by the shard shape.

    5 over 2 leaves a one-cell edge shard on every axis, so the clipping
    is exercised by default rather than only in the test that names it.
    """
    root.create_sharded_chunk_array(name, grid_shape, shard_shape=shard_shape)
    return name


def _write(root, name, coords, payload=b"payload"):
    for c in coords:
        root.write_bytes(name, ".".join(str(x) for x in c), payload)
    return sorted(".".join(str(x) for x in c) for c in coords)


def _sharded_level(tmp_store_path, shard_shape=(2, 2, 2)):
    """A real level whose ``vertices`` is sharded, for the coordinator verbs."""
    root = create_store(
        str(tmp_store_path), bounds=[[0, 0, 0], [64, 64, 64]],
        chunk_shape=(16, 16, 16), geometry_types=["point_cloud"],
    )
    lg = get_resolution_level(root, 0)
    # create_store warm-creates these unsharded; drop them so this is about
    # the sharded allocation rather than what already existed.
    lg.delete_subtree("vertices")
    lg.delete_subtree("vertex_fragments")
    with lg.native_sharded_arrays(shard_shape, (4, 4, 4)):
        create_vertices_array(lg, dtype="float32")
    return lg


# --- the rebuild ------------------------------------------------------


def test_derive_on_sharded_array_recovers_every_cell(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _sharded(root)
    # Spread across interior shards AND the partial edge shard at (4,4,4).
    expected = _write(root, name, [
        (0, 0, 0), (1, 1, 1), (2, 3, 4), (4, 4, 4), (3, 0, 2),
    ])
    arr = root._sharded_chunk_array(name)
    assert arr.shards == (2, 2, 2)

    arr.attrs["nonempty_chunks"] = []      # as a lost update would leave it
    assert root.list_chunks(name) == []

    assert root.derive_nonempty_chunks(name) == expected
    assert root.list_chunks(name) == expected


def test_derive_recovers_a_cell_from_a_partial_edge_shard(tmp_store_path):
    """The last shard on each axis is only partly in bounds."""
    root = create_store(str(tmp_store_path))
    name = _sharded(root, grid_shape=(3, 3, 3), shard_shape=(2, 2, 2))
    expected = _write(root, name, [(2, 2, 2)])

    root._sharded_chunk_array(name).attrs["nonempty_chunks"] = []
    assert root.derive_nonempty_chunks(name) == expected


def test_derive_handles_a_shard_larger_than_the_array(tmp_store_path):
    """Legal, and the layout a coarse pyramid level ends up with."""
    root = create_store(str(tmp_store_path))
    name = _sharded(root, grid_shape=(2, 2, 2), shard_shape=(8, 8, 8))
    expected = _write(root, name, [(0, 0, 0), (1, 1, 1)])

    root._sharded_chunk_array(name).attrs["nonempty_chunks"] = []
    assert root.derive_nonempty_chunks(name) == expected


def test_derive_excludes_empty_payload_cells(tmp_store_path):
    """An emptied cell is not present, however it is stored."""
    root = create_store(str(tmp_store_path))
    name = _sharded(root)
    expected = _write(root, name, [(0, 0, 0)])
    root.write_bytes(name, "0.0.1", b"")

    root._sharded_chunk_array(name).attrs["nonempty_chunks"] = []
    assert root.derive_nonempty_chunks(name) == expected


def test_derive_honours_the_chunk_grid_origin(tmp_store_path):
    """Keys come back as absolute coords, not cell indices."""
    root = create_store(str(tmp_store_path))
    root.create_sharded_chunk_array(
        "sh", (4, 4, 4), shard_shape=(2, 2, 2), origin=(10, 20, 30),
    )
    expected = _write(root, "sh", [(10, 20, 30), (13, 23, 33)])

    root._sharded_chunk_array("sh").attrs["nonempty_chunks"] = []
    assert root.derive_nonempty_chunks("sh") == expected


def test_derive_on_an_empty_sharded_array_is_empty(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _sharded(root)
    assert root.derive_nonempty_chunks(name) == []


def test_derive_reads_one_object_per_present_shard(tmp_store_path):
    """The cost must stay O(shards), not O(cells).

    A per-cell fallback would pass every other test in this file, so the
    read count is asserted rather than inferred.
    """
    root = create_store(str(tmp_store_path))
    name = _sharded(root, grid_shape=(4, 4, 4), shard_shape=(4, 4, 4))
    # 8 cells, all inside the SINGLE shard that covers the whole grid.
    _write(root, name, [
        (i, j, k) for i in (0, 3) for j in (0, 3) for k in (0, 3)
    ])
    root._sharded_chunk_array(name).attrs["nonempty_chunks"] = []

    store = root._zarr.store
    original = store.get
    gets: list[str] = []

    async def counting_get(key, *a, **kw):
        if "/c/" in key:
            gets.append(key)
        return await original(key, *a, **kw)

    store.get = counting_get
    try:
        got = root.derive_nonempty_chunks(name)
    finally:
        store.get = original

    assert len(got) == 8
    # One shard object holds all 8 cells. A sharded read legitimately makes
    # a couple of partial reads for the index, but nothing like 8.
    assert len(gets) <= 3, f"expected O(shards) reads, got {len(gets)}: {gets}"


# --- the other two modes ----------------------------------------------


def test_on_sharded_raise_still_raises(tmp_store_path):
    """Kept as an assertion for a caller that wants one."""
    root = create_store(str(tmp_store_path))
    name = _sharded(root)
    with pytest.raises(ShardedPresenceError):
        root.derive_nonempty_chunks(name, on_sharded="raise")


def test_on_sharded_skip_returns_the_recorded_manifest(tmp_store_path):
    root = create_store(str(tmp_store_path))
    name = _sharded(root)
    expected = _write(root, name, [(0, 0, 0)])
    assert root.derive_nonempty_chunks(name, on_sharded="skip") == expected

    root._sharded_chunk_array(name).attrs["nonempty_chunks"] = []
    assert root.derive_nonempty_chunks(name, on_sharded="skip") == []


def test_an_unsharded_array_is_unaffected(tmp_store_path):
    """The flat branch keeps its own path through the listing."""
    root = create_store(str(tmp_store_path))
    root.create_sharded_chunk_array("flat", (4, 4, 4))
    expected = _write(root, "flat", [(0, 0, 0), (2, 2, 2)])

    root._sharded_chunk_array("flat").attrs["nonempty_chunks"] = []
    assert root.derive_nonempty_chunks("flat") == expected


# --- the coordinator verbs --------------------------------------------


def test_rebuild_presence_level_wide_includes_sharded_arrays(tmp_store_path):
    """It used to skip them, which is the opposite of repairing them."""
    lg = _sharded_level(tmp_store_path)
    lg.write_bytes("vertices", "0.0.0", b"payload")
    assert lg.list_chunks("vertices") == ["0.0.0"]

    lg._sharded_chunk_array("vertices").attrs["nonempty_chunks"] = []
    rebuilt = rebuild_presence(lg)

    assert "vertices" in rebuilt, f"sharded array was skipped: {rebuilt}"
    assert lg.list_chunks("vertices") == ["0.0.0"]


def test_rebuild_presence_single_array_form_rebuilds_a_sharded_array(
    tmp_store_path,
):
    lg = _sharded_level(tmp_store_path)
    lg.write_bytes("vertices", "1.2.3", b"payload")
    lg._sharded_chunk_array("vertices").attrs["nonempty_chunks"] = []

    assert rebuild_presence(lg, "vertices") == ["1.2.3"]


def test_rebuild_presence_skip_still_leaves_sharded_arrays_alone(
    tmp_store_path,
):
    """And still reports them as not rebuilt."""
    lg = _sharded_level(tmp_store_path)
    lg.write_bytes("vertices", "0.0.0", b"payload")
    lg._sharded_chunk_array("vertices").attrs["nonempty_chunks"] = []

    rebuilt = rebuild_presence(lg, on_sharded="skip")
    assert "vertices" not in rebuilt
    assert lg.list_chunks("vertices") == []
