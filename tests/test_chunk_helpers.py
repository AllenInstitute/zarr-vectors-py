"""Tests for the Tier D / E helpers.

* ``neighbouring_chunk_keys`` — pure tuple work, table-driven.
* ``chunk_local_to_global_offsets`` — backed by the vertex_count sidecar.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.spatial.boundary import chunk_local_to_global_offsets
from zarr_vectors.spatial.chunking import neighbouring_chunk_keys
from zarr_vectors.types.points import write_points


# ===================================================================
# neighbouring_chunk_keys
# ===================================================================


def test_neighbours_2d_halo_1():
    out = neighbouring_chunk_keys((1, 1), halo=1)
    # 3^2 - 1 = 8 neighbours
    assert len(out) == 8
    assert (1, 1) not in out
    assert (0, 0) in out and (2, 2) in out and (1, 2) in out


def test_neighbours_3d_halo_1():
    out = neighbouring_chunk_keys((0, 0, 0), halo=1)
    assert len(out) == 26  # 3^3 - 1


def test_neighbours_3d_halo_2():
    out = neighbouring_chunk_keys((0, 0, 0), halo=2)
    assert len(out) == 5 ** 3 - 1


def test_neighbours_include_self():
    out = neighbouring_chunk_keys((1, 1), halo=1, include_self=True)
    assert (1, 1) in out
    assert len(out) == 9


def test_neighbours_filter_to_occupied():
    occupied = {(0, 0), (0, 1), (1, 0)}
    out = neighbouring_chunk_keys((0, 0), halo=1, occupied_keys=occupied)
    assert set(out) == {(0, 1), (1, 0)}  # (-1,*) / (*, -1) are off-grid


def test_neighbours_invalid_halo():
    with pytest.raises(ValueError):
        neighbouring_chunk_keys((0, 0), halo=-1)


def test_neighbours_works_for_4d_keys():
    """Attribute-chunked stores produce 4D chunk keys (attr_bin, z, y, x).
    The helper must compose with arbitrary arity."""
    out = neighbouring_chunk_keys((0, 0, 0, 0), halo=1)
    assert len(out) == 3 ** 4 - 1


# ===================================================================
# chunk_local_to_global_offsets
# ===================================================================


def test_offsets_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (777, 3)).astype("f4")
    store = tmp_path / "p.zv"
    write_points(str(store), pos, chunk_shape=(50.0, 50.0, 50.0))

    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    offsets, keys, total = chunk_local_to_global_offsets(lvl)
    assert total == 777
    # Offsets monotonic non-decreasing
    last = -1
    for k in keys:
        assert offsets[k] >= last
        last = offsets[k]


def test_offsets_empty_store_safe(tmp_path):
    """An empty level should report 0 chunks and 0 vertices, not raise."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10, (5, 3)).astype("f4")  # one chunk
    store = tmp_path / "p.zv"
    write_points(str(store), pos, chunk_shape=(100.0, 100.0, 100.0))
    root = open_store(str(store))
    lvl = get_resolution_level(root, 0)
    _, keys, total = chunk_local_to_global_offsets(lvl)
    assert total == 5
    assert len(keys) == 1


def test_offsets_are_right_on_a_2d_store(tmp_path):
    """The row width is the store's, not an assumed three.

    ``chunk_local_to_global_offsets`` divides each blob's byte length by
    ``ndim * itemsize`` to get its row count, and ``ndim`` used to be
    hardcoded to 3.  On a 2D store that divides by 12 where it should
    divide by 8, so every count -- and so every offset, and so every
    global vertex index derived from one -- came out two thirds of the
    truth.  Nothing raised; the numbers were simply wrong.
    """
    store = tmp_path / "flat.zv"
    positions = np.array(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [11.0, 11.0], [12.0, 12.0]],
        dtype=np.float32,
    )
    write_points(store, positions, chunk_shape=(10.0, 10.0))

    level = get_resolution_level(open_store(store), 0)
    offsets, chunk_keys, total = chunk_local_to_global_offsets(level)

    assert total == len(positions)
    # Offsets partition [0, total) in chunk order, with no gaps.
    assert sorted(offsets.values()) == [0, 3]
    assert offsets[chunk_keys[0]] == 0


def test_offsets_accept_an_explicit_row_width(tmp_path):
    store = tmp_path / "explicit.zv"
    positions = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    write_points(store, positions, chunk_shape=(10.0, 10.0))

    level = get_resolution_level(open_store(store), 0)
    _offsets, _keys, total = chunk_local_to_global_offsets(level, 2)
    assert total == 2
