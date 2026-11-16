"""A bin that does not tile the chunk is rejected at the call declaring it.

Before, ``write_points(bin_shape=60)`` in a chunk of 100 wrote a store that
failed level-2 validation, ``bin_shape=70`` raised an unrelated out-of-grid
``StoreError`` from the chunk writer, and ``create_store(base_bin_shape=)``
accepted either.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zarr_vectors.core.metadata import check_bin_shape_tiles_chunk
from zarr_vectors.core.store import create_store
from zarr_vectors.exceptions import MetadataError
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.types.points import write_points
from zarr_vectors.validate import validate

PTS = np.random.default_rng(0).uniform(0, 100, (500, 3)).astype("f4")


@pytest.mark.parametrize("b", [60, 70, 150])
def test_write_points_rejects_a_bin_that_does_not_tile(tmp_path, b):
    path = str(tmp_path / "s.zv")
    with pytest.raises(MetadataError, match="does not divide chunk_shape"):
        write_points(path, PTS, chunk_shape=(100,) * 3, bin_shape=(b,) * 3)
    assert not os.path.exists(path) or not os.listdir(path)


def test_write_points_accepts_a_bin_that_tiles(tmp_path):
    path = str(tmp_path / "s.zv")
    write_points(path, PTS, chunk_shape=(100,) * 3, bin_shape=(25,) * 3)
    assert validate(path, level=3).ok


@pytest.mark.parametrize("b", [60, 70])
def test_create_store_rejects_a_bin_that_does_not_tile(tmp_path, b):
    path = str(tmp_path / "c.zv")
    with pytest.raises(MetadataError, match="base_bin_shape"):
        create_store(path, chunk_shape=(100,) * 3, bounds=([0] * 3, [100] * 3),
                     base_bin_shape=(b,) * 3)
    assert not os.path.exists(path)


def test_other_writers_reject_it_through_store_creation(tmp_path):
    with pytest.raises(MetadataError, match="does not divide"):
        write_polylines(str(tmp_path / "l.zv"), [PTS[:5]],
                        chunk_shape=(100,) * 3, bin_shape=(30,) * 3)


def test_helper_rank_and_sign():
    with pytest.raises(MetadataError, match="axes"):
        check_bin_shape_tiles_chunk((10, 10), (5,))
    with pytest.raises(MetadataError, match="> 0"):
        check_bin_shape_tiles_chunk((10,), (0,))
    check_bin_shape_tiles_chunk((10.0, 9.0), (2.5, 3.0))
