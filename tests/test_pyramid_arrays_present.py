"""``build_pyramid`` keeps every level's ``arrays_present`` true to disk."""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
)
from zarr_vectors.multiresolution.coarsen import build_pyramid, coarsen_level
from zarr_vectors.types.polylines import write_polylines


def _store(tmp_path):
    rng = np.random.default_rng(0)
    lines = [
        np.clip(np.cumsum(rng.normal(0, 2, (10, 3)), 0) + 50, 0, 100)
        .astype(np.float32)
        for _ in range(30)
    ]
    path = str(tmp_path / "l.zarr")
    write_polylines(
        path, lines, chunk_shape=(25.0,) * 3, bin_shape=(5.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
        object_attributes={"w": rng.random(30)},
    )
    return path


def _assert_true_to_disk(path, levels):
    root = open_store(path)
    for level in levels:
        on_disk = set(get_resolution_level(root, level).children())
        declared = set(read_level_metadata(root, level).arrays_present)
        assert declared == on_disk, level


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_build_pyramid_declares_what_it_wrote(tmp_path, depth):
    path = _store(tmp_path)
    build_pyramid(path, factors=[(2, 1), (2, 2)], cross_level_depth=depth)
    root = open_store(path)
    for level in (1, 2):
        present = read_level_metadata(root, level).arrays_present
        assert {"object_attributes", "vertex_fragments"} <= set(present)
        assert ("links" in present) == (depth > 0)
    _assert_true_to_disk(path, (0, 1, 2))


def test_coarsen_level_declares_what_it_wrote(tmp_path):
    path = _store(tmp_path)
    coarsen_level(path, 0, 1, coarsen_factor=2.0, cross_level_storage="explicit")
    _assert_true_to_disk(path, (0, 1))
