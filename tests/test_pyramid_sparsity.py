"""Object sparsity across a pyramid: nested selections, honest metadata."""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_object_manifest_rows
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
)
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.polylines import write_polylines


def _lines(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return [
        np.clip(
            np.cumsum(rng.normal(0, 2, (10, 3)), 0) + rng.uniform(10, 90, 3),
            0, 100,
        ).astype(np.float32)
        for _ in range(n)
    ]


def _present(store, level):
    ids, manifests = read_object_manifest_rows(
        get_resolution_level(open_store(store), level),
    )
    return {int(i) for i, m in zip(ids.tolist(), manifests) if m}


@pytest.mark.parametrize("seed", [3, None])
def test_each_level_selects_from_the_previous_survivors(tmp_path, seed):
    store = str(tmp_path / "l.zarr")
    write_polylines(
        store, _lines(), chunk_shape=(25.0,) * 3, bin_shape=(5.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
    )
    summary = build_pyramid(
        store, factors=[(1, 2), (1, 2), (1, 2)], sparsity_seed=seed,
    )
    previous = _present(store, 0)
    assert len(previous) == 400
    for level, expected in ((1, 200), (2, 100), (3, 50)):
        now = _present(store, level)
        assert len(now) == expected, level
        assert now <= previous, level
        assert summary["level_specs"][level - 1]["objects_kept"] == expected
        previous = now


def test_point_cloud_without_objects_records_nothing_dropped(tmp_path):
    from zarr_vectors.types.points import read_points, write_points

    store = str(tmp_path / "p.zarr")
    pos = np.random.default_rng(1).uniform(0, 100, (500, 3)).astype(np.float32)
    write_points(
        store, pos, chunk_shape=(25.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
    )
    build_pyramid(store, factors=[(2, 4)])
    # One implicit object: nothing to select from, so nothing is dropped.
    assert len(read_points(store, level=1)["positions"]) > 0
    assert read_level_metadata(open_store(store), 1).object_sparsity == 1.0


def test_object_sparsity_records_a_selection_that_ran(tmp_path):
    store = str(tmp_path / "l.zarr")
    write_polylines(
        store, _lines(), chunk_shape=(25.0,) * 3, bin_shape=(5.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
    )
    build_pyramid(store, factors=[(1, 4)], sparsity_seed=0)
    assert read_level_metadata(open_store(store), 1).object_sparsity == 0.25
