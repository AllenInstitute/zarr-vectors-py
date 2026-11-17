"""The NGFF transform of a coarse level places its vertices where they are.

Coarse vertices are centroids of level-0 positions, stored in level-0
coordinates.  The multiscales entry used to say ``scale = bin_ratio`` and
``translation = bin_shape / 2``, so an NGFF viewer drew level 1 at twice
its size and level 2 at four times.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.multiscale import (
    read_level_transform,
)
from zarr_vectors.core.store import (
    open_store,
    read_level_metadata,
)
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.points import read_points, write_points


def _apply(positions, scale, translation):
    out = positions * np.asarray(scale)
    if translation is not None:
        out = out + np.asarray(translation)
    return out


@pytest.fixture
def pyramid(tmp_path):
    store = str(tmp_path / "p.zarr")
    write_points(
        store,
        np.random.default_rng(0).uniform(0, 100, (2000, 3)).astype(np.float32),
        chunk_shape=(50.0,) * 3, bin_shape=(10.0,) * 3,
        bounds=([0.0] * 3, [100.0] * 3),
    )
    build_pyramid(store, factors=[(2, 1), (2, 1)])
    return store


def test_coarse_levels_carry_level_zero_transform(pyramid):
    root = open_store(pyramid)
    s0, t0 = read_level_transform(root, 0)
    world0 = _apply(read_points(pyramid, level=0)["positions"], s0, t0)
    for level in (1, 2):
        s, t = read_level_transform(root, level)
        assert s == s0 and t == t0
        world = _apply(read_points(pyramid, level=level)["positions"], s, t)
        # Centroids lie inside the level-0 extent once both are placed.
        assert np.all(world.min(0) >= world0.min(0) - 1e-4)
        assert np.all(world.max(0) <= world0.max(0) + 1e-4)


def test_bin_geometry_still_reads_back(pyramid):
    root = open_store(pyramid)
    for level, ratio in ((1, 2.0), (2, 4.0)):
        lm = read_level_metadata(root, level)
        assert lm.bin_ratio == pytest.approx((ratio,) * 3)
        assert lm.bin_shape == pytest.approx((10.0 * ratio,) * 3)


def test_migration_moves_ngff_only_bins_into_the_level():
    """Stores older than the per-level bin fields kept a coarse level's
    bins only in its NGFF entry.  Format 0.10.0 drops that block, so the
    0.10.0 migration copies them into the level's own object."""
    from zarr_vectors.core.migrate import _migrate_level, _ngff_bins

    root_attrs = {"multiscales": [{"datasets": [
        {"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0] * 3},
            {"type": "translation", "translation": [5.0] * 3}]},
        {"path": "2", "coordinateTransformations": [
            {"type": "scale", "scale": [4.0] * 3},
            {"type": "translation", "translation": [20.0] * 3}]},
    ]}]}
    bins = _ngff_bins(root_attrs)
    assert bins == {"2": {"bin_ratio": [4.0] * 3, "bin_shape": [40.0] * 3}}

    level = _migrate_level("2", {"zarr_vectors_level": {"level": 2}}, bins["2"])
    assert level["zarr_vectors"] == {
        "kind": "level", "level": 2, "bin_ratio": [4.0] * 3, "bin_shape": [40.0] * 3,
    }
    # A level that already carries its bins keeps them.
    own = {"zarr_vectors_level": {"level": 2, "bin_ratio": [3.0] * 3}}
    assert _migrate_level("2", own, bins["2"])["zarr_vectors"]["bin_ratio"] == [3.0] * 3
