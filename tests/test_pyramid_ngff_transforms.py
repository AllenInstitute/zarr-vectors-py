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
    write_multiscale_metadata,
)
from zarr_vectors.core.store import (
    get_resolution_level,
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


def _make_old_layout(store):
    """Rewrite a store the way it was written before: bins only in NGFF."""
    root = open_store(store, mode="r+")
    attrs = root.attrs.to_dict()
    for ds in attrs["multiscales"][0]["datasets"]:
        level = int(ds["path"])
        if level == 0:
            continue
        lg = get_resolution_level(root, level)
        zvl = dict(lg.attrs.to_dict()["zarr_vectors_level"])
        ratio = zvl.pop("bin_ratio")
        shape = zvl.pop("bin_shape")
        # Top-level keys are replaced whole, so this drops the two fields.
        lg.attrs.update({"zarr_vectors_level": zvl})
        ds["coordinateTransformations"] = [
            {"type": "scale", "scale": ratio},
            {"type": "translation", "translation": [b / 2 for b in shape]},
        ]
    root.attrs.update(attrs)
    return root


def test_old_layout_reads_and_regenerates(pyramid):
    root = _make_old_layout(pyramid)
    lg = get_resolution_level(root, 2)
    assert "bin_ratio" not in lg.attrs.to_dict()["zarr_vectors_level"]
    assert read_level_metadata(root, 2).bin_ratio == pytest.approx((4.0,) * 3)

    write_multiscale_metadata(root)
    root = open_store(pyramid)
    s0, t0 = read_level_transform(root, 0)
    assert read_level_transform(root, 2) == (s0, t0)
    lm = read_level_metadata(root, 2)
    assert lm.bin_ratio == pytest.approx((4.0,) * 3)
    assert lm.bin_shape == pytest.approx((40.0,) * 3)
