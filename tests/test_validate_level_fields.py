"""Per-level validation reads the fields where the writer keeps them.

Level fields live under ``zarr_vectors_level`` and the spatial transform
(``bin_shape`` / ``bin_ratio``) only in the NGFF ``multiscales`` block.
The level-2/3/5 checks used to read the level's top-level attributes,
found nothing there, and passed every store.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.types.points import write_points
from zarr_vectors.validate import validate


@pytest.fixture
def pyramid(tmp_path):
    path = str(tmp_path / "s.zv")
    rng = np.random.default_rng(0)
    write_points(
        path, rng.uniform(0, 100, (2000, 3)).astype("f4"),
        chunk_shape=(50, 50, 50), bin_shape=(25, 25, 25),
        object_ids=np.arange(2000) // 10,
    )
    build_pyramid(path, factors=[(2, 2)])
    return path


def _set(path, level, key, value):
    g = zarr.open_group(path, mode="r+")
    block = dict(g[str(level)].attrs["zarr_vectors_level"])
    block[key] = value
    g[str(level)].attrs["zarr_vectors_level"] = block


def _set_transform(path, level, kind, value):
    g = zarr.open_group(path, mode="r+")
    ms = g.attrs["multiscales"]
    for ds in ms[0]["datasets"]:
        if ds["path"] == str(level):
            for t in ds["coordinateTransformations"]:
                if t["type"] == kind:
                    t[kind] = value
    g.attrs["multiscales"] = ms


def test_written_pyramid_validates(pyramid):
    r = validate(pyramid, level=5)
    assert r.ok, r.summary()


@pytest.mark.parametrize("level", [0, 1])
def test_wrong_vertex_count_fails(pyramid, level):
    _set(pyramid, level, "vertex_count", 9999)
    r = validate(pyramid, level=5)
    assert any("vertex_count=9999" in e for e in r.errors), r.summary()


def test_negative_vertex_count_fails_level_2(pyramid):
    _set(pyramid, 1, "vertex_count", -3)
    r = validate(pyramid, level=2)
    assert any("vertex_count=-3 invalid" in e for e in r.errors), r.summary()


def test_vertex_count_growing_up_the_pyramid_fails_level_5(pyramid):
    # Level 3 would also flag the count itself; level 5's own check is the
    # pyramid ordering, so assert that message.
    _set(pyramid, 1, "vertex_count", 5000)
    r = validate(pyramid, level=5)
    assert any("5000 > resolution_0" in e for e in r.errors), r.summary()


@pytest.mark.parametrize("value", [5.0, 0.0, -1.0])
def test_object_sparsity_out_of_range_fails(pyramid, value):
    _set(pyramid, 1, "object_sparsity", value)
    r = validate(pyramid, level=2)
    assert any("object_sparsity" in e for e in r.errors), r.summary()


def test_bin_ratio_below_one_fails(pyramid):
    _set_transform(pyramid, 1, "scale", [0.5, 2.0, 2.0])
    r = validate(pyramid, level=2)
    assert any("bin_ratio[0]=0.5 < 1" in e for e in r.errors), r.summary()


def test_bin_shape_wrong_rank_fails(pyramid):
    _set_transform(pyramid, 1, "translation", [25.0, 25.0])
    r = validate(pyramid, level=2)
    assert any("bin_shape has 2 dims" in e for e in r.errors), r.summary()


def test_zero_placeholder_vertex_count_only_warns(pyramid):
    # Several build paths stamp 0 and never refresh it; that is stale
    # bookkeeping, reported, but not a reason to reject the store.
    _set(pyramid, 0, "vertex_count", 0)
    r = validate(pyramid, level=5)
    assert r.ok, r.summary()
    assert any("vertex_count=0 (not recorded)" in w for w in r.warnings)
