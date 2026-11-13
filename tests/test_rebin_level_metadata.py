"""``rebin_level`` keeps the bin shape where it belongs.

It used to write ``bin_shape`` / ``bin_ratio`` into level 0's own attrs
(which ``LevelMetadata.validate`` forbids) and leave the NGFF
translation at the old ``bin / 2``.  It also wrote vertices back as
float32 and left per-vertex attributes in the old vertex order.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.core.arrays import read_chunk_attributes, read_chunk_vertices
from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.multiscale import read_level_transform
from zarr_vectors.core.store import (
    get_resolution_level,
    open_store,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.exceptions import StoreError
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.rechunk import rebin_level
from zarr_vectors.types.points import write_points
from zarr_vectors.validate import validate


@pytest.fixture
def cloud(tmp_path: Path):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (600, 3))
    path = tmp_path / "cloud.zarr"
    write_points(path, pos, chunk_shape=(50.0,) * 3,
                 bounds=([0.0] * 3, [100.0] * 3), dtype="float64",
                 vertex_attributes={"i": np.arange(600, dtype=np.float32)})
    return path, pos


def _level_block(path, level):
    return json.loads((Path(path) / str(level) / "zarr.json").read_text())[
        "attributes"]["zarr_vectors_level"]


def test_level_zero_rebin_updates_root_and_ngff_only(cloud):
    path, _ = cloud
    rebin_level(path, (10.0,) * 3)
    block = _level_block(path, 0)
    assert "bin_shape" not in block and "bin_ratio" not in block
    LevelMetadata.from_dict({"zarr_vectors_level": block}).validate()
    root = open_store(str(path), mode="r")
    assert read_root_metadata(root).base_bin_shape == (10.0,) * 3
    assert read_level_transform(root, 0) == ([1.0] * 3, [5.0] * 3)
    assert validate(str(path), level=5).ok


def test_attributes_follow_vertices_and_dtype_is_kept(cloud):
    path, pos = cloud
    rebin_level(path, (10.0,) * 3)
    lg = get_resolution_level(open_store(str(path), mode="r"), 0)
    seen = Counter()
    for key in lg.list_chunks("vertices"):
        cc = tuple(int(x) for x in key.split("."))
        groups = read_chunk_vertices(lg, cc, dtype="float64", ndim=3)
        attrs = read_chunk_attributes(lg, "i", cc, dtype="float32", ncols=1)
        assert len(groups) > 1
        for g, a in zip(groups, attrs):
            for p, i in zip(g, np.ravel(a)):
                # float64 positions survive exactly, next to their own index
                np.testing.assert_array_equal(p, pos[int(i)])
                seen[int(i)] += 1
    assert seen == Counter(range(600))


def test_rebinning_the_base_rescales_coarser_levels(cloud):
    path, _ = cloud
    build_pyramid(path, factors=[(2.0, 1.0)])
    root = open_store(str(path), mode="r")
    before = read_level_metadata(root, 1).bin_shape
    rebin_level(path, (10.0,) * 3)
    root = open_store(str(path), mode="r")
    lm1 = read_level_metadata(root, 1)
    assert lm1.bin_shape == before
    assert lm1.bin_ratio == tuple(b / 10.0 for b in before)

    rebin_level(path, (25.0,) * 3, level=1)
    root = open_store(str(path), mode="r")
    lm1 = read_level_metadata(root, 1)
    assert lm1.bin_shape == (25.0,) * 3
    assert lm1.bin_ratio == (2.5,) * 3
    assert "bin_shape" not in _level_block(path, 1)


def test_a_store_with_manifests_is_refused(tmp_path):
    path = tmp_path / "objs.zarr"
    rng = np.random.default_rng(0)
    write_points(path, rng.uniform(0, 100, (100, 3)).astype("float32"),
                 chunk_shape=(50.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
                 object_ids=np.arange(100) % 7)
    with pytest.raises(StoreError, match="object_index"):
        rebin_level(path, (10.0,) * 3)
