"""Level 2 runs RootMetadata.validate().

Reading parses root metadata without checking values, and the validator
never ran the rules either, so a store with ``reduction_factor=1`` or an
unknown ``cross_level_storage`` validated at level 5.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

import zarr_vectors.api as zv
from zarr_vectors.types.points import write_points
from zarr_vectors.validate import validate


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "s.zv")
    write_points(path, np.random.default_rng(0).uniform(0, 100, (300, 3)).astype("f4"),
                 chunk_shape=(50,) * 3)
    return path


def _set(path, key, value):
    g = zarr.open_group(path, mode="r+")
    block = dict(g.attrs["zarr_vectors"])
    block[key] = value
    g.attrs["zarr_vectors"] = block


def test_valid_store_passes_root_validation(store):
    r = validate(store, level=2)
    assert r.ok, r.summary()
    assert "Root metadata validated" in r.passed


def test_fresh_api_store_passes_root_validation(tmp_path):
    path = str(tmp_path / "f.zv")
    zv.create(path, schema=zv.Schema(kind="graph", bounds=([0] * 3, [100] * 3)))
    r = validate(path, level=2)
    assert r.ok, r.summary()


@pytest.mark.parametrize(
    ("key", "value", "words"),
    [
        ("reduction_factor", 1, "reduction_factor"),
        ("cross_level_storage", "bogus", "cross_level_storage"),
        ("cross_level_depth", -5, "cross_level_depth"),
        ("manifest_layout", "weird", "manifest_layout"),
        ("geometry_types", ["blob"], "geometry_type 'blob'"),
        ("zv_version", "0.8.0", "zv_version"),
    ],
)
def test_invalid_root_value_fails_level_2(store, key, value, words):
    _set(store, key, value)
    r = validate(store, level=2)
    assert any(
        e.startswith("Root metadata invalid") and words in e for e in r.errors
    ), r.summary()
