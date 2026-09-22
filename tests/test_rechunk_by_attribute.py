"""Tests for the categorical attribute rechunking wrapper.

Covers:

* ``RechunkSpec(categorical=True)`` produces one bin per unique value,
  no quartile fallback, even with many uniques.
* The ergonomic ``rechunk_by_attribute`` wrapper sets the right
  RechunkSpec.
* The output store's level metadata records ``chunk_dims``,
  ``chunk_attribute_name``, ``chunk_attribute_values``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.core.store import open_store, read_level_metadata
from zarr_vectors.rechunk import RechunkSpec, rechunk, rechunk_by_attribute
from zarr_vectors.rechunk.spec import DimensionMapper
from zarr_vectors.types.points import write_points


# ===================================================================
# DimensionMapper categorical mode
# ===================================================================


def test_dimension_mapper_categorical_many_uniques():
    """With categorical=True, 20 unique values produce 20 bins (no quartiles)."""
    values = np.arange(20, dtype=np.int64)
    spec = RechunkSpec(by="attribute:x", categorical=True)
    mapping = DimensionMapper(spec).map_objects(
        n_objects=20,
        object_attributes={"x": values},
    )
    assert len(set(mapping.values())) == 20


def test_dimension_mapper_non_categorical_falls_back_to_quartiles():
    """Legacy path: without categorical and >10 uniques, quartile binning kicks in."""
    values = np.arange(20, dtype=np.float64)
    spec = RechunkSpec(by="attribute:x")  # categorical defaults to False
    mapping = DimensionMapper(spec).map_objects(
        n_objects=20,
        object_attributes={"x": values},
    )
    # Quartile binning produces at most 4 bins.
    assert len(set(mapping.values())) <= 4


def test_dimension_mapper_categorical_string_values():
    values = np.array(["gene_a", "gene_b", "gene_c", "gene_a"])
    spec = RechunkSpec(by="attribute:gene", categorical=True)
    mapping = DimensionMapper(spec).map_objects(
        n_objects=4,
        object_attributes={"gene": values},
    )
    # 3 unique values → 3 bins.  Two objects share bin (for "gene_a").
    assert len(set(mapping.values())) == 3
    assert mapping[0] == mapping[3]


# ===================================================================
# End-to-end rechunk_by_attribute on a points store with per-object attrs
# ===================================================================


def _make_points_store_with_object_attr(tmp_path, n_objects=30, attr="cluster"):
    """Create a points store where every vertex is its own object and
    has a per-object integer attribute (the cluster ID)."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (n_objects, 3)).astype("f4")
    obj_ids = np.arange(n_objects, dtype=np.int64)
    # Assign each object to one of 5 cluster IDs.
    clusters = rng.integers(0, 5, size=n_objects)

    store = tmp_path / "src.zv"
    write_points(
        str(store), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=obj_ids,
        object_attributes={attr: clusters},
    )
    return store, clusters


def test_rechunk_by_attribute_wrapper_categorical(tmp_path):
    src, clusters = _make_points_store_with_object_attr(tmp_path, n_objects=40)
    out = tmp_path / "rechunked.zv"

    summary = rechunk_by_attribute(str(src), "cluster", output=str(out))

    assert summary["bins_created"] == len(np.unique(clusters))

    # Level metadata records the attribute mapping.
    root = open_store(str(out))
    lm = read_level_metadata(root, 0)
    assert lm.chunk_attribute_name == "cluster"
    assert lm.chunk_attribute_values is not None
    assert lm.chunk_attribute_values == sorted(set(int(c) for c in clusters))
    assert lm.chunk_dims is not None
    assert lm.chunk_dims[0] == "cluster"


def test_rechunk_by_attribute_matches_explicit_spec(tmp_path):
    """The wrapper should produce the same output as a hand-built RechunkSpec."""
    src1, _ = _make_points_store_with_object_attr(tmp_path / "a", n_objects=20)
    src2, _ = _make_points_store_with_object_attr(tmp_path / "b", n_objects=20)

    out1 = tmp_path / "out1.zv"
    out2 = tmp_path / "out2.zv"

    rechunk_by_attribute(str(src1), "cluster", output=str(out1))
    rechunk(
        str(src2),
        RechunkSpec(
            by="attribute:cluster",
            categorical=True,
            prefix_dim_name="cluster",
        ),
        output=str(out2),
    )

    lm1 = read_level_metadata(open_store(str(out1)), 0)
    lm2 = read_level_metadata(open_store(str(out2)), 0)
    assert lm1.chunk_dims == lm2.chunk_dims
    assert lm1.chunk_attribute_name == lm2.chunk_attribute_name
    # The two source stores use the same RNG seed so values are equal too.
    assert lm1.chunk_attribute_values == lm2.chunk_attribute_values


def test_rechunk_by_attribute_high_cardinality(tmp_path):
    """A store with ~40 unique cluster IDs should not collapse to 4 quartile bins."""
    rng = np.random.default_rng(0)
    n_objects = 40
    pos = rng.uniform(0, 100, (n_objects, 3)).astype("f4")
    clusters = np.arange(n_objects, dtype=np.int64)  # 40 unique values
    obj_ids = np.arange(n_objects, dtype=np.int64)

    src = tmp_path / "src.zv"
    write_points(
        str(src), pos,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=obj_ids,
        object_attributes={"cluster": clusters},
    )

    out = tmp_path / "rechunked.zv"
    summary = rechunk_by_attribute(str(src), "cluster", output=str(out))
    # Without categorical=True this would have been 4 (quartile fallback).
    assert summary["bins_created"] == n_objects
