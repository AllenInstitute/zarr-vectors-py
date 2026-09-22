"""Level 1 checks a level's per-chunk arrays share one rank, keys included.

A store corrupted by the rank bug fixed in ``cf4c2f1`` -- an attribute
array allocated one rank short of its level -- used to pass every
validator clean. These build that corruption directly, and the other
shapes of it, and check level 1 now names each one.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.arrays import level_grid_layout
from zarr_vectors.core.store import open_store
from zarr_vectors.rechunk import RechunkSpec, rechunk
from zarr_vectors.types.points import write_points
from zarr_vectors.types.polylines import write_polylines
from zarr_vectors.validate import validate
from zarr_vectors.validate.structure import validate_structure

from tests.test_attr_chunked_allocation import (
    _CHUNK,
    _OFFSET_BOUNDS,
    _attr_store,
    _level,
)


def _errors_mentioning(result, *words):
    return [e for e in result.errors if all(w in e for w in words)]


# --- corrupt stores ---------------------------------------------------


def test_a_rank_short_attribute_array_is_an_error(tmp_path):
    """The cf4c2f1 store: rank-3 array, rank-4 keys, on a rank-4 level."""
    path = _attr_store(tmp_path / "g.zarrvectors", bounds=_OFFSET_BOUNDS)
    lg = _level(path)
    origin, grid = level_grid_layout(_OFFSET_BOUNDS, _CHUNK)
    lg.create_sharded_chunk_array("vertex_attributes/score", grid, origin=origin)
    # Stamped directly: write_bytes now refuses the mismatch outright.
    lg.write_array_meta(
        "vertex_attributes/score", {"nonempty_chunks": ["1.2.2.2", "1.2.2.3"]},
    )

    result = validate_structure(path)

    assert not result.ok
    assert _errors_mentioning(result, "vertex_attributes/score", "presence keys")
    assert _errors_mentioning(result, "vertex_attributes/score", "vertices has rank 4")


def test_a_rank_short_array_is_caught_after_a_manifest_rebuild(tmp_path):
    """A rebuilt manifest agrees with its own array; only the level catches it."""
    path = _attr_store(tmp_path / "g.zarrvectors", bounds=_OFFSET_BOUNDS)
    lg = _level(path)
    origin, grid = level_grid_layout(_OFFSET_BOUNDS, _CHUNK)
    lg.create_sharded_chunk_array("vertex_attributes/score", grid, origin=origin)

    result = validate_structure(path)

    assert not _errors_mentioning(result, "presence keys")
    assert _errors_mentioning(result, "vertex_attributes/score", "rank 3")


def test_a_key_of_the_wrong_arity_is_an_error(tmp_path):
    path = _attr_store(tmp_path / "g.zarrvectors")
    lg = _level(path)
    keys = lg.list_chunks("vertices")
    lg.write_array_meta("vertices", {"nonempty_chunks": [*keys, "0.0.0"]})

    result = validate_structure(path)

    assert _errors_mentioning(result, "0/vertices", "'0.0.0'")


def test_a_bin_axis_that_disagrees_with_the_values_is_an_error(tmp_path):
    """A reader maps a value to a bin by position, so the lengths must agree."""
    path = _attr_store(tmp_path / "g.zarrvectors")
    root = open_store(path, mode="r+")
    level_attrs = root["0"].attrs.to_dict()
    block = dict(level_attrs["zarr_vectors_level"])
    block["chunk_attribute_values"] = ["A", "B"]  # the axis has 3
    root["0"].attrs.update({"zarr_vectors_level": block})

    result = validate_structure(path)

    assert _errors_mentioning(result, "3 bins", "names 2")


# --- clean controls ---------------------------------------------------


def test_an_attribute_chunked_store_passes(tmp_path):
    path = _attr_store(tmp_path / "g.zarrvectors", bounds=_OFFSET_BOUNDS)
    result = validate_structure(path)
    assert result.ok, result.summary()
    assert any("share rank 4" in p for p in result.passed)


def test_a_spatial_store_passes(tmp_path):
    rng = np.random.default_rng(0)
    path = str(tmp_path / "p.zarrvectors")
    write_points(
        path, rng.uniform(0, 100, (50, 3)).astype("float32"),
        chunk_shape=_CHUNK,
    )
    result = validate(path, level=5)
    assert result.ok, result.summary()


def test_a_group_rechunk_passes(tmp_path):
    """A leading axis with no chunk_attribute_values is still one rank."""
    rng = np.random.default_rng(1)
    polys = [
        rng.uniform(0, 100, (5, 3)).astype("float32") for _ in range(10)
    ]
    src = str(tmp_path / "src.zarrvectors")
    write_polylines(
        src, polys, chunk_shape=_CHUNK,
        groups={0: list(range(5)), 1: list(range(5, 10))},
    )
    out = str(tmp_path / "grouped.zarrvectors")
    rechunk(src, RechunkSpec(by="group"), output=out)

    result = validate_structure(out)
    assert result.ok, result.summary()
