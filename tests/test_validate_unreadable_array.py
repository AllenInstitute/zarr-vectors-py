"""Level 1 reports an array whose metadata zarr cannot parse.

A ``zarr.json`` whose ``shape`` rank disagrees with its ``chunk_grid``
made ``validate`` raise zarr's raw ``ValueError`` (on ``vertices``) or
pass silently (on an attribute array nothing opened).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from zarr_vectors.types.points import write_points
from zarr_vectors.validate import validate


def _store(tmp_path):
    path = str(tmp_path / "s.zv")
    write_points(
        path, np.random.default_rng(0).uniform(0, 100, (300, 3)).astype("f4"),
        chunk_shape=(50,) * 3, vertex_attributes={"a": np.arange(300.0)},
    )
    return path


def _drop_an_axis(path, array):
    f = os.path.join(path, array, "zarr.json")
    with open(f) as fh:
        meta = json.load(fh)
    meta["shape"] = meta["shape"][:-1]
    with open(f, "w") as fh:
        json.dump(meta, fh)


@pytest.mark.parametrize(
    "array", ["0/vertices", "0/vertex_fragments", "0/vertex_attributes/a"],
)
def test_rank_mismatch_is_a_validation_error(tmp_path, array):
    path = _store(tmp_path)
    _drop_an_axis(path, array)
    r = validate(path, level=5)   # must not raise
    assert not r.ok
    assert any(
        e.startswith(f"{array} metadata cannot be read") and "chunk_grid" in e
        for e in r.errors
    ), r.summary()


def test_intact_store_has_no_unreadable_nodes(tmp_path):
    r = validate(_store(tmp_path), level=1)
    assert r.ok, r.summary()
    assert not any("cannot be read" in e for e in r.errors)
