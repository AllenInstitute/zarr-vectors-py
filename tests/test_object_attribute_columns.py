"""``write_object_attribute_columns``: many columns, the store one-per-column leaves."""

from __future__ import annotations

import numpy as np
import pytest

from tests._store_compare import assert_stores_identical
from zarr_vectors.core.arrays import (
    read_object_attributes,
    write_object_attribute_columns,
    write_object_attributes,
)
from zarr_vectors.exceptions import ArrayError


def _level(tmp_path, name):
    from zarr_vectors.building import create_store, get_resolution_level

    (tmp_path / name).mkdir()
    path = tmp_path / name / "s.zarrvectors"
    root = create_store(path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3)
    return path, get_resolution_level(root, 0)


def _flush(rng, n):
    return {
        "length": rng.uniform(0, 100, n).astype(np.float32),
        "count": rng.integers(0, 9, n).astype(np.int32),
        "rgb": rng.integers(0, 255, (n, 3)).astype(np.uint8),
    }


def test_columns_leave_the_store_single_columns_do(tmp_path):
    path_a, lg_a = _level(tmp_path, "single")
    path_b, lg_b = _level(tmp_path, "columns")
    rng = np.random.default_rng(3)
    n_total = 0
    # First write, plain appends, a gap, and a torn flush that reaches back.
    for step, gap in enumerate([0, 0, 5, 0, -4, 0]):
        cols = _flush(rng, int(rng.integers(1, 30)))
        at = max(0, n_total + gap)
        fills = {"count": -1}
        for name, data in cols.items():
            write_object_attributes(
                lg_a, name, data, mode="append", at=at, fill_value=fills.get(name),
            )
        write_object_attribute_columns(lg_b, cols, at=at, fill_values=fills)
        n_total = at + len(cols["length"])
    assert_stores_identical(path_a, path_b)
    np.testing.assert_array_equal(
        read_object_attributes(lg_a, "rgb"), read_object_attributes(lg_b, "rgb"),
    )


def test_it_looks_each_column_up_once(tmp_path, monkeypatch):
    from zarr_vectors.core.group import Group

    _, lg = _level(tmp_path, "ops")
    rng = np.random.default_rng(1)
    write_object_attribute_columns(lg, _flush(rng, 10))
    lookups = []
    real = Group._lookup_node

    def _count(self, path, *a, **kw):
        lookups.append(path)
        return real(self, path, *a, **kw)

    monkeypatch.setattr(Group, "_lookup_node", _count)
    write_object_attribute_columns(lg, _flush(rng, 10), at=12)
    per_column = {p for p in lookups if p.startswith("object_attributes/")}
    assert len(per_column) == 3
    assert all(lookups.count(p) == 1 for p in per_column)


def test_a_tail_shape_mismatch_raises(tmp_path):
    _, lg = _level(tmp_path, "bad")
    write_object_attribute_columns(lg, {"rgb": np.zeros((2, 3), np.uint8)})
    with pytest.raises(ArrayError, match="tail dimensions"):
        write_object_attribute_columns(lg, {"rgb": np.zeros((2, 4), np.uint8)})


def test_device_columns_are_copied_off_once(tmp_path):
    from tests._fake_device import FakeDeviceArray
    from zarr_vectors import _xp

    _, lg = _level(tmp_path, "dev")
    cols = _flush(np.random.default_rng(2), 8)
    with _xp.count_transfers() as stats:
        write_object_attribute_columns(lg, {k: FakeDeviceArray(v) for k, v in cols.items()})
    assert stats.d2h_calls == 3
    np.testing.assert_array_equal(read_object_attributes(lg, "count"), cols["count"])
