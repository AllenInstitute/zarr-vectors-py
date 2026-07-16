"""Regressions for the lazy ZVLevel accessors.

Both bugs here were pre-existing and unrelated to the links merge; the
post-merge cleanup audit surfaced them. Each test hangs or raises against
the pre-fix code.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from zarr_vectors.lazy import open_zv
from zarr_vectors.types.points import write_points


def _point_store(with_attrs: bool = True) -> str:
    path = os.path.join(tempfile.mkdtemp(), "p.zv")
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    kw = {}
    if with_attrs:
        kw["vertex_attributes"] = {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([3.0, 4.0], dtype=np.float32),
        }
    write_points(path, pos, **kw)
    return path


class TestAttributeAccessorIsAMapping:
    """``level.attributes`` must behave as a Mapping, not hang.

    It defined ``__getitem__`` but neither ``__iter__`` nor ``__len__``, so
    ``list()`` fell back to the legacy sequence protocol (``acc[0]``,
    ``acc[1]``, …); and ``__getitem__`` never raised for an unknown name,
    so the sequence never terminated. ``list(level.attributes)`` hung the
    interpreter.
    """

    def test_iteration_terminates(self) -> None:
        acc = open_zv(_point_store())[0].attributes
        # If __iter__/__len__ are missing this hangs rather than fails; the
        # test process cap turns that into a failure.
        assert sorted(acc) == ["a", "b"]
        assert sorted(dict(acc)) == ["a", "b"]
        assert len(acc) == 2

    def test_membership(self) -> None:
        acc = open_zv(_point_store())[0].attributes
        assert "a" in acc
        assert "missing" not in acc

    def test_unknown_key_raises_keyerror(self) -> None:
        acc = open_zv(_point_store())[0].attributes
        with pytest.raises(KeyError):
            acc["missing"]

    def test_known_key_returns_collection(self) -> None:
        acc = open_zv(_point_store())[0].attributes
        got = acc["a"].compute()
        np.testing.assert_allclose(np.asarray(got).ravel(), [1.0, 2.0])

    def test_empty_when_no_attributes(self) -> None:
        acc = open_zv(_point_store(with_attrs=False))[0].attributes
        assert list(acc) == []
        assert len(acc) == 0


class TestHasObjectOnPointCloud:
    """``has_object`` must return False on a store with no object index.

    A plain point cloud has no ``object_index``; ``read_object_manifest``
    reads ``sid_ndim`` off its meta and raised ``KeyError('sid_ndim')``.
    An existence query must answer, not crash.
    """

    def test_returns_false_not_keyerror(self) -> None:
        lvl = open_zv(_point_store())[0]
        assert lvl.has_object(0) is False
        assert lvl.has_object(999) is False
