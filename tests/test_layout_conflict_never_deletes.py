"""A layout disagreement must never cost data.

``_ensure_array_dir`` recreates an array when an explicit write session
asks for a layout it does not have, and recreating means deleting the
node first.  That is correct for the array ``create_store`` warm-creates
and nobody has written to — it is how a later session applies its shard
shape and codecs — but it is catastrophic for an array that holds data.

Two things put real data on that path.  ``shard_shape=None`` was the
default of every type writer, so a plain second write into a store built
with ``shard_shape=`` read as "unsharded wanted, sharded found".  And the
emptiness test was ``nonempty_chunks``, which is empty in exactly the
cases where it must not be believed: a decentralised writer passing
``record_presence=False`` leaves it so while the payloads are on disk.

The first of those is gone -- the writers default to ``"inherit"`` now,
so an unsaid shard shape takes the store's own and there is nothing to
disagree about.  The guard still has to hold for a caller who asks for a
different layout outright, which is what these tests do.

Covers:

* An explicitly differently-laid-out write preserves the first.
* An ordinary second write neither deletes nor warns.
* An unstamped-but-populated array is not treated as empty.
* An array that really is empty is still re-laid-out (no over-correction).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from zarr_vectors.building import array_is_sharded
from zarr_vectors.core.arrays import create_vertices_array
from zarr_vectors.core.store import create_store, get_resolution_level
from zarr_vectors.types.points import read_points, write_points

_BOUNDS = [[0, 0, 0], [64, 64, 64]]
_CHUNK = (16, 16, 16)


def _level(root):
    return get_resolution_level(root, 0)


def test_an_explicitly_unsharded_write_into_a_sharded_store_preserves_it(
    tmp_path,
):
    """The headline case: this used to delete the first write outright.

    The second write says ``shard_shape=None`` -- *explicitly* unsharded,
    against a store that is sharded. That is a real disagreement, and the
    caller is told about it; what must not happen is the array being
    recreated to satisfy it.

    The default no longer reaches here: an unsaid ``shard_shape`` inherits
    the store's own declaration, so the ordinary second write has nothing
    to disagree about (see
    ``test_declared_shard_layout.py::test_a_second_write_inherits_rather_than_unsharding``).
    Asking for the conflict outright is the only way to exercise the
    guard now, which is the shape a regression test for it should have.
    """
    path = str(tmp_path / "s.zarrvectors")
    first = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype="float32")
    second = np.array([[40.0, 40.0, 40.0]], dtype="float32")

    write_points(path, first, bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2)
    assert len(read_points(path)["positions"]) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_points(
            path, second, bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=None,
        )

    assert len(read_points(path)["positions"]) == 3, (
        "the first write's vertices were destroyed by the second"
    )
    assert any(
        "holds data with a different layout" in str(w.message)
        for w in caught
    ), "reused a mismatched array without saying so"


def test_an_ordinary_second_write_neither_deletes_nor_warns(tmp_path):
    """Inheriting means there is no disagreement to report."""
    path = str(tmp_path / "s.zarrvectors")
    write_points(
        path, np.array([[1.0, 1.0, 1.0]], dtype="float32"),
        bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_points(
            path, np.array([[40.0, 40.0, 40.0]], dtype="float32"),
            bounds=_BOUNDS, chunk_shape=_CHUNK,
        )

    assert len(read_points(path)["positions"]) == 2
    assert not [
        w for w in caught if "different layout" in str(w.message)
    ]


def test_the_reused_array_keeps_its_own_layout(tmp_path):
    """Reuse means reuse — the store does not quietly become flat."""
    path = str(tmp_path / "s.zarrvectors")
    pts = np.array([[1.0, 1.0, 1.0]], dtype="float32")

    write_points(path, pts, bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        write_points(
            path, np.array([[40.0, 40.0, 40.0]], dtype="float32"),
            bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=None,
        )

    from zarr_vectors.core.store import open_store

    lg = _level(open_store(path, mode="r"))
    assert array_is_sharded(lg, "vertices")


def test_an_unstamped_but_populated_array_is_not_treated_as_empty(
    tmp_store_path,
):
    """``record_presence=False`` leaves the manifest empty on purpose.

    The payloads are on disk; believing the manifest here is what deleted
    them. The store is asked instead.
    """
    root = create_store(
        str(tmp_store_path), bounds=_BOUNDS, chunk_shape=_CHUNK,
        geometry_types=["point_cloud"],
    )
    lg = _level(root)
    lg.write_bytes("vertices", "0.0.0", b"payload", record_presence=False)
    assert lg.list_chunks("vertices") == []          # manifest says empty
    assert lg._array_has_stored_data("vertices")     # the store disagrees

    # An explicit session wanting a different layout must not recreate it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with lg.native_sharded_arrays((2, 2, 2), (4, 4, 4)):
            create_vertices_array(lg, dtype="float32")

    assert lg.read_bytes("vertices", "0.0.0") == b"payload"
    assert any(
        "holds data with a different layout" in str(w.message)
        for w in caught
    )


def test_an_empty_array_is_still_recreated_to_the_session_layout(
    tmp_store_path,
):
    """Guards against over-correcting: the warm-create path must still work.

    ``create_store`` makes ``vertices`` unsharded and uncompressed; a
    later session with ``shard_shape=`` is supposed to replace it. There
    is nothing to lose, so that must keep happening.
    """
    root = create_store(
        str(tmp_store_path), bounds=_BOUNDS, chunk_shape=_CHUNK,
        geometry_types=["point_cloud"],
    )
    lg = _level(root)
    assert not array_is_sharded(lg, "vertices")
    assert not lg._array_has_stored_data("vertices")

    with lg.native_sharded_arrays((2, 2, 2), (4, 4, 4)):
        create_vertices_array(lg, dtype="float32")

    assert array_is_sharded(lg, "vertices"), (
        "an empty array was not re-laid-out; the warm-create path is broken"
    )


def test_a_matching_layout_is_reused_without_warning(tmp_store_path):
    """No warning when there is no disagreement."""
    root = create_store(
        str(tmp_store_path), bounds=_BOUNDS, chunk_shape=_CHUNK,
        geometry_types=["point_cloud"],
    )
    lg = _level(root)
    with lg.native_sharded_arrays((2, 2, 2), (4, 4, 4)):
        create_vertices_array(lg, dtype="float32")
    lg.write_bytes("vertices", "0.0.0", b"payload")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with lg.native_sharded_arrays((2, 2, 2), (4, 4, 4)):
            create_vertices_array(lg, dtype="float32")

    assert lg.read_bytes("vertices", "0.0.0") == b"payload"
    assert not [
        w for w in caught
        if "holds data with a different layout" in str(w.message)
    ]


@pytest.mark.parametrize("record_presence", [True, False])
def test_stored_data_probe_agrees_with_reality(
    tmp_store_path, record_presence,
):
    """The probe answers about the store, whatever the manifest says."""
    root = create_store(str(tmp_store_path))
    root.create_sharded_chunk_array("cells", (4, 4, 4))
    assert not root._array_has_stored_data("cells")

    root.write_bytes(
        "cells", "0.0.0", b"payload", record_presence=record_presence,
    )
    assert root._array_has_stored_data("cells")
