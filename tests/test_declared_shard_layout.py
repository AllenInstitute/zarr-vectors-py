"""A store declares its shard shape, and everything it allocates obeys.

``create_store(shard_shape=N)`` used to be accepted, dropped and
reported as success: there was no such parameter, so the value fell into
``**backend_kwargs`` and, on a local path, was discarded without reaching
anything.  Nothing recorded it either, so there was no way to ask a store
whether it was sharded, and no way for the arrays created later — in
workers that never saw the argument — to find out.

Recording it on the root is what closes that.  The units are grid cells,
not coordinates, so one declaration holds at every level of a pyramid
even though their grids differ.

Covers:

* The declaration round-trips, scalar and per-axis.
* Warm-created AND lazily created arrays honour it.
* A second write inherits rather than silently unsharding.
* Bad shapes are refused at the call that declared them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.building import (
    array_is_sharded,
    create_attribute_array,
    create_store,
    get_resolution_level,
    open_store,
    per_chunk_array_paths,
    read_root_metadata,
    update_root_metadata,
)
from zarr_vectors.exceptions import MetadataError
from zarr_vectors.types.points import read_points, write_points

_BOUNDS = ((0, 0, 0), (64, 64, 64))
_CHUNK = (16, 16, 16)


def _store(path, **kw):
    return create_store(
        str(path), bounds=_BOUNDS, chunk_shape=_CHUNK,
        geometry_types=["point_cloud"], **kw,
    )


def _root_block(path) -> dict:
    return json.loads(
        (Path(path) / "zarr.json").read_text(encoding="utf-8")
    )["attributes"]["zarr_vectors"]


# --- the declaration --------------------------------------------------


def test_create_store_records_a_scalar_shard_shape(tmp_path):
    """Stored as a bare int: the compact form a reader looks for."""
    path = tmp_path / "s.zarrvectors"
    _store(path, shard_shape=2)

    assert _root_block(path)["shard_shape"] == 2
    assert read_root_metadata(open_store(str(path), mode="r")).shard_shape == 2


def test_create_store_records_a_per_axis_shard_shape(tmp_path):
    path = tmp_path / "s.zarrvectors"
    _store(path, shard_shape=[2, 4, 8])

    assert _root_block(path)["shard_shape"] == [2, 4, 8]
    assert read_root_metadata(
        open_store(str(path), mode="r")
    ).shard_shape == [2, 4, 8]


def test_an_undeclared_store_stays_unsharded(tmp_path):
    """Sharding is opt-in; the default must not move."""
    path = tmp_path / "s.zarrvectors"
    root = _store(path)

    assert "shard_shape" not in _root_block(path)
    lg = get_resolution_level(root, 0)
    assert not any(array_is_sharded(lg, n) for n in per_chunk_array_paths(lg))


def test_update_root_metadata_accepts_shard_shape(tmp_path):
    """It raised MetadataError, because the field did not exist."""
    path = tmp_path / "s.zarrvectors"
    _store(path)

    update_root_metadata(open_store(str(path), mode="r+"), shard_shape=4)
    assert read_root_metadata(open_store(str(path), mode="r")).shard_shape == 4


# --- allocation -------------------------------------------------------


def test_warm_created_arrays_are_born_sharded(tmp_path):
    """``create_store`` makes these two itself, before any writer runs."""
    path = tmp_path / "s.zarrvectors"
    lg = get_resolution_level(_store(path, shard_shape=2), 0)

    assert array_is_sharded(lg, "vertices")
    assert array_is_sharded(lg, "vertex_fragments")


def test_a_lazily_created_array_honours_the_declaration(tmp_path):
    """The case the declaration exists for.

    No write session is open: this is the path a worker takes when it
    allocates an attribute array or a links segment on first write,
    having never seen ``create_store``'s arguments. It read a hardcoded
    ``shard_shape=None``, so a store came out sharded in whatever the
    coordinator made and flat in everything else.
    """
    path = tmp_path / "s.zarrvectors"
    _store(path, shard_shape=2)
    lg = get_resolution_level(open_store(str(path), mode="r+"), 0)

    create_attribute_array(lg, "intensity", dtype="float32")

    assert array_is_sharded(lg, "vertex_attributes/intensity")
    flat = [n for n in per_chunk_array_paths(lg) if not array_is_sharded(lg, n)]
    assert not flat, f"created unsharded despite a sharded store: {flat}"


def test_every_array_a_real_write_creates_is_sharded(tmp_path):
    path = str(tmp_path / "s.zarrvectors")
    rng = np.random.default_rng(0)
    write_points(
        path, rng.uniform(0, 64, (50, 3)).astype("float32"),
        bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2,
        vertex_attributes={"intensity": np.arange(50, dtype="float32")},
    )

    lg = get_resolution_level(open_store(path, mode="r"), 0)
    names = per_chunk_array_paths(lg)
    assert names
    flat = [n for n in names if not array_is_sharded(lg, n)]
    assert not flat, f"created unsharded despite a sharded store: {flat}"
    assert any(n.startswith("vertex_attributes/") for n in names)


def test_write_points_shard_shape_declares_it_on_the_store(tmp_path):
    """Otherwise only the arrays this call made would be sharded."""
    path = str(tmp_path / "s.zarrvectors")
    write_points(
        path, np.array([[1.0, 1.0, 1.0]], dtype="float32"),
        bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2,
    )
    assert read_root_metadata(open_store(path, mode="r")).shard_shape == 2


def test_a_second_write_inherits_rather_than_unsharding(tmp_path):
    """``shard_shape`` unsaid must not read as ``shard_shape=None``."""
    path = str(tmp_path / "s.zarrvectors")
    rng = np.random.default_rng(0)
    write_points(
        path, rng.uniform(0, 64, (20, 3)).astype("float32"),
        bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_points(
            path, rng.uniform(0, 64, (5, 3)).astype("float32"),
            bounds=_BOUNDS, chunk_shape=_CHUNK,
        )

    lg = get_resolution_level(open_store(path, mode="r"), 0)
    assert array_is_sharded(lg, "vertices")
    # Inherited, so there is no layout disagreement to warn about.
    assert not [
        w for w in caught if "different layout" in str(w.message)
    ]


def test_explicit_none_still_means_unsharded(tmp_path):
    """The sentinel exists so "did not say" and "said no" differ."""
    path = tmp_path / "s.zarrvectors"
    _store(path)
    lg = get_resolution_level(open_store(str(path), mode="r+"), 0)

    from zarr_vectors.core.arrays import create_vertices_array, open_write_session

    with open_write_session(
        lg, shard_shape=None, bounds=[list(_BOUNDS[0]), list(_BOUNDS[1])],
        chunk_shape=_CHUNK,
    ):
        create_vertices_array(lg, dtype="float32")

    assert not array_is_sharded(lg, "vertices")


# --- units: cells, not coordinates ------------------------------------


def test_a_coarser_level_inherits_cells_per_shard(tmp_path):
    """Grids differ between levels; the declaration is grid-independent."""
    from zarr_vectors.core.metadata import LevelMetadata
    from zarr_vectors.core.store import create_resolution_level

    path = tmp_path / "s.zarrvectors"
    root = _store(path, shard_shape=2)
    create_resolution_level(
        root, 1,
        LevelMetadata(
            level=1, vertex_count=0, arrays_present=[],
            chunk_shape=(32, 32, 32), bin_shape=(32, 32, 32),
            bin_ratio=(2, 2, 2), parent_level=0,
        ),
    )
    lg1 = get_resolution_level(root, 1)
    create_attribute_array(lg1, "intensity", dtype="float32")

    arr = lg1._sharded_chunk_array("vertex_attributes/intensity")
    assert arr.shards == (2, 2, 2)
    # The coarser grid is smaller than level 0's, which is fine: a shard
    # larger than the array is legal and must not be clipped.
    assert arr.shape <= get_resolution_level(root, 0)._sharded_chunk_array(
        "vertices"
    ).shape


# --- refusal ----------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, [2, 4], [2, 0, 4], True])
def test_a_bad_shard_shape_is_refused_at_declaration(tmp_path, bad):
    """At the call that declared it, not inside a later worker."""
    with pytest.raises(MetadataError, match="shard_shape"):
        _store(tmp_path / "s.zarrvectors", shard_shape=bad)


def test_read_points_round_trips_through_a_sharded_store(tmp_path):
    path = str(tmp_path / "s.zarrvectors")
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 64, (40, 3)).astype("float32")
    write_points(
        path, pts, bounds=_BOUNDS, chunk_shape=_CHUNK, shard_shape=2,
    )
    assert len(read_points(path)["positions"]) == 40
