"""Regression tests for the 0.9.0 single-array per-chunk layout.

Every per-spatial-chunk array (``vertices``, ``vertex_fragments``,
``links/<delta>``, attribute arrays) is ONE Zarr v3 vlen-bytes array
whose cells are spatial chunks — chunk files at ``<array>/c/i/j/k`` —
instead of the old "Option G" group of single-chunk sub-arrays.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zarr_vectors.constants import VERTICES
from zarr_vectors.core.arrays import list_chunk_keys
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.points import read_points, write_points


def _store(prefix: str) -> str:
    return str(Path(tempfile.mkdtemp(prefix=f"sal_{prefix}_")) / "s.zarrvectors")


def _vmeta(store: str) -> dict:
    return json.loads(
        (Path(store) / "0" / "vertices" / "zarr.json").read_text()
    )


def test_vertices_is_single_array_not_per_chunk_subgroups() -> None:
    """The core ask: ``vertices/`` is one array (``zarr.json`` + ``c/``),
    not a group with one sub-array per chunk key."""
    store = _store("layout")
    pts = np.random.RandomState(0).rand(300, 3).astype("float32") * 250.0
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))

    vdir = Path(store) / "0" / "vertices"
    assert (vdir / "zarr.json").is_file()
    assert (vdir / "c").is_dir()

    meta = _vmeta(store)
    assert meta["node_type"] == "array"
    assert meta["data_type"] == "variable_length_bytes"
    # No dotted per-chunk sub-array directories (Option-G) remain.
    subdirs = [
        d for d in os.listdir(vdir)
        if (vdir / d).is_dir() and d != "c"
    ]
    assert subdirs == [], f"unexpected per-chunk sub-dirs: {subdirs}"

    # A real chunk file exists at the nested c/i/j/k path.
    cfiles = list((vdir / "c").rglob("*"))
    assert any(f.is_file() for f in cfiles)


def test_nonempty_chunks_matches_list_chunk_keys() -> None:
    store = _store("manifest")
    pts = np.random.RandomState(1).rand(400, 3).astype("float32") * 300.0
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))

    meta = _vmeta(store)
    manifest = set(meta["attributes"]["nonempty_chunks"])

    lg = get_resolution_level(open_store(store), 0)
    listed = {".".join(map(str, c)) for c in list_chunk_keys(lg, VERTICES)}
    assert manifest == listed and manifest


def test_round_trip_matches_input() -> None:
    store = _store("rt")
    pts = np.random.RandomState(2).rand(500, 3).astype("float32") * 400.0
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))
    out = np.asarray(read_points(store)["positions"])
    a = pts[np.lexsort(pts.T)]
    b = out[np.lexsort(out.T)]
    assert np.allclose(a, b, atol=1e-4)


def test_negative_coordinates_round_trip() -> None:
    """Positions below the origin produce negative chunk coords; the
    grid origin offset must keep them addressable and round-tripping."""
    store = _store("neg")
    rs = np.random.RandomState(3)
    pts = (rs.rand(400, 3).astype("float32") * 400.0) - 150.0  # spans <0
    assert pts.min() < 0
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))

    meta = _vmeta(store)
    origin = meta["attributes"].get("chunk_grid_origin")
    assert origin is not None and min(origin) < 0

    out = np.asarray(read_points(store)["positions"])
    a = pts[np.lexsort(pts.T)]
    b = out[np.lexsort(out.T)]
    assert np.allclose(a, b, atol=1e-4)


def test_chunk_by_attribute_adds_leading_grid_axis() -> None:
    store = _store("attr")
    rs = np.random.RandomState(4)
    pts = rs.rand(300, 3).astype("float32") * 200.0
    labels = rs.randint(0, 3, size=300)
    write_points(
        store, pts, vertex_attributes={"lab": labels},
        chunk_by_attribute="lab", chunk_shape=(100.0, 100.0, 100.0),
    )
    meta = _vmeta(store)
    # Rank = 1 (attr-bin axis) + 3 spatial; leading axis spans the bins.
    assert len(meta["shape"]) == 4
    assert meta["shape"][0] == 3
    out = np.asarray(read_points(store)["positions"])
    assert out.shape[0] == 300


def test_list_chunk_keys_numeric_order_stable() -> None:
    """Enumeration order is numeric (``2 < 10``), not lexicographic, so
    downstream flat-index reconstruction stays stable."""
    store = _store("order")
    # Wide extent so chunk coords reach double digits along x.
    rs = np.random.RandomState(5)
    pts = rs.rand(2000, 3).astype("float32")
    pts[:, 0] *= 1500.0  # x in [0, 1500) -> chunk coords 0..14
    pts[:, 1:] *= 100.0
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))

    lg = get_resolution_level(open_store(store), 0)
    keys = list_chunk_keys(lg, VERTICES)
    xs = [k[0] for k in keys]
    assert xs == sorted(xs), "chunk keys must be in numeric order"
    assert max(xs) >= 10, "test needs a double-digit chunk coord"


@pytest.mark.parametrize("shard_shape", [None, 2])
def test_sharding_opt_in(shard_shape) -> None:
    store = _store(f"shard_{shard_shape}")
    pts = np.random.RandomState(6).rand(500, 3).astype("float32") * 300.0
    write_points(
        store, pts, chunk_shape=(100.0, 100.0, 100.0), shard_shape=shard_shape,
    )
    meta = _vmeta(store)
    names = [c["name"] for c in meta["codecs"]]
    if shard_shape is None:
        assert "sharding_indexed" not in names
        assert "vlen-bytes" in names
    else:
        assert "sharding_indexed" in names
    # Round-trips either way.
    out = np.asarray(read_points(store)["positions"])
    assert out.shape[0] == 500
