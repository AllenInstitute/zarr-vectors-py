"""Store creation and metadata bookkeeping regressions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from zarr_vectors.exceptions import ArrayError, DracoError


# --- pre-built Zarr stores keep the writer's create kwargs --------------

def test_prebuilt_store_gets_writer_bounds():
    from zarr.storage import MemoryStore

    from zarr_vectors.core.store import open_store, read_root_metadata
    from zarr_vectors.types.points import read_points, write_points

    pts = np.random.default_rng(0).random((50, 3)).astype("f4") * 500
    store = MemoryStore()
    write_points(store, pts, chunk_shape=(100.0, 100.0, 100.0))
    meta = read_root_metadata(open_store(store))
    assert meta.chunk_shape == (100.0, 100.0, 100.0)
    lo, hi = meta.bounds
    assert np.all(np.asarray(lo) <= pts.min(0))
    assert np.all(np.asarray(hi) >= pts.max(0))
    assert read_points(store)["positions"].shape == (50, 3)


# --- Draco: honest encoding stamp, no half-written store -----------------

def _vertices_attrs(path):
    with open(path / "0" / "vertices" / "zarr.json") as fh:
        return json.load(fh)["attributes"]


def test_draco_on_2d_mesh_is_stamped_raw(tmp_path):
    from zarr_vectors.types.meshes import read_mesh, write_mesh

    path = tmp_path / "m2d.zv"
    v = np.random.default_rng(1).random((20, 2)).astype("f4") * 100
    f = np.array([[0, 1, 2], [3, 4, 5]])
    write_mesh(path, v, f, chunk_shape=(50.0, 50.0), encoding="draco")
    assert _vertices_attrs(path)["encoding"] == "raw"
    out = read_mesh(path, object_ids=[0])  # refused on a Draco store
    assert out["vertices"].shape == (20, 2)


def test_failed_draco_write_creates_nothing(tmp_path, monkeypatch):
    import zarr_vectors.encoding.draco as draco
    from zarr_vectors.types.meshes import write_mesh

    monkeypatch.setattr(draco, "draco_backend_available", lambda: False)
    path = tmp_path / "d.zv"
    v = np.random.default_rng(2).random((6, 3)).astype("f4") * 10
    with pytest.raises(DracoError):
        write_mesh(path, v, np.array([[0, 1, 2]]), chunk_shape=(50.0,) * 3,
                   encoding="draco")
    assert not path.exists()


def test_draco_quads_rejected_before_creation(tmp_path):
    from zarr_vectors.types.meshes import write_mesh

    path = tmp_path / "q.zv"
    v = np.random.default_rng(3).random((8, 3)).astype("f4") * 10
    with pytest.raises(ArrayError, match="triangles"):
        write_mesh(path, v, np.array([[0, 1, 2, 3]]), chunk_shape=(50.0,) * 3,
                   encoding="draco")
    assert not path.exists()
