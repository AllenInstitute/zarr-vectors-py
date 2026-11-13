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
