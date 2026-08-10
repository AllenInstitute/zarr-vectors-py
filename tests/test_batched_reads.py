"""Tests for ``Group.batched_reads`` and the ``_batch_reader`` helper.

Covers:

* Empty plan (no-op).
* Round-trip: bytes written via the normal path read back identically
  through a ``batched_reads`` block.
* End-to-end: ``read_points`` against a store written by ``write_points``
  (the read path internally wraps the chunk loop in ``batched_reads``).
* Cache-miss safety: a read for a key NOT in the plan falls through to
  the sync ``read_bytes`` path.
* Nesting is rejected with ``StoreError``.
* Icechunk-style fallback (monkeypatched detector) round-trips correctly
  via the serial sync path.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr.storage import MemoryStore

from zarr_vectors import open_store
from zarr_vectors.core.store import create_store
from zarr_vectors.exceptions import StoreError
from zarr_vectors.types.points import read_points, write_points


def _cell_array(root, name, grid_shape=(3, 4, 5)):
    """Allocate a per-chunk vlen array to read cells from.

    ``write_bytes`` no longer conjures a per-cell group for an
    unallocated name — every per-chunk array is one grid-shaped vlen
    array — so keys are coord tuples within ``grid_shape``.
    """
    root.create_sharded_chunk_array(name, grid_shape)
    return name


def test_batched_reads_empty_plan_is_noop(tmp_store_path):
    root = create_store(str(tmp_store_path))
    with root.batched_reads([]):
        pass


def test_batched_reads_round_trip(tmp_store_path):
    """Bytes written via the sync path read back identically through a
    batched_reads block (cache hit)."""
    root = create_store(str(tmp_store_path))
    _cell_array(root, "read_test")
    payloads = {
        "0.0.0": b"first chunk bytes",
        "1.0.0": b"\x00" * 32,
        "2.3.4": np.arange(100, dtype=np.uint8).tobytes(),
    }
    for k, v in payloads.items():
        root.write_bytes("read_test", k, v)

    plan = [("read_test", list(payloads.keys()))]
    with root.batched_reads(plan):
        for k, v in payloads.items():
            assert root.read_bytes("read_test", k) == v


def test_batched_reads_cache_miss_falls_through(tmp_store_path):
    """A read for an (array, key) not in the plan still returns correct
    data — the cache miss drops through to the sync path."""
    root = create_store(str(tmp_store_path))
    _cell_array(root, "planned_arr")
    _cell_array(root, "unplanned_arr")
    root.write_bytes("planned_arr", "0.0.0", b"planned-data")
    root.write_bytes("unplanned_arr", "0.0.0", b"unplanned-data")

    plan = [("planned_arr", ["0.0.0"])]
    with root.batched_reads(plan):
        # Hits the cache.
        assert root.read_bytes("planned_arr", "0.0.0") == b"planned-data"
        # Cache miss: falls back to sync read.
        assert root.read_bytes("unplanned_arr", "0.0.0") == b"unplanned-data"


def test_batched_reads_nesting_rejected(tmp_store_path):
    root = create_store(str(tmp_store_path))
    with root.batched_reads([]):
        with pytest.raises(StoreError, match="does not support nesting"):
            with root.batched_reads([]):
                pass


def test_batched_reads_via_read_points_memory_store():
    """End-to-end: read_points against a MemoryStore exercises the
    batched_reads path internally (read_points wraps its chunk loop)
    and must produce identical results to the unbatched sync path."""
    mem = MemoryStore()
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, (300, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, 300).astype(np.float32)
    write_points(mem, positions, vertex_attributes={"intensity": intensity})

    result = read_points(mem, attribute_names=["intensity"])
    assert result["vertex_count"] == 300
    assert result["positions"].shape == (300, 3)
    assert result["vertex_attributes"]["intensity"].shape == (300,)


def test_batched_reads_via_read_points_localstore(tmp_path):
    """Same end-to-end against a LocalStore — covers the path that the
    benchmark notebooks (zarr-vectors-tools) exercise."""
    url = str(tmp_path / "batch_read_points.zarr")
    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 1000, (1000, 3)).astype(np.float32)
    score = positions[:, 0].astype(np.float32).copy()
    write_points(
        url, positions,
        chunk_shape=(100., 100., 100.),
        vertex_attributes={"score": score},
    )
    out = read_points(url, attribute_names=["score"])
    assert out["vertex_count"] == 1000
    assert out["positions"].shape == (1000, 3)
    assert out["vertex_attributes"]["score"].shape == (1000,)


def test_batched_reads_falls_back_to_sync_for_icechunk_like_store(
    tmp_store_path, monkeypatch,
):
    """Stores that look like icechunk take the sync fallback inside
    ``flush_prefetch``.  Force the detector to return True and verify
    the round-trip still works."""
    from zarr_vectors.core import _batch_reader

    monkeypatch.setattr(_batch_reader, "_is_icechunk_store", lambda _store: True)

    root = create_store(str(tmp_store_path))
    _cell_array(root, "fallback_arr")
    payloads = {
        "0.0.0": b"icechunk-fallback-data",
        "1.0.0": np.arange(32, dtype=np.uint8).tobytes(),
    }
    for k, v in payloads.items():
        root.write_bytes("fallback_arr", k, v)

    plan = [("fallback_arr", list(payloads.keys()))]
    with root.batched_reads(plan):
        for k, v in payloads.items():
            assert root.read_bytes("fallback_arr", k) == v


def test_batched_reads_clears_cache_on_exception(tmp_store_path):
    """If the block raises, the cache is dropped so subsequent reads
    don't accidentally serve stale data."""
    root = create_store(str(tmp_store_path))
    _cell_array(root, "err_arr")
    root.write_bytes("err_arr", "0.0.0", b"data")
    plan = [("err_arr", ["0.0.0"])]
    with pytest.raises(RuntimeError, match="boom"):
        with root.batched_reads(plan):
            raise RuntimeError("boom")
    # Cache cleared, sync path still works.
    assert root._prefetch_cache is None
    assert root.read_bytes("err_arr", "0.0.0") == b"data"


def test_batched_reads_missing_chunk_omitted_from_cache(tmp_store_path):
    """A plan entry for an unwritten in-grid cell is skipped in the cache,
    and the batched read agrees with the unbatched one.

    Note the semantics changed with the single-array layout: an
    allocated-but-never-written cell reads back ``b""`` — the vlen fill
    value — rather than raising.  It used to raise because each cell was
    its own sub-array, so absence was a missing node.  Now absence and an
    explicitly-written empty payload are byte-identical; ``chunk_exists``
    (the presence manifest) is what separates them.

    What matters here is that batching does not change the answer.
    """
    root = create_store(str(tmp_store_path))
    _cell_array(root, "partial_arr")
    root.write_bytes("partial_arr", "0.0.0", b"present")

    unbatched = root.read_bytes("partial_arr", "2.3.4")
    assert unbatched == b""
    assert root.chunk_exists("partial_arr", "0.0.0") is True
    assert root.chunk_exists("partial_arr", "2.3.4") is False

    plan = [("partial_arr", ["0.0.0", "2.3.4"])]
    with root.batched_reads(plan):
        assert root.read_bytes("partial_arr", "0.0.0") == b"present"
        # The batched path must agree with the sync path exactly.
        assert root.read_bytes("partial_arr", "2.3.4") == unbatched


def test_read_bytes_raises_for_out_of_grid_coords(tmp_store_path):
    """Out-of-grid coords still raise — that is a real error, not absence.

    Keeps the distinction the test above relies on: `b""` means "in the
    grid, nothing written"; StoreError means "not addressable at all".
    """
    root = create_store(str(tmp_store_path))
    _cell_array(root, "bounded_arr", grid_shape=(2, 2, 2))
    with pytest.raises(StoreError):
        root.read_bytes("bounded_arr", "9.9.9")
