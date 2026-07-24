"""Tests for ``Group.batched_writes`` and the ``_batch_writer`` helper.

Covers:

* Empty context (no-op flush).
* Single-flush round-trip — write a handful of bytes blobs in batched
  mode and confirm they read back via the normal ``read_bytes`` path.

Cell writes go through an explicitly allocated chunk array with
coord-tuple keys.  ``write_bytes`` no longer conjures a per-cell group
for an unallocated name — every per-chunk array is a single grid-shaped
vlen array, and the grid has to come from somewhere.
* ``write_points`` end-to-end against a ``MemoryStore`` while batching
  is active.
* Nesting is rejected with ``StoreError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr.storage import MemoryStore

from zarr_vectors import open_store
from zarr_vectors.core.store import create_store
from zarr_vectors.exceptions import StoreError
from zarr_vectors.types.points import read_points, write_points


def test_batched_writes_empty_block_is_noop(tmp_store_path):
    root = create_store(str(tmp_store_path))
    # Just open/close — should not raise, should not write anything new.
    with root.batched_writes():
        pass
    # Sanity: re-open and confirm root attrs are intact.
    re = open_store(str(tmp_store_path), mode="r")
    assert "zarr_vectors" in re.attrs.to_dict()


def _cell_array(root, name="batch_test", grid_shape=(3, 4, 5)):
    """Allocate a per-chunk vlen array to write cells into.

    ``write_bytes`` no longer conjures a per-cell group for an
    unallocated name — every per-chunk array is one grid-shaped vlen
    array, and the grid has to come from somewhere.  Keys are therefore
    coord tuples within ``grid_shape``.
    """
    root.create_sharded_chunk_array(name, grid_shape)
    return name


def test_batched_writes_round_trip(tmp_store_path):
    """Bytes written inside a batched_writes block read back identically."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    payloads = {
        "0.0.0": b"first chunk bytes",
        "1.0.0": b"\x00" * 32,
        "2.3.4": np.arange(100, dtype=np.uint8).tobytes(),
    }
    with root.batched_writes():
        for k, v in payloads.items():
            root.write_bytes(name, k, v)

    # Read back through the normal sync path.
    for k, v in payloads.items():
        assert root.read_bytes(name, k) == v


def test_batched_writes_empty_payload_round_trips(tmp_store_path):
    """An explicitly-written empty cell reads back as empty bytes.

    b"" is also the vlen fill value, so an empty write and an absent cell
    are byte-identical — presence is what ``nonempty_chunks`` tracks, not
    the payload.
    """
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    with root.batched_writes():
        root.write_bytes(name, "0.0.1", b"")
    assert root.read_bytes(name, "0.0.1") == b""


def test_batched_writes_nesting_rejected(tmp_store_path):
    root = create_store(str(tmp_store_path))
    with root.batched_writes():
        with pytest.raises(StoreError, match="does not support nesting"):
            with root.batched_writes():
                pass


def test_batched_writes_via_write_points_memory_store():
    """End-to-end: write_points against a MemoryStore exercises the
    batched path internally (write_points wraps its chunk loop) and
    must produce a readable store."""
    mem = MemoryStore()
    positions = np.random.default_rng(0).uniform(0, 100, (200, 3)).astype(np.float32)
    intensity = np.random.default_rng(1).uniform(0, 1, 200).astype(np.float32)
    write_points(mem, positions, vertex_attributes={"intensity": intensity})

    result = read_points(mem)
    assert result["vertex_count"] == 200
    assert result["positions"].shape == (200, 3)


def test_batched_writes_via_write_points_localstore(tmp_path):
    """Same end-to-end on a local file store, then re-open via URL."""
    url = str(tmp_path / "batch_points.zarr")
    positions = np.random.default_rng(0).uniform(0, 100, (500, 3)).astype(np.float32)
    write_points(url, positions, vertex_attributes={"score": positions[:, 0].copy()})
    # read_points reads attributes only when names are requested explicitly.
    result = read_points(url, attribute_names=["score"])
    assert result["vertex_count"] == 500
    assert "score" in result["vertex_attributes"]
    assert result["vertex_attributes"]["score"].shape == (500,)


def test_batched_writes_defers_write_array_meta(tmp_store_path):
    """``write_array_meta`` inside a batched block is deferred and the
    flushed parent ``zarr.json`` carries the queued attributes."""
    root = create_store(str(tmp_store_path))
    with root.batched_writes():
        root.write_array_meta("custom_arr", {"zv_array": "vertices", "dtype": "float32"})
        # Within the block, the inner attrs dict is queued, not visible yet
        # on the underlying zarr group.
        assert "custom_arr" not in list(root._zarr.group_keys())
    # After exit, the array exists and its attributes match what we queued.
    meta = root.read_array_meta("custom_arr")
    assert meta["zv_array"] == "vertices"
    assert meta["dtype"] == "float32"


def test_batched_write_array_meta_composes_multiple_updates(tmp_store_path):
    """Successive ``write_array_meta`` calls inside one batch should
    merge their attribute dicts (matching the sync ``attrs.update``
    semantics)."""
    root = create_store(str(tmp_store_path))
    with root.batched_writes():
        root.write_array_meta("foo", {"a": 1, "b": 2})
        root.write_array_meta("foo", {"b": 99, "c": 3})  # overrides b
    meta = root.read_array_meta("foo")
    assert meta == {"a": 1, "b": 99, "c": 3}


def test_batched_writes_falls_back_to_sync_for_icechunk_like_store(
    tmp_store_path, monkeypatch
):
    """Stores that look like icechunk (by class name) take the sync
    fallback path: raw ``store.set`` PUTs on ``zarr.json`` skip
    icechunk's array registry, so flush_batch must replay each write
    through ``zarr.Array.create_array`` + ``array[:] = …``.

    We can't install icechunk in every CI matrix slot, so this test
    forces the detection via monkeypatch and verifies the resulting
    round-trip rather than the icechunk-specific commit machinery.
    """
    from zarr_vectors.core import _batch_writer

    monkeypatch.setattr(_batch_writer, "_is_icechunk_store", lambda _store: True)

    root = create_store(str(tmp_store_path))
    root.create_sharded_chunk_array("fallback_arr", (3, 4, 5))
    payloads = {
        "0.0.0": b"chunk-bytes-via-fallback",
        "1.0.0": np.arange(64, dtype=np.uint8).tobytes(),
    }
    with root.batched_writes():
        root.write_array_meta("fallback_arr", {"zv_array": "vertices", "dtype": "float32"})
        for k, v in payloads.items():
            root.write_bytes("fallback_arr", k, v)

    # Round-trip via the normal sync read path.
    assert root.read_array_meta("fallback_arr")["zv_array"] == "vertices"
    for k, v in payloads.items():
        assert root.read_bytes("fallback_arr", k) == v


def test_batched_writes_handles_exception_cleanly(tmp_store_path):
    """If the batch block raises, the queue is cleared and the Group
    stays usable for subsequent writes."""
    root = create_store(str(tmp_store_path))
    name = _cell_array(root)
    with pytest.raises(RuntimeError):
        with root.batched_writes():
            root.write_bytes(name, "0.0.0", b"queued")
            raise RuntimeError("simulated mid-batch failure")
    # Queue must be cleared so the next batched block works.
    assert root._pending_writes is None
    with root.batched_writes():
        root.write_bytes(name, "1.0.0", b"after-recovery")
    assert root.read_bytes(name, "1.0.0") == b"after-recovery"
