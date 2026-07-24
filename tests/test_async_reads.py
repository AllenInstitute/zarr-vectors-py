"""Tests for the async read path (:mod:`zarr_vectors.core.aio`).

The consumer for this path is a browser (Pyodide), where a synchronous
zarr call can only work via WASM stack switching and deadlocks the JS
event loop under concurrency.  So the property under test is not just
"the result is right" but "``sync()`` was never reached" — a read that
returns correct data while still bridging through ``sync()`` would pass
a naive test and hang in production.

Async is driven with a local ``asyncio.run`` rather than pytest-asyncio,
matching ``tests/test_lazy_writer.py``.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import zarr.core.sync as zsync
from zarr.storage import MemoryStore

from zarr_vectors.core.aio import open_store_async, read_async
from zarr_vectors.exceptions import StoreError
from zarr_vectors.types.points import read_points, write_points
from zarr_vectors.types.polylines import read_polylines, write_polylines


def _run(coro):
    return asyncio.run(coro)


class _SyncTrap:
    """Context manager that makes any use of zarr's ``sync()`` bridge fail.

    Monkeypatching the bridge is blunt, but it is the property that
    matters: the async path must never reach it.
    """

    def __init__(self) -> None:
        self.hits = 0

    def __enter__(self) -> _SyncTrap:
        self._real = zsync.sync

        def trap(coro=None, *args, **kwargs):
            self.hits += 1
            # Close the coroutine we are refusing to run, else it is
            # garbage-collected un-awaited and emits a RuntimeWarning
            # that has nothing to do with what is being tested.
            if hasattr(coro, "close"):
                coro.close()
            raise AssertionError("sync() was reached on the async read path")

        zsync.sync = trap
        return self

    def __exit__(self, *exc) -> None:
        zsync.sync = self._real


def _points_store(n=300, seed=0):
    store = MemoryStore()
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 100, (n, 3)).astype("f4")
    write_points(
        store, pos,
        chunk_shape=(50.0, 50.0, 50.0),
        object_ids=np.arange(n, dtype=np.int64),
    )
    return store, pos


def _polyline_store(n=20, seed=1):
    store = MemoryStore()
    rng = np.random.default_rng(seed)
    lines = [rng.uniform(0, 100, (12, 3)).astype("f4") for _ in range(n)]
    write_polylines(store, lines, chunk_shape=(50.0, 50.0, 50.0))
    return store, lines


# ===================================================================
# Equivalence with the sync path
# ===================================================================


def test_async_points_matches_sync():
    store, _ = _points_store()
    expected = read_points(store)

    async def go():
        root = await open_store_async(store)
        return await read_async(read_points, root)

    got = _run(go())
    assert got["vertex_count"] == expected["vertex_count"]
    np.testing.assert_array_equal(
        np.sort(got["positions"], axis=0),
        np.sort(expected["positions"], axis=0),
    )


def test_async_polylines_matches_sync():
    store, lines = _polyline_store()
    expected = read_polylines(store)

    async def go():
        root = await open_store_async(store)
        return await read_async(read_polylines, root)

    got = _run(go())
    assert len(got["polylines"]) == len(expected["polylines"]) == len(lines)


def test_async_respects_reader_kwargs():
    """kwargs reach the reader, and filtering still narrows the result."""
    store, pos = _points_store()
    lo = [0.0, 0.0, 0.0]
    hi = [50.0, 50.0, 50.0]
    expected = read_points(store, bbox=(lo, hi))

    async def go():
        root = await open_store_async(store)
        return await read_async(read_points, root, bbox=(lo, hi))

    got = _run(go())
    assert got["vertex_count"] == expected["vertex_count"]
    assert got["vertex_count"] < len(pos)  # the filter really bit


# ===================================================================
# The actual requirement: no sync() bridge
# ===================================================================


def test_async_points_never_reaches_sync():
    store, _ = _points_store()

    async def go():
        root = await open_store_async(store)
        return await read_async(read_points, root)

    with _SyncTrap() as trap:
        got = _run(go())
    assert trap.hits == 0
    assert got["vertex_count"] == 300


def test_async_polylines_never_reaches_sync():
    """Polylines exercise object manifests and the links families, which
    reach store surfaces ``read_points`` never touches."""
    store, _ = _polyline_store()

    async def go():
        root = await open_store_async(store)
        return await read_async(read_polylines, root)

    with _SyncTrap() as trap:
        got = _run(go())
    assert trap.hits == 0
    assert len(got["polylines"]) == 20


def test_sync_path_still_uses_sync_bridge():
    """Guard against the trap silently testing nothing.

    If the sync reader stopped routing through ``sync()``, the
    no-sync assertions above would pass for the wrong reason.
    """
    store, _ = _points_store(n=50)
    with pytest.raises(AssertionError, match="sync\\(\\) was reached"):
        with _SyncTrap():
            read_points(store)


# ===================================================================
# Concurrency — the failure this path exists to prevent is load-dependent
# ===================================================================


def test_async_concurrent_reads():
    """Many reads in flight at once, which is what deadlocked the fork's
    sync+JSPI path; small stores pass serially and hang under load."""
    stores = [_points_store(n=200, seed=s)[0] for s in range(8)]

    async def go():
        roots = await asyncio.gather(*(open_store_async(s) for s in stores))
        return await asyncio.gather(
            *(read_async(read_points, r) for r in roots)
        )

    with _SyncTrap() as trap:
        results = _run(go())
    assert trap.hits == 0
    assert [r["vertex_count"] for r in results] == [200] * 8


# ===================================================================
# Store injection
# ===================================================================


def test_open_store_async_accepts_prebuilt_store():
    """A pre-built Store is the only way in for a browser host, which
    cannot build one from a URL."""
    store, _ = _points_store(n=100)

    async def go():
        root = await open_store_async(store)
        return await read_async(read_points, root)

    assert _run(go())["vertex_count"] == 100


def test_open_store_async_passes_group_through():
    store, _ = _points_store(n=100)

    async def go():
        root = await open_store_async(store)
        again = await open_store_async(root)
        return root is again

    assert _run(go()) is True


def test_read_async_accepts_raw_source():
    """``read_async`` opens the store itself when handed one."""
    store, _ = _points_store(n=100)

    async def go():
        return await read_async(read_points, store)

    assert _run(go())["vertex_count"] == 100


# ===================================================================
# Offline snapshot contract
# ===================================================================


def test_offline_reads_rejects_nesting():
    from zarr_vectors.core.group import _OfflineSession
    from zarr_vectors.core.store import open_store

    store, _ = _points_store(n=50)
    root = open_store(store)
    session = _OfflineSession()
    with root.offline_reads(session):
        with pytest.raises(StoreError, match="does not support nesting"):
            with root.offline_reads(session):
                pass


def test_offline_gap_raises_rather_than_reading_store():
    """An incomplete snapshot must fail loudly.  Falling back to the
    store would issue exactly the sync read the path exists to avoid —
    correct-looking locally, deadlocking in a browser.
    """
    from zarr_vectors.core.group import _OfflineSession
    from zarr_vectors.core.store import open_store

    store, _ = _points_store(n=50)
    root = open_store(store)
    with root.offline_reads(_OfflineSession()):
        with pytest.raises(StoreError, match="[Oo]ffline read"):
            root.read_bytes("vertices", "0.0.0")
