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


# ---------------------------------------------------------------------------
# Local-filesystem direct read path (_direct_spec / _direct_read)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compressor", [None, "zstd", "blosc"])
def test_direct_and_gather_paths_agree(tmp_path, compressor):
    root = create_store(str(tmp_path / "store"), ndim=3)
    name = _cell_array(root, "cells")
    payloads = {
        "0.0.0": b"",
        "1.2.3": bytes(range(256)) * 7,
        "2.3.4": b"a single short cell",
    }
    with root.chunk_array_codecs(compressor):
        for key, blob in payloads.items():
            root.write_bytes(name, key, blob)
    # A cell nobody wrote, and a coord off the grid: both paths have to
    # agree on those too, not just on the ones with bytes behind them.
    probes = [*sorted(payloads), "0.0.1", "9.9.9"]

    plan = [(name, probes)]
    with root.batched_reads(plan):
        direct = {k: root.read_bytes(name, k) for k in probes[:-1]}

    import zarr_vectors.core._batch_reader as batch_reader

    real = batch_reader._direct_spec
    batch_reader._direct_spec = lambda *a, **k: None
    try:
        with root.batched_reads(plan):
            gathered = {k: root.read_bytes(name, k) for k in probes[:-1]}
    finally:
        batch_reader._direct_spec = real

    assert direct == gathered
    for key, blob in payloads.items():
        assert direct[key] == blob
    assert direct["0.0.1"] == b""


def test_direct_path_declines_a_memory_store(tmp_path):
    """Only a LocalStore has files to open; anything else must fall
    through rather than guess at a path."""
    import zarr_vectors.core._batch_reader as batch_reader

    root = create_store(MemoryStore(), ndim=3)
    name = _cell_array(root, "cells")
    root.write_bytes(name, "0.0.0", b"payload")

    assert batch_reader._direct_spec(root._zarr, name) is None
    with root.batched_reads([(name, ["0.0.0"])]):
        assert root.read_bytes(name, "0.0.0") == b"payload"


def test_direct_path_declines_a_missing_array(tmp_path):
    import zarr_vectors.core._batch_reader as batch_reader

    root = create_store(str(tmp_path / "store"), ndim=3)
    assert batch_reader._direct_spec(root._zarr, "nope") is None


# ---------------------------------------------------------------------------
# Group.cached_nodes
# ---------------------------------------------------------------------------


def test_cached_nodes_resolves_each_node_once(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    name = _cell_array(root, "cells")
    root.write_bytes(name, "0.0.0", b"payload")

    calls = []
    real = type(root._zarr).__getitem__

    def counting(self, key):
        calls.append(key)
        return real(self, key)

    type(root._zarr).__getitem__ = counting
    try:
        with root.cached_nodes():
            for _ in range(5):
                root.read_array_meta(name)
                root.list_chunks(name)
        inside = list(calls)

        calls.clear()
        for _ in range(5):
            root.read_array_meta(name)
            root.list_chunks(name)
        outside = list(calls)
    finally:
        type(root._zarr).__getitem__ = real

    assert len(inside) == 1, inside
    assert len(outside) == 10, outside


def test_cached_nodes_shares_with_derived_groups(tmp_path):
    """A reader opens the level group inside the block; it must hit the
    same cache the root does, or the saving stops at the root."""
    root = create_store(str(tmp_path / "store"), ndim=3)
    level = root.require_group("0")
    level.create_sharded_chunk_array("cells", (2, 2, 2))

    with root.cached_nodes():
        derived = root["0"]
        assert derived._node_cache is root._node_cache
        assert derived._listing_cache is root._listing_cache
    assert root._node_cache is None
    assert root["0"]._node_cache is None


def test_cached_nodes_nesting_is_a_noop(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    with root.cached_nodes():
        outer = root._node_cache
        with root.cached_nodes():
            assert root._node_cache is outer
        assert root._node_cache is outer
    assert root._node_cache is None


def test_cached_listing_survives_only_the_block(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    name = _cell_array(root, "cells")
    root.write_bytes(name, "0.0.0", b"a")

    with root.cached_nodes():
        assert root.list_chunk_coords(name) == [(0, 0, 0)]
        assert root._listing_cache
    assert root._listing_cache is None
    # Outside the block a later write is visible immediately.
    root.write_bytes(name, "1.1.1", b"b")
    assert root.list_chunk_coords(name) == [(0, 0, 0), (1, 1, 1)]


# ---------------------------------------------------------------------------
# Group.prime_nodes
# ---------------------------------------------------------------------------


def _count_lookups(root, fn):
    """Run ``fn``; return the zarr node lookups it made."""
    calls = []
    real = type(root._zarr).__getitem__

    def counting(self, key):
        calls.append(key)
        return real(self, key)

    type(root._zarr).__getitem__ = counting
    try:
        fn()
    finally:
        type(root._zarr).__getitem__ = real
    return calls


def test_prime_nodes_fills_the_cache_in_one_pass(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    level = root.require_group("0")
    level.create_sharded_chunk_array("vertices", (2, 2, 2))
    level.create_sharded_chunk_array("vertex_fragments", (2, 2, 2))

    with root.cached_nodes():
        root.prime_nodes(["0", "0/vertices", "0/vertex_fragments"])
        assert sorted(root._node_cache) == [
            "0", "0/vertex_fragments", "0/vertices",
        ]
        # Everything the reader goes on to ask for is already answered,
        # including from the derived level handle.
        after = _count_lookups(root, lambda: (
            root["0"].read_array_meta("vertices"),
            root["0"].read_array_meta("vertex_fragments"),
        ))
        assert after == []


def test_prime_nodes_records_an_absent_path_as_absent(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    with root.cached_nodes():
        root.prime_nodes(["nope"])
        assert "nope" in root._node_cache
        # And answering "no" from the cache costs no lookup.
        assert _count_lookups(root, lambda: root.array_exists("nope")) == []
        assert root.array_exists("nope") is False


def test_prime_nodes_is_a_noop_without_a_cache_block(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    root.require_group("0")
    root.prime_nodes(["0"])          # must not raise, must not cache
    assert root._node_cache is None


def test_prime_nodes_skips_paths_already_cached(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    root.require_group("0")
    with root.cached_nodes():
        root.prime_nodes(["0"])
        cached = root._node_cache["0"]
        root.prime_nodes(["0"])
        assert root._node_cache["0"] is cached


def test_prime_nodes_failure_does_not_fail_the_read(tmp_path, monkeypatch):
    """Priming is an optimisation; if the gather blows up the reader must
    still get its data by resolving nodes the ordinary way."""
    root = create_store(str(tmp_path / "store"), ndim=3)
    name = _cell_array(root, "cells")
    root.write_bytes(name, "0.0.0", b"payload")

    import zarr_vectors.core.aio as aio

    def boom(*a, **k):
        raise RuntimeError("gather failed")

    monkeypatch.setattr(aio, "_resolve_nodes", boom)
    with root.cached_nodes():
        root.prime_nodes([name])
        assert root.read_bytes(name, "0.0.0") == b"payload"


def test_read_points_resolves_no_nodes_inside_a_warm_block(tmp_path):
    """The pattern a viewer uses: hold one block open across many picks.
    After the first read every node is cached, so a pick is chunk I/O."""
    store = str(tmp_path / "pts")
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (2000, 3)).astype(np.float32)
    write_points(store, pos, chunk_shape=(50.0, 50.0, 50.0),
                 bounds=([0.0] * 3, [100.0] * 3))

    root = open_store(store)
    with root.cached_nodes():
        read_points(root, bbox=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)))
        later = _count_lookups(root, lambda: [
            read_points(root, bbox=((float(x), 0.0, 0.0),
                                    (float(x) + 5.0, 10.0, 10.0)))
            for x in range(0, 50, 10)
        ])
    assert later == [], later


# ---------------------------------------------------------------------------
# Every reader, not just read_points, resolves its nodes once per session
# ---------------------------------------------------------------------------


def _warm_block_costs_no_lookups(store, read, **kw):
    """Read twice inside one block; the second must resolve nothing."""
    root = open_store(store)
    with root.cached_nodes():
        read(root, **kw)
        return _count_lookups(root, lambda: read(root, **kw))


def test_read_polylines_resolves_no_nodes_inside_a_warm_block(tmp_path):
    from zarr_vectors.types.polylines import read_polylines, write_polylines

    rng = np.random.default_rng(0)
    lines = [rng.uniform(0, 100, (10, 3)).astype(np.float32) for _ in range(30)]
    store = str(tmp_path / "pl.zv")
    write_polylines(store, lines, chunk_shape=(50.0, 50.0, 50.0))
    assert _warm_block_costs_no_lookups(
        store, read_polylines, bbox=([0.0] * 3, [40.0] * 3),
    ) == []


def test_read_mesh_resolves_no_nodes_inside_a_warm_block(tmp_path):
    from zarr_vectors.types.meshes import read_mesh, write_mesh

    rng = np.random.default_rng(1)
    verts = rng.uniform(0, 100, (300, 3)).astype(np.float32)
    faces = rng.integers(0, 300, (200, 3)).astype(np.int64)
    store = str(tmp_path / "m.zv")
    write_mesh(store, verts, faces, chunk_shape=(50.0, 50.0, 50.0))
    assert _warm_block_costs_no_lookups(
        store, read_mesh, bbox=([0.0] * 3, [40.0] * 3),
    ) == []


def test_read_graph_resolves_no_nodes_inside_a_warm_block(tmp_path):
    from zarr_vectors.types.graphs import read_graph, write_graph

    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 100, (200, 3)).astype(np.float32)
    edges = rng.integers(0, 200, (300, 2)).astype(np.int64)
    store = str(tmp_path / "g.zv")
    write_graph(store, pos, edges, chunk_shape=(50.0, 50.0, 50.0))
    assert _warm_block_costs_no_lookups(
        store, read_graph, bbox=([0.0] * 3, [40.0] * 3),
    ) == []


def test_read_lines_resolves_no_nodes_inside_a_warm_block(tmp_path):
    from zarr_vectors.types.lines import read_lines, write_lines

    rng = np.random.default_rng(3)
    endpoints = rng.uniform(0, 100, (80, 2, 3)).astype(np.float32)
    store = str(tmp_path / "l.zv")
    write_lines(store, endpoints, chunk_shape=(50.0, 50.0, 50.0))
    assert _warm_block_costs_no_lookups(
        store, read_lines, bbox=([0.0] * 3, [40.0] * 3),
    ) == []


def test_read_lines_bbox_keeps_a_line_spanning_untouched_chunks(tmp_path):
    """``read_lines`` keeps a line if EITHER endpoint is in the box.

    So its chunk set must not be narrowed to the box: a line stored in a
    chunk the box never touches can still have its other endpoint inside,
    and narrowing would silently drop it.
    """
    from zarr_vectors.types.lines import read_lines, write_lines

    endpoints = np.array([
        [[10.0, 10.0, 10.0], [180.0, 180.0, 180.0]],   # one end in the box
        [[170.0, 170.0, 170.0], [190.0, 190.0, 190.0]],  # neither end
    ], dtype=np.float32)
    store = str(tmp_path / "span.zv")
    write_lines(store, endpoints, chunk_shape=(50.0, 50.0, 50.0))

    out = read_lines(store, bbox=([0.0] * 3, [50.0] * 3))
    assert out["line_count"] == 1
    np.testing.assert_allclose(out["endpoints"][0], endpoints[0])
