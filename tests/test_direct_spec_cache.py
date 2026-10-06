"""A direct-read spec is resolved from metadata alone, never from presence.

``batched_reads`` asks for the spec of every array in its plan. Inside
``cached_nodes`` that spec used to be kept on the array's presence
listing, and building the listing on a deferred level derives presence
from the store -- for a sharded array, by reading every shard. A task
prefetching its own few cells therefore read every shard written so far
(BRIDGE's 50 um graph stage: 9 h). The spec now has a cache of its own,
with the same lifetime and invalidation as the node cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from zarr_vectors.building import defer_presence, get_resolution_level, write_chunk_vertices
from zarr_vectors.core import _batch_reader
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import create_store, open_store

CELLS = [(0, 0, 0), (1, 0, 0), (2, 3, 1), (4, 4, 4)]
ARRAYS = ("vertices", "vertex_fragments")


def _key(cell):
    return ".".join(str(c) for c in cell)


def _deferred_level(path, **kw):
    create_store(
        str(path),
        bounds=([0.0, 0.0, 0.0], [500.0, 500.0, 500.0]),
        chunk_shape=(100.0, 100.0, 100.0),
        geometry_types=["point_cloud"],
        ndim=3,
        **kw,
    )
    level = get_resolution_level(open_store(str(path), mode="r+"), 0)
    defer_presence(level)
    for cell in CELLS:
        pts = np.full((3, 3), 50.0, dtype=np.float32) + np.asarray(cell) * 100.0
        write_chunk_vertices(level, cell, [pts])
    # A fresh handle, as a worker opening the level would have.
    return get_resolution_level(open_store(str(path), mode="r+"), 0)


@pytest.fixture(params=[None, 2], ids=["unsharded", "sharded"])
def level(request, tmp_path):
    kw = {} if request.param is None else {"shard_shape": request.param}
    return _deferred_level(tmp_path / "s.zv", **kw)


@pytest.fixture
def no_presence(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("resolving an array derived its presence")

    monkeypatch.setattr(Group, "_presence_from_store", refuse)


def _expected(level, keys):
    return {(a, k): level.read_bytes(a, k) for a in ARRAYS for k in keys}


def test_a_prefetch_in_cached_nodes_derives_no_presence(level, no_presence):
    keys = [_key(CELLS[0]), _key(CELLS[2])]
    want = _expected(level, keys)
    assert level.presence_deferred()
    with level.cached_nodes():
        with level.batched_reads([(a, keys) for a in ARRAYS]):
            got = {(a, k): level.read_bytes(a, k) for a in ARRAYS for k in keys}
        # A second prefetch in the same session is served the cached spec.
        with level.batched_reads([(a, keys) for a in ARRAYS]):
            again = {(a, k): level.read_bytes(a, k) for a in ARRAYS for k in keys}
    assert got == want and again == want
    assert all(got.values())


def test_the_spec_is_resolved_once_per_session(level, no_presence, monkeypatch):
    calls: list[str] = []
    real = _batch_reader._direct_spec

    def counting(group, name, resolved=None):
        calls.append(name)
        return real(group, name, resolved)

    monkeypatch.setattr(_batch_reader, "_direct_spec", counting)
    keys = [_key(CELLS[1])]
    with level.cached_nodes():
        for _ in range(3):
            with level.batched_reads([(a, keys) for a in ARRAYS]):
                pass
        assert sorted(calls) == sorted(ARRAYS)
        # Invalidating a node drops its spec with it.
        level._invalidate_node("vertices")
        with level.batched_reads([(a, keys) for a in ARRAYS]):
            pass
        assert sorted(calls) == sorted([*ARRAYS, "vertices"])
    assert level._spec_cache is None

    # Outside a session the spec is derived per call, as before.
    calls.clear()
    with level.batched_reads([("vertices", keys)]):
        pass
    with level.batched_reads([("vertices", keys)]):
        pass
    assert calls == ["vertices", "vertices"]


def test_derived_groups_share_the_spec_cache(tmp_path):
    root = create_store(str(tmp_path / "store"), ndim=3)
    root.require_group("0").create_sharded_chunk_array("cells", (2, 2, 2))
    with root.cached_nodes():
        derived = root["0"]
        assert derived._spec_cache is root._spec_cache is not None
        derived._direct_spec_cached("cells")
        assert "0/cells" in root._spec_cache
    assert root._spec_cache is None


def test_listing_still_derives_presence_on_a_deferred_level(level):
    # The fix moves the spec, not presence: a listing asked for is still
    # the store's answer.
    with level.cached_nodes():
        assert level.list_chunks("vertices") == sorted(_key(c) for c in CELLS)
