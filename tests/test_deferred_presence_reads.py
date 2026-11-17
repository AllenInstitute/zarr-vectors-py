"""A level whose presence is deferred reads what it holds through the API.

:func:`~zarr_vectors.building.defer_presence` drops every array's
``nonempty_chunks``, and readers are to ask the store instead.  The
query engine replays a reader offline, where the store cannot be asked,
so ``select().read()`` used to answer an empty level.  A deferred level
is now read by its reader directly, sync and async.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import zarr_vectors as zv
from zarr_vectors.building import defer_presence, rebuild_presence
from zarr_vectors.core.store import get_resolution_level, open_store

BOX = 100.0
QUERIES = [
    {},
    {"bbox": ([10.0, 10.0, 10.0], [60.0, 60.0, 60.0])},
    {"objects": [3]},
    {"objects": [0, 7, 19]},
]


def _store(path, kind, shard):
    rng = np.random.default_rng(1)
    n = 1_500
    pts = rng.uniform(0, BOX - 1e-3, (n, 3)).astype(np.float32)
    ids = np.arange(n) // 50
    ds = zv.create(str(path), schema=zv.Schema(
        bounds=([0, 0, 0], [BOX] * 3), kind=kind,
        layout=zv.Layout(cells=4, shard_cells=shard, compression="zstd")))
    if kind == "point_cloud":
        ds.add_points(pts, object_ids=ids)
    elif kind == "mesh":
        tri = np.stack([np.arange(0, n - 2), np.arange(1, n - 1), np.arange(2, n)], 1)
        ds.add_mesh(pts, tri[ids[tri[:, 0]] == ids[tri[:, 2]]], object_ids=ids)
    else:
        e = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
        ds.add_graph(pts, e[ids[e[:, 0]] == ids[e[:, 1]]], object_ids=ids)
    return path


def _answer(r):
    out = [np.sort(np.asarray(r.positions), axis=0)]
    for name in ("faces", "edges"):
        a = getattr(r, name, None)
        out.append(None if a is None else len(a))
    return out


def _same(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["point_cloud", "mesh", "graph"])
@pytest.mark.parametrize("shard", [None, 2])
def test_a_deferred_level_reads_what_it_holds(tmp_path, kind, shard):
    path = _store(tmp_path / "s.zv", kind, shard)
    want = [_answer(zv.open(str(path)).select(**q).read()) for q in QUERIES]
    assert len(want[0][0]) == 1_500

    lg = get_resolution_level(open_store(str(path), mode="r+"), 0)
    defer_presence(lg)
    ds = zv.open(str(path))
    assert ds.level(0).store.presence_deferred()
    for q, ref in zip(QUERIES, want):
        _same(_answer(ds.select(**q).read()), ref)
        _same(_answer(asyncio.run(ds.select(**q).aread())), ref)

    rebuild_presence(get_resolution_level(open_store(str(path), mode="r+"), 0))
    ds = zv.open(str(path))
    assert not ds.level(0).store.presence_deferred()
    for q, ref in zip(QUERIES, want):
        _same(_answer(ds.select(**q).read()), ref)
