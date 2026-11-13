"""``get_shard_info`` lists arrays in a fixed order, on one grid per level.

The arrays came back in store-listing order.  Link arrays were also
reported on a different grid from vertices; a link record lives in its
source chunk's cell, so every per-chunk array of a level shares the
level's grid -- this pins that too.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.sharding.io import get_shard_info, shard_store
from zarr_vectors.types.graphs import write_graph


def _sharded_graph(tmp_path, *, bounds):
    path = tmp_path / "g.zarr"
    rng = np.random.default_rng(0)
    n = 200
    pos = rng.uniform(25, 99, (n, 3)).astype(np.float32)
    kw = {"bounds": bounds} if bounds else {}
    write_graph(path, positions=pos,
                edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1),
                object_ids=np.zeros(n, np.int64), chunk_shape=(20.0,) * 3, **kw)
    build_pyramid(path, factors=[(2.0, 1.0)])
    shard_store(path, shard_shape=2)
    return path


def test_arrays_are_sorted_by_level_then_path(tmp_path):
    info = get_shard_info(_sharded_graph(tmp_path, bounds=([0.0] * 3, [100.0] * 3)))
    keys = [(int(a["name"].split("/", 1)[0]), a["name"].split("/", 1)[1])
            for a in info["arrays"]]
    assert len(keys) > 10
    assert keys == sorted(keys)


def test_link_arrays_share_the_level_grid(tmp_path):
    for i, bounds in enumerate((([0.0] * 3, [100.0] * 3), None)):
        sub = tmp_path / str(i)
        sub.mkdir()
        grids = defaultdict(set)
        for a in get_shard_info(_sharded_graph(sub, bounds=bounds))["arrays"]:
            grids[a["name"].split("/", 1)[0]].add(tuple(a["grid_shape"]))
        assert all(len(g) == 1 for g in grids.values()), dict(grids)
