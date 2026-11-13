"""``unshard_store`` clears the root ``shard_shape``.

It used to leave it, so the next array allocated -- a pyramid level, say
-- came out sharded again in a store that had just been unsharded.
"""

from __future__ import annotations

import numpy as np

from zarr_vectors.core.store import open_store, read_root_metadata
from zarr_vectors.multiresolution.coarsen import build_pyramid
from zarr_vectors.sharding.io import get_shard_info, is_sharded, unshard_store
from zarr_vectors.types.points import write_points


def _store(tmp_path):
    path = tmp_path / "pts.zarr"
    rng = np.random.default_rng(0)
    write_points(path, rng.uniform(0, 100, (500, 3)).astype(np.float32),
                 chunk_shape=(25.0,) * 3, bounds=([0.0] * 3, [100.0] * 3),
                 shard_shape=2)
    return path


def test_later_writes_stay_unsharded(tmp_path):
    path = _store(tmp_path)
    assert read_root_metadata(open_store(str(path))).shard_shape == 2
    unshard_store(path)
    assert read_root_metadata(open_store(str(path))).shard_shape is None
    assert not is_sharded(path)

    build_pyramid(path, factors=[(2.0, 1.0)])
    assert not is_sharded(path), get_shard_info(path)["arrays"]


def test_a_partial_unshard_keeps_the_store_default(tmp_path):
    path = _store(tmp_path)
    unshard_store(path, arrays=["vertices"])
    assert read_root_metadata(open_store(str(path))).shard_shape == 2
