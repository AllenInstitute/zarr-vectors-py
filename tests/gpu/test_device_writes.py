"""Writers fed device arrays: one download per argument, identical bytes."""

from __future__ import annotations

import numpy as np

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors import _xp

pytestmark = CUDA


def _level(tmp_path, name):
    from zarr_vectors.core.store import get_resolution_level, open_store
    from zarr_vectors.types.points import write_points

    (tmp_path / name).mkdir()
    path = str(tmp_path / name / "s.zarrvectors")
    write_points(
        path, np.random.default_rng(1).uniform(0, 100, (300, 3)).astype("float32"),
        chunk_shape=(50.0, 50.0, 50.0), bounds=[[0, 0, 0], [100, 100, 100]],
    )
    return path, get_resolution_level(open_store(path, mode="r+"), 0)


def test_csr_fragments_from_the_device(tmp_path):
    from tests._store_compare import assert_stores_identical
    from zarr_vectors.core.arrays import write_chunk_fragments

    rng = np.random.default_rng(4)
    counts = rng.integers(0, 6, 40)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = rng.integers(0, 90, int(offsets[-1])).astype(np.int64)
    host_path, host = _level(tmp_path, "host")
    dev_path, dev = _level(tmp_path, "dev")
    write_chunk_fragments(host, (1, 0, 1), csr=(indices, offsets), mode="append")
    with _xp.count_transfers() as stats:
        write_chunk_fragments(
            dev, (1, 0, 1), csr=(cupy.asarray(indices), cupy.asarray(offsets)),
            mode="append",
        )
    assert stats.d2h_calls == 2
    assert_stores_identical(host_path, dev_path)
