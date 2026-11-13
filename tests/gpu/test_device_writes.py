"""Writers fed device arrays: identical bytes, encoded where the arrays are.

Encoding runs on the device, so what is downloaded is the encoded form
(fragment sections, grouped link rows), not each argument.
"""

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
    assert stats.d2h_calls == 5  # checks, then the four sections
    assert_stores_identical(host_path, dev_path)


def test_manifests_from_the_device_and_read_back_to_it(tmp_path):
    from tests._store_compare import assert_stores_identical
    from zarr_vectors.building import (
        create_store,
        get_resolution_level,
        read_all_object_manifests_csr,
        write_object_manifests,
    )
    from zarr_vectors.constants import OBJECT_INDEX
    from zarr_vectors.core.arrays import OBJECT_INDEX_LAYOUT_V1

    rng = np.random.default_rng(5)
    counts = rng.integers(0, 4, 50)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    coords = rng.integers(0, 2, (int(offsets[-1]), 3)).astype(np.int64)
    frags = rng.integers(0, 100, int(offsets[-1])).astype(np.int64)

    def _write(name, *arrays):
        (tmp_path / name).mkdir()
        path = tmp_path / name / "s.zarrvectors"
        lg = get_resolution_level(create_store(
            path, bounds=([0.0] * 3, [100.0] * 3), chunk_shape=(50.0,) * 3,
        ), 0)
        write_object_manifests(
            lg, chunk_coords=arrays[0], fragment_idx=arrays[1],
            manifest_offsets=arrays[2], mode="append",
        )
        lg.write_array_meta(OBJECT_INDEX, {
            "zv_array": "object_index", "num_objects": 50, "num_present": 50,
            "sid_ndim": 3, "layout": OBJECT_INDEX_LAYOUT_V1,
        })
        return path, lg

    host_path, _ = _write("host", coords, frags, offsets)
    dev_path, dev = _write("dev", *(cupy.asarray(a) for a in (coords, frags, offsets)))
    assert_stores_identical(host_path, dev_path)

    csr = read_all_object_manifests_csr(dev, device="cuda")
    assert all(isinstance(a, cupy.ndarray) for a in csr)
    np.testing.assert_array_equal(csr.offsets.get(), offsets)
    np.testing.assert_array_equal(csr.chunk_coords.get(), coords)
    np.testing.assert_array_equal(csr.fragment_idx.get(), frags)


def test_link_cells_from_the_device(tmp_path):
    from tests._store_compare import assert_stores_identical
    from zarr_vectors.building import (
        create_store,
        finalize_links,
        get_resolution_level,
        write_link_cells,
    )

    rng = np.random.default_rng(6)
    n = 80
    base = rng.integers(0, 3, (n, 1, 3))
    chunks = np.clip(base + rng.integers(-1, 2, (n, 2, 3)), 0, 3).astype(np.int64)
    vids = rng.integers(0, 50, (n, 2)).astype(np.int64)
    attrs = {"w": rng.uniform(0, 1, n).astype(np.float32)}

    def _level(name):
        (tmp_path / name).mkdir()
        path = tmp_path / name / "s.zarrvectors"
        return path, get_resolution_level(create_store(
            path, bounds=([0.0] * 3, [200.0] * 3), chunk_shape=(50.0,) * 3,
        ), 0)

    host_path, host = _level("host")
    dev_path, dev = _level("dev")
    write_link_cells(host, chunks=chunks, vids=vids, attributes=attrs)
    with _xp.count_transfers() as stats:
        write_link_cells(
            dev, chunks=cupy.asarray(chunks), vids=cupy.asarray(vids),
            attributes={k: cupy.asarray(v) for k, v in attrs.items()},
        )
    assert stats.d2h_calls == 6  # the grouped partition (5), the attribute (1)
    for lg in (host, dev):
        finalize_links(lg, delta=0)
    assert_stores_identical(host_path, dev_path)


def test_dense_manifests_from_the_device(tmp_path):
    """A dense index takes device arrays as they are: one copy each, no blobs."""
    from zarr_vectors.building import (
        create_store,
        get_resolution_level,
        read_all_object_manifests_csr,
        write_object_manifests,
    )
    from zarr_vectors.constants import OBJECT_INDEX

    rng = np.random.default_rng(7)
    counts = rng.integers(0, 4, 500)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    coords = rng.integers(0, 2, (int(offsets[-1]), 3)).astype(np.int64)
    frags = rng.integers(0, 100, int(offsets[-1])).astype(np.int64)
    lg = get_resolution_level(create_store(
        tmp_path / "s.zarrvectors", bounds=([0.0] * 3, [100.0] * 3),
        chunk_shape=(50.0,) * 3, manifest_layout="dense",
    ), 0)
    with _xp.count_transfers() as stats:
        write_object_manifests(
            lg, chunk_coords=cupy.asarray(coords), fragment_idx=cupy.asarray(frags),
            manifest_offsets=cupy.asarray(offsets), mode="append",
        )
    assert stats.d2h_calls == 3
    meta = lg.read_array_meta(OBJECT_INDEX)
    lg.write_array_meta(OBJECT_INDEX, {**meta, "num_objects": 500, "num_present": 500})
    csr = read_all_object_manifests_csr(lg, device="cuda")
    np.testing.assert_array_equal(csr.offsets.get(), offsets)
    np.testing.assert_array_equal(csr.chunk_coords.get(), coords)
    np.testing.assert_array_equal(csr.fragment_idx.get(), frags)
