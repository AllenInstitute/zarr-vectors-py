"""Reads with ``device="cuda"``: the host result, one upload per array."""

from __future__ import annotations

import numpy as np

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors import _xp
from zarr_vectors.core import arrays
from zarr_vectors.core.cells import read_cells
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points

pytestmark = CUDA


def _points(tmp_path):
    rng = np.random.default_rng(0)
    path = str(tmp_path / "p.zarrvectors")
    write_points(
        path, rng.uniform(0, 100, (500, 3)).astype("float32"),
        chunk_shape=(25.0, 25.0, 25.0), bounds=[[0, 0, 0], [100, 100, 100]],
        vertex_attributes={"i": rng.uniform(0, 1, 500).astype("float32")},
    )
    return get_resolution_level(open_store(path), 0)


def test_read_cells_on_the_device(tmp_path):
    lg = _points(tmp_path)
    cells = np.asarray(arrays.list_chunk_keys(lg), dtype=np.int64)
    host = read_cells(lg, cells, ["vertices", "vertex_attributes/i"])
    with _xp.count_transfers() as stats:
        dev = read_cells(lg, cells, ["vertices", "vertex_attributes/i"], device="cuda")
    assert stats.h2d_calls == 4  # data + offsets, per array
    for name in host:
        assert isinstance(dev[name].data, cupy.ndarray)
        np.testing.assert_array_equal(dev[name].data.get(), host[name].data)
        np.testing.assert_array_equal(dev[name].offsets.get(), host[name].offsets)
    assert isinstance(dev["vertices"].cell, cupy.ndarray)


def test_device_coords_mean_device_results(tmp_path):
    lg = _points(tmp_path)
    cells = cupy.asarray(np.asarray(arrays.list_chunk_keys(lg), dtype=np.int64))
    batch = read_cells(lg, cells, ["vertices"])
    assert batch.device == "cuda" and isinstance(batch["vertices"].data, cupy.ndarray)


def test_flat_readers_and_link_reads_on_the_device(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    path = str(tmp_path / "g.zarrvectors")
    write_graph(
        path, positions=rng.uniform(0, 100, (n, 3)).astype("float32"),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1),
        object_ids=np.zeros(n, dtype=np.int64),
        chunk_shape=(40.0, 40.0, 40.0), bounds=([0, 0, 0], [100, 100, 100]),
    )
    lg = get_resolution_level(open_store(path), 0)
    cc = tuple(arrays.list_chunk_keys(lg)[0])
    v = arrays.read_chunk_vertex_buffer(lg, cc, device="cuda")
    np.testing.assert_array_equal(v.get(), arrays.read_chunk_vertex_buffer(lg, cc))
    with _xp.count_transfers() as stats:
        chunks, vi = arrays.read_link_arrays(lg, device="cuda")
    assert stats.h2d_calls == 2
    host_chunks, host_vi = arrays.read_link_arrays(lg)
    np.testing.assert_array_equal(chunks.get(), host_chunks)
    np.testing.assert_array_equal(vi.get(), host_vi)


def test_read_result_to_the_device(tmp_path):
    import zarr_vectors as zv

    lg = _points(tmp_path)
    del lg
    result = zv.open(str(tmp_path / "p.zarrvectors")).level(0).read()
    dev = result.to_device("cuda")
    assert isinstance(dev.positions, cupy.ndarray)
    np.testing.assert_array_equal(dev.positions.get(), result.positions)
    back = dev.to_device("cpu")
    np.testing.assert_array_equal(back.positions, result.positions)
