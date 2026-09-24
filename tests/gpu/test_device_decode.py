"""``read_cells(device="cuda")`` decoding on the device: the host result, exactly.

The host decode is the reference. Every combination of codec (none, zstd),
layout (flat, sharded) and read route (pinned host reads, kvikio, a
non-local zarr store) must give the same data, offsets and errors for
vertices, stamped and unstamped vertex attributes, flat and ragged link
cells, and link attributes -- over present cells, repeated cells, cells
nobody wrote, and cells off the grid.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors.core.arrays import list_chunk_keys, list_link_offsets
from zarr_vectors.core.cells import read_cells
from zarr_vectors.core.paths import is_intra, parse_offsets
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ArrayError
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points

pytestmark = CUDA

_CHUNK = (25.0, 25.0, 25.0)
_BOUNDS = [[0, 0, 0], [100, 100, 100]]
_CODECS = [None, "zstd"]
_SHARDS = [None, (2, 2, 2)]


def _needs(compressor):
    """zstd on the device needs nvCOMP's Python bindings."""
    if compressor == "zstd":
        pytest.importorskip("nvidia.nvcomp")


def _points(path, compressor, shard):
    _needs(compressor)
    rng = np.random.default_rng(0)
    write_points(
        str(path), rng.uniform(0, 100, (700, 3)).astype("float32"),
        chunk_shape=_CHUNK, bounds=_BOUNDS, compressor=compressor, shard_shape=shard,
        vertex_attributes={
            "i": rng.uniform(0, 1, 700).astype("float32"),
            "rgb": rng.integers(0, 255, (700, 3)).astype("uint8"),
        },
    )
    return get_resolution_level(open_store(str(path)), 0)


def _graph(path, compressor, shard):
    _needs(compressor)
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 100, (300, 3)).astype("float32")
    edges = rng.integers(0, 300, (900, 2)).astype(np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    write_graph(
        str(path), pos, edges, chunk_shape=_CHUNK, bounds=_BOUNDS,
        compressor=compressor, shard_shape=shard,
        link_attributes={"w": rng.uniform(0, 1, len(edges)).astype("float32")},
    )
    return get_resolution_level(open_store(str(path)), 0)


def _cells(lg):
    present = [tuple(k) for k in list_chunk_keys(lg)]
    extra = [present[0], (0, 3, 3), (3, 3, 0), (7, 7, 7), (-1, 0, 0)]
    return np.asarray(present + extra, dtype=np.int64)


def _link_arrays(lg):
    names = []
    for seg in list_link_offsets(lg, 0):
        names.append(f"links/0/{seg}")
        if lg.read_array_meta(f"link_attributes/w/0/{seg}") is not None:
            names.append(f"link_attributes/w/0/{seg}")
    return names


def _assert_same(host, dev):
    assert set(host) == set(dev)
    assert sorted(host.errors) == sorted(dev.errors)
    for name in host:
        h, d = host[name], dev[name]
        assert isinstance(d.data, cupy.ndarray), name
        assert d.data.dtype == h.data.dtype, name
        assert d.data.shape == h.data.shape, name
        np.testing.assert_array_equal(d.data.get(), h.data, err_msg=name)
        np.testing.assert_array_equal(d.offsets.get(), h.offsets, err_msg=name)


@pytest.mark.parametrize("compressor", _CODECS)
@pytest.mark.parametrize("shard", _SHARDS)
@pytest.mark.parametrize("io", ["host", "kvikio"])
def test_points_decode_on_the_device(tmp_path, compressor, shard, io, monkeypatch):
    if io == "kvikio":
        pytest.importorskip("kvikio")
    monkeypatch.setenv("ZARR_VECTORS_GPU_IO", io)
    lg = _points(tmp_path / "p.zarrvectors", compressor, shard)
    arrays = ["vertices", "vertex_attributes/i", "vertex_attributes/rgb"]
    cells = _cells(lg)
    host = read_cells(lg, cells, arrays)
    dev = read_cells(lg, cells, arrays, device="cuda", decode="device")
    _assert_same(host, dev)
    # An unstamped attribute alone still finds its width from the vertices.
    alone = read_cells(lg, cells, ["vertex_attributes/i"], device="cuda", decode="device")
    _assert_same(read_cells(lg, cells, ["vertex_attributes/i"]), alone)


@pytest.mark.parametrize("compressor", _CODECS)
@pytest.mark.parametrize("shard", _SHARDS)
def test_links_decode_on_the_device(tmp_path, compressor, shard):
    lg = _graph(tmp_path / "g.zarrvectors", compressor, shard)
    arrays = _link_arrays(lg)
    segs = [a.split("/")[2] for a in arrays if a.startswith("links/")]
    assert any(is_intra(parse_offsets(s, sid_ndim=3, link_width=2)) for s in segs)
    assert any(not is_intra(parse_offsets(s, sid_ndim=3, link_width=2)) for s in segs)
    assert any(a.startswith("link_attributes/") for a in arrays)
    cells = _cells(lg)
    _assert_same(
        read_cells(lg, cells, arrays),
        read_cells(lg, cells, arrays, device="cuda", decode="device"),
    )


@pytest.mark.parametrize("compressor", _CODECS)
@pytest.mark.parametrize("shard", _SHARDS)
def test_a_non_local_store_decodes_on_the_device(tmp_path, compressor, shard):
    import zarr

    path = tmp_path / "p.zarrvectors"
    _points(path, compressor, shard)
    mem = zarr.storage.MemoryStore()
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            key = os.path.relpath(full, path).replace(os.sep, "/")
            with open(full, "rb") as fh:
                zarr.core.sync.sync(mem.set(key, zarr.core.buffer.cpu.Buffer.from_bytes(fh.read())))
    lg = get_resolution_level(open_store(mem), 0)
    cells = _cells(lg)
    arrays = ["vertices", "vertex_attributes/rgb"]
    _assert_same(
        read_cells(lg, cells, arrays),
        read_cells(lg, cells, arrays, device="cuda", decode="device"),
    )


def _one_cell_file(lg, name="vertices"):
    root = lg._sharded_chunk_array(name).store_path
    base = os.path.join(str(root.store.root), root.path, "c")
    for dirpath, _, files in os.walk(base):
        for f in files:
            return os.path.join(dirpath, f)
    raise AssertionError("no cell files")


@pytest.mark.parametrize("compressor", _CODECS)
def test_a_corrupt_cell_is_an_error_not_garbage(tmp_path, compressor):
    lg = _points(tmp_path / "p.zarrvectors", compressor, None)
    victim = _one_cell_file(lg)
    with open(victim, "r+b") as fh:
        fh.seek(0)
        fh.write(b"\x07\x00\x00\x00garbage!")
    cells = _cells(lg)
    host = read_cells(lg, cells, ["vertices"])
    dev = read_cells(lg, cells, ["vertices"], device="cuda", decode="device")
    assert host.errors and dev.errors
    assert {(e.array, e.chunk_key) for e in dev.errors} == {
        (e.array, e.chunk_key) for e in host.errors
    }
    np.testing.assert_array_equal(dev["vertices"].offsets.get(), host["vertices"].offsets)
    np.testing.assert_array_equal(dev["vertices"].data.get(), host["vertices"].data)
    with pytest.raises(ArrayError):
        read_cells(lg, cells, ["vertices"], device="cuda", decode="device", on_error="raise")


def test_a_truncated_zstd_frame_is_caught(tmp_path):
    lg = _points(tmp_path / "p.zarrvectors", "zstd", None)
    victim = _one_cell_file(lg)
    data = open(victim, "rb").read()
    with open(victim, "wb") as fh:
        fh.write(data[: len(data) // 2])
    dev = read_cells(lg, _cells(lg), ["vertices"], device="cuda", decode="device")
    assert len(dev.errors) == 1


def test_auto_decodes_uncompressed_on_the_device_and_zstd_on_the_host(tmp_path, monkeypatch):
    from zarr_vectors.gpu import _read as device_read

    fetched = []
    real = device_read.fetch_payloads

    def spy(items, *a, **kw):
        fetched.extend((src.name, src.zstd) for src, *_ in items)
        return real(items, *a, **kw)

    monkeypatch.setattr(device_read, "fetch_payloads", spy)
    for compressor in (None, "zstd"):
        rng = np.random.default_rng(0)
        path = tmp_path / f"{compressor}.zarrvectors"
        write_points(
            str(path), rng.uniform(0, 100, (300, 3)).astype("float32"),
            chunk_shape=_CHUNK, bounds=_BOUNDS, compressor=compressor,
        )
        lg = get_resolution_level(open_store(str(path)), 0)
        cells = _cells(lg)
        _assert_same(
            read_cells(lg, cells, ["vertices"]),
            read_cells(lg, cells, ["vertices"], device="cuda"),
        )
    assert fetched == [("vertices", False)]


def test_an_unsupported_codec_falls_back_or_raises(tmp_path):
    lg = _points(tmp_path / "p.zarrvectors", "blosc", None)
    cells = _cells(lg)
    host = read_cells(lg, cells, ["vertices"])
    _assert_same(host, read_cells(lg, cells, ["vertices"], device="cuda"))
    with pytest.raises(ArrayError, match="cannot be decoded on the device"):
        read_cells(lg, cells, ["vertices"], device="cuda", decode="device")


def test_a_host_read_ignores_decode(tmp_path):
    lg = _points(tmp_path / "p.zarrvectors", None, None)
    batch = read_cells(lg, _cells(lg), ["vertices"], decode="device")
    assert isinstance(batch["vertices"].data, np.ndarray)


def test_a_missing_shard_reads_as_empty(tmp_path):
    lg = _points(tmp_path / "p.zarrvectors", "zstd", (2, 2, 2))
    shard = os.path.dirname(_one_cell_file(lg))
    shutil.rmtree(shard)
    cells = _cells(lg)
    _assert_same(
        read_cells(lg, cells, ["vertices"]),
        read_cells(lg, cells, ["vertices"], device="cuda", decode="device"),
    )
