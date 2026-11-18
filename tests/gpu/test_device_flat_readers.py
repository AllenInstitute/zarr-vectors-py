"""The flat readers read through ``read_cells``' device path with ``device="cuda"``.

``read_chunk_vertex_buffer``, ``read_chunk_vertex_rows``,
``read_chunk_fragment_attributes`` and ``read_link_attributes`` must return
exactly what the host reader returns (dtype, shape, values), raise what it
raises, and fall back to it wherever the device path cannot answer the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors.core import arrays
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.encoding.fragments import encode_fragments
from zarr_vectors.exceptions import ArrayError, StoreError
from zarr_vectors.gpu import _readers
from zarr_vectors.types.graphs import write_graph
from zarr_vectors.types.points import write_points

pytestmark = CUDA

_CHUNK = (25.0, 25.0, 25.0)
_BOUNDS = [[0, 0, 0], [100, 100, 100]]
_CASES = [(None, None), (None, (2, 2, 2)), ("zstd", None), ("zstd", (2, 2, 2))]


@pytest.fixture
def served(monkeypatch):
    """Names of the device readers that answered (not NotImplemented)."""
    answered = []
    for name in ("read_chunk_vertex_buffer", "read_chunk_vertex_rows",
                 "read_chunk_fragment_attributes", "read_link_attributes"):
        real = getattr(_readers, name)

        def spy(*a, _real=real, _name=name, **k):
            out = _real(*a, **k)
            if out is not NotImplemented:
                answered.append(_name)
            return out

        monkeypatch.setattr(_readers, name, spy)
    return answered


def _same(dev, host):
    assert isinstance(dev, cupy.ndarray)
    assert dev.dtype == host.dtype and dev.shape == host.shape
    np.testing.assert_array_equal(dev.get(), host)


def _outcome(fn):
    try:
        return fn(), None
    except (ArrayError, StoreError) as exc:
        return None, (type(exc), str(exc))


def _points(path, compressor, shard, ndim=3):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (600, ndim)).astype("float32")
    write_points(
        str(path), pos, chunk_shape=_CHUNK[:ndim],
        bounds=[b[:ndim] for b in _BOUNDS], compressor=compressor, shard_shape=shard,
    )
    return get_resolution_level(open_store(str(path), mode="r+"), 0)


def _probe_cells(lg):
    present = [tuple(k) for k in arrays.list_chunk_keys(lg)]
    return present + [(0, 0, 0), (3, 3, 3), (7, 7, 7), (-1, 0, 0)]


@pytest.mark.parametrize("compressor,shard", _CASES)
def test_vertex_buffer_and_rows(tmp_path, compressor, shard, served):
    lg = _points(tmp_path / "p.zarrvectors", compressor, shard)
    for cc in _probe_cells(lg):
        for reader in (arrays.read_chunk_vertex_buffer, arrays.read_chunk_vertex_rows):
            host, host_err = _outcome(lambda: reader(lg, cc))
            dev, dev_err = _outcome(lambda: reader(lg, cc, device="cuda"))
            assert dev_err == host_err, (reader.__name__, cc)
            if host_err is None:
                _same(dev, host)
    # Default handling is the host's.
    assert arrays.read_chunk_vertex_buffer(lg, (7, 7, 7), default=None, device="cuda") is None
    assert {"read_chunk_vertex_buffer", "read_chunk_vertex_rows"} <= set(served)


def _negative_index():
    """An explicit fragment ``[-1, 2]``, which the encoder refuses to write."""
    blob = bytearray(encode_fragments([np.array([7, 2])], force_explicit=True))
    at = len(blob) - 16
    blob[at:at + 8] = np.int64(-1).tobytes()
    return bytes(blob)


def test_vertex_rows_follow_every_kind_of_fragment(tmp_path, served):
    lg = _points(tmp_path / "p.zarrvectors", None, None)
    present = [tuple(k) for k in arrays.list_chunk_keys(lg)]
    rows = {cc: len(arrays.read_chunk_vertex_buffer(lg, cc)) for cc in present}
    cells = [cc for cc in present if rows[cc] >= 6][:6]
    assert len(cells) == 6
    n = [rows[cc] for cc in cells]
    blobs = [
        encode_fragments([(2, 3), np.array([0, 5, 0]), (0, 0), (1, n[0] - 1)]),
        encode_fragments([np.array([n[1] - 1, 0])] * 5, force_explicit=True),
        encode_fragments([(n[2] - 2, 2), (0, 2)]),            # out of order
        encode_fragments([]),                                 # no fragments
        encode_fragments([(n[4] - 1, 5)]),                    # runs off the end
        _negative_index(),
    ]
    for cc, blob in zip(cells, blobs):
        lg.write_bytes("vertex_fragments", ".".join(map(str, cc)), blob)
    for cc in cells:
        _same(arrays.read_chunk_vertex_rows(lg, cc, device="cuda"),
              arrays.read_chunk_vertex_rows(lg, cc))
    # The first three are gathered on the device, the rest by the host.
    assert served.count("read_chunk_vertex_rows") == 3


def test_vertex_buffer_with_one_dimension_and_another_dtype(tmp_path, served):
    lg = _points(tmp_path / "p.zarrvectors", None, None, ndim=1)
    for cc in [tuple(k) for k in arrays.list_chunk_keys(lg)][:4]:
        _same(arrays.read_chunk_vertex_buffer(lg, cc, ndim=1, device="cuda"),
              arrays.read_chunk_vertex_buffer(lg, cc, ndim=1))
        _same(arrays.read_chunk_vertex_buffer(lg, cc, "float32", 1, device="cuda"),
              arrays.read_chunk_vertex_buffer(lg, cc, "float32", 1))
        _same(arrays.read_chunk_vertex_buffer(lg, cc, "int32", 1, device="cuda"),
              arrays.read_chunk_vertex_buffer(lg, cc, "int32", 1))
    assert served.count("read_chunk_vertex_buffer") == 8   # int32 is the host's


def _graph(path, compressor, shard):
    rng = np.random.default_rng(1)
    n = 300
    pos = rng.uniform(0, 100, (n, 3)).astype("float32")
    edges = rng.integers(0, n, (900, 2)).astype(np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    write_graph(
        str(path), pos, edges, chunk_shape=_CHUNK, bounds=_BOUNDS,
        compressor=compressor, shard_shape=shard,
        link_attributes={
            "w": rng.uniform(0, 1, len(edges)).astype("float32"),
            "rgb": rng.integers(0, 255, (len(edges), 3)).astype("uint8"),
        },
    )
    return get_resolution_level(open_store(str(path)), 0)


@pytest.mark.parametrize("compressor,shard", _CASES)
def test_link_attributes(tmp_path, compressor, shard, served):
    lg = _graph(tmp_path / "g.zarrvectors", compressor, shard)
    for name in ("w", "rgb"):
        host = arrays.read_link_attributes(lg, name)
        assert len(host) > 100
        _same(arrays.read_link_attributes(lg, name, device="cuda"), host)
    _same(arrays.read_link_attributes(lg, "w", "int32", device="cuda"),
          arrays.read_link_attributes(lg, "w", "int32"))
    _same(arrays.read_link_attributes(lg, "absent", device="cuda"),
          arrays.read_link_attributes(lg, "absent"))
    assert served.count("read_link_attributes") == 2


@pytest.mark.parametrize("compressor,shard", _CASES)
def test_fragment_attributes(tmp_path, compressor, shard, served):
    rng = np.random.default_rng(4)
    path = tmp_path / "p.zarrvectors"
    lg = _points(path, compressor, shard)
    present = [tuple(k) for k in arrays.list_chunk_keys(lg)]
    arrays.create_fragment_attribute_array(lg, "score", ncols=2)
    for cc in present[:5]:
        f = arrays.read_vertex_fragment_index(lg, cc).num_fragments
        arrays.write_chunk_fragment_attributes(
            lg, "score", cc, rng.uniform(0, 1, (f, 2)).astype("float32"),
        )
    lg = get_resolution_level(open_store(str(path)), 0)
    for cc in present[:7] + [(7, 7, 7)]:
        host, host_err = _outcome(
            lambda: arrays.read_chunk_fragment_attributes(lg, "score", cc))
        dev, dev_err = _outcome(
            lambda: arrays.read_chunk_fragment_attributes(lg, "score", cc, device="cuda"))
        assert dev_err == host_err, cc
        if host_err is None:
            _same(dev, host)
    assert served.count("read_chunk_fragment_attributes") == 5


def test_a_shard_transaction_is_read_on_the_host(tmp_path, served):
    """Staged cells are only visible through Group.read_bytes."""
    from zarr_vectors import building as zb
    from zarr_vectors.core.cells import read_cells

    bounds, chunk = ([0.0] * 3, [400.0] * 3), (100.0,) * 3
    root = zb.create_store(str(tmp_path / "s.zv"), bounds=bounds, chunk_shape=chunk,
                           shard_shape=2)
    lg = zb.get_resolution_level(root, 0)
    with zb.open_write_session(lg, bounds=bounds, chunk_shape=chunk):
        zb.create_vertices_array(lg, dtype="float32")
        zb.write_chunk_vertices(lg, (0, 0, 0), [np.zeros((3, 3), np.float32)])
    zb.defer_presence(lg)
    with zb.shard_transaction(lg, (0, 0, 0)):
        zb.write_chunk_vertices(lg, (0, 0, 0), [np.ones((5, 3), np.float32)])
        host = arrays.read_chunk_vertex_buffer(lg, (0, 0, 0))
        assert host.shape == (5, 3)
        _same(arrays.read_chunk_vertex_buffer(lg, (0, 0, 0), device="cuda"), host)
        batch = read_cells(lg, np.asarray([(0, 0, 0)]), ["vertices"], device="cuda")
        _same(batch["vertices"].data, host)
        assert batch.io[0].path == "host"
    assert served == []
