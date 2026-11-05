"""``device=`` on the flat readers, batched link reads, ReadResult.to_device."""

from __future__ import annotations

import inspect
import sys

import numpy as np
import pytest

import zarr_vectors as zv
from tests._fake_device import FakeDeviceArray
from zarr_vectors.core import arrays
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import get_resolution_level, open_store
from zarr_vectors.exceptions import ZVError
from zarr_vectors.types.graphs import write_graph

_READERS = [
    "read_chunk_vertex_buffer", "read_chunk_vertex_rows",
    "read_chunk_attribute_rows", "read_chunk_fragment_attributes",
    "read_link_arrays", "read_link_attributes",
]


@pytest.fixture
def graph(tmp_path):
    rng = np.random.default_rng(2)
    n = 150
    path = str(tmp_path / "g.zarrvectors")
    write_graph(
        path, positions=rng.uniform(0, 100, (n, 3)).astype("float32"),
        edges=np.stack([np.arange(n - 1), np.arange(1, n)], axis=1),
        object_ids=np.zeros(n, dtype=np.int64),
        chunk_shape=(40.0, 40.0, 40.0), bounds=([0, 0, 0], [100, 100, 100]),
        link_attributes={"w": rng.uniform(0, 1, n - 1).astype("float32")},
    )
    return path


@pytest.mark.parametrize("name", _READERS)
def test_every_flat_reader_takes_a_keyword_device(name):
    param = inspect.signature(getattr(arrays, name)).parameters["device"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None


def test_host_results_are_unchanged(graph):
    lg = get_resolution_level(open_store(graph), 0)
    cc = tuple(arrays.list_chunk_keys(lg)[0])
    np.testing.assert_array_equal(
        arrays.read_chunk_vertex_buffer(lg, cc, device="cpu"),
        arrays.read_chunk_vertex_buffer(lg, cc),
    )
    assert arrays.read_chunk_vertex_buffer(lg, (9, 9, 9), default=None, device="cpu") is None


def test_cuda_without_the_extension_is_an_error(graph, monkeypatch):
    lg = get_resolution_level(open_store(graph), 0)
    monkeypatch.setitem(sys.modules, "zarr_vectors.gpu", None)
    with pytest.raises(ZVError, match="gpu"):
        arrays.read_link_arrays(lg, device="cuda")


@pytest.mark.parametrize("reader", ["read_link_arrays", "read_link_attributes"])
def test_link_reads_prefetch_every_segment_once(graph, monkeypatch, reader):
    lg = get_resolution_level(open_store(graph), 0)
    expected = (
        arrays.read_link_arrays(lg) if reader == "read_link_arrays"
        else arrays.read_link_attributes(lg, "w")
    )
    calls = []
    real = Group.batched_reads

    def _count(self, plan, **kw):
        calls.append(len(plan))
        return real(self, plan, **kw)

    monkeypatch.setattr(Group, "batched_reads", _count)
    got = (
        arrays.read_link_arrays(lg) if reader == "read_link_arrays"
        else arrays.read_link_attributes(lg, "w")
    )
    assert len(calls) == 1 and calls[0] > 1  # one prefetch, several segments
    if isinstance(expected, tuple):
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_array_equal(got, expected)


def test_read_result_to_cpu_and_restrict_guard(graph):
    result = zv.open(graph).level(0).read()
    moved = result.to_device("cpu")
    np.testing.assert_array_equal(moved.positions, result.positions)
    assert moved.edges is not None
    on_device = type(result)(kind=result.kind, positions=FakeDeviceArray(result.positions))
    with pytest.raises(ZVError, match="to_device"):
        on_device.restrict(np.ones(result.vertex_count, dtype=bool))
