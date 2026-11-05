"""The host/device contract, checked on a machine without a GPU."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests._fake_device import FakeDeviceArray
from zarr_vectors import _xp
from zarr_vectors.exceptions import ZVError


def test_device_arrays_are_recognised_by_their_interface():
    assert _xp.is_device_array(FakeDeviceArray([1, 2]))
    assert not _xp.is_device_array(np.arange(2))
    assert not _xp.is_device_array([1, 2])


@pytest.mark.parametrize("device, like, expected", [
    (None, (), "cpu"),
    (None, (np.arange(2),), "cpu"),
    (None, (np.arange(2), FakeDeviceArray([1])), "cuda"),
    ("cpu", (FakeDeviceArray([1]),), "cpu"),
    ("cuda", (), "cuda"),
])
def test_resolve_device(device, like, expected):
    assert _xp.resolve_device(device, *like) == expected


def test_an_unknown_device_is_an_error():
    with pytest.raises(ZVError, match="device='tpu'"):
        _xp.resolve_device("tpu")


def test_to_host_copies_a_device_array_once():
    data = np.arange(6, dtype=np.int64)
    with _xp.count_transfers() as stats:
        host = _xp.to_host(FakeDeviceArray(data))
    np.testing.assert_array_equal(host, data)
    assert (stats.d2h_calls, stats.d2h_bytes) == (1, data.nbytes)


def test_to_host_leaves_a_host_array_alone():
    data = np.arange(6)
    with _xp.count_transfers() as stats:
        assert _xp.to_host(data) is data
    assert stats.d2h_calls == 0


def test_to_host_casts_on_request():
    assert _xp.to_host(np.arange(3), dtype=np.int32).dtype == np.int32


def test_cuda_without_the_extension_is_an_error_not_a_host_array(monkeypatch):
    # A None entry makes the import fail whether or not cupy is installed.
    monkeypatch.setitem(sys.modules, "zarr_vectors.gpu", None)
    with pytest.raises(ZVError, match=r"zarr-vectors\[gpu\]"):
        _xp.to_device(np.arange(3), "cuda")


def test_to_device_none_and_cpu():
    data = np.arange(3)
    assert _xp.to_device(data, None) is data
    np.testing.assert_array_equal(_xp.to_device(FakeDeviceArray(data), "cpu"), data)


def test_namespace_is_numpy_for_host_arrays():
    assert _xp.namespace(np.arange(2), [1]) is np


def test_counters_nest():
    with _xp.count_transfers() as outer:
        _xp.to_host(FakeDeviceArray([1]))
        with _xp.count_transfers() as inner:
            _xp.to_host(FakeDeviceArray([1, 2]))
    assert (outer.d2h_calls, inner.d2h_calls) == (2, 1)
