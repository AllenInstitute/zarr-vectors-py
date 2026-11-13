"""Host/device copies with a real device."""

from __future__ import annotations

import numpy as np

from tests.gpu._cuda import CUDA, cupy
from zarr_vectors import _xp, runtime_capabilities

pytestmark = CUDA


def test_cupy_arrays_are_device_arrays():
    assert _xp.is_device_array(cupy.arange(3))
    assert _xp.resolve_device(None, cupy.arange(3)) == "cuda"


def test_a_round_trip_is_one_copy_each_way():
    data = np.arange(12, dtype=np.float64).reshape(4, 3)
    with _xp.count_transfers() as stats:
        dev = _xp.to_device(data, "cuda")
        back = _xp.to_host(dev)
    assert isinstance(dev, cupy.ndarray)
    np.testing.assert_array_equal(back, data)
    assert (stats.h2d_calls, stats.d2h_calls) == (1, 1)


def test_a_device_array_is_not_copied_to_the_device_again():
    dev = cupy.arange(3)
    with _xp.count_transfers() as stats:
        assert _xp.to_device(dev, "cuda") is dev
    assert stats.h2d_calls == 0


def test_the_namespace_follows_the_data():
    assert _xp.namespace(cupy.arange(2)) is cupy


def test_the_probe_sees_the_device():
    assert runtime_capabilities(probe_device=True)["device_arrays"] is True
