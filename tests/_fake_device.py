"""A stand-in for a device array, for testing the host/device contract on CPU.

It exposes ``__cuda_array_interface__`` and ``.get()`` like cupy, but its
``__array__`` raises: any code path that reaches for ``np.asarray`` on a
device input, instead of going through ``_xp.to_host``, fails loudly on a
machine with no GPU.
"""

from __future__ import annotations

import numpy as np


class FakeDeviceArray:
    def __init__(self, host):
        self._host = np.asarray(host)

    @property
    def __cuda_array_interface__(self):  # pragma: no cover - never read
        raise AssertionError("the interface itself should never be read")

    def get(self):
        return self._host.copy()

    def __array__(self, *args, **kwargs):
        raise TypeError("implicit host conversion of a device array")

    @property
    def shape(self):
        return self._host.shape

    @property
    def dtype(self):
        return self._host.dtype

    def __len__(self):
        return len(self._host)
