"""Skip a GPU test module unless cupy imports and a CUDA device is visible.

Each GPU test module starts with ``from tests.gpu._cuda import cupy`` and
``pytestmark = CUDA``. CI has no GPU, so these run locally, from the repo
root, in an environment with cupy (e.g. ``bridge-gpu-zv3``).
"""

from __future__ import annotations

import pytest

cupy = pytest.importorskip("cupy")

from zarr_vectors.gpu import device_count  # noqa: E402

CUDA = [
    pytest.mark.gpu,
    pytest.mark.skipif(device_count() == 0, reason="no CUDA device"),
]
