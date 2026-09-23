"""Optional GPU extension: device arrays for zarr-vectors reads and writes.

Installed with ``pip install "zarr-vectors[gpu]"``, or in a conda
environment by installing cupy from conda-forge (the extra would add a
second, pip-built cupy). Nothing in the core imports this module at import
time; it is loaded the first time a device array is involved, through
``device="cuda"`` on a reader or a device array passed to a writer.

What it does today: moves arrays between host and device. Cells are still
fetched and decoded on the host (they are compressed variable-length
bytes, which zarr's GPU buffers cannot carry), so a device read is one
host decode and one upload per returned array, and a device write is one
download per argument and a host encode. The bytes on disk are the same
either way. Device-side decode and GPUDirect Storage are future work;
:func:`zarr_vectors.runtime_capabilities` reports what this install can do.

Callers should reach this through ``device=`` and ``runtime_capabilities``
rather than import it: its own surface is not yet promised.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - exercised without cupy
    raise ImportError(
        "zarr_vectors.gpu needs cupy. Install it with "
        "pip install 'zarr-vectors[gpu]', or in a conda environment "
        "conda install -c conda-forge cupy."
    ) from exc

#: The device array module.
xp = cupy


def upload(a: Any) -> Any:
    """A host array as a device array (one host-to-device copy)."""
    return cupy.asarray(np.asarray(a))


def to_host(a: Any) -> np.ndarray:
    """Any object exposing ``__cuda_array_interface__`` as a numpy array."""
    return cupy.asarray(a).get()


def device_count() -> int:
    """How many CUDA devices are visible; 0 if the runtime cannot say.

    Initialises the CUDA driver, so a process that will fork should call
    it in the children.
    """
    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except cupy.cuda.runtime.CUDARuntimeError:
        return 0


__all__ = ["device_count", "to_host", "upload", "xp"]
