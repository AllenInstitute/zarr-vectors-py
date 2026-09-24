"""Optional GPU extension: device arrays for zarr-vectors reads and writes.

Installed with ``pip install "zarr-vectors[gpu]"``, or in a conda
environment by installing cupy from conda-forge (the extra would add a
second, pip-built cupy). Nothing in the core imports this module at import
time; it is loaded the first time a device array is involved, through
``device="cuda"`` on a reader or a device array passed to a writer.

What it does:

- reads (:func:`zarr_vectors.core.cells.read_cells` with
  ``device="cuda"``) fetch cells' stored bytes into device memory and
  decode them there (:mod:`._read`, :mod:`._fetch`, :mod:`._kernels`),
  with zstd through nvCOMP when asked (:mod:`._codecs`);
- writers handed device arrays run the fragment, manifest and link
  partition encoders on the device and download the encoded form
  (:mod:`._encode`);
- everything else moves arrays between host and device once per array.

Compression and the write itself stay on the host, so the bytes on disk
are the same whether the arrays came from numpy or from the device.
:func:`zarr_vectors.runtime_capabilities` reports what this install can
do (``device_decode``, ``gpu_encode``, ``gpu_codecs``, ``gpu_io``).

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
