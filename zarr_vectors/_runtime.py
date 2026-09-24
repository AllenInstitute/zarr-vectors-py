"""What this installation can do, for a caller that branches on it.

:func:`runtime_capabilities` answers for the *install*: which array-form
entry points this build has, and whether device arrays are usable here.
It is not :attr:`~zarr_vectors.api.Dataset.capabilities`, which reports
what a *store* declares on disk, and it is not ``FEATURES``, which names
API features fixed at build time. A caller probes it instead of parsing a
version string, because a shared checkout can change branch under a
running process.
"""

from __future__ import annotations

import functools

#: Entry points and guarantees, each True once this build has it. Keys
#: are only ever added, so a caller can rely on one being present.
_STATIC: dict[str, bool] = {
    # Array-form writers and readers (BRIDGE R1/R3/R12).
    "csr_fragments": True,
    "array_manifests": True,
    "manifests_csr_read": True,
    "object_attribute_columns": True,
    "array_link_cells": True,
    "read_cells": True,
    "read_neighbourhood": True,
    "batched_link_reads": True,
    # Already true of this build.
    "defer_presence": True,
    "append_safe_sharding": True,
    # The dense object-index layout (create_store(manifest_layout="dense")).
    "dense_manifests": True,
}

#: Keys that depend on what is installed alongside, filled in per call.
_DYNAMIC = ("device_arrays", "device_decode", "gpu_encode", "gpu_io", "gpu_codecs")


@functools.cache
def _gpu_extension() -> bool:
    """Whether the optional GPU extension imports (does not touch CUDA)."""
    try:
        import zarr_vectors.gpu  # noqa: F401
    except Exception:
        # A probe must answer, not raise: a cupy that imports but cannot
        # find its CUDA libraries is as unusable as a missing one.
        return False
    return True


@functools.cache
def _importable(module: str) -> bool:
    import importlib

    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


@functools.cache
def _device_count() -> int:
    import zarr_vectors.gpu as gpu

    return gpu.device_count()


def runtime_capabilities(*, probe_device: bool = False) -> dict[str, bool]:
    """What this installation of zarr-vectors can do.

    Every value is a bool, True only when usable in this process:

    - ``device_arrays``: the optional GPU extension imports (and, with
      ``probe_device=True``, a CUDA device is visible), so readers take
      ``device="cuda"`` and writers take device arrays;
    - ``device_decode``: ``read_cells`` can decode uncompressed cells on
      the device rather than on the host (same condition);
    - ``gpu_encode``: writers handed device arrays encode fragments,
      manifests and link partitions on the device, and download the
      encoded form (same condition; the bytes written are unchanged);
    - ``gpu_codecs``: nvCOMP is installed too, so ``read_cells(...,
      decode="device")`` can decompress zstd cells on the device;
    - ``gpu_io``: kvikio is installed too, so local files can be read
      straight into device memory (GPUDirect Storage where the system
      has it; see ``docs/how_to/gpu.md``).

    Probing a device initialises the CUDA driver, so a process that will
    fork should probe in its children, not before forking.

    Returns:
        A new dict each call; the key set only grows between releases.
    """
    caps = dict(_STATIC)
    usable = _gpu_extension()
    if usable and probe_device:
        usable = _device_count() > 0
    caps["device_arrays"] = usable
    caps["device_decode"] = usable
    caps["gpu_encode"] = usable
    caps["gpu_codecs"] = usable and _importable("nvidia.nvcomp")
    caps["gpu_io"] = usable and _importable("kvikio")
    return caps
