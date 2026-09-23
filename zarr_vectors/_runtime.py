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
    "csr_fragments": False,
    "array_manifests": False,
    "manifests_csr_read": False,
    "object_attribute_columns": False,
    "array_link_cells": False,
    "read_cells": True,
    "read_neighbourhood": True,
    "batched_link_reads": True,
    # Already true of this build.
    "defer_presence": True,
    "append_safe_sharding": True,
    # Not yet: a dense manifest layout, and anything done on the device
    # rather than copied to or from it.
    "dense_manifests": False,
    "gpu_encode": False,
    "gpu_io": False,
    "gpu_codecs": False,
}


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
def _device_count() -> int:
    import zarr_vectors.gpu as gpu

    return gpu.device_count()


def runtime_capabilities(*, probe_device: bool = False) -> dict[str, bool]:
    """What this installation of zarr-vectors can do.

    Every value is a bool, True only when usable in this process.
    ``device_arrays`` is True when the optional GPU extension imports,
    and with ``probe_device=True`` also only when a CUDA device is
    visible. Probing initialises the CUDA driver, so a process that will
    fork should probe in its children, not before forking.

    Returns:
        A new dict each call; the key set only grows between releases.
    """
    caps = dict(_STATIC)
    usable = _gpu_extension()
    if usable and probe_device:
        usable = _device_count() > 0
    caps["device_arrays"] = usable
    return caps
