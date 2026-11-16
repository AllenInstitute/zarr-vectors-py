"""What this installation can do, for a caller that branches on it.

:func:`runtime_capabilities` answers for the *install*, unlike
:attr:`~zarr_vectors.api.Dataset.capabilities` (what a store declares) and
``FEATURES`` (API features fixed at build time).
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
    # Appends into a shared per-chunk shard keep the cells already there.
    # Per-chunk arrays only: the object layer's at= writes are not covered.
    "append_safe_sharding": True,
    # A second write into a link cell appends its records (and their
    # attribute rows) after the first, list form or array form, presence
    # deferred or not -- mirrored records of a canonical seam included
    # (tests/test_link_cell_writers.py::TestRepeatBatchesIntoOneCell).
    # Serial writes only: two processes writing one cell at once can
    # still lose rows.
    "link_cells_append": True,
    # zarr_vectors.concurrency_contract(): which writes several processes
    # may make at once -- per-chunk shards, and the object layer's
    # reserved rows written with mode="place".
    "concurrency_contract": True,
    # The dense object-index layout (create_store(manifest_layout="dense")).
    "dense_manifests": True,
    # shard_transaction(io_threads=, sweep=): the publish from a pool, and
    # the partial sweep on entry made optional.
    "shard_transaction_io_threads": True,
    "transaction_sweep": True,
    # An object index may hold more than 2**26 objects; only the empty
    # rows total_objects= declares are bounded.
    "object_count_unbounded": True,
    # batched_reads inside cached_nodes derives no presence.
    "spec_without_presence": True,
}

#: Keys that depend on what is installed alongside, filled in per call.
_DYNAMIC = ("device_arrays", "device_decode", "gpu_encode", "gpu_io", "gpu_codecs", "gds")


@functools.cache
def _gpu_extension() -> bool:
    """Whether the optional GPU extension imports (does not touch CUDA)."""
    try:
        import zarr_vectors.gpu  # noqa: F401
    except Exception:  # a cupy without its CUDA libraries is as unusable as none
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


def _gds_available() -> bool:
    from zarr_vectors._gds import gds_status

    return gds_status().available


def runtime_capabilities(*, probe_device: bool = False) -> dict[str, bool]:
    """What this installation of zarr-vectors can do.

    Every value is a bool, True only when usable in this process; the keys
    are listed in ``docs/how_to/gpu.md``. The device keys (``device_arrays``,
    ``device_decode``, ``gpu_encode``) need the GPU extension to import and,
    with ``probe_device=True``, a visible CUDA device; ``gpu_codecs`` and
    ``gpu_io`` also need nvCOMP and kvikio. ``gds`` is only probed with
    ``probe_device=True`` (it opens the cuFile driver). Probing initialises
    CUDA, so a process that will fork should probe in its children.

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
    caps["gds"] = caps["gpu_io"] and probe_device and _gds_available()
    return caps
