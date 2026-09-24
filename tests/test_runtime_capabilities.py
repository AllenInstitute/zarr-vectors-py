"""``zarr_vectors.runtime_capabilities()``: what this install can do."""

from __future__ import annotations

import subprocess
import sys

import zarr_vectors as zv

#: Keys a caller may rely on. Keys are only ever added.
_KEYS = {
    "csr_fragments", "array_manifests", "manifests_csr_read",
    "object_attribute_columns", "array_link_cells", "read_cells",
    "read_neighbourhood", "batched_link_reads", "defer_presence",
    "append_safe_sharding", "dense_manifests", "gpu_encode", "gpu_io",
    "gpu_codecs", "device_arrays", "device_decode",
}


def test_every_promised_key_is_present_and_a_bool():
    caps = zv.runtime_capabilities()
    assert _KEYS <= set(caps)
    assert all(isinstance(v, bool) for v in caps.values())


def test_each_call_returns_a_fresh_dict():
    a = zv.runtime_capabilities()
    a["csr_fragments"] = "tampered"
    assert isinstance(zv.runtime_capabilities()["csr_fragments"], bool)


def test_it_is_advertised():
    assert "runtime_capabilities" in zv.__all__
    assert "runtime-capabilities" in zv.FEATURES


def test_importing_the_package_loads_no_gpu_code():
    """The core never imports the extension, nor cupy beyond what zarr does.

    zarr itself imports cupy when it is installed (``zarr.core.buffer.gpu``),
    so "cupy is not loaded" is only ours to promise where zarr alone does
    not load it.
    """
    code = (
        "import sys, zarr; by_zarr = 'cupy' in sys.modules; "
        "import zarr_vectors, zarr_vectors.building; "
        "zarr_vectors.runtime_capabilities; "
        "print(by_zarr, 'cupy' in sys.modules, 'zarr_vectors.gpu' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    by_zarr, cupy_loaded, extension_loaded = out.stdout.split()
    assert extension_loaded == "False"
    assert cupy_loaded == by_zarr


def test_device_keys_follow_what_is_installed(monkeypatch):
    from zarr_vectors import _runtime

    monkeypatch.setattr(_runtime, "_gpu_extension", lambda: False)
    caps = zv.runtime_capabilities()
    assert not any(caps[k] for k in (
        "device_arrays", "device_decode", "gpu_encode", "gpu_io", "gpu_codecs",
    ))

    monkeypatch.setattr(_runtime, "_gpu_extension", lambda: True)
    monkeypatch.setattr(_runtime, "_importable", lambda m: m == "kvikio")
    caps = zv.runtime_capabilities()
    assert caps["device_arrays"] and caps["device_decode"] and caps["gpu_encode"]
    assert caps["gpu_io"]
    assert not caps["gpu_codecs"]
