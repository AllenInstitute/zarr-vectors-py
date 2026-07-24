"""Tests for the Zarr-Store-level dispatch in ``_make_zarr_store_with_session``.

Covers the obstore + fsspec backends and the ome-zarr-py-style
pass-through of pre-built ``zarr.abc.store.Store`` and
``obstore.store.ObjectStore`` instances.
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr.abc.store import Store as _ZStore
from zarr.storage import LocalStore as _ZarrLocalStore

from zarr_vectors import open_store
from zarr_vectors.core.store import (
    _make_fsspec_zarr_store,
    _make_obstore_zarr_store,
    _make_zarr_store_with_session,
    create_store,
)
from zarr_vectors.exceptions import StoreError


obstore = pytest.importorskip("obstore")


def test_obstore_helper_returns_objectstore():
    # obstore.store.from_url handles the scheme dispatch; memory:// is a
    # dependency-free target that yields a usable ObjectStore.
    store, session = _make_obstore_zarr_store("memory:///x", mode="r+")
    assert session is None
    assert isinstance(store, _ZStore)
    assert store.read_only is False


def test_obstore_helper_read_only():
    store, _ = _make_obstore_zarr_store("memory:///x", mode="r")
    assert store.read_only is True


def test_fsspec_helper_returns_fsspecstore(tmp_path):
    pytest.importorskip("fsspec")
    store, session = _make_fsspec_zarr_store(f"file://{tmp_path}", mode="r+")
    assert session is None
    assert isinstance(store, _ZStore)


def test_dispatch_passes_through_prebuilt_zarr_store(tmp_store_path):
    inner = _ZarrLocalStore(str(tmp_store_path))
    store, session = _make_zarr_store_with_session(inner, mode="r+")
    assert store is inner
    assert session is None


def test_dispatch_prebuilt_store_honors_read_only(tmp_store_path):
    inner = _ZarrLocalStore(str(tmp_store_path))
    store, _ = _make_zarr_store_with_session(inner, mode="r")
    # LocalStore supports with_read_only; result should be read-only.
    assert store.read_only is True


def test_dispatch_wraps_bare_obstore_object():
    from obstore.store import MemoryStore

    ms = MemoryStore()
    store, session = _make_zarr_store_with_session(ms, mode="r+")
    assert session is None
    assert isinstance(store, _ZStore)
    assert store.read_only is False


def test_dispatch_wraps_bare_obstore_object_read_only():
    from obstore.store import MemoryStore

    ms = MemoryStore()
    store, _ = _make_zarr_store_with_session(ms, mode="r")
    assert store.read_only is True


def test_create_then_open_via_obstore_local(tmp_path):
    """End-to-end create + reopen via backend='obstore' against a local
    path.

    Note: the dispatch in ``_make_zarr_store_with_session`` always routes
    local schemes (``""`` / ``file://``) to :class:`zarr.storage.LocalStore`
    regardless of the ``backend`` kwarg.  ``backend='obstore'`` only
    selects the obstore-wrapped ``ObjectStore`` for cloud schemes; for
    local paths it's silently a no-op.  This test asserts the round-trip
    works, not the underlying store class.
    """
    from zarr_vectors.core.store import create_store

    url = f"file://{tmp_path / 'obstore_local.zarr'}"
    root = create_store(url, backend="obstore")
    assert root is not None
    assert "zarr_vectors" in root.attrs.to_dict()

    ro = open_store(url, mode="r", backend="obstore")
    assert ro._zarr.store.read_only is True
    assert "zarr_vectors" in ro.attrs.to_dict()


def test_open_store_read_mode_makes_store_read_only(tmp_path):
    """``mode='r'`` on the obstore backend must produce a read_only zarr
    store regardless of underlying scheme."""
    from zarr_vectors.core.store import create_store

    url = f"file://{tmp_path / 'ro.zarr'}"
    create_store(url, backend="obstore")

    ro = open_store(url, mode="r", backend="obstore")
    assert ro._zarr.store.read_only is True


def test_open_store_accepts_prebuilt_local_store(tmp_path):
    """Caller can construct a zarr LocalStore and pass it directly."""
    from zarr_vectors.core.store import create_store

    store_dir = tmp_path / "prebuilt.zarr"
    create_store(str(store_dir))

    inner = _ZarrLocalStore(str(store_dir))
    ro = open_store(inner, mode="r")
    # Sanity: root attrs are reachable and the store-level read_only is set.
    assert "zarr_vectors" in ro.attrs.to_dict()
    assert ro._zarr.store.read_only is True


def test_obstore_requires_zarr_v3(monkeypatch):
    """If ``zarr.storage.ObjectStore`` is unavailable, the helper raises
    a StoreError that mentions the zarr version."""
    import zarr.storage as zs

    monkeypatch.delattr(zs, "ObjectStore", raising=False)
    # Force a fresh ImportError by clearing any cached lookups.
    with pytest.raises(StoreError, match="requires zarr>=3"):
        _make_obstore_zarr_store("file:///tmp/x", mode="r")
