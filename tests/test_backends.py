"""Tests for the storage backend selection + Zarr-store construction.

Covers:

* URL-scheme detection
* Backend resolution precedence (explicit kwarg / env var / auto)
* Thin native store builders (obstore / fsspec) + ``storage_options``
* Helpful error when an optional backend is missing
* ``rebind()`` semantics
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zarr_vectors.core.backends import (
    SCHEMES_LOCAL,
    SCHEMES_OBJECT_STORE,
    detect_scheme,
    resolve_backend_name,
)
from zarr_vectors.core.group import Group
from zarr_vectors.core.store import (
    _make_zarr_store_with_session,
    create_store,
    open_store,
    rebind,
)
from zarr_vectors.exceptions import StoreError


# ===================================================================
# URL scheme detection
# ===================================================================


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/abs/path/store.zv", ""),
        ("relative/path", ""),
        (r"C:\Users\me\store.zv", ""),   # bare Windows drive — not a scheme
        ("file:///C:/Users/me/store.zv", "file"),
        ("file:///tmp/store.zv", "file"),
        ("s3://bucket/path", "s3"),
        ("gs://bucket/path", "gs"),
        ("gcs://bucket/path", "gcs"),
        ("az://container/path", "az"),
        ("azure://container/path", "azure"),
        ("abfs://container/path", "abfs"),
        ("http://host/path", "http"),
        ("https://host/path", "https"),
    ],
)
def test_detect_scheme(url, expected):
    assert detect_scheme(url) == expected


def test_detect_scheme_path_object(tmp_path):
    assert detect_scheme(tmp_path) == ""


def test_scheme_categories_disjoint():
    assert SCHEMES_LOCAL.isdisjoint(SCHEMES_OBJECT_STORE)


# ===================================================================
# Backend resolution precedence
# ===================================================================


def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setenv("ZARR_VECTORS_BACKEND", "obstore")
    # Explicit kwarg overrides env.
    assert resolve_backend_name("/local/path", explicit="local") == "local"


def test_resolve_env_var_wins_over_auto(monkeypatch):
    monkeypatch.setenv("ZARR_VECTORS_BACKEND", "fsspec")
    assert resolve_backend_name("/local/path") == "fsspec"


def test_resolve_auto_local_for_filesystem_path():
    assert resolve_backend_name("/some/path", env_override="") == "local"
    assert resolve_backend_name("file:///tmp/x", env_override="") == "local"


def test_resolve_cloud_without_extras_raises(monkeypatch):
    """s3:// URL with neither obstore nor fsspec installed must error helpfully."""
    monkeypatch.setitem(sys.modules, "obstore", None)
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(StoreError, match="requires a cloud backend"):
        resolve_backend_name("s3://bucket/key", env_override="")


def test_cloud_url_routes_to_obstore_when_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "fsspec", None)
    monkeypatch.setitem(sys.modules, "obstore", type(sys)("obstore"))
    assert resolve_backend_name("s3://bucket/path", env_override="") == "obstore"


def test_cloud_url_falls_back_to_fsspec_when_obstore_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "obstore", None)
    monkeypatch.setitem(sys.modules, "fsspec", type(sys)("fsspec"))
    assert resolve_backend_name("s3://bucket/path", env_override="") == "fsspec"


# ===================================================================
# Native store builders (obstore / fsspec) + storage_options
# ===================================================================


def _make_group(store):
    import zarr
    return zarr.open_group(store, mode="w")


def test_obstore_builder_returns_working_zarr_store():
    """`backend="obstore"` on a memory:// URL yields a usable zarr Store."""
    pytest.importorskip("obstore")
    from zarr.abc.store import Store

    store, session = _make_zarr_store_with_session(
        "memory:///x", backend="obstore", mode="w",
    )
    assert isinstance(store, Store)
    assert session is None
    g = _make_group(store)
    g.attrs["hello"] = "world"
    assert dict(g.attrs)["hello"] == "world"


def test_fsspec_builder_returns_working_zarr_store():
    """`backend="fsspec"` on a memory:// URL yields a usable zarr Store."""
    pytest.importorskip("fsspec")
    from zarr.abc.store import Store

    store, session = _make_zarr_store_with_session(
        "memory:///y", backend="fsspec", mode="w",
    )
    assert isinstance(store, Store)
    assert session is None
    g = _make_group(store)
    g.attrs["k"] = 1
    assert dict(g.attrs)["k"] == 1


def test_storage_options_merged_and_forwarded(monkeypatch):
    """`storage_options=` and loose `**backend_kwargs` merge and reach the
    builder verbatim."""
    import zarr_vectors.core.store as store_mod

    seen: dict = {}

    def _spy(url, *, mode="r+", storage_options=None):
        seen.clear()
        seen.update(storage_options or {})
        return ("DUMMY_STORE", None)

    monkeypatch.setattr(store_mod, "_make_fsspec_zarr_store", _spy)
    store, session = _make_zarr_store_with_session(
        "s3://bucket/key", backend="fsspec", mode="w",
        storage_options={"key": "K", "anon": True}, secret="S",
    )
    assert store == "DUMMY_STORE" and session is None
    assert seen == {"key": "K", "anon": True, "secret": "S"}


def test_obstore_missing_dep_message(monkeypatch):
    """Explicit obstore backend when not installed → helpful StoreError."""
    monkeypatch.setitem(sys.modules, "obstore", None)
    with pytest.raises(StoreError, match="obstore is not installed"):
        _make_zarr_store_with_session("s3://bucket/x", backend="obstore", mode="w")


def test_explicit_backend_honored_on_local_path(monkeypatch):
    """An explicit backend= is honored even for a local/file path (previously
    the local short-circuit silently ignored it)."""
    pytest.importorskip("fsspec")
    from zarr.storage import FsspecStore

    store, _ = _make_zarr_store_with_session(
        str("memory:///z"), backend="fsspec", mode="w",
    )
    assert isinstance(store, FsspecStore)


# ===================================================================
# Group smoke tests (local store)
# ===================================================================


def test_group_create_subgroup(tmp_path):
    from zarr_vectors.core.store import FsGroup
    root = FsGroup(tmp_path, create=True)
    child = root.create_group("child")
    assert "child" in root
    assert isinstance(child, Group)
    assert child.prefix == "child"


def test_group_url_property(tmp_path):
    from zarr_vectors.core.store import FsGroup
    root = FsGroup(tmp_path, create=True)
    child = root.create_group("a").create_group("b")
    assert child.url.endswith("/a/b")


def test_group_path_only_for_local(tmp_path):
    from zarr_vectors.core.store import FsGroup
    root = FsGroup(tmp_path, create=True)
    assert isinstance(root.path, Path)


# ===================================================================
# rebind + create_store / open_store backend= routing
# ===================================================================


def _minimal_root_kwargs():
    return dict(
        axes=[
            {"name": "x", "type": "space", "unit": "unit"},
            {"name": "y", "type": "space", "unit": "unit"},
            {"name": "z", "type": "space", "unit": "unit"},
        ],
        chunk_shape=(100.0, 100.0, 100.0),
        bounds=([0, 0, 0], [100, 100, 100]),
        geometry_types=["point_cloud"],
    )


def test_rebind_swap_local_for_local(tmp_path):
    """Same-URL rebind preserves the URL and keeps handles resolving."""
    store_path = tmp_path / "test.zv"
    root = create_store(store_path, **_minimal_root_kwargs())
    original_url = root.url

    rebind(root, "local")
    assert root.url == original_url

    reopened = open_store(store_path)
    assert reopened.attrs["zarr_vectors"]["zv_version"]


def test_rebind_rejects_non_backend_object(tmp_path):
    """rebind takes a backend name or a zarr Store — anything else errors."""
    store_path = tmp_path / "test.zv"
    root = create_store(store_path, **_minimal_root_kwargs())
    with pytest.raises(StoreError, match="backend name string or a"):
        rebind(root, object())


def test_create_store_with_explicit_local_backend(tmp_path):
    root = create_store(tmp_path / "x.zv", **_minimal_root_kwargs(), backend="local")
    assert "zarr_vectors" in root.attrs


def test_open_store_with_explicit_local_backend(tmp_path):
    p = tmp_path / "x.zv"
    create_store(p, **_minimal_root_kwargs())
    root = open_store(p, backend="local")
    assert "zarr_vectors" in root.attrs


def test_create_store_accepts_storage_options_local(tmp_path):
    """storage_options is accepted (and harmlessly ignored) for local stores."""
    root = create_store(
        tmp_path / "so.zv", **_minimal_root_kwargs(),
        storage_options={"ignored": True},
    )
    assert "zarr_vectors" in root.attrs
