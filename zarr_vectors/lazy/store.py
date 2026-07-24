"""ZVStore — lazy wrapper around an open zarr vectors store.

No data is read until ``.compute()`` is called on a collection.
Metadata (root attrs, level attrs, chunk listings) is loaded on
first access and cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.storage import StoreLike

from zarr_vectors.core.group import Group
from zarr_vectors.core.store import (
    FsGroup,
    list_resolution_levels,
    open_store,
    read_root_metadata,
    read_level_metadata,
    rebind,
)
from zarr_vectors.core.metadata import RootMetadata, LevelMetadata
from zarr_vectors.lazy.level import ZVLevel


class ZVStore:
    """Lazy handle to a zarr vectors store.

    Attributes are read from ``.zattrs`` on first access and cached.
    Resolution levels are accessed by integer index via ``__getitem__``.

    Args:
        root: An open :class:`FsGroup` for the store root.
        meta: Parsed :class:`RootMetadata`.
    """

    def __init__(self, root: Group, meta: RootMetadata) -> None:
        self._root = root
        self._meta = meta
        self._levels_cache: dict[int, ZVLevel] = {}
        self._level_list: list[int] | None = None

    # ---------------------------------------------------------------
    # Metadata properties (no data I/O)
    # ---------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Filesystem path of the store (local ``LocalStore`` only)."""
        return self._root.path

    @property
    def url(self) -> str:
        """Canonical URL of the store root.  Portable across backends."""
        return self._root.url

    @property
    def chunk_shape(self) -> tuple[float, ...]:
        return self._meta.chunk_shape

    @property
    def bin_shape(self) -> tuple[float, ...]:
        """Effective bin shape (base_bin_shape or chunk_shape)."""
        return self._meta.effective_bin_shape

    @property
    def base_bin_shape(self) -> tuple[float, ...] | None:
        return self._meta.base_bin_shape

    @property
    def bins_per_chunk(self) -> tuple[int, ...]:
        return self._meta.bins_per_chunk

    @property
    def geometry_types(self) -> list[str]:
        return self._meta.geometry_types

    @property
    def bounds(self) -> tuple[list[float], list[float]]:
        return self._meta.bounds

    @property
    def ndim(self) -> int:
        return self._meta.sid_ndim

    @property
    def levels(self) -> list[int]:
        """Sorted list of available resolution level indices."""
        if self._level_list is None:
            self._level_list = list_resolution_levels(self._root)
        return self._level_list

    @property
    def zv_version(self) -> str:
        return self._meta.zv_version

    @property
    def headers(self) -> dict[str, dict[str, Any]]:
        """Dict of stored format headers, keyed by format name.

        Returns:
            ``{format_name: header_dict}`` for each stored header.
            Empty dict if no headers are stored.  Typed deserialisation
            of header dicts is handled by the format package.
        """
        from zarr_vectors.headers.registry import HeaderRegistry
        try:
            reg = HeaderRegistry(self._root)
            return {fmt: reg.get(fmt) for fmt in reg.available_formats}
        except Exception:
            return {}

    # ---------------------------------------------------------------
    # Level access
    # ---------------------------------------------------------------

    def __getitem__(self, level: int) -> ZVLevel:
        """Get a lazy handle to a resolution level."""
        if level not in self._levels_cache:
            from zarr_vectors.core.store import get_resolution_level
            level_group = get_resolution_level(self._root, level)
            try:
                level_meta = read_level_metadata(self._root, level)
            except Exception:
                level_meta = None
            self._levels_cache[level] = ZVLevel(
                level_group, level, self._meta, level_meta,
            )
        return self._levels_cache[level]

    # ---------------------------------------------------------------
    # Repr
    # ---------------------------------------------------------------

    def __repr__(self) -> str:
        types = ", ".join(self.geometry_types)
        return (
            f"ZVStore('{self._root.url}', "
            f"levels={self.levels}, "
            f"geometry=[{types}], "
            f"chunk={self.chunk_shape}, "
            f"bin={self.bin_shape})"
        )

    # ---------------------------------------------------------------
    # Backend rebinding
    # ---------------------------------------------------------------

    def set_backend(
        self,
        backend: str,
        *,
        storage_options: dict[str, Any] | None = None,
        **backend_kwargs: Any,
    ) -> None:
        """Swap the underlying storage backend in place (no data movement).

        Useful for switching driver (e.g. ``"fsspec"`` → ``"obstore"``)
        or credentials on a store you already have open.  Any cached
        level / array handles are invalidated and will be rebuilt on the
        next access using the new backend.

        Args:
            backend: Backend name (``"local"`` / ``"obstore"`` /
                ``"fsspec"`` / ``"icechunk"``).  The URL is unchanged; only
                the driver/credentials are re-bound.
            storage_options: Credentials / options forwarded to the store
                constructor (e.g. ``key`` / ``secret`` / ``anon``).
            **backend_kwargs: Additional options merged into
                ``storage_options`` (back-compat with loose kwargs).
        """
        rebind(
            self._root, backend,
            storage_options=storage_options, **backend_kwargs,
        )
        self._levels_cache.clear()

    def object_levels(self, oid: int) -> list[int]:
        """Monotonically-increasing list of levels at which ``oid`` is
        present.

        For an ID-preserving pyramid, this is the set of LODs a viewer
        can pick for the object — the object's OID is stable across
        levels and the object is present at level $L$ iff its manifest
        at level $L$ is non-empty.

        Returns ``[]`` if the object is absent from every level.
        """
        out: list[int] = []
        for L in self.levels:
            try:
                if self[L].has_object(oid):
                    out.append(L)
            except Exception:
                continue
        return out


def open_zv(
    path: StoreLike,
    *,
    backend: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **backend_kwargs: Any,
) -> ZVStore:
    """Open a zarr vectors store lazily.

    Reads only root metadata (a few KB).  No vertex data is loaded.

    Args:
        path: URL, filesystem path, or a pre-built ``zarr.abc.store.Store``
            (or any other ``StoreLike``).  A Store instance is passed
            through to :func:`open_store` untouched, which is the only way
            to drive the lazy API from an environment that cannot
            construct a store from a URL — a browser fetch-backed Store
            under Pyodide, say.
        backend: Force a backend (``"local"`` / ``"obstore"`` /
            ``"fsspec"`` / ``"icechunk"``).  Auto-detected from the URL
            scheme by default.
        storage_options: Credentials / options forwarded to the store
            constructor (e.g. ``key`` / ``secret`` / ``anon``).
        **backend_kwargs: Additional options merged into
            ``storage_options``.

    Returns:
        A :class:`ZVStore` handle for lazy access.
    """
    # mode="r+" so the writer() handles returned by ZVLevel / ZVStore
    # can mutate without an extra reopen.  Pure readers pay no cost for
    # this — the actual reads still touch only the chunks they need.
    root = open_store(
        path, mode="r+", backend=backend,
        storage_options=storage_options, **backend_kwargs,
    )
    meta = read_root_metadata(root)
    return ZVStore(root, meta)


