"""HeaderRegistry — manages format-specific headers within a store.

Headers are stored under ``/headers/<format>/.zattrs`` as raw
JSON-compatible dicts.  The registry provides ``add``, ``get``,
``remove``, and ``available_formats`` for managing them.

Typed (de)serialisation of header dicts is the responsibility of the
format package; core only round-trips opaque dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zarr_vectors.core.group import Group
from zarr_vectors.core.store import open_store


class HeaderRegistry:
    """Manages format-specific headers within a zarr vectors store.

    Args:
        store_path_or_root: Either a filesystem path (str/Path) to the
            store, or an already-open :class:`Group` root handle.

    The handle check is against ``Group``, not ``FsGroup``.  ``FsGroup``
    is returned only when the backing store happens to be a
    ``LocalStore``, so testing for it meant a cloud-backed root fell
    through to ``open_store(str(root))`` -- which stringifies a Group and
    then tries to open the result as a path.
    """

    def __init__(self, store_path_or_root: str | Path | Group) -> None:
        if isinstance(store_path_or_root, Group):
            self._root = store_path_or_root
        else:
            self._root = open_store(str(store_path_or_root), mode="r+")

    def _headers_group(self, create: bool = False) -> Group:
        """Get or create the /headers/ group."""
        if create:
            return self._root.require_group("headers")
        if "headers" not in self._root:
            raise KeyError("No /headers/ group in store")
        return self._root["headers"]

    @property
    def available_formats(self) -> list[str]:
        """List of format names with stored headers."""
        try:
            hg = self._headers_group()
        except KeyError:
            return []
        return sorted(
            name for name in hg
            if not name.startswith(".")
        )

    def has(self, format_name: str) -> bool:
        """Check if a header exists for the given format."""
        return format_name in self.available_formats

    def get(self, format_name: str) -> dict[str, Any]:
        """Read a format header as a raw dict.

        Args:
            format_name: Format identifier (e.g. ``"trk"``, ``"swc"``).

        Returns:
            The JSON-compatible attributes dict for this format.

        Raises:
            KeyError: If no header exists for this format.
        """
        try:
            hg = self._headers_group()
        except KeyError:
            raise KeyError(f"No header stored for format '{format_name}'")

        if format_name not in hg:
            raise KeyError(f"No header stored for format '{format_name}'")

        fmt_group = hg[format_name]
        return fmt_group.attrs.to_dict()

    def add(self, format_name: str, header: dict[str, Any]) -> None:
        """Store a format header as a dict.

        If a header for this format already exists, it is overwritten.

        Args:
            format_name: Format identifier.
            header: JSON-compatible dict of header fields.
        """
        hg = self._headers_group(create=True)
        fmt_group = hg.require_group(format_name)
        fmt_group.attrs.update(header)

    def remove(self, format_name: str) -> None:
        """Remove a stored header.

        Args:
            format_name: Format to remove.

        Raises:
            KeyError: If no header exists for this format.
        """
        try:
            hg = self._headers_group()
        except KeyError:
            raise KeyError(f"No header stored for format '{format_name}'")

        if format_name not in hg:
            raise KeyError(f"No header stored for format '{format_name}'")

        # delete_subtree, not shutil.rmtree: ``hg.path`` raises for any
        # store that is not local, so removing a header worked on disk
        # and nowhere else.
        hg.delete_subtree(format_name)

    def __repr__(self) -> str:
        fmts = self.available_formats
        return f"HeaderRegistry(formats={fmts})"
