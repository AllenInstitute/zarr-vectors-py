"""Pyramid-refresh hook (dependency-injection slot).

Re-coarsening a pyramid after edits is *multi-scale coordination*, which lives in
``zarr-vectors-tools`` — not in this core data-access SDK.  But
:class:`zarr_vectors.ops.edit.EditSession` still needs to trigger it when the
caller asks for ``refresh_pyramid``.

So core exposes this small registry slot.  ``zarr-vectors-tools`` registers its
``rebuild_pyramid_from_level`` implementation on import (mirroring the coarsener
registry).  If a refresh is requested without a refresher registered (e.g.
core installed without tools), :class:`EditSession` raises a clear error.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

# Signature: (root, source_level) -> Any
PyramidRefresher = Callable[[Any, int], Any]

_PYRAMID_REFRESHER: Optional[PyramidRefresher] = None


def register_pyramid_refresher(fn: PyramidRefresher) -> None:
    """Register the pyramid-refresh implementation (called by zarr-vectors-tools)."""
    global _PYRAMID_REFRESHER
    _PYRAMID_REFRESHER = fn


def get_pyramid_refresher() -> Optional[PyramidRefresher]:
    """Return the registered pyramid refresher, or ``None`` if not installed."""
    return _PYRAMID_REFRESHER
