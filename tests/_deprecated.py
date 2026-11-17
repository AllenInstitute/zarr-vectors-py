"""Calls into the deprecated lazy layer, asserting their DeprecationWarning.

pyproject.toml turns DeprecationWarnings raised from zarr_vectors into
errors, so new code cannot drift onto a deprecated API unnoticed. The tests
that exist to cover the lazy layer itself (``open_zv`` / ``ZVStore`` /
``ZVLevel`` / ``ZVWriter``) go through these helpers instead.
"""

from __future__ import annotations

from typing import Any

import pytest

from zarr_vectors.lazy.store import open_zv as _open_zv


def open_zv(*args: Any, **kw: Any) -> Any:
    """``zarr_vectors.lazy.open_zv``, asserting its DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="open_zv"):
        return _open_zv(*args, **kw)


def zv_writer(level: Any) -> Any:
    """``ZVLevel.writer()``, asserting the ZVWriter DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ZVWriter"):
        return level.writer()
