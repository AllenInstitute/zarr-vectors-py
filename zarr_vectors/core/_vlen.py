"""Shared vlen-bytes cell decoding.

The single place the zarr 3.x slice-then-extract rule lives, so the sync,
async, and manifest cell readers cannot drift.  The hazard this guards:
zarr 3.x scalar-indexing a vlen-bytes array (``arr[i]``) returns a 0-d
object ndarray whose ``bytes()`` is the array *header*, not the payload.
The fix is to slice a one-cell region and pull the element out of it.

This module deliberately imports nothing from ``zarr_vectors`` — it is a
leaf, so ``core.group`` and ``core._batch_reader`` (which keeps no other
package imports to avoid an import cycle) can both import it directly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def cell_region(coords: Sequence[int]) -> tuple[slice, ...]:
    """The ``(slice(c, c+1), …)`` region selecting exactly one cell.

    Never index a vlen array with a scalar (``arr[i]``) — see the module
    docstring.  Use this region so the read stays a slice.
    """
    return tuple(slice(int(c), int(c) + 1) for c in coords)


def region_to_bytes(region: Any) -> bytes:
    """Extract the single vlen payload from a one-cell region.

    ``region`` is the result of indexing a vlen array with
    :func:`cell_region` (or any 1-element slice, e.g. ``arr[i:i+1]``).  An
    empty region, or a fill value (``None``), decodes to ``b""``.
    """
    arr = np.asarray(region)
    if arr.size == 0:
        return b""
    val = arr.flat[0]
    return b"" if val is None else bytes(val)
