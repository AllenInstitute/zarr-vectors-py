"""N-dimensional rechunking for zarr vectors stores.

Rechunking reorganises data so that objects sharing a common
dimension value (group, attribute bin, object ID range) are
physically contiguous on disk. This turns O(N) manifest scans
into O(1) prefix-based reads for that dimension.

Usage::

    from zarr_vectors.rechunk import rechunk, RechunkSpec

    rechunk("tracts.zarrvectors", RechunkSpec(by="group"))

    # Categorical attribute-based rechunking (one chunk per unique value):
    from zarr_vectors.rechunk import rechunk_by_attribute
    rechunk_by_attribute("cells.zarrvectors", "gene")
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.rechunk.engine import rechunk
from zarr_vectors.rechunk.rebin import rebin_level
from zarr_vectors.rechunk.spec import RechunkSpec

__all__ = [
    "rechunk_spatial",
    "suggest_chunk_shape","RechunkSpec", "rebin_level", "rechunk", "rechunk_by_attribute"]


def rechunk_by_attribute(
    store_path: str,
    attribute_name: str,
    *,
    output: str | None = None,
    spatial_chunk_shape: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Rechunk a store so that one chunk == one attribute value.

    Categorical only — every unique value of the named per-object
    attribute becomes its own bin, regardless of how many unique values
    there are.  Resulting chunk keys gain a leading dim:
    ``(attr_bin, z, y, x)``.

    Args:
        store_path: Source store path or URL.
        attribute_name: Name of a per-object attribute already present
            on the source store (under ``object_attributes/<name>``).
        output: Output path; if ``None``, rechunks in place.
        spatial_chunk_shape: Optional new spatial chunk shape for the
            output store.

    Returns:
        The summary dict produced by :func:`rechunk`.
    """
    spec = RechunkSpec(
        by=f"attribute:{attribute_name}",
        categorical=True,
        spatial_chunk_shape=spatial_chunk_shape,
        prefix_dim_name=attribute_name,
    )
    return rechunk(store_path, spec, output=output)

from zarr_vectors.rechunk.spatial import rechunk_spatial, suggest_chunk_shape  # noqa: E402,F401
