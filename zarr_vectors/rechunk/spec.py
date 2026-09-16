"""Rechunk specification and dimension mapping.

``RechunkSpec`` describes how to rechunk a store: which dimension to
chunk by (group, object_id, attribute) and optional bin edges for
continuous values.

``DimensionMapper`` resolves each object to a rechunk bin index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class RechunkSpec:
    """Specification for rechunking a store along a non-spatial dimension.

    Args:
        by: Dimension to rechunk by. One of:
            - ``"group"`` — chunk by group membership
            - ``"object_id"`` — chunk by object ID ranges
            - ``"attribute:<name>"`` — chunk by attribute value bins
            - ``"spatial"`` — re-spatial-chunk only (change chunk_shape)
        bins: Explicit bin edges for continuous values. For
            ``by="object_id"``, these are ID boundaries. For
            ``by="attribute:length"``, these are value boundaries.
            If None *and* ``categorical`` is False, the legacy
            quartile-based auto-binning is used for attributes with >10
            unique values.
        spatial_chunk_shape: Override spatial chunk shape for the output
            store. If None, keeps the source chunk shape.
        prefix_dim_name: Custom name for the prefix dimension in
            metadata. Defaults to the ``by`` value.
        categorical: If True, treat the binning dimension as categorical
            — every unique value gets its own bin regardless of how many
            unique values there are.  Set this for gene labels, bundle
            labels, etc.  Ignored when ``bins`` is also set.
    """

    by: str
    bins: list[float] | None = None
    spatial_chunk_shape: tuple[float, ...] | None = None
    prefix_dim_name: str | None = None
    categorical: bool = False

    @property
    def dimension_name(self) -> str:
        return self.prefix_dim_name or self.by.split(":")[0]


class DimensionMapper:
    """Maps each object to a rechunk bin index.

    Args:
        spec: The rechunk specification.
    """

    def __init__(self, spec: RechunkSpec) -> None:
        self.spec = spec

    def map_objects(
        self,
        *,
        n_objects: int,
        groupings: list[list[int]] | None = None,
        object_attributes: dict[str, npt.NDArray] | None = None,
    ) -> dict[int, int]:
        """Assign each object to a rechunk bin.

        Returns:
            ``{object_id: bin_index}`` mapping.
        """
        by = self.spec.by

        if by == "group":
            return self._map_by_group(n_objects, groupings)
        elif by == "object_id":
            return self._map_by_object_id(n_objects)
        elif by.startswith("attribute:"):
            attr_name = by.split(":", 1)[1]
            if object_attributes is None or attr_name not in object_attributes:
                raise ValueError(
                    f"Attribute '{attr_name}' not found in object attributes"
                )
            return self._map_by_attribute(
                n_objects, object_attributes[attr_name],
            )
        elif by == "spatial":
            # All objects in bin 0 (no prefix dimension)
            return {oid: 0 for oid in range(n_objects)}
        else:
            raise ValueError(f"Unknown rechunk dimension: '{by}'")

    def _map_by_group(
        self,
        n_objects: int,
        groupings: list[list[int]] | None,
    ) -> dict[int, int]:
        """Assign objects to bins by group membership."""
        if groupings is None:
            # No groupings → all in bin 0
            return {oid: 0 for oid in range(n_objects)}

        mapping: dict[int, int] = {}
        for group_idx, members in enumerate(groupings):
            for oid in members:
                mapping[oid] = group_idx

        # Objects not in any group → bin -1 (ungrouped)
        for oid in range(n_objects):
            if oid not in mapping:
                mapping[oid] = -1

        return mapping

    def _map_by_object_id(self, n_objects: int) -> dict[int, int]:
        """Assign objects to bins by ID ranges."""
        if self.spec.bins is None:
            # Each object is its own bin
            return {oid: oid for oid in range(n_objects)}

        edges = sorted(self.spec.bins)
        mapping: dict[int, int] = {}
        for oid in range(n_objects):
            bin_idx = int(np.searchsorted(edges, oid, side="right")) - 1
            bin_idx = max(0, min(bin_idx, len(edges) - 1))
            mapping[oid] = bin_idx

        return mapping

    def _map_by_attribute(
        self,
        n_objects: int,
        values: npt.NDArray,
    ) -> dict[int, int]:
        """Assign objects to bins by attribute value."""
        if self.spec.bins is None:
            unique_vals = np.unique(values)
            # Categorical path: one bin per unique value, no quartile
            # fallback.  Also used implicitly when there are few enough
            # unique values that quartile-binning would be silly.
            if self.spec.categorical or len(unique_vals) <= 10:
                val_to_bin = {_hashable(v): i for i, v in enumerate(unique_vals)}
                return {
                    oid: val_to_bin[_hashable(values[oid])]
                    for oid in range(n_objects)
                }
            # Continuous fallback: auto-bin to quartiles.
            q = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
            edges = np.unique(q)
            if len(edges) < 2:
                return {oid: 0 for oid in range(n_objects)}
            indices = np.searchsorted(edges[1:], values, side="right")
            indices = np.clip(indices, 0, len(edges) - 2)
            return {oid: int(indices[oid]) for oid in range(n_objects)}

        edges = np.array(sorted(self.spec.bins), dtype=np.float64)
        indices = np.searchsorted(edges, values, side="right") - 1
        indices = np.clip(indices, 0, len(edges) - 1)
        return {oid: int(indices[oid]) for oid in range(n_objects)}

    @property
    def n_bins(self) -> int | None:
        """Number of bins (if determinable from spec alone)."""
        if self.spec.bins is not None:
            return len(self.spec.bins)
        return None


def _hashable(v: Any) -> Any:
    """Reduce a numpy scalar to a Python value usable as a dict key."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v
