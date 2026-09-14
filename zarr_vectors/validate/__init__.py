"""Unified validation — runs levels 1–5 cumulatively."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from zarr_vectors.validate.conformance import validate_conformance, validate_multiresolution
from zarr_vectors.validate.consistency import validate_consistency
from zarr_vectors.validate.metadata import validate_metadata
from zarr_vectors.validate.structure import ValidationResult, validate_structure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group


def validate(store_path: str | Path | Group, *, level: int = 3) -> ValidationResult:
    """Validate a zarr vectors store at the specified conformance level (1–5)."""
    if level < 1 or level > 5:
        raise ValueError(f"Conformance level must be 1–5, got {level}")

    combined = ValidationResult(level=level)

    r1 = validate_structure(store_path)
    combined.merge(r1)
    if not r1.ok:
        return combined

    if level >= 2:
        r2 = validate_metadata(store_path)
        combined.merge(r2)
        if not r2.ok and level >= 3:
            return combined

    if level >= 3:
        combined.merge(validate_consistency(store_path))

    if level >= 4:
        combined.merge(validate_conformance(store_path))

    if level >= 5:
        combined.merge(validate_multiresolution(store_path))

    return combined
