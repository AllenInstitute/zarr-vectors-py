"""Level 1 structural validation — verify the store layout on disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from zarr_vectors.constants import (
    FRAGMENT_ATTRIBUTES,
    GROUPS,
    LINKS,
    OBJECT_ATTRIBUTES,
    OBJECT_INDEX,
    PARAMETRIC_GROUP,
    VERTEX_ATTRIBUTES,
    VERTEX_FRAGMENTS,
    VERTICES,
)
from zarr_vectors.core.paths import format_delta

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group


@dataclass
class ValidationResult:
    """Accumulated validation outcome."""

    level: int
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add_pass(self, msg: str) -> None:
        self.passed.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def merge(self, other: "ValidationResult") -> None:
        self.passed.extend(other.passed)
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        parts = [
            f"Level {self.level} validation: {status}",
            f"  {len(self.passed)} passed, {len(self.warnings)} warnings, {len(self.errors)} errors",
        ]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts)


def validate_structure(store_path: str | Path | Group) -> ValidationResult:
    """Level 1: verify the store's structure.

    Asked of the store, not of a filesystem.  This used to be pure
    ``pathlib`` -- ``Path(store_path).exists()`` and ``iterdir()`` -- which
    made level 1 the one validator that could only ever see a local
    directory.  Worse, it could not see the local ones either when reached
    through the api: ``Dataset.validate()`` passes ``Group.url``, and
    ``Path("file:///C:/...")`` does not exist, so every api-driven
    validation failed at level 1 and returned before doing anything else.

    Accepts a path, a URL or an open :class:`~zarr_vectors.core.group.Group`.
    """
    result = ValidationResult(level=1)

    from zarr_vectors.core.arrays import list_link_deltas
    from zarr_vectors.core.store import (
        list_resolution_levels,
        open_store,
    )

    # ``require_zv=False``: reporting a missing root marker is this
    # validator's job, so it must not be pre-empted by open_store raising
    # on the same condition.
    try:
        root = open_store(store_path, require_zv=False)
    except Exception as e:
        result.add_error(f"Cannot open store: {e}")
        return result
    result.add_pass("Store root opened")

    if "zarr_vectors" in root.attrs:
        result.add_pass("Root metadata file found")
    else:
        result.add_error(
            "No root metadata found (expected a 'zarr_vectors' block in the "
            "root attributes)"
        )

    # Resolution levels are bare integer group names (``0``, ``1``, ...)
    # under the 0.4.1+ layout; anything else at the root is some other
    # entity (``parametric``, ``headers``).
    levels = list_resolution_levels(root)
    if not levels:
        result.add_error("No resolution level directories found")
        return result
    result.add_pass(f"Found {len(levels)} resolution level(s)")

    for lv in levels:
        ln = str(lv)
        try:
            level = root[ln]
        except Exception as e:
            result.add_error(f"{ln}/ cannot be opened: {e}")
            continue

        if level.array_exists(VERTICES):
            result.add_pass(f"{ln}/vertices/ exists")
        else:
            result.add_error(f"{ln}/vertices/ missing")

        if level.array_exists(VERTEX_FRAGMENTS):
            result.add_pass(f"{ln}/vertex_fragments/ exists")
        else:
            result.add_warning(f"{ln}/vertex_fragments/ missing")

        if level.attrs.to_dict():
            result.add_pass(f"{ln}/ has metadata")
        else:
            result.add_warning(f"{ln}/ has no metadata file")

        for opt in [VERTEX_ATTRIBUTES, FRAGMENT_ATTRIBUTES,
                    OBJECT_INDEX, OBJECT_ATTRIBUTES, GROUPS]:
            if level.array_exists(opt):
                result.add_pass(f"{ln}/{opt}/ exists")

        # Multiscale link layout (0.4+): list every <delta> segment.  All
        # connectivity lives under the single ``links/`` family since 0.9.0.
        if level.array_exists(LINKS):
            deltas = list_link_deltas(level)
            if deltas:
                names = ",".join(format_delta(d) for d in deltas)
                result.add_pass(f"{ln}/links/ exists (deltas: {names})")
            else:
                result.add_warning(f"{ln}/links/ exists but has no <delta> subdirs")

    if root.array_exists(PARAMETRIC_GROUP):
        result.add_pass("parametric/ group exists")

    return result
