"""Level 1 structural validation — verify the store layout on disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


def validate_structure(store_path: str | Path) -> ValidationResult:
    """Level 1: verify store directory layout."""
    result = ValidationResult(level=1)
    root = Path(store_path)

    if not root.exists():
        result.add_error(f"Store path does not exist: {root}")
        return result
    if not root.is_dir():
        result.add_error(f"Store path is not a directory: {root}")
        return result
    result.add_pass("Store root exists and is a directory")

    has_meta = any((root / f).exists() for f in [".zattrs", "zarr.json", "metadata.json"])
    if has_meta:
        result.add_pass("Root metadata file found")
    else:
        result.add_error("No root metadata found (expected .zattrs, zarr.json, or metadata.json)")

    # Level directories are bare integer names (``0/``, ``1/``, ...) under
    # the 0.4.1+ layout.  Anything that doesn't parse as an int is some
    # other top-level group (e.g. ``parametric/``).
    def _is_level_dir(d):
        if not d.is_dir():
            return False
        try:
            int(d.name)
            return True
        except ValueError:
            return False
    level_dirs = sorted(
        (d for d in root.iterdir() if _is_level_dir(d)),
        key=lambda d: int(d.name),
    )
    if not level_dirs:
        result.add_error("No resolution level directories found")
        return result
    result.add_pass(f"Found {len(level_dirs)} resolution level(s)")

    for level_dir in level_dirs:
        ln = level_dir.name
        vd = level_dir / "vertices"
        if vd.exists() and vd.is_dir():
            result.add_pass(f"{ln}/vertices/ exists")
        else:
            result.add_error(f"{ln}/vertices/ missing")

        vfg = level_dir / "vertex_fragments"
        if vfg.exists() and vfg.is_dir():
            result.add_pass(f"{ln}/vertex_fragments/ exists")
        else:
            result.add_warning(f"{ln}/vertex_fragments/ missing")

        if any((level_dir / f).exists() for f in [".zattrs", "zarr.json"]):
            result.add_pass(f"{ln}/ has metadata")
        else:
            result.add_warning(f"{ln}/ has no metadata file")

        for opt in ["vertex_attributes", "fragment_attributes",
                     "object_index", "object_attributes", "groups"]:
            if (level_dir / opt).exists():
                result.add_pass(f"{ln}/{opt}/ exists")
        # Multiscale link layout (0.4+): list every <delta> segment.  All
        # connectivity lives under the single ``links/`` family since 0.9.0.
        links_dir = level_dir / "links"
        if links_dir.exists():
            deltas = sorted(d.name for d in links_dir.iterdir() if d.is_dir())
            if deltas:
                result.add_pass(f"{ln}/links/ exists (deltas: {','.join(deltas)})")
            else:
                result.add_warning(f"{ln}/links/ exists but has no <delta> subdirs")

    if (root / "parametric").exists():
        result.add_pass("parametric/ group exists")

    return result
