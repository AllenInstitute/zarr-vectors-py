"""The root ``multiscales`` block: the registered multiscales convention.

Format 0.10.0 lists the resolution levels with the multiscales convention
(https://github.com/zarr-conventions/multiscales, v0.1)::

    "multiscales": {"layout": [{"asset": "0"}, {"asset": "1"}, ...]}

Each entry names a level group and carries nothing else.  The convention's
``transform`` describes how *coordinates* change between levels, and in a
Zarr Vectors store they do not: every level's vertices are stored in
level-0 coordinates.  Bin geometry is Zarr Vectors' own and is stored
once, outside this block -- level 0's bin is the root ``base_bin_shape``,
and a coarse level's ``bin_shape`` / ``bin_ratio`` sit in its own
``zarr_vectors`` level object.

Until 0.10.0 this block was a bare-root OME-NGFF 0.4 ``multiscales`` list
whose every dataset repeated level 0's transform (``scale = 1``,
``translation = base_bin_shape / 2``).  :func:`read_level_transform` keeps
returning exactly that pair, derived from the root, so callers written
against the NGFF block are unchanged.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.core.store import (
    FsGroup,
    list_resolution_levels,
    read_root_metadata,
)


def empty_multiscales() -> dict[str, Any]:
    """The block a new store starts with: level 0 only."""
    return {"layout": [{"asset": "0"}]}


def _block(root: FsGroup) -> dict[str, Any] | None:
    ms = root.attrs.to_dict().get("multiscales")
    return dict(ms) if isinstance(ms, dict) else None


def _assets(block: dict[str, Any] | None) -> list[str]:
    if not block:
        return []
    return [
        str(e.get("asset")) for e in block.get("layout") or []
        if isinstance(e, dict) and "asset" in e
    ]


def _write_levels(root: FsGroup, levels: list[int]) -> dict[str, Any]:
    block = _block(root) or {}
    kept = {
        str(e["asset"]): dict(e) for e in block.get("layout") or []
        if isinstance(e, dict) and "asset" in e
    }
    layout = [kept.get(str(lv), {"asset": str(lv)}) for lv in sorted(set(levels))]
    block["layout"] = layout
    root.attrs.update({"multiscales": block})
    return block


def upsert_level_transform(
    root: FsGroup,
    level: int,
    *,
    scale: list[float] | None = None,
    translation: list[float] | None = None,
) -> None:
    """Ensure ``level`` is listed in the multiscales layout.

    ``scale`` and ``translation`` are accepted for compatibility and not
    stored: in format 0.10.0 every level's transform is level 0's, which
    is derived from the root's ``base_bin_shape`` (see the module
    docstring).
    """
    del scale, translation
    block = _block(root)
    if block is None:
        return
    if str(level) in _assets(block):
        return
    levels = [int(a) for a in _assets(block) if a.isdigit()]
    _write_levels(root, [*levels, int(level)])


def _level0_transform(root: FsGroup) -> tuple[list[float], list[float] | None]:
    meta = read_root_metadata(root)
    ndim = meta.sid_ndim
    base_bin = meta.effective_bin_shape
    translation = [float(b) / 2.0 for b in base_bin] if base_bin else None
    if translation is not None and not any(t != 0 for t in translation):
        translation = None
    return [1.0] * ndim, translation


def read_level_transform(
    root: FsGroup,
    level: int,
) -> tuple[list[float] | None, list[float] | None]:
    """``(scale, translation)`` for a level: level 0's transform when the
    level is listed, ``(None, None)`` when it is not."""
    if str(level) not in _assets(_block(root)):
        return None, None
    return _level0_transform(root)


def write_multiscale_metadata(root: FsGroup) -> dict[str, Any]:
    """Rewrite the layout to list exactly the store's resolution levels.

    Returns:
        The ``multiscales`` block written to the root attributes.
    """
    return _write_levels(root, list_resolution_levels(root))


def read_multiscale_metadata(root: FsGroup) -> dict[str, Any] | None:
    """The root's ``multiscales`` block, or None if it has none."""
    return _block(root)


def get_level_scale(root: FsGroup, level: int) -> list[float] | None:
    """A level's scale (level 0's, ``[1.0] * ndim``), or None if unlisted."""
    return read_level_transform(root, level)[0]


def get_level_translation(root: FsGroup, level: int) -> list[float] | None:
    """A level's translation (``base_bin_shape / 2``), or None if unlisted."""
    return read_level_transform(root, level)[1]
