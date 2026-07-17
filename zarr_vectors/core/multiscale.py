"""OME-Zarr compatible multiscale metadata for zarr vectors stores.

Generates and reads ``multiscales`` metadata blocks in the root
``.zattrs``, following the OME-NGFF spec (v0.4).  This allows
OME-Zarr-aware viewers to discover the resolution pyramid structure.

NGFF v0.5 nests OME metadata under ``attributes.ome`` inside Zarr v3
``zarr.json``; ZV writes ``multiscales`` at bare root, which matches
the v0.4 layout — hence the ``version: "0.4"`` declaration.

The ZV format discriminator lives in
``multiscales[].metadata.format = "zarr_vectors"``, NOT in
``multiscales[].type`` — NGFF reserves ``type`` for the downsampling
method (``"gaussian"``, ``"nearest"``, ...).

The metadata is informational and coexists with zarr vectors-specific
metadata — it does not replace the ``zarr_vectors_level`` entries.
"""

from __future__ import annotations

from typing import Any

from zarr_vectors.core.metadata import LevelMetadata
from zarr_vectors.core.store import (
    FsGroup,
    list_resolution_levels,
    read_level_metadata,
    read_root_metadata,
)
from zarr_vectors.constants import RESOLUTION_PREFIX
from zarr_vectors.exceptions import MetadataError


def upsert_level_transform(
    root: FsGroup,
    level: int,
    *,
    scale: list[float],
    translation: list[float] | None = None,
) -> None:
    """Upsert one level's entry in the NGFF ``multiscales[0].datasets`` list.

    This is the **authoritative** writer for per-level spatial transforms
    under the 0.5+ format: ``bin_ratio`` lives as the ``scale`` factor
    and ``bin_shape / 2`` lives as the ``translation`` offset.  Callers
    in :mod:`zarr_vectors.core.store` invoke this inside
    :func:`create_resolution_level` and :func:`add_resolution_level`
    after writing the level's other attrs.

    Args:
        root: Root store group.
        level: Resolution level index (``0`` for full resolution).
        scale: Per-axis scale factor (= ``bin_ratio`` for that level).
        translation: Optional per-axis translation offset (= ``bin_shape / 2``).
            When all entries are zero, the translation transform is omitted.
    """
    attrs = root.attrs.to_dict()
    ms = attrs.get("multiscales") or []
    if not ms or not isinstance(ms, list):
        # No NGFF block yet — nothing to upsert into.  This shouldn't
        # happen for stores created by the current ``create_store`` (it
        # seeds the block at create time), but handle it gracefully.
        return
    ms_entry = dict(ms[0])
    datasets = list(ms_entry.get("datasets") or [])

    transforms: list[dict[str, Any]] = [
        {"type": "scale", "scale": [float(s) for s in scale]},
    ]
    if translation is not None and any(t != 0 for t in translation):
        transforms.append({
            "type": "translation",
            "translation": [float(t) for t in translation],
        })

    path = f"{RESOLUTION_PREFIX}{level}"
    new_entry = {"path": path, "coordinateTransformations": transforms}
    found = False
    for i, ds in enumerate(datasets):
        if ds.get("path") == path:
            datasets[i] = new_entry
            found = True
            break
    if not found:
        datasets.append(new_entry)
        # Keep datasets sorted by level for deterministic on-disk output.
        datasets.sort(key=lambda d: int(d.get("path", "0").lstrip(RESOLUTION_PREFIX) or 0))

    ms_entry["datasets"] = datasets
    attrs["multiscales"] = [ms_entry] + list(ms[1:])
    root.attrs.update(attrs)


def read_level_transform(
    root: FsGroup,
    level: int,
) -> tuple[list[float] | None, list[float] | None]:
    """Read ``(scale, translation)`` for a level from the NGFF block.

    Returns ``(None, None)`` when the level has no entry in the NGFF
    ``multiscales[0].datasets`` list — callers should fall back to the
    legacy ``zarr_vectors_level.bin_ratio`` / ``bin_shape`` fields on
    the level's own attrs.
    """
    ms = read_multiscale_metadata(root)
    if ms is None:
        return None, None
    path = f"{RESOLUTION_PREFIX}{level}"
    for ms_entry in ms:
        for ds in ms_entry.get("datasets", []):
            if ds.get("path") != path:
                continue
            scale: list[float] | None = None
            translation: list[float] | None = None
            for t in ds.get("coordinateTransformations", []):
                if t.get("type") == "scale":
                    scale = [float(s) for s in t.get("scale") or []]
                elif t.get("type") == "translation":
                    translation = [float(s) for s in t.get("translation") or []]
            return scale, translation
    return None, None


def write_multiscale_metadata(root: FsGroup) -> dict[str, Any]:
    """Generate and write OME-Zarr multiscale metadata to root .zattrs.

    Reads all existing resolution levels and their bin shapes to
    compute scale and translation transforms.

    Args:
        root: Root store group.

    Returns:
        The ``multiscales`` list written to ``.zattrs``.
    """
    meta = read_root_metadata(root)
    base_bin = meta.effective_bin_shape
    ndim = meta.sid_ndim
    levels = list_resolution_levels(root)

    # Build axes from spatial_index_dims.  ``unit`` is omitted unless
    # the source axis carries a non-empty value — NGFF requires units to
    # be UDUNITS-2 names (no placeholder strings).
    axes: list[dict[str, str]] = []
    for ax in meta.spatial_index_dims:
        out: dict[str, str] = {
            "name": ax.get("name", f"dim{len(axes)}"),
            "type": ax.get("type", "space"),
        }
        unit = ax.get("unit")
        if unit:
            out["unit"] = unit
        axes.append(out)

    # Build datasets array (one per level)
    datasets: list[dict[str, Any]] = []
    for lvl in levels:
        path = f"{RESOLUTION_PREFIX}{lvl}"

        if lvl == 0:
            # Level 0: scale = 1.0, translation = base_bin/2
            scale = [1.0] * ndim
            translation = [bs / 2.0 for bs in base_bin]
        else:
            try:
                lm = read_level_metadata(root, lvl)
                if lm.bin_ratio is not None:
                    scale = [float(r) for r in lm.bin_ratio]
                elif lm.bin_shape is not None:
                    # Plain float ratio, not compute_bin_ratio: NGFF scale
                    # has no integer requirement, unlike the separately-
                    # typed `bin_ratio: tuple[int, ...]` field — a
                    # fractional cumulative coarsen_factor must not raise
                    # here. See the identical fallback in
                    # store.create_resolution_level.
                    scale = [
                        float(bs) / float(bb)
                        for bs, bb in zip(lm.bin_shape, base_bin)
                    ]
                else:
                    scale = [1.0] * ndim

                if lm.bin_shape is not None:
                    translation = [bs / 2.0 for bs in lm.bin_shape]
                else:
                    translation = [bs / 2.0 for bs in base_bin]
            except Exception:
                scale = [1.0] * ndim
                translation = [bs / 2.0 for bs in base_bin]

        transforms: list[dict[str, Any]] = [
            {"type": "scale", "scale": scale},
        ]
        if any(t != 0 for t in translation):
            transforms.append({"type": "translation", "translation": translation})

        datasets.append({
            "path": path,
            "coordinateTransformations": transforms,
        })

    multiscales = [{
        "version": "0.4",
        "name": "default",
        "axes": axes,
        "datasets": datasets,
        # NGFF reserves ``type`` for the downsampling method
        # (``"gaussian"``, ``"nearest"``, ...).  Stash the ZV format
        # discriminator under ``metadata.format`` instead.
        "metadata": {"format": "zarr_vectors"},
    }]

    # Write to root attrs (alongside existing zarr_vectors metadata)
    attrs = root.attrs.to_dict()
    attrs["multiscales"] = multiscales
    root.attrs.update(attrs)

    return multiscales


def read_multiscale_metadata(root: FsGroup) -> list[dict[str, Any]] | None:
    """Read OME-Zarr multiscale metadata from root .zattrs.

    Args:
        root: Root store group.

    Returns:
        The ``multiscales`` list, or None if not present.
    """
    attrs = root.attrs.to_dict()
    return attrs.get("multiscales")


def get_level_scale(
    root: FsGroup,
    level: int,
) -> list[float] | None:
    """Get the scale factors for a specific level from multiscale metadata.

    Args:
        root: Root store group.
        level: Resolution level index.

    Returns:
        List of scale factors per dimension, or None if not available.
    """
    ms = read_multiscale_metadata(root)
    if ms is None:
        return None

    path = f"{RESOLUTION_PREFIX}{level}"
    for ms_entry in ms:
        for ds in ms_entry.get("datasets", []):
            if ds.get("path") == path:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale":
                        return t.get("scale")
    return None


def get_level_translation(
    root: FsGroup,
    level: int,
) -> list[float] | None:
    """Get the translation offset for a specific level.

    Args:
        root: Root store group.
        level: Resolution level index.

    Returns:
        List of translation offsets per dimension, or None.
    """
    ms = read_multiscale_metadata(root)
    if ms is None:
        return None

    path = f"{RESOLUTION_PREFIX}{level}"
    for ms_entry in ms:
        for ds in ms_entry.get("datasets", []):
            if ds.get("path") == path:
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "translation":
                        return t.get("translation")
    return None
