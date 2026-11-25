"""In-place migration of a format 0.9.x store's metadata to 0.10.0.

Format 0.10.0 changes where metadata lives, not what any cell holds (see
:data:`zarr_vectors.constants.FORMAT_VERSION`), so a 0.9.x store is
migrated by rewriting its ``zarr.json`` documents.  No chunk is read or
written.

The root is rewritten last.  A migration interrupted part-way therefore
leaves a store this package still refuses as 0.9.x, and running the
migration again finishes it: every step skips a node that is already in
the 0.10.0 layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import zarr

from zarr_vectors.constants import FORMAT_VERSION
from zarr_vectors.core import convention as zvc

__all__ = ["migrate_metadata_0_10"]

_LEGACY_LEVEL_KEY = "zarr_vectors_level"
_LEGACY_PRESENCE_KEY = "zarr_vectors_presence"


def migrate_metadata_0_10(
    store: str | Path | zarr.Group,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Rewrite a 0.9.x store's metadata in the 0.10.0 layout, in place.

    Args:
        store: Path or URL of the store, or its open root group.
        dry_run: Count what would change without writing anything.

    Returns:
        Counts of the nodes rewritten, by role: ``root``, ``levels``,
        ``arrays`` and ``other``.  All zero for a store already at 0.10.0.

    Raises:
        ValueError: If ``store`` is not a Zarr Vectors store.
    """
    root = store if isinstance(store, zarr.Group) else zarr.open_group(
        str(store), mode="r" if dry_run else "r+", zarr_format=3,
    )
    root_attrs = dict(root.attrs)
    zv = root_attrs.get(zvc.ZV_KEY)
    if not isinstance(zv, dict):
        raise ValueError(f"{store!r} is not a Zarr Vectors store (no root 'zarr_vectors')")
    counts = {"root": 0, "levels": 0, "arrays": 0, "other": 0}
    if zv.get("kind") == "root":
        return counts

    # Coarse levels whose bins live only in the old NGFF entry (stores
    # older than the per-level bin fields) get them copied into their own
    # block first; the new multiscales layout carries no bin geometry.
    ngff_bins = _ngff_bins(root_attrs)

    for path, node in root.members(max_depth=None):
        attrs = dict(node.attrs)
        new, role = _migrate_node(path, attrs, ngff_bins)
        if new is None:
            continue
        counts[role] += 1
        if not dry_run:
            node.attrs.put(new)

    new_root = _migrate_root(root_attrs)
    counts["root"] = 1
    if not dry_run:
        root.attrs.put(new_root)
    return counts


def _migrate_node(
    path: str, attrs: dict[str, Any], ngff_bins: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, Any] | None, str]:
    if _LEGACY_LEVEL_KEY in attrs:
        return _migrate_level(path, attrs, ngff_bins.get(path)), "levels"
    if "zv_array" in attrs or "zvf_array" in attrs:
        flat = dict(attrs)
        if "zvf_array" in flat and "zv_array" not in flat:
            flat["zv_array"] = flat["zvf_array"]
        return zvc.wrap_array_meta(flat), "arrays"
    if "parametric_types" in attrs:
        out = {k: v for k, v in attrs.items() if k != "parametric_types"}
        out[zvc.ZV_KEY] = {"kind": "parametric", "parametric_types": attrs["parametric_types"]}
        return zvc.with_registration(out), "other"
    return None, "other"


def _migrate_level(
    path: str, attrs: dict[str, Any], bins: dict[str, list[float]] | None,
) -> dict[str, Any]:
    out = dict(attrs)
    block = dict(out.pop(_LEGACY_LEVEL_KEY) or {})
    presence = out.pop(_LEGACY_PRESENCE_KEY, None)
    if presence is not None:
        block["presence"] = presence
    if bins and not (block.get("bin_shape") or block.get("bin_ratio")):
        block.update(bins)
    out[zvc.ZV_KEY] = {"kind": "level", **block}
    return zvc.with_registration(out)


def _ngff_bins(root_attrs: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """``{level path: {bin_ratio, bin_shape}}`` for coarse levels whose NGFF
    entry differs from level 0's (the pre-per-level-bins encoding)."""
    ms = root_attrs.get("multiscales")
    if not (isinstance(ms, list) and ms):
        return {}
    transforms: dict[str, tuple[list[float] | None, list[float] | None]] = {}
    for ds in ms[0].get("datasets") or []:
        scale = translation = None
        for t in ds.get("coordinateTransformations") or []:
            if t.get("type") == "scale":
                scale = [float(x) for x in t.get("scale") or []]
            elif t.get("type") == "translation":
                translation = [float(x) for x in t.get("translation") or []]
        transforms[str(ds.get("path"))] = (scale, translation)
    level0 = transforms.get("0")
    out: dict[str, dict[str, list[float]]] = {}
    for path, (scale, translation) in transforms.items():
        if path == "0" or scale is None or (scale, translation) == level0:
            continue
        entry: dict[str, list[float]] = {"bin_ratio": scale}
        if translation is not None:
            entry["bin_shape"] = [2.0 * t for t in translation]
        out[path] = entry
    return out


def _migrate_root(attrs: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    zv = dict(out.pop(zvc.ZV_KEY))
    ms = out.pop("multiscales", None)

    axes: list[dict[str, Any]] = []
    levels: list[str] = []
    if isinstance(ms, list) and ms:
        axes = [dict(a) for a in ms[0].get("axes") or []]
        levels = [str(ds.get("path")) for ds in ms[0].get("datasets") or [] if "path" in ds]
    if not levels:
        levels = ["0"]

    crs = zv.pop("crs", None)
    zv = {"kind": "root", **zv, "zv_version": FORMAT_VERSION, "axes": axes}
    out[zvc.ZV_KEY] = zv
    out["multiscales"] = {"layout": [{"asset": lv} for lv in sorted(levels, key=int)]}
    extra: tuple[dict[str, str], ...] = (zvc.MULTISCALES_CMO,)
    if crs is not None:
        out[zvc.PROJ_PROJJSON_KEY] = crs
        extra += (zvc.PROJ_CMO,)
    return zvc.with_registration(out, full=True, extra=extra)
