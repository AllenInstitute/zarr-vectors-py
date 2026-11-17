"""On-disk metadata layout of format 0.10.0: the Zarr Vectors convention.

Format 0.10.0 stores every Zarr Vectors property of a node in one
attribute, ``zarr_vectors``, discriminated by ``kind``, and registers the
convention in ``zarr_conventions`` on every node that carries it (the Zarr
Conventions specification requires registration per node).  See
https://github.com/AllenInstitute/zarr-vectors-convention.

Inside the package, array metadata keeps its historical *flat* shape --
``{"zv_array": "vertices", "dtype": ..., "nonempty_chunks": [...]}`` --
because dozens of call sites read and write it that way.  This module is
the single translation point between that in-memory shape and the on-disk
one:

* :func:`wrap_array_meta` turns a flat dict into the attributes to write;
* :func:`unwrap_array_meta` turns stored attributes back into a flat dict;
* :func:`get_field` / :func:`set_field` / :func:`del_field` /
  :func:`has_field` read and update one field of a node in place, merging
  into the nested object instead of replacing it.

The ``zarr_vectors`` key on the store root and on level groups is a plain
dict with ``kind: "root"`` / ``kind: "level"`` and is handled by
:mod:`zarr_vectors.core.metadata`; only arrays and family groups go through
the wrap/unwrap here.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

#: The attribute that holds every Zarr Vectors property of a node.
ZV_KEY = "zarr_vectors"

#: The Zarr Conventions registration attribute.
CONVENTIONS_KEY = "zarr_conventions"

CONVENTION_UUID = "66268a61-7833-4995-bbcb-233bb417dae5"
CONVENTION_VERSION = "0.1.0"

#: Full convention metadata object, written on the store root.
CONVENTION_CMO: dict[str, str] = {
    "schema_url": (
        "https://raw.githubusercontent.com/AllenInstitute/"
        f"zarr-vectors-convention/refs/tags/v{CONVENTION_VERSION}/schema.json"
    ),
    "spec_url": (
        "https://github.com/AllenInstitute/zarr-vectors-convention/"
        f"blob/v{CONVENTION_VERSION}/README.md"
    ),
    "uuid": CONVENTION_UUID,
    "name": "zarr_vectors",
    "description": (
        "Spatially chunked, multi-resolution N-dimensional vector data "
        "(point clouds, meshes, skeletons, streamlines, graphs)"
    ),
}

#: Short registration, written on every other node.
CONVENTION_CMO_SHORT: dict[str, str] = {"uuid": CONVENTION_UUID}

MULTISCALES_CMO: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}

PROJ_CMO: dict[str, str] = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data",
}

#: Root attribute holding a CRS, from the proj convention.
PROJ_PROJJSON_KEY = "proj:projjson"

#: Keys on a ``multiscales.layout`` entry carrying a level's bin geometry.
#: The convention keeps them prefixed because they extend another
#: convention's object.
LAYOUT_BIN_RATIO_KEY = "zarr_vectors:bin_ratio"
LAYOUT_BIN_SHAPE_KEY = "zarr_vectors:bin_shape"

# In-memory ``zv_array`` tag -> on-disk ``kind``.  Two tags are renamed;
# every other tag is its own kind.
_TAG_TO_KIND: dict[str, str] = {
    "attribute": "vertex_attribute",
    "groupings_attribute": "group_attribute",
}
_KIND_TO_TAG: dict[str, str] = {v: k for k, v in _TAG_TO_KIND.items()}

#: Kinds that are root or level groups, not arrays or family groups.
STRUCTURAL_KINDS = frozenset({"root", "level", "parametric"})


def tag_to_kind(tag: str) -> str:
    return _TAG_TO_KIND.get(tag, tag)


def kind_to_tag(kind: str) -> str:
    return _KIND_TO_TAG.get(kind, kind)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _has_entry(entries: list[Any], uuid: str) -> bool:
    return any(isinstance(e, Mapping) and e.get("uuid") == uuid for e in entries)


def with_registration(
    attrs: Mapping[str, Any],
    *,
    full: bool = False,
    extra: tuple[Mapping[str, str], ...] = (),
) -> dict[str, Any]:
    """``attrs`` with Zarr Vectors (and ``extra``) registered in
    ``zarr_conventions``.  Existing entries, including other conventions',
    are kept; ours is replaced so a stale URL is not left behind."""
    out = dict(attrs)
    entries = [
        e for e in (out.get(CONVENTIONS_KEY) or [])
        if not (isinstance(e, Mapping) and e.get("uuid") == CONVENTION_UUID)
    ]
    entries.insert(0, dict(CONVENTION_CMO if full else CONVENTION_CMO_SHORT))
    for cmo in extra:
        if not _has_entry(entries, cmo["uuid"]):
            entries.append(dict(cmo))
    out[CONVENTIONS_KEY] = entries
    return out


# ---------------------------------------------------------------------------
# Array / family-group metadata
# ---------------------------------------------------------------------------


def is_array_block(block: Any) -> bool:
    """True when ``block`` is an array or family-group ``zarr_vectors``
    object (not the root's or a level's)."""
    return (
        isinstance(block, Mapping)
        and isinstance(block.get("kind"), str)
        and block["kind"] not in STRUCTURAL_KINDS
    )


def wrap_array_meta(
    flat: Mapping[str, Any],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attributes to store for an array or family group.

    ``flat`` is the in-memory shape (``zv_array`` + fields).  ``existing``
    is the node's current attributes, if any; its Zarr Vectors fields are
    kept and ``flat`` is merged over them, matching the key-by-key
    ``attrs.update`` semantics callers relied on before the fields were
    nested.  Keys that belong to other conventions or to the user stay at
    the top level.
    """
    merged = unwrap_array_meta(existing or {})
    merged.update(flat)
    return _nest(merged)


def _nest(flat: Mapping[str, Any]) -> dict[str, Any]:
    flat = dict(flat)
    tag = flat.pop("zv_array", None)
    flat.pop("zvf_array", None)  # pre-0.5 spelling, never written back
    if tag is None:
        # Not a Zarr Vectors node: store as given.
        return flat
    passthrough = {
        k: flat.pop(k) for k in list(flat)
        if k == CONVENTIONS_KEY or k == ZV_KEY or _is_foreign_key(k)
    }
    passthrough.pop(ZV_KEY, None)
    block = {"kind": tag_to_kind(str(tag)), **flat}
    return with_registration({**passthrough, ZV_KEY: block})


def _is_foreign_key(key: str) -> bool:
    """Top-level keys that are not Zarr Vectors fields: other conventions'
    prefixed keys and the package's own user-metadata block."""
    from zarr_vectors.core.user_metadata import USER_METADATA_KEY

    return ":" in key or key == USER_METADATA_KEY


def unwrap_array_meta(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """The in-memory flat dict for stored ``attrs``.

    Accepts both the nested 0.10.0 layout and an already-flat dict (an
    in-flight batch, or a node that carries no Zarr Vectors block), so it
    is safe to call on anything.
    """
    attrs = dict(attrs)
    block = attrs.pop(ZV_KEY, None)
    attrs.pop(CONVENTIONS_KEY, None)
    if not is_array_block(block):
        if block is not None:
            attrs[ZV_KEY] = block
        return attrs
    block = dict(block)
    kind = block.pop("kind")
    return {**attrs, "zv_array": kind_to_tag(kind), **block}


# ---------------------------------------------------------------------------
# One field of a node, in place
# ---------------------------------------------------------------------------


def _block(attrs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    block = attrs.get(ZV_KEY)
    return block if is_array_block(block) else None


def get_field(attrs: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read ``key`` from a node's Zarr Vectors block (or, for a node that
    has none, from its top-level attributes)."""
    block = _block(attrs)
    if block is not None:
        return block.get(key, default)
    return attrs.get(key, default)


def has_field(attrs: Mapping[str, Any], key: str) -> bool:
    block = _block(attrs)
    return key in block if block is not None else key in attrs


def field_update(attrs: Mapping[str, Any], key: str, value: Any) -> dict[str, Any]:
    """The top-level attribute update that sets one field: for a Zarr
    Vectors node, the whole rewritten block; otherwise ``{key: value}``.
    For callers that apply updates themselves (``update_attributes``)."""
    block = _block(attrs)
    if block is None:
        return {key: value}
    return {ZV_KEY: {**block, key: value}}


def set_field(attrs: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set one field without disturbing the rest of the block.  ``attrs``
    is a zarr ``Attributes`` object or a dict; the whole block is written
    back in one assignment."""
    block = _block(attrs)
    if block is None:
        attrs[key] = value
        return
    new = dict(block)
    new[key] = value
    attrs[ZV_KEY] = new


def del_field(attrs: MutableMapping[str, Any], key: str) -> None:
    block = _block(attrs)
    if block is None:
        if key in attrs:
            del attrs[key]
        return
    if key in block:
        new = dict(block)
        del new[key]
        attrs[ZV_KEY] = new
