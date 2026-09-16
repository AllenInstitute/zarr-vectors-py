"""The ``ome`` root-attribute block, so a store can join an OME collection.

OME-Zarr RFC 8 gives NGFF a common **Node** interface and a ``collection``
node type whose children may be named by ``Path`` — which is how a manifest
elsewhere points at this store::

    {"type": "collection", "name": "skeleton",
     "path": {"type": "zarr", "path": "./skeleton.zarrvectors"}}

A resolver following that path fetches this store's root ``zarr.json`` and
looks for an ``ome`` object that is a legal node.  Without one it finds
``zarr_vectors`` and ``multiscales``, neither of which it knows, and
resolution fails.  Supplying the block is the whole of what membership
requires.

**This module adds; it moves nothing.**  ``zarr_vectors`` and
``multiscales`` are untouched and remain the source of truth for every
field this package reads.  The block is a second, parallel description
aimed at a reader that speaks RFC 8 and nothing else.  Re-seating the
format *on* RFC 8 — dropping the repurposed ``coordinateTransformations``,
minting node types for the arrays, collapsing the attribute discriminators
— is the separate 0.10.0 change, and this block is forward-compatible with
it: every key here survives that change, and the level entries become
``collection`` nodes carrying a ``path``.

Two shapes are written:

``nodes``
    One leaf per resolution level, typed ``zv:level``.  A collection MUST
    carry ``nodes`` or ``path`` (not both), so the level list cannot be
    omitted.  Declaring levels as *prefixed leaf* nodes with no ``path``
    is what stops the cascade: RFC 8 constrains the structure of its own
    node types, not of prefixed ones, so no level group and no array needs
    a block of its own.  The cost is that a generic tool cannot descend
    into a level, which costs nothing today — there is nothing in a level
    it could render.

``attributes.scene``
    An RFC 8 ``scene`` declaring one RFC 5 coordinate system, ``world``,
    carrying the store's axes and units.  This is what makes membership
    useful rather than merely legal: with it a viewer can place this store
    beside an image pyramid.  ZV vertices are already stored in world
    coordinates, so there are no edges and
    ``coordinateTransformations`` is empty.

Nothing here reads the bare-root ``multiscales`` block, whose ``scale`` is
a dimensionless bin ratio sitting in a slot NGFF defines as
world-units-per-index.  That block is why ``world`` is declared afresh
rather than derived from a transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from zarr_vectors.core.group import Group

__all__ = [
    "OME_ATTRS_KEY",
    "OME_VERSION",
    "WORLD",
    "ZV_PREFIX",
    "build_root_node",
    "derive_store_name",
    "read_root_node",
    "refresh_root_node",
]

#: Root-attribute key the block lives under.  Reserved by NGFF 0.5+.
OME_ATTRS_KEY: str = "ome"

OME_VERSION: str = "0.6"
"""NGFF version stamped on the node.

The one field here that cannot yet be got right.  RFC 8 is a proposal, so
the version it lands under is not settled; ``"0.5"`` is the current NGFF
release and the version RFC 8's own examples carry.  It is a single
constant precisely so that correcting it is a one-line change.

RFC 8 requires ``version`` on the root node of a document and forbids it
on any other, so it appears here and nowhere else in the store.
"""

#: Extension prefix registered for this format.  RFC 8 reserves unprefixed
#: identifiers for the core spec and lets a third party use a prefixed one
#: without going through the OME RFC process.
ZV_PREFIX: str = "zv"

#: Node type of a resolution-level entry in the root's ``nodes`` list.
#: Prefixed, and therefore a leaf whose structure RFC 8 does not constrain.
NODE_TYPE_LEVEL: str = f"{ZV_PREFIX}:level"

#: Node type of the store root.  Core, not prefixed: a ZV store really is
#: a collection of levels.
NODE_TYPE_COLLECTION: str = "collection"

#: Name of the one coordinate system every ZV store declares.  Vertices are
#: stored in it directly.
WORLD: str = "world"

#: Used when a store's name cannot be derived from its URL (a store opened
#: through a backend whose ``repr`` is not a path).  RFC 8 requires a
#: non-empty name; this keeps the block legal rather than pretty.
DEFAULT_STORE_NAME: str = "zarr_vectors"

#: Store-path suffixes stripped when deriving a name.  Nothing in the
#: format reads the extension, so both spellings mean the same thing.
_NAME_SUFFIXES: tuple[str, ...] = (".zarrvectors", ".zv", ".zarr")

#: Axis keys copied into a coordinate system.  RFC 5 also defines
#: ``discrete`` and ``longName``; ZV axes carry neither today, and copying
#: only what is known keeps a stray key out of the declaration.
_AXIS_KEYS: tuple[str, ...] = ("name", "type", "unit")


def derive_store_name(url: str) -> str:
    """A human-readable store name from its URL.

    The last path segment with a store suffix stripped, so
    ``file:///data/minnie65/skeleton.zarrvectors`` gives ``"skeleton"``.
    Falls back to :data:`DEFAULT_STORE_NAME` when the URL carries no
    usable segment, which is the case for a backend whose ``repr`` stands
    in for a path.

    The name is not an identifier.  A manifest referencing this store
    supplies its own ``name`` for the node, and that is what addresses the
    store within that collection; this one is what the store calls itself
    when nothing else does.
    """
    text = str(url or "").strip()
    if not text:
        return DEFAULT_STORE_NAME

    # ``urlparse`` on a bare Windows path reads ``C:`` as the scheme, so
    # only take the parsed path when there is a real scheme to speak of.
    parsed = urlparse(text)
    candidate = unquote(parsed.path) if len(parsed.scheme) > 1 else text

    segment = candidate.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    for suffix in _NAME_SUFFIXES:
        if len(segment) > len(suffix) and segment.lower().endswith(suffix):
            segment = segment[: -len(suffix)]
            break
    segment = segment.strip()
    return segment or DEFAULT_STORE_NAME


def _clean_axes(axes: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Axis descriptors as an RFC 5 coordinate system wants them.

    Copies ``name`` / ``type`` / ``unit``, dropping an empty ``unit``:
    NGFF requires UDUNITS-2 names and rejects a placeholder, so an axis
    with no declared unit must carry no key rather than an empty string.
    """
    out: list[dict[str, str]] = []
    for i, axis in enumerate(axes or []):
        cleaned: dict[str, str] = {}
        for key in _AXIS_KEYS:
            value = axis.get(key)
            if value:
                cleaned[key] = str(value)
        cleaned.setdefault("name", f"dim{i}")
        cleaned.setdefault("type", "space")
        out.append(cleaned)
    return out


def build_scene(axes: list[dict[str, str]] | None) -> dict[str, Any]:
    """The ``scene`` attribute: one coordinate system, ``world``, no edges.

    ZV vertices are stored in world coordinates already, so there is
    nothing to transform and ``coordinateTransformations`` is empty.  The
    list is written rather than omitted because its emptiness is the
    claim: a reader learns the store needs no edge, instead of having to
    guess whether one is missing.
    """
    return {
        "coordinateSystems": [
            {"name": WORLD, "axes": _clean_axes(axes)},
        ],
        "coordinateTransformations": [],
    }


def build_root_node(
    *,
    name: str,
    axes: list[dict[str, str]] | None,
    levels: list[int],
) -> dict[str, Any]:
    """The complete ``ome`` block for a store root.

    Args:
        name: Non-empty store name.  RFC 8 requires one on every node.
        axes: NGFF axis descriptors, as stored in ``multiscales[0].axes``.
        levels: Resolution level indices present in the store.

    Returns:
        A dict to write at root ``attributes.ome``.
    """
    if not name:
        raise ValueError("an RFC 8 node name must be non-empty")
    return {
        "version": OME_VERSION,
        "type": NODE_TYPE_COLLECTION,
        "name": str(name),
        "attributes": {"scene": build_scene(axes)},
        "nodes": [
            {"type": NODE_TYPE_LEVEL, "name": str(level)}
            for level in sorted(levels)
        ],
    }


def read_root_node(attrs: dict[str, Any]) -> dict[str, Any] | None:
    """The store's ``ome`` node from a root-attrs dict, or ``None``.

    ``None`` for a store written before 0.9.2, which is not an error:
    the block is additive and every field this package reads lives
    elsewhere.  :func:`refresh_root_node` is how such a store gets one.
    """
    node = attrs.get(OME_ATTRS_KEY)
    return dict(node) if isinstance(node, dict) else None


def refresh_root_node(
    root: Group,
    *,
    name: str | None = None,
    axes: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Rebuild the root ``ome`` block from what is on disk, and write it.

    The level list is derived from the store's level groups rather than
    maintained alongside them, so it cannot drift: every path that adds or
    removes a level calls this and the answer comes from the same place a
    reader would look.  (The same reasoning as
    :func:`zarr_vectors.building.refresh_arrays_present`.)

    Args:
        root: Store root group.
        name: Store name.  ``None`` keeps the name already recorded, and
            derives one from the store URL when there is none — so a name
            set once is not overwritten by a later level being added.
        axes: NGFF axis descriptors.  ``None`` reads them from the
            existing ``world`` declaration, falling back to
            ``multiscales[0].axes``.

    Returns:
        The block as written.
    """
    from zarr_vectors.core.store import list_resolution_levels

    attrs = root.attrs.to_dict()
    existing = read_root_node(attrs) or {}

    if name is None:
        name = existing.get("name") or derive_store_name(root.url)

    if axes is None:
        axes = _axes_from_node(existing)
    if axes is None:
        multiscales = attrs.get("multiscales") or []
        if multiscales and isinstance(multiscales, list):
            axes = multiscales[0].get("axes")

    node = build_root_node(
        name=name, axes=axes, levels=list_resolution_levels(root),
    )

    # Preserve anything a caller put on the node that this function does
    # not own -- a ``zv:companionImage`` reference, say.  Only the four
    # keys built above are authoritative.
    merged_attributes = dict(existing.get("attributes") or {})
    merged_attributes.update(node["attributes"])
    node = {**existing, **node, "attributes": merged_attributes}

    root.attrs.update({OME_ATTRS_KEY: node})
    return node


def _axes_from_node(node: dict[str, Any]) -> list[dict[str, str]] | None:
    """The ``world`` axes already declared on a node, if any."""
    scene = (node.get("attributes") or {}).get("scene") or {}
    for system in scene.get("coordinateSystems") or []:
        if system.get("name") == WORLD:
            axes = system.get("axes")
            return list(axes) if axes else None
    return None
