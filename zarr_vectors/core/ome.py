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

    The level leaves are the only entries this package owns.  Anything
    else in the list — a node some other tool put there, such as a
    back-reference to the collection this store sits in — is *foreign*,
    and :func:`refresh_root_node` carries it through a level refresh
    unchanged.  See :func:`is_owned_node` for the exact rule.

``attributes.scene``
    An RFC 8 ``scene`` declaring one RFC 5 coordinate system, ``world``,
    carrying the store's axes and units.  This is what makes membership
    useful rather than merely legal: with it a viewer can place this store
    beside an image pyramid.  ZV vertices are already stored in world
    coordinates, so there are no edges and
    ``coordinateTransformations`` is empty.

    The system carries ``"id": "world"`` as well as ``"name": "world"``.
    RFC 8 identifies a coordinate system by ``id`` — ``name`` is optional
    and descriptive there — so the ``id`` is what lets a collection
    *elsewhere* bind to this frame with a ``Reference``::

        {"path": {"type": "zarr", "path": "./skeleton.zarrvectors"},
         "id": "world"}

    ``name`` stays because it is the RFC 5 spelling of the identifier,
    and a reader written against that draft looks for it.

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
    "is_owned_node",
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

#: Identifier of the one coordinate system every ZV store declares.
#: Vertices are stored in it directly.  Written as both ``id`` and
#: ``name``: RFC 8 identifies a coordinate system by ``id`` -- it is what a
#: ``Reference`` from another document binds to -- and leaves ``name``
#: descriptive, while RFC 5 readers know only ``name``.
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

    The system carries ``id`` as well as ``name`` so a collection
    elsewhere can bind an edge to it -- ``{"path": ..., "id": "world"}``
    -- instead of redeclaring the frame.  See :data:`WORLD`.
    """
    return {
        "coordinateSystems": [
            {"id": WORLD, "name": WORLD, "axes": _clean_axes(axes)},
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

    **What is rebuilt and what is kept.**  This function owns the node's
    ``version``, ``type``, ``name``, ``attributes.scene``, and the
    *level entries* of ``nodes``.  Everything else is kept as found:

    * any other key of ``attributes`` (a ``zv:companionImage``, say);
    * any other top-level key of the node;
    * every **foreign** child in ``nodes`` — one for which
      :func:`is_owned_node` is false — verbatim and in its original
      relative order, after the level entries.

    So a tool that files its own node here (a back-reference from the
    store to the collection that contains it is the motivating case)
    keeps it across ``build_pyramid``, ``remove_resolution_level`` and
    :func:`zarr_vectors.building.stamp_ome_node`, all of which call this.

    Two things are *not* kept, both with a :class:`UserWarning` naming
    them, because keeping them would make the written node illegal:

    * a foreign child that is not a JSON object (it is not a node);
    * a foreign child whose ``name`` equals a level's name.  RFC 8
      requires child names to be unique within a collection, and the
      level list is the side derived from disk, so it wins.  Level names
      are the decimal level indices, so a foreign node never collides
      with one unless it is named like a level.

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
    # not own -- a ``zv:companionImage`` reference, say.  Only the keys
    # built above are authoritative, and of ``nodes`` only the level
    # entries.
    merged_attributes = dict(existing.get("attributes") or {})
    merged_attributes.update(node["attributes"])
    merged_nodes = node["nodes"] + _foreign_nodes(
        existing.get("nodes"),
        level_names={child["name"] for child in node["nodes"]},
    )
    node = {
        **existing, **node,
        "attributes": merged_attributes, "nodes": merged_nodes,
    }

    root.attrs.update({OME_ATTRS_KEY: node})
    return node


def is_owned_node(child: Any) -> bool:
    """Whether a child of the root's ``nodes`` list is one this package writes.

    **The rule:** a child is owned iff it is a JSON object whose ``type``
    is exactly :data:`NODE_TYPE_LEVEL` (``"zv:level"``).  Owned children
    are regenerated from the level groups on disk by every refresh, so a
    stale one disappears and anything a caller added *to* one is lost.
    Every other child is foreign and is carried through unchanged — a
    core ``collection`` or ``multiscale``, another tool's prefixed type,
    or a ``zv:``-prefixed type other than ``zv:level``.

    Keyed on the type rather than on the ``zv:`` prefix as a whole for
    the same reason the attributes merge is keyed on ``scene`` rather
    than on every ``zv:`` key: the refresh owns what it builds and
    nothing more, so a ``zv:`` identifier some other part of this
    package (or a caller) files here is not silently discarded.

    When the level entries become ``collection`` nodes carrying a
    ``path`` (the 0.10.0 re-seating described above), a core type can no
    longer say "this is a level", and this predicate must grow a marker
    those entries carry.  That is why the rule lives in one function.
    """
    return isinstance(child, dict) and child.get("type") == NODE_TYPE_LEVEL


def _foreign_nodes(
    existing: Any, *, level_names: set[str],
) -> list[dict[str, Any]]:
    """The children of an existing ``nodes`` list that a refresh keeps.

    See :func:`refresh_root_node` for why the two dropped cases are
    dropped rather than kept.
    """
    import warnings

    kept: list[dict[str, Any]] = []
    for child in existing if isinstance(existing, list) else []:
        if is_owned_node(child):
            continue
        if not isinstance(child, dict):
            warnings.warn(
                f"dropping {child!r} from the root 'ome' node's children: "
                f"an RFC 8 node is a JSON object",
                UserWarning, stacklevel=3,
            )
            continue
        if child.get("name") in level_names:
            warnings.warn(
                f"dropping the foreign {child.get('type')!r} node named "
                f"{child.get('name')!r} from the root 'ome' node: a "
                f"resolution level has that name, and RFC 8 requires "
                f"child names to be unique within a collection",
                UserWarning, stacklevel=3,
            )
            continue
        kept.append(dict(child))
    return kept


def _axes_from_node(node: dict[str, Any]) -> list[dict[str, str]] | None:
    """The ``world`` axes already declared on a node, if any.

    Matched by ``id`` or by ``name``: a store stamped before the ``id``
    was added declares the system by ``name`` alone.
    """
    scene = (node.get("attributes") or {}).get("scene") or {}
    for system in scene.get("coordinateSystems") or []:
        if WORLD in (system.get("id"), system.get("name")):
            axes = system.get("axes")
            return list(axes) if axes else None
    return None
