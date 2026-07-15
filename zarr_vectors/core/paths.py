"""On-disk path helpers for the links layout.

Connectivity lives in a single family under a resolution-level group,
parameterised by a signed level-delta segment and a relative-offset
segment:

    /resolution_N/links/<delta>/<offsets>/<chunk_key>
    /resolution_N/link_attributes/<name>/<delta>/<offsets>/<chunk_key>

The delta segment is signed: ``"0"``, ``"+1"``, ``"-1"``, ``"+2"``, ...
It says how many pyramid levels the record spans.

The ``<offsets>`` segment says where the *other* endpoints sit relative
to the record's **source** chunk, which is the array cell holding it.
It carries ``link_width - 1`` offsets, each a ``sid_ndim``-tuple:

    links/0/0.0.0/           intra-chunk edges (both endpoints in the cell)
    links/0/0.0.+1/          edges to the neighbour one chunk along +z
    links/0/0.0.+1_0.+1.0/   triangle spanning the cell, +z, and +y
    links/0/self/            link_width=1 (parent refs); no other endpoint

Because the relationship is factored into the path, each array is a
plain rank-D vlen array over the level's chunk grid — one cell per
source chunk — and record ``vi_k`` is local to chunk ``src + o_k``
(with ``o_0 = 0`` by definition).  An intra-chunk link is simply a link
whose offsets are all zero, which is why there is no separate
``cross_chunk_links`` family.

This module is intentionally values-only: callers compose paths via the
helpers below and never assemble the raw f-strings inline, so the
delta and offset conventions have exactly one definition each.
"""

from __future__ import annotations

from typing import Sequence

from zarr_vectors.constants import (
    LINK_ATTRIBUTES,
    LINKS,
)
from zarr_vectors.typing import ChunkCoords

#: Path segment used when ``link_width == 1`` and there are therefore no
#: relative offsets to encode.  Keeps ``links/<delta>/`` uniformly a group.
SELF_OFFSETS_SEGMENT = "self"

_OFFSET_SEP = "_"
_COMPONENT_SEP = "."


def format_delta(delta: int) -> str:
    """Format a level delta as its on-disk path segment.

    ``0 -> "0"``, ``+N -> "+N"``, ``-N -> "-N"``.  The leading ``+``
    is preserved so a directory listing distinguishes positive deltas
    from the unsigned ``0`` at a glance.
    """
    if delta == 0:
        return "0"
    return f"+{delta}" if delta > 0 else str(delta)


def parse_delta(segment: str) -> int:
    """Inverse of :func:`format_delta`.

    Accepts ``"0"``, ``"+N"``, ``"-N"``.  Raises ``ValueError`` for
    anything else (including stray whitespace or empty input) so a
    malformed on-disk listing fails fast.
    """
    if segment == "0":
        return 0
    if not segment or segment[0] not in "+-":
        raise ValueError(f"invalid level-delta segment: {segment!r}")
    return int(segment)


def format_offsets(offsets: Sequence[ChunkCoords]) -> str:
    """Format ``link_width - 1`` relative chunk offsets as a path segment.

    Each offset is a ``sid_ndim``-tuple of signed ints written with the
    same convention as :func:`format_delta` (``"0"``, ``"+1"``, ``"-1"``),
    components joined by ``"."`` and offsets joined by ``"_"``.  An empty
    sequence (``link_width == 1``) formats as :data:`SELF_OFFSETS_SEGMENT`
    so ``links/<delta>/`` stays uniformly a group.

    Offsets are relative to the record's source chunk; the implicit
    ``o_0`` (the source itself) is never encoded.
    """
    if not offsets:
        return SELF_OFFSETS_SEGMENT
    return _OFFSET_SEP.join(
        _COMPONENT_SEP.join(format_delta(int(c)) for c in offset)
        for offset in offsets
    )


def parse_offsets(
    segment: str, *, sid_ndim: int, link_width: int,
) -> tuple[ChunkCoords, ...]:
    """Inverse of :func:`format_offsets`.

    Returns the ``link_width - 1`` offset tuples.  Raises ``ValueError``
    when the segment's arity does not match ``sid_ndim`` /
    ``link_width``, so a malformed on-disk listing fails fast rather
    than decoding to the wrong geometry.
    """
    if segment == SELF_OFFSETS_SEGMENT:
        if link_width != 1:
            raise ValueError(
                f"offsets segment {segment!r} implies link_width=1, "
                f"got link_width={link_width}"
            )
        return ()
    if link_width == 1:
        raise ValueError(
            f"link_width=1 requires the {SELF_OFFSETS_SEGMENT!r} segment, "
            f"got {segment!r}"
        )

    parts = segment.split(_OFFSET_SEP)
    expected = link_width - 1
    if len(parts) != expected:
        raise ValueError(
            f"offsets segment {segment!r} has {len(parts)} offsets; "
            f"expected {expected} (link_width={link_width} - 1)"
        )
    out: list[ChunkCoords] = []
    for part in parts:
        comps = part.split(_COMPONENT_SEP)
        if len(comps) != sid_ndim:
            raise ValueError(
                f"offset {part!r} in segment {segment!r} has {len(comps)} "
                f"components; expected sid_ndim={sid_ndim}"
            )
        out.append(tuple(parse_delta(c) for c in comps))
    return tuple(out)


def intra_offsets(sid_ndim: int, link_width: int) -> tuple[ChunkCoords, ...]:
    """The all-zero offsets identifying the intra-chunk array.

    ``links/<delta>/<intra_offsets(...)>/`` is the array holding records
    whose endpoints all live in the same chunk — the family that was a
    standalone ``links/<delta>/`` array before the offset layout.
    """
    zero: ChunkCoords = tuple(0 for _ in range(sid_ndim))
    return tuple(zero for _ in range(max(0, link_width - 1)))


def is_intra(offsets: Sequence[ChunkCoords]) -> bool:
    """Whether every offset is zero (all endpoints in the source chunk).

    ``link_width == 1`` (empty offsets) counts as intra: the single
    endpoint is by definition in the source chunk.
    """
    return all(all(int(c) == 0 for c in offset) for offset in offsets)


def links_group_path(delta: int = 0) -> str:
    """Path of the ``links/<delta>/`` **group** within a resolution level.

    The group's children are one rank-D vlen array per distinct offsets
    segment; see :func:`links_path`.
    """
    return f"{LINKS}/{format_delta(delta)}"


def links_path(delta: int, offsets: Sequence[ChunkCoords]) -> str:
    """Path of a ``links/<delta>/<offsets>/`` array within a level.

    ``offsets`` is required: under the offset layout there is no single
    array at ``links/<delta>/`` to address, only the group
    (:func:`links_group_path`).
    """
    return f"{links_group_path(delta)}/{format_offsets(offsets)}"


def link_attributes_group_path(name: str, delta: int = 0) -> str:
    """Path of the ``link_attributes/<name>/<delta>/`` **group**."""
    return f"{LINK_ATTRIBUTES}/{name}/{format_delta(delta)}"


def link_attributes_path(
    name: str, delta: int, offsets: Sequence[ChunkCoords],
) -> str:
    """Path of a ``link_attributes/<name>/<delta>/<offsets>/`` array.

    Mirrors :func:`links_path` exactly — same delta, same offsets
    segment — so attribute cells align 1:1 with link cells.
    """
    return f"{link_attributes_group_path(name, delta)}/{format_offsets(offsets)}"
