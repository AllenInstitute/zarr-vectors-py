"""On-disk path helpers for the 0.4+ multiscale links layout.

Pre-0.4 the link-family arrays (``links``, ``cross_chunk_links``,
``link_attributes``) lived directly under the resolution-level group.
The 0.4 layout interposes a ``<level_delta>`` path segment so cross-
pyramid-level edges become first-class.  v0.8 then partitions the
``cross_chunk_links/<delta>/`` family into per-(sorted-unique-chunks)
leaves:

    /resolution_N/links/<delta>/<chunk_key>
    /resolution_N/cross_chunk_links/<delta>/<chunk_sorted_0>/.../<chunk_sorted_{K-1}>/data
    /resolution_N/link_attributes/<name>/<delta>/<chunk_key>
    /resolution_N/cross_chunk_link_attributes/<name>/<delta>/<chunk_sorted_0>/.../<chunk_sorted_{K-1}>/data

The delta segment is signed: ``"0"``, ``"+1"``, ``"-1"``, ``"+2"``, ...
Endpoints in a ``links/<delta>/...`` array have source side at the
owning level and target side at ``this_level + delta``.

This module is intentionally values-only: callers compose paths via
the helpers below and never assemble the raw f-strings inline, so the
delta convention and the K-deep leaf-path shape each have exactly one
definition.
"""

from __future__ import annotations

from zarr_vectors.constants import (
    CROSS_CHUNK_LINK_ATTRIBUTES,
    CROSS_CHUNK_LINKS,
    LINK_ATTRIBUTES,
    LINKS,
)
from zarr_vectors.typing import ChunkCoords


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


def links_path(delta: int = 0) -> str:
    """Path of a ``links/<delta>/`` array within a resolution level."""
    return f"{LINKS}/{format_delta(delta)}"


def cross_chunk_links_path(delta: int = 0) -> str:
    """Path of a ``cross_chunk_links/<delta>/`` array within a level."""
    return f"{CROSS_CHUNK_LINKS}/{format_delta(delta)}"


def link_attributes_path(name: str, delta: int = 0) -> str:
    """Path of a ``link_attributes/<name>/<delta>/`` array within a level."""
    return f"{LINK_ATTRIBUTES}/{name}/{format_delta(delta)}"


def cross_chunk_link_attributes_path(name: str, delta: int = 0) -> str:
    """Path of a ``cross_chunk_link_attributes/<name>/<delta>/`` array."""
    return f"{CROSS_CHUNK_LINK_ATTRIBUTES}/{name}/{format_delta(delta)}"


# -----------------------------------------------------------------------
# v0.8 partitioned cross-chunk-link encoding helpers.
#
# Records are partitioned by the sorted unique chunks each record touches.
# The on-disk layout is a single 1-D vlen-bytes zarr Array per
# ``cross_chunk_links/<delta>/`` (and ``cross_chunk_link_attributes/<name>/<delta>/``),
# where each cell holds one leaf's payload.  The sidecar
# ``.zattrs.leaf_index`` maps the sorted-chunks tuple to a cell ordinal,
# so a reader looking up records between chunks A and B sorts (A, B) and
# indexes into the Array via the leaf_index map.
# -----------------------------------------------------------------------


def _chunk_segment(coords: ChunkCoords) -> str:
    """Dot-separated chunk-coord string for a single chunk segment.

    ``(0, 1, 2)`` → ``"0.1.2"``.  Mirrors ``_chunk_key()`` in
    ``core/arrays.py``; lives here so path / index helpers don't import
    from arrays.
    """
    return ".".join(str(int(c)) for c in coords)


def _parse_chunk_segment(segment: str) -> ChunkCoords:
    """Inverse of :func:`_chunk_segment` — ``"0.1.2"`` → ``(0, 1, 2)``."""
    return tuple(int(x) for x in segment.split("."))


def sort_chunks_lex(chunks: tuple[ChunkCoords, ...]) -> tuple[ChunkCoords, ...]:
    """Lex-sort a tuple of chunk-coord tuples (element-wise int compare).

    For a record with endpoints in chunks ``[A, B, A, C]`` the leaf key
    is ``sort_chunks_lex((A, B, C))`` → the K = 3 distinct chunks in
    lex-min order.  Callers first deduplicate via ``set()`` and then
    pass the unique tuple here.
    """
    return tuple(sorted(chunks))


def leaf_index_key(sorted_chunks: tuple[ChunkCoords, ...]) -> str:
    """Serialize a sorted-unique-chunks tuple to its leaf-index key string.

    The on-disk leaf_index dict is keyed by ``"|"``-joined chunk
    segments so JSON round-trips cleanly (tuple keys aren't JSON-native).
    Mirrors the inverse :func:`parse_leaf_index_key`.

    Examples::

        leaf_index_key(((0, 0, 0),))                    → "0.0.0"
        leaf_index_key(((0, 0, 0), (1, 0, 0)))          → "0.0.0|1.0.0"
        leaf_index_key(((0, 0, 0), (0, 1, 0), (1, 0, 0))) → "0.0.0|0.1.0|1.0.0"
    """
    if not sorted_chunks:
        raise ValueError("leaf_index_key: sorted_chunks must be non-empty")
    return "|".join(_chunk_segment(c) for c in sorted_chunks)


def parse_leaf_index_key(key: str) -> tuple[ChunkCoords, ...]:
    """Inverse of :func:`leaf_index_key`."""
    if not key:
        raise ValueError("parse_leaf_index_key: empty key")
    return tuple(_parse_chunk_segment(seg) for seg in key.split("|"))
