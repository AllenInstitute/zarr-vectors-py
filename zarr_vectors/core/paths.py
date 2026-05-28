"""On-disk path helpers for the multiscale links layout.

The link-family arrays live under a resolution-level group, each
parameterised by a signed level-delta segment:

    /resolution_N/links/<delta>/<chunk_key>
    /resolution_N/cross_chunk_links/<delta>/<cell_key>
    /resolution_N/link_attributes/<name>/<delta>/<chunk_key>
    /resolution_N/cross_chunk_link_attributes/<name>/<delta>/<cell_key>

The delta segment is signed: ``"0"``, ``"+1"``, ``"-1"``, ``"+2"``, ...

For ``cross_chunk_links/<delta>/`` (and the parallel attribute array),
the ``<cell_key>`` is the dotted concatenation of the ``link_width``
canonical-sorted chunk coordinates that anchor each record (6-D for
edges, 9-D for triangles, 12-D for quads, 3-D for parent refs at
``link_width=1``).  Each cell holds only the records spanning exactly
that L-tuple of chunks, so reads scale with cell size and not store
size.

This module is intentionally values-only: callers compose paths via
the helpers below and never assemble the raw f-strings inline, so the
delta convention has exactly one definition.
"""

from __future__ import annotations

from typing import Sequence

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


def format_cell_key(chunks: Sequence[ChunkCoords]) -> str:
    """Concatenate L pre-canonical-sorted chunk coords into a single key.

    For an array with ``link_width = L`` and ``sid_ndim = D`` the key
    has ``L * D`` dotted components.  Callers must canonical-sort the
    L chunk-tuples before calling — this helper does not enforce
    ordering, only formatting.
    """
    if not chunks:
        raise ValueError("format_cell_key requires at least one chunk")
    parts: list[str] = []
    for c in chunks:
        parts.extend(str(x) for x in c)
    return ".".join(parts)


def parse_cell_key(
    key: str, *, sid_ndim: int, link_width: int,
) -> tuple[ChunkCoords, ...]:
    """Inverse of :func:`format_cell_key`.

    Splits a ``cross_chunk_links`` cell key into its L canonical-sorted
    chunk-coord tuples.  Raises ``ValueError`` if the dotted-component
    count does not equal ``sid_ndim * link_width``.
    """
    parts = key.split(".")
    expected = sid_ndim * link_width
    if len(parts) != expected:
        raise ValueError(
            f"cell key {key!r} has {len(parts)} components; expected "
            f"{expected} (sid_ndim={sid_ndim} * link_width={link_width})"
        )
    ints = [int(p) for p in parts]
    return tuple(
        tuple(ints[i * sid_ndim : (i + 1) * sid_ndim])
        for i in range(link_width)
    )


def cross_chunk_links_cell_path(delta: int, key: str) -> str:
    """Path of a single cell under ``cross_chunk_links/<delta>/``."""
    return f"{CROSS_CHUNK_LINKS}/{format_delta(delta)}/{key}"


def cross_chunk_link_attributes_cell_path(
    name: str, delta: int, key: str,
) -> str:
    """Path of a single cell under ``cross_chunk_link_attributes/<name>/<delta>/``."""
    return f"{CROSS_CHUNK_LINK_ATTRIBUTES}/{name}/{format_delta(delta)}/{key}"
