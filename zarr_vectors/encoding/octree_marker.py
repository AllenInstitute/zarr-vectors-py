"""The ``zarr_vectors_octree_chunks`` codec: an octree level's per-chunk arrays carry it.

An octree-chunked level (:mod:`zarr_vectors.octree_chunks`) stores each
leaf at its anchor cell, so its ``vertices`` array is a valid zarr array
whose cells are *not* the uniform cells its grid suggests.  A reader that
does not know octree chunks -- an older zarr-vectors, or anything else
that reads zarr -- would take each leaf for one cell and return the
wrong points and edges without an error.  Metadata cannot stop it: such
a reader ignores level attributes and capabilities it does not know, and
zarr-vectors before octree chunks never refuses a newer ``zv_version``.

Zarr itself can.  A codec in an array's chain is something every zarr
reader must understand to decode a single cell, and one it does not know
fails the array at open (``UnknownCodecError`` in zarr-python).  So the
conversion to octree chunks puts this codec in the cell codecs of every
per-chunk array it writes, and the conversion back writes them without
it.  It changes no bytes (encode and decode are the identity), so the
cells are exactly what they would be without it; registering it is all
a reader that understands octree chunks needs, and importing
:mod:`zarr_vectors` does that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zarr.abc.codec import BytesBytesCodec
from zarr.core.common import parse_named_configuration
from zarr.registry import register_codec

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

#: The codec's name in ``zarr.json``.
OCTREE_MARKER_CODEC = "zarr_vectors_octree_chunks"


@dataclass(frozen=True)
class OctreeChunksMarker(BytesBytesCodec):
    """Identity bytes-to-bytes codec marking an octree level's arrays."""

    is_fixed_size = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OctreeChunksMarker:
        parse_named_configuration(data, OCTREE_MARKER_CODEC, require_configuration=False)
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {"name": OCTREE_MARKER_CODEC}

    def _decode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer:
        return chunk_bytes

    def _encode_sync(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        return chunk_bytes

    async def _encode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> Buffer | None:
        return chunk_bytes

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length


def is_marker(codec: Any) -> bool:
    """Whether ``codec`` (a codec object, or its ``zarr.json`` dict) is the marker."""
    if isinstance(codec, dict):
        return codec.get("name") == OCTREE_MARKER_CODEC
    return isinstance(codec, OctreeChunksMarker)


register_codec(OCTREE_MARKER_CODEC, OctreeChunksMarker)

__all__ = ["OCTREE_MARKER_CODEC", "OctreeChunksMarker", "is_marker"]
